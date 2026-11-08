#pragma once

// Static analysis of a PRX module (module-relative addresses, i.e. load base 0): function discovery and
// recovery of names from the game's own registration calls.

#include <cstdint>
#include <optional>

#include "psprecore/formats/elf.hpp"
#include "psprecore/formats/prx.hpp"
#include "psprecore/mips/allegrex.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::analysis {

class Program {
public:
    Program(const elf::File& file, prx::Module module);

    const prx::Module& module() const { return module_; }
    std::uint32_t text_begin() const { return text_begin_; }
    std::uint32_t text_end() const { return text_end_; }
    bool in_text(std::uint32_t addr) const { return addr >= text_begin_ && addr < text_end_; }
    // PROGBITS sections inside executable segments besides .text (writable code, e.g. init routines in .data).
    struct CodeSection {
        std::uint32_t begin = 0, end = 0;  // [begin, end)
    };
    const nx::vector<CodeSection>& code_sections() const { return code_sections_; }
    // .text or any such section: everywhere recompiled code may live.
    bool in_code(std::uint32_t addr) const {
        if (in_text(addr)) return true;
        for (const auto& s : code_sections_)
            if (addr >= s.begin && addr < s.end) return true;
        return false;
    }

    std::optional<std::uint32_t> word(std::uint32_t addr) const;  // file-backed memory only
    mips::Instruction insn(std::uint32_t addr) const;
    // NUL-terminated printable string at `addr` (empty if not a plausible string).
    nx::string string_at(std::uint32_t addr, std::size_t max_len = 128) const;

    // Relocation patching the word at `addr`, if any.
    const prx::Relocation* relocation_at(std::uint32_t addr) const;
    std::uint32_t segment_base(std::uint8_t index) const;

private:
    prx::Module module_;
    nx::vector<std::byte> image_;  // unrelocated loadable image, BSS zero-filled
    std::uint32_t text_begin_ = 0, text_end_ = 0;
    nx::vector<CodeSection> code_sections_;
    nx::map<std::uint32_t, std::size_t> reloc_by_addr_;
};

struct Function {
    std::uint32_t start = 0;
    std::uint32_t end = 0;  // exclusive
    nx::string name;       // empty if unknown
    enum class Origin : std::uint8_t { Entry, Export, Import, Call, Pointer, Gap } origin = Origin::Gap;
};

// Switch jump tables: `jr rX` (rX != ra) whose target is loaded from a table of R32-relocated code addresses.
// The table address is recovered by evaluating the load's base register backwards (lui/addiu/ori/addu);
// the entry count comes from a preceding `sltiu idx, N` bound, or else the run of consecutive R32 entries.
struct JumpTable {
    std::uint32_t jr = 0;     // address of the jr instruction
    std::uint32_t table = 0;  // address of the table
    nx::vector<std::uint32_t> targets;
};
nx::map<std::uint32_t, JumpTable> find_jump_tables(const Program& program);

// Every instruction of .text ends up in exactly one function (sorted by start).
// Jump-table targets are never treated as function starts.
nx::vector<Function> discover_functions(const Program& program);
// Sorts by start and assigns ends (next start, else the end of the containing code region). Use after appending
// manual seeds to discover_functions' output.
void finalize_functions(const Program& program, nx::vector<Function>& functions);

struct Registration {
    std::uint32_t call_site = 0;  // address of the jal
    std::uint32_t registrar = 0;  // called function
    nx::string name;             // string argument
    std::uint32_t handler = 0;    // code-pointer argument
};

// Calls of the form f("Name", handler, ...) — string in a0 and code pointer in a1 — found by simulating
// constant loads (lui/addiu/ori + relocations) linearly through each function.
nx::vector<Registration> find_registrations(const Program& program, const nx::vector<Function>& functions);

}  // namespace psprecore::analysis
