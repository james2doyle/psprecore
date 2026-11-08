#include "psprecore/analysis/program.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <ranges>

#include "psprecore/error.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::analysis {

using mips::Op;

namespace {

constexpr std::uint32_t kRa = 31, kSp = 29;

bool is_jr_ra(const mips::Instruction& in) { return in.op == Op::Jr && in.rs() == kRa; }

// GPR written by an instruction (0 = none / $zero). Covers everything the simulation can't model exactly.
std::uint32_t written_gpr(const mips::Instruction& in) {
    switch (in.op) {
        case Op::Sll: case Op::Srl: case Op::Rotr: case Op::Sra: case Op::Sllv: case Op::Srlv: case Op::Rotrv:
        case Op::Srav: case Op::Movz: case Op::Movn: case Op::Mfhi: case Op::Mflo: case Op::Clz: case Op::Clo:
        case Op::Add: case Op::Addu: case Op::Sub: case Op::Subu: case Op::And: case Op::Or: case Op::Xor:
        case Op::Nor: case Op::Slt: case Op::Sltu: case Op::Max: case Op::Min: case Op::Wsbh: case Op::Wsbw:
        case Op::Seb: case Op::Seh: case Op::Bitrev: case Op::Jalr:
            return in.rd();
        case Op::Addi: case Op::Addiu: case Op::Slti: case Op::Sltiu: case Op::Andi: case Op::Ori: case Op::Xori:
        case Op::Lui: case Op::Lb: case Op::Lh: case Op::Lwl: case Op::Lw: case Op::Lbu: case Op::Lhu: case Op::Lwr:
        case Op::Ll: case Op::Sc: case Op::Ext: case Op::Ins: case Op::Mfc1: case Op::Cfc1: case Op::Mfv:
        case Op::Mfvc: case Op::Mfic:
            return in.rt();
        case Op::Jal: case Op::Bltzal: case Op::Bgezal: case Op::Bltzall: case Op::Bgezall:
            return kRa;
        default:
            return 0;
    }
}

struct Value {
    bool known = false;
    std::uint32_t value = 0;
    std::uint8_t segment = 0;  // address segment from relocations (value is relative to its base)
};

}  // namespace

Program::Program(const elf::File& file, prx::Module module) : module_(std::move(module)) {
    std::uint32_t end = 0;
    for (const auto& s : module_.segments)
        if (s.type == 1) end = std::max(end, s.vaddr + s.memsz);
    image_.resize(end);
    for (const auto& s : module_.segments) {
        if (s.type != 1) continue;
        const auto src = file.data().subspan(s.offset, s.filesz);
        std::copy(src.begin(), src.end(), image_.begin() + s.vaddr);
    }
    const auto* text = file.find_section(".text");
    if (!text) throw FormatError("module has no .text section");
    text_begin_ = text->addr;
    text_end_ = text->addr + text->size;
    for (const auto& s : file.sections())
        if (s.name.starts_with(".sceStub.text")) text_end_ = std::max(text_end_, s.addr + s.size);
    // Code the game keeps in writable sections (init routines in .data, called through tables): any PROGBITS
    // section inside an executable PT_LOAD segment. discover_functions scans them for address-taken functions.
    for (const auto& s : file.sections()) {
        if (s.name.starts_with(".text") || s.name.starts_with(".sceStub.text") || s.type != 1 /* PROGBITS */)
            continue;
        const auto seg = std::ranges::find_if(module_.segments, [&](const elf::Segment& g) {
            return g.type == 1 /* PT_LOAD */ && (g.flags & 1) /* PF_X */ && s.addr >= g.vaddr &&
                   s.addr + s.size <= g.vaddr + g.memsz;
        });
        if (seg != module_.segments.end()) code_sections_.push_back({s.addr, s.addr + s.size});
    }
    for (std::size_t i = 0; i < module_.relocations.size(); ++i) {
        const auto& r = module_.relocations[i];
        reloc_by_addr_.emplace(segment_base(r.offset_segment) + r.offset, i);
    }
}

std::uint32_t Program::segment_base(std::uint8_t index) const {
    return index < module_.segments.size() ? module_.segments[index].vaddr : 0;
}

std::optional<std::uint32_t> Program::word(std::uint32_t addr) const {
    if (addr % 4 || addr + 4 > image_.size()) return std::nullopt;
    std::uint32_t w;
    std::memcpy(&w, image_.data() + addr, 4);
    return w;
}

mips::Instruction Program::insn(std::uint32_t addr) const { return mips::decode(word(addr).value_or(0), addr); }

nx::string Program::string_at(std::uint32_t addr, std::size_t max_len) const {
    nx::string s;
    for (std::uint32_t a = addr; a < image_.size() && s.size() <= max_len; ++a) {
        const auto c = std::to_integer<unsigned char>(image_[a]);
        if (c == 0) return s.size() >= 2 ? s : nx::string();
        if (c < 0x20 || c >= 0x7F) return {};
        s += static_cast<char>(c);
    }
    return {};
}

const prx::Relocation* Program::relocation_at(std::uint32_t addr) const {
    const auto it = reloc_by_addr_.find(addr);
    return it == reloc_by_addr_.end() ? nullptr : &module_.relocations[it->second];
}

nx::map<std::uint32_t, JumpTable> find_jump_tables(const Program& p) {
    constexpr std::uint32_t kWindow = 24 * 4;  // how far back to look for the table load / bounds check

    // Constant value of `reg` just before `pc`, following lui/addiu/ori/addu chains. For addu, the constant
    // operand is returned (the other one is the scaled index).
    const auto constant = [&](auto&& self, std::uint32_t reg, std::uint32_t pc, int depth) -> std::optional<std::uint32_t> {
        if (reg == 0) return 0u;
        if (depth > 6) return std::nullopt;
        for (std::uint32_t q = pc; q >= p.text_begin() + 4 && q + kWindow > pc;) {
            q -= 4;
            const auto in = p.insn(q);
            if (written_gpr(in) != reg) continue;
            switch (in.op) {
                case Op::Lui: return in.uimm() << 16;
                case Op::Addiu:
                    if (auto v = self(self, in.rs(), q, depth + 1)) return *v + static_cast<std::uint32_t>(in.simm());
                    return std::nullopt;
                case Op::Ori:
                    if (auto v = self(self, in.rs(), q, depth + 1)) return *v | in.uimm();
                    return std::nullopt;
                case Op::Addu:
                    if (auto v = self(self, in.rs(), q, depth + 1)) return v;
                    return self(self, in.rt(), q, depth + 1);
                default: return std::nullopt;
            }
        }
        return std::nullopt;
    };

    // Pass 1: table addresses and entry limits.
    struct Candidate {
        std::uint32_t jr, table, limit;
    };
    nx::vector<Candidate> candidates;
    for (std::uint32_t pc = p.text_begin(); pc < p.text_end(); pc += 4) {
        const auto jr = p.insn(pc);
        if (jr.op != Op::Jr || jr.rs() == kRa) continue;
        // The load that produced the jump register.
        std::optional<std::uint32_t> table;
        for (std::uint32_t q = pc; q >= p.text_begin() + 4 && q + kWindow > pc && !table;) {
            q -= 4;
            const auto in = p.insn(q);
            if (written_gpr(in) != jr.rs()) continue;
            if (in.op != Op::Lw) break;
            if (auto base = constant(constant, in.rs(), q, 0)) table = *base + static_cast<std::uint32_t>(in.simm());
            break;
        }
        if (!table) continue;

        // Index rebase: `sll x, idx, 2` fed by `addiu idx, src, -K` means the bound check compares src with K + N.
        std::uint32_t rebase = 0, rebase_src = 0;
        for (std::uint32_t q = pc; q >= p.text_begin() + 4 && q + kWindow > pc;) {
            q -= 4;
            const auto sll = p.insn(q);
            if (sll.op != Op::Sll || sll.sa() != 2) continue;
            for (std::uint32_t k = q; k >= p.text_begin() + 4 && k + kWindow > q;) {
                k -= 4;
                const auto in = p.insn(k);
                if (written_gpr(in) != sll.rt()) continue;
                if (in.op == Op::Addiu && in.simm() < 0 && in.rs() != in.rt()) {
                    rebase = static_cast<std::uint32_t>(-in.simm());
                    rebase_src = in.rs();
                }
                break;
            }
            break;
        }
        std::uint32_t limit = 0;
        for (std::uint32_t q = pc; q >= p.text_begin() + 4 && q + kWindow > pc;) {
            q -= 4;
            const auto in = p.insn(q);
            if ((in.op == Op::Sltiu || in.op == Op::Slti) && in.simm() > 0) {  // index bound check
                // Subtract the rebase only if the check tests the pre-rebase register.
                const auto n = static_cast<std::uint32_t>(in.simm());
                limit = rebase && in.rs() == rebase_src && n > rebase ? n - rebase : n;
                break;
            }
        }
        candidates.push_back({pc, *table, limit});
    }

    // Pass 2: entries, never running into the next table.
    nx::set<std::uint32_t> starts;
    for (const auto& c : candidates) starts.insert(c.table);
    nx::map<std::uint32_t, JumpTable> tables;
    for (const auto& c : candidates) {
        JumpTable jt{c.jr, c.table, {}};
        for (std::uint32_t i = 0; c.limit == 0 || i < c.limit; ++i) {
            const std::uint32_t at = c.table + 4 * i;
            if (i > 0 && starts.contains(at)) break;
            const auto* r = p.relocation_at(at);
            const auto target = p.word(at);
            if (!r || r->type != prx::kRel32 || !target || !p.in_code(*target)) break;
            if (c.limit == 0 && (*target + 0x10000 < c.jr || *target > c.jr + 0x10000)) break;  // unbounded: stay local
            jt.targets.push_back(*target);
        }
        if (!jt.targets.empty()) tables.emplace(c.jr, std::move(jt));
    }
    return tables;
}

nx::vector<Function> discover_functions(const Program& p) {
    nx::map<std::uint32_t, Function> funcs;
    const auto seed = [&](std::uint32_t addr, Function::Origin origin, nx::string name = {}) {
        if (!p.in_code(addr) || addr % 4) return;
        auto [it, inserted] = funcs.try_emplace(addr, Function{addr, 0, std::move(name), origin});
        if (!inserted && static_cast<int>(origin) < static_cast<int>(it->second.origin)) {
            it->second.origin = origin;
            if (!name.empty()) it->second.name = std::move(name);
        }
    };

    seed(p.module().entry, Function::Origin::Entry, "module_entry");
    for (const auto& exp : p.module().exports)
        for (const auto& f : exp.functions) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "export_%08X", f.nid);
            seed(f.address, Function::Origin::Export, f.nid == 0xD632ACDB ? "module_start" : f.nid == 0xCEE8593C ? "module_stop" : buf);
        }
    for (const auto& imp : p.module().imports)
        for (const auto& f : imp.functions) {
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s_%08X", imp.library.c_str(), f.nid);
            seed(f.stub, Function::Origin::Import, buf);
        }

    // Direct calls, in .text and the writable code sections alike (.data init routines call .text).
    // Plain jumps AND branch targets seed too: a branch/jump to an undiscovered function would trap at run
    // time (the generator only emits intra-function gotos), so every static control-flow edge must land on a
    // function start. Splits are always safe: entering mid-code with live registers is what the hardware does.
    const auto seed_calls = [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t pc = begin; pc < end; pc += 4) {
            const auto in = p.insn(pc);
            if (in.op == Op::Jal) seed(in.jump_target(), Function::Origin::Call);
            else if (in.op == Op::J) seed(in.jump_target(), Function::Origin::Call);
            else if (in.op == Op::Bgezal && in.rs() == 0) seed(in.branch_target(), Function::Origin::Call);
            else if (in.is_branch()) seed(in.branch_target(), Function::Origin::Call);
        }
    };
    seed_calls(p.text_begin(), p.text_end());
    for (const auto& sec : p.code_sections()) seed_calls(sec.begin, sec.end);

    // Code pointers (R32 in data, HI16/LO16 pairs in code). Jump-table entries also point into .text,
    // so a pointer only counts if it looks like a function start.
    const auto looks_like_start = [&](std::uint32_t addr) {
        if (funcs.contains(addr)) return true;
        const auto first = p.insn(addr);
        if (first.op == Op::Addiu && first.rt() == kSp && first.rs() == kSp && first.simm() < 0) return true;
        if (addr < p.text_begin() + 8) return false;
        // The previous code cannot fall through: a return, an indirect jump, or an unconditional jump/branch
        // (e.g. a function ending in a tail branch back into its own body), followed by its delay slot.
        const auto prev = p.insn(addr - 8);
        return prev.op == Op::Jr || prev.op == Op::J ||
               (prev.op == Op::Beq && prev.rs() == 0 && prev.rt() == 0);
    };
    nx::vector<std::uint32_t> pointers;
    std::uint32_t hi_value = 0;
    for (const auto& r : p.module().relocations) {
        const std::uint32_t at = p.segment_base(r.offset_segment) + r.offset;
        const auto w = p.word(at);
        if (!w) continue;
        if (r.type == prx::kRel32 && r.address_segment == 0) pointers.push_back(*w);
        if (r.type == prx::kRelHi16) hi_value = (*w & 0xFFFF) << 16;
        if (r.type == prx::kRelLo16 && r.address_segment == 0)
            pointers.push_back(hi_value + static_cast<std::uint32_t>(static_cast<std::int16_t>(*w & 0xFFFF)));
    }
    nx::set<std::uint32_t> case_labels;
    for (const auto& [_, jt] : find_jump_tables(p)) case_labels.insert(jt.targets.begin(), jt.targets.end());
    for (auto ptr : pointers)
        if (p.in_text(ptr) && ptr % 4 == 0 && !case_labels.contains(ptr) && looks_like_start(ptr))
            seed(ptr, Function::Origin::Pointer);

    // Function tables in writable code sections (init routines in .data, called through a table): runs of at
    // least 3 consecutive aligned words pointing into those sections seed every entry. Static tables only
    // (file-backed words); BSS filled at run time is invisible here.
    const auto seed_run = [&](std::uint32_t first, std::uint32_t count) {
        for (std::uint32_t e = first; e < first + count * 4; e += 4)
            if (const auto t = p.word(e)) seed(*t, Function::Origin::Pointer);
    };
    for (const auto& sec : p.code_sections()) {
        std::uint32_t run = 0;
        for (std::uint32_t a = sec.begin; a < sec.end; a += 4) {
            const auto v = p.word(a);
            if (v && *v % 4 == 0 && p.in_code(*v)) {
                ++run;
            } else {
                if (run >= 3) seed_run(a - run * 4, run);
                run = 0;
            }
        }
        if (run >= 3) seed_run(sec.end - run * 4, run);
    }

    // Code addresses handed to the system (thread entries, callbacks, handlers): a constant in a0-a3 at a
    // jal to an import stub. The stub call itself runs HLE, so the address never appears as a jal target.
    // No start-shape check: the game would not hand the system a non-entry (and .text holds no data).
    nx::set<std::uint32_t> stubs;
    for (const auto& imp : p.module().imports)
        for (const auto& f : imp.functions) stubs.insert(f.stub);
    // Register value just before pc, following lui/addiu/ori chains (depth-limited, this function only).
    const auto constant = [&](auto&& self, std::uint32_t reg, std::uint32_t pc, int depth) -> std::optional<std::uint32_t> {
        if (reg == 0) return 0u;
        if (depth > 6) return std::nullopt;
        for (std::uint32_t q = pc; q >= p.text_begin() + 4 && q + 24 * 4 > pc;) {
            q -= 4;
            const auto in = p.insn(q);
            if (written_gpr(in) != reg) continue;
            switch (in.op) {
                case Op::Lui: return in.uimm() << 16;
                case Op::Addiu:
                    if (auto v = self(self, in.rs(), q, depth + 1)) return *v + static_cast<std::uint32_t>(in.simm());
                    return std::nullopt;
                case Op::Ori:
                    if (auto v = self(self, in.rs(), q, depth + 1)) return *v | in.uimm();
                    return std::nullopt;
                default: return std::nullopt;
            }
        }
        return std::nullopt;
    };
    const auto seed_hle_args = [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t pc = begin; pc < end; pc += 4) {
            const auto in = p.insn(pc);
            if (in.op != Op::Jal || !stubs.contains(in.jump_target())) continue;
            for (std::uint32_t r = 4; r <= 7; ++r)
                if (const auto v = constant(constant, r, pc, 0))
                    if (p.in_code(*v) && *v % 4 == 0 && !case_labels.contains(*v))
                        seed(*v, Function::Origin::Pointer);
        }
    };
    seed_hle_args(p.text_begin(), p.text_end());
    for (const auto& sec : p.code_sections()) seed_hle_args(sec.begin, sec.end);

    // Register-indirect calls with a constant target (shared tails, table-driven code reached through a
    // register): the target never appears as a jal, so seed it. Returns (jalr ra) are excluded. Both .text
    // and the writable code sections run this scan: the .data init routines dispatch the same way.
    const auto seed_const_jalr = [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t pc = begin; pc < end; pc += 4) {
            const auto in = p.insn(pc);
            if (in.op != Op::Jalr || in.rs() == kRa) continue;
            if (const auto v = constant(constant, in.rs(), pc, 0))
                if (p.in_code(*v) && *v % 4 == 0) seed(*v, Function::Origin::Pointer);
        }
    };
    seed_const_jalr(p.text_begin(), p.text_end());
    for (const auto& sec : p.code_sections()) seed_const_jalr(sec.begin, sec.end);

    // Gap filling: a function ends after `jr ra` + delay slot when no earlier branch jumps past it and no
    // computed jump (switch) precedes it; the next non-nop instruction then starts a new function. Runs per
    // code region (.text and each writable code section); region ends bound the scan like the next start does.
    seed(p.text_begin(), Function::Origin::Gap);
    for (const auto& sec : p.code_sections()) seed(sec.begin, Function::Origin::Gap);
    // Walked by key: seed() inserts into funcs, which invalidates iterators (nx::map is a B-tree).
    for (auto at = funcs.begin(); at != funcs.end();) {
        const std::uint32_t start = at->first;
        const auto next_it = std::next(at);
        std::uint32_t limit = next_it == funcs.end() ? p.text_end() : next_it->first;
        if (next_it == funcs.end()) {
            for (const auto& s : p.code_sections())
                if (start >= s.begin && start < s.end) {
                    limit = s.end;
                    break;
                }
        }
        at = funcs.end();  // not used past this point: the scan below may insert
        std::uint32_t furthest = 0;
        bool computed_jump = false;
        for (std::uint32_t pc = start; pc < limit; pc += 4) {
            const auto in = p.insn(pc);
            if (in.is_branch()) furthest = std::max(furthest, in.branch_target());
            if (in.op == Op::J && in.jump_target() >= start && in.jump_target() < limit)
                furthest = std::max(furthest, in.jump_target());
            if (in.op == Op::Jr && in.rs() != kRa) computed_jump = true;
            if (!is_jr_ra(in) || computed_jump || furthest > pc + 4) continue;
            std::uint32_t next = pc + 8;
            while (next < limit && p.word(next).value_or(1) == 0) next += 4;  // nop padding
            if (next < limit) seed(next, Function::Origin::Gap);  // the next function after start, scanned next
            break;
        }
        at = funcs.upper_bound(start);
    }

    nx::vector<Function> out;
    for (auto& [a, f] : funcs) out.push_back(std::move(f));
    finalize_functions(p, out);
    return out;
}

void finalize_functions(const Program& p, nx::vector<Function>& out) {
    std::sort(out.begin(), out.end(),
              [](const Function& a, const Function& b) { return a.start < b.start; });
    const auto region_end = [&](std::uint32_t addr) {
        if (p.in_text(addr)) return p.text_end();
        for (const auto& s : p.code_sections())
            if (addr >= s.begin && addr < s.end) return s.end;
        return p.text_end();
    };
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i].end = i + 1 < out.size() ? out[i + 1].start : region_end(out[i].start);
}

nx::vector<Registration> find_registrations(const Program& p, const nx::vector<Function>& functions) {
    nx::vector<Registration> out;
    for (const auto& f : functions) {
        if (f.origin == Function::Origin::Import) continue;
        std::array<Value, 32> regs{};
        const auto exec = [&](std::uint32_t pc) {
            const auto in = p.insn(pc);
            const auto* reloc = p.relocation_at(pc);
            const std::uint8_t seg = reloc ? reloc->address_segment : 0;
            Value result;
            std::uint32_t dest = 0;
            switch (in.op) {
                case Op::Lui:
                    dest = in.rt();
                    result = {true, in.uimm() << 16, seg};
                    break;
                case Op::Addiu:
                    dest = in.rt();
                    if (in.rs() == 0) result = {true, static_cast<std::uint32_t>(in.simm()), 0};
                    else if (regs[in.rs()].known)
                        result = {true, regs[in.rs()].value + static_cast<std::uint32_t>(in.simm()),
                                  reloc ? seg : regs[in.rs()].segment};
                    break;
                case Op::Ori:
                    dest = in.rt();
                    if (in.rs() == 0) result = {true, in.uimm(), 0};
                    else if (regs[in.rs()].known) result = {true, regs[in.rs()].value | in.uimm(), regs[in.rs()].segment};
                    break;
                case Op::Addu: case Op::Or:
                    dest = in.rd();
                    if (in.rt() == 0 && in.rs() != 0) result = regs[in.rs()];
                    else if (in.rs() == 0 && in.rt() != 0) result = regs[in.rt()];
                    break;
                default:
                    dest = written_gpr(in);
                    break;
            }
            if (dest != 0) regs[dest] = result;
        };
        const auto address = [&](const Value& v) { return v.value + p.segment_base(v.segment); };

        for (std::uint32_t pc = f.start; pc < f.end; pc += 4) {
            const auto in = p.insn(pc);
            if (in.op != Op::Jal && in.op != Op::Jalr) {
                exec(pc);
                continue;
            }
            exec(pc + 4);  // delay slot runs before the call
            if (in.op == Op::Jal && regs[4].known && regs[5].known) {
                const auto name = p.string_at(address(regs[4]));
                const auto handler = address(regs[5]);
                if (!name.empty() && p.in_code(handler))
                    out.push_back(Registration{pc, in.jump_target(), name, handler});
            }
            for (std::uint32_t r : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u, 24u, 25u, 31u})
                regs[r] = {};
            pc += 4;  // skip the delay slot (already executed)
        }
    }
    return out;
}

}  // namespace psprecore::analysis
