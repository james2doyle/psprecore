// psprecomp — translate BOOT.BIN into C (see docs/recompiler.md).
//
//   psprecomp <BOOT.BIN> <output dir> [--overrides DIR] [--per-file N]     (--help)

#include <algorithm>
#include <cstdio>
#include <exception>

#include "psprecore/analysis/program.hpp"
#include "psprecore/cli.hpp"
#include "psprecore/error.hpp"
#include "psprecore/formats/elf.hpp"
#include "psprecore/formats/prx.hpp"
#include "psprecore/io/file.hpp"
#include "psprecore/io/path.hpp"
#include "psprecore/recomp/generator.hpp"
#include "psprecore/text.hpp"
#include "core/foundation/containers/map.h"
#include "core/foundation/platform/filesystem.h"
#include "core/foundation/strings/utf8_string.h"

using namespace psprecore;

namespace {

void write_text(nx::fs::path_view path, nx::string_view text) {
    io::write_all(path, std::as_bytes(std::span(text.data(), text.size())));
}

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'; }

bool is_hex_digit(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

// Adds the address of every `GAME_FUNC\s*\(\s*([0-9A-Fa-f]{8})\s*,` in `text`.
void scan_game_funcs(nx::string_view text, nx::set<std::uint32_t>& out) {
    constexpr nx::string_view kMacro = "GAME_FUNC";
    for (std::size_t at = text.find(kMacro); at != nx::string_view::npos; at = text.find(kMacro, at + 1)) {
        std::size_t i = at + kMacro.size();
        const auto skip_spaces = [&] {
            while (i < text.size() && is_space(text[i])) ++i;
        };
        skip_spaces();
        if (i >= text.size() || text[i] != '(') continue;
        ++i;
        skip_spaces();
        const std::size_t digits = i;
        while (i < text.size() && i - digits < 8 && is_hex_digit(text[i])) ++i;
        if (i - digits != 8) continue;
        skip_spaces();
        if (i >= text.size() || text[i] != ',') continue;
        out.insert(static_cast<std::uint32_t>(parse_u64(text.substr(digits, 8), 16)));
        at = i;
    }
}

}  // namespace

// Addresses of GAME_FUNC(XXXXXXXX, name) replacements in the C/C++ sources under `dir`.
nx::set<std::uint32_t> scan_overrides(nx::fs::path_view dir) {
    nx::set<std::uint32_t> out;
    const auto entries = nx::fs::recursive_directory_entries(dir);
    if (!entries) throw IoError(nx::string("cannot list directory '") + dir.native() + "'");
    for (const auto& entry : entries.value()) {
        const auto ext = nx::fs::path_view(entry.path).extension();
        if (entry.type != nx::fs::file_type::regular) continue;
        if (ext != ".c" && ext != ".cpp" && ext != ".h" && ext != ".hpp") continue;
        const auto text = nx::fs::file_read_text(entry.path);
        if (!text) throw IoError("cannot read '" + entry.path + "'");
        scan_game_funcs(text.value(), out);
    }
    return out;
}

int main(int argc, char** argv) {
    nx::string input, out_dir, overrides, seeds;
    std::size_t per_file = 200;
    nx::cli::Parser cli("psprecomp", "Translates a PSP executable (an unencrypted PRX such as BOOT.BIN) into C.");
    cli.root()
        .positional("module", input, "the executable (BOOT.BIN)")
        .positional("out", out_dir, "output folder: funcs.h, lookup.c and funcs_NNN.c")
        .option("overrides", overrides, "folder scanned for GAME_FUNC(XXXXXXXX, ...) replacements: their originals are "
                                        "generated as f_XXXXXXXX_orig", {.value_name = "DIR"})
        .option("seeds", seeds, "file with extra function starts (hex addresses, whitespace-separated, # comments): "
                                "callbacks and table entries the analysis cannot see", {.value_name = "FILE"})
        .option("per-file", per_file, "functions per .c file (default: 200)", {.value_name = "N"});
    if (const int code = parse_command_line(cli, argc, argv); code >= 0) return code;
    try {
        recomp::Options options;
        if (!overrides.empty()) options.overrides = scan_overrides(overrides);
        if (per_file == 0) per_file = 1;

        const auto data = io::read_all(input);
        const elf::File file(data);
        const auto module = prx::parse(file);
        const analysis::Program program(file, module);
        auto functions = analysis::discover_functions(program);
        if (!seeds.empty()) {
            // Manual seeds for code addresses the analysis cannot see (callbacks stored to memory, runtime-built
            // tables). The generator splits functions at them like at any discovered start.
            const auto text = nx::fs::file_read_text(seeds);
            if (!text) throw IoError(nx::string("cannot read seeds file '") + seeds + "'");
            nx::set<std::uint32_t> known;
            for (const auto& f : functions) known.insert(f.start);
            std::size_t pos = 0;
            const auto sv = nx::string_view(text.value());
            while (pos < sv.size()) {
                while (pos < sv.size() && (sv[pos] == ' ' || sv[pos] == '\t' || sv[pos] == '\n' || sv[pos] == '\r')) ++pos;
                if (pos >= sv.size() || sv[pos] == '#') {
                    while (pos < sv.size() && sv[pos] != '\n') ++pos;
                    continue;
                }
                std::size_t end = pos;
                while (end < sv.size() && sv[end] != ' ' && sv[end] != '\t' && sv[end] != '\n' && sv[end] != '\r') ++end;
                const auto addr = static_cast<std::uint32_t>(parse_u64(sv.substr(pos, end - pos), 16));
                if (!program.in_code(addr) || addr % 4 != 0)
                    throw Error(nx::string("seed ") + nx::string(sv.substr(pos, end - pos)) + " is not code");
                if (known.insert(addr).second)
                    functions.push_back(analysis::Function{addr, 0, "seeded", analysis::Function::Origin::Pointer});
                pos = end;
            }
            analysis::finalize_functions(program, functions);
        }
        recomp::Generator gen(program, prx::load_image(file, module, options.base), std::move(functions), options);

        if (!nx::fs::create_directories(out_dir))
            throw IoError(nx::string("cannot create directory '") + out_dir + "'");
        const auto old_files = nx::fs::directory_entries(out_dir);
        if (!old_files) throw IoError(nx::string("cannot list directory '") + out_dir + "'");
        for (const auto& old : old_files.value())  // drop stale chunks from earlier runs
            if (nx::fs::path_view(old.path).filename().native().starts_with("funcs_") && !nx::fs::remove(old.path))
                throw IoError("cannot remove '" + old.path + "'");

        write_text(path_join(out_dir, "funcs.h"), gen.header());
        write_text(path_join(out_dir, "lookup.c"), gen.lookup_source());
        const auto& generated = gen.functions();
        std::size_t files = 0;
        for (std::size_t i = 0; i < generated.size(); i += per_file, ++files) {
            nx::string source = nx::string("/* Generated by psprecomp from ") + nx::fs::path_view(input).filename().native() +
                                ". Do not edit. */\n#include \"funcs.h\"\n";
            for (std::size_t j = i; j < std::min(generated.size(), i + per_file); ++j)
                source += gen.function_source(generated[j]);
            char name[32];
            std::snprintf(name, sizeof name, "funcs_%03zu.c", files);
            write_text(path_join(out_dir, name), source);
        }

        const auto& s = gen.stats();
        std::printf("%zu functions, %zu instructions, %zu jump tables, %zu tail calls, %zu weak-symbol calls"
                    " -> %zu files in %s\n",
                    s.functions, s.instructions, s.jump_tables, s.tail_calls, s.null_calls, files, out_dir.c_str());
        if (!options.overrides.empty())
            std::printf("%zu of %zu overrides matched a function\n", s.overrides, options.overrides.size());
        std::printf("%zu warnings\n", s.warnings.size());
        for (std::size_t i = 0; i < s.warnings.size() && i < 100; ++i) std::printf("  %s\n", s.warnings[i].c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "psprecomp: %s\n", e.what());
        return 1;
    }
}
