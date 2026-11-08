#include "psprecore/recomp/generator.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "psprecore/error.hpp"
#include "psprecore/mips/allegrex.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/strings/format.h"

namespace psprecore::recomp {

using mips::Instruction;
using mips::Op;

namespace {

constexpr std::uint32_t kRa = 31;

nx::string hex(std::uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "0x%08Xu", v);
    return buf;
}
nx::string label(std::uint32_t abs) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "L_%08X", abs);
    return buf;
}
nx::string R(std::uint32_t n) { return n == 0 ? nx::string("0u") : "ctx->r[" + nx::format("{}", n) + "]"; }
nx::string F(std::uint32_t n) { return "ctx->f[" + nx::format("{}", n) + "]"; }
nx::string FU(std::uint32_t n) { return "ctx->fu[" + nx::format("{}", n) + "]"; }
nx::string S(std::uint32_t n) { return "(int32_t)" + R(n); }
nx::string addr(const Instruction& in) { return "(" + R(in.rs()) + " + " + hex(static_cast<std::uint32_t>(in.simm())) + ")"; }
nx::string assign(std::uint32_t dest, const nx::string& expr) {
    return dest == 0 ? nx::string() : R(dest) + " = " + expr + ";";
}

bool is_control(const Instruction& in) { return in.is_branch() || in.is_jump(); }

// Branch condition (evaluated before the delay slot); empty string = always.
nx::string branch_condition(const Instruction& in) {
    const auto rs = in.rs(), rt = in.rt();
    switch (in.op) {
        case Op::Beq: case Op::Beql: return rs == rt ? "" : R(rs) + " == " + R(rt);
        case Op::Bne: case Op::Bnel: return R(rs) + " != " + R(rt);
        case Op::Blez: case Op::Blezl: return S(rs) + " <= 0";
        case Op::Bgtz: case Op::Bgtzl: return S(rs) + " > 0";
        case Op::Bltz: case Op::Bltzl: case Op::Bltzal: case Op::Bltzall: return S(rs) + " < 0";
        case Op::Bgez: case Op::Bgezl: case Op::Bgezal: case Op::Bgezall: return rs == 0 ? "" : S(rs) + " >= 0";
        case Op::Bc1f: case Op::Bc1fl: return "!fpu_cond(ctx)";
        case Op::Bc1t: case Op::Bc1tl: return "fpu_cond(ctx)";
        case Op::Bvf: case Op::Bvfl: return "!vfpu_cond(ctx, " + nx::format("{}", (in.word >> 18) & 7) + ")";
        case Op::Bvt: case Op::Bvtl: return "vfpu_cond(ctx, " + nx::format("{}", (in.word >> 18) & 7) + ")";
        default: return "/*?*/0";
    }
}

nx::string ext_mask(std::uint32_t size) { return size >= 32 ? "0xFFFFFFFFu" : hex((1u << size) - 1); }

// Translation of a non-control instruction.
nx::string simple(const Instruction& in, std::uint32_t pc) {
    const auto rs = in.rs(), rt = in.rt(), rd = in.rd(), sa = in.sa();
    const auto w = in.word;
    switch (in.op) {
        case Op::Sll: return w == 0 ? "" : assign(rd, R(rt) + " << " + nx::format("{}", sa));
        case Op::Srl: return assign(rd, R(rt) + " >> " + nx::format("{}", sa));
        case Op::Sra: return assign(rd, "(uint32_t)(" + S(rt) + " >> " + nx::format("{}", sa) + ")");
        case Op::Rotr: return assign(rd, "cpu_rotr(" + R(rt) + ", " + nx::format("{}", sa) + ")");
        case Op::Sllv: return assign(rd, R(rt) + " << (" + R(rs) + " & 31)");
        case Op::Srlv: return assign(rd, R(rt) + " >> (" + R(rs) + " & 31)");
        case Op::Srav: return assign(rd, "(uint32_t)(" + S(rt) + " >> (" + R(rs) + " & 31))");
        case Op::Rotrv: return assign(rd, "cpu_rotr(" + R(rt) + ", " + R(rs) + ")");
        case Op::Movz: return rd == 0 ? "" : "if (" + R(rt) + " == 0) " + assign(rd, R(rs));
        case Op::Movn: return rd == 0 ? "" : "if (" + R(rt) + " != 0) " + assign(rd, R(rs));
        case Op::Mfhi: return assign(rd, "ctx->hi");
        case Op::Mflo: return assign(rd, "ctx->lo");
        case Op::Mthi: return "ctx->hi = " + R(rs) + ";";
        case Op::Mtlo: return "ctx->lo = " + R(rs) + ";";
        case Op::Clz: return assign(rd, "cpu_clz(" + R(rs) + ")");
        case Op::Clo: return assign(rd, "cpu_clz(~" + R(rs) + ")");
        case Op::Mult: return "cpu_set_hilo(ctx, (uint64_t)((int64_t)" + S(rs) + " * " + S(rt) + "));";
        case Op::Multu: return "cpu_set_hilo(ctx, (uint64_t)" + R(rs) + " * " + R(rt) + ");";
        case Op::Madd: return "cpu_set_hilo(ctx, cpu_hilo(ctx) + (uint64_t)((int64_t)" + S(rs) + " * " + S(rt) + "));";
        case Op::Maddu: return "cpu_set_hilo(ctx, cpu_hilo(ctx) + (uint64_t)" + R(rs) + " * " + R(rt) + ");";
        case Op::Msub: return "cpu_set_hilo(ctx, cpu_hilo(ctx) - (uint64_t)((int64_t)" + S(rs) + " * " + S(rt) + "));";
        case Op::Msubu: return "cpu_set_hilo(ctx, cpu_hilo(ctx) - (uint64_t)" + R(rs) + " * " + R(rt) + ");";
        case Op::Div: return "cpu_div(ctx, " + R(rs) + ", " + R(rt) + ");";
        case Op::Divu: return "cpu_divu(ctx, " + R(rs) + ", " + R(rt) + ");";
        case Op::Add: case Op::Addu: return assign(rd, R(rs) + " + " + R(rt));
        case Op::Sub: case Op::Subu: return assign(rd, R(rs) + " - " + R(rt));
        case Op::And: return assign(rd, R(rs) + " & " + R(rt));
        case Op::Or: return assign(rd, R(rs) + " | " + R(rt));
        case Op::Xor: return assign(rd, R(rs) + " ^ " + R(rt));
        case Op::Nor: return assign(rd, "~(" + R(rs) + " | " + R(rt) + ")");
        case Op::Slt: return assign(rd, "(uint32_t)(" + S(rs) + " < " + S(rt) + ")");
        case Op::Sltu: return assign(rd, "(uint32_t)(" + R(rs) + " < " + R(rt) + ")");
        case Op::Max: return assign(rd, "(" + S(rs) + " > " + S(rt) + " ? " + R(rs) + " : " + R(rt) + ")");
        case Op::Min: return assign(rd, "(" + S(rs) + " < " + S(rt) + " ? " + R(rs) + " : " + R(rt) + ")");
        case Op::Addi: case Op::Addiu: return assign(rt, R(rs) + " + " + hex(static_cast<std::uint32_t>(in.simm())));
        case Op::Slti: return assign(rt, "(uint32_t)(" + S(rs) + " < " + nx::format("{}", in.simm()) + ")");
        case Op::Sltiu: return assign(rt, "(uint32_t)(" + R(rs) + " < " + hex(static_cast<std::uint32_t>(in.simm())) + ")");
        case Op::Andi: return assign(rt, R(rs) + " & " + hex(in.uimm()));
        case Op::Ori: return assign(rt, R(rs) + " | " + hex(in.uimm()));
        case Op::Xori: return assign(rt, R(rs) + " ^ " + hex(in.uimm()));
        case Op::Lui: return assign(rt, hex(in.uimm() << 16));
        case Op::Ext: return assign(rt, "(" + R(rs) + " >> " + nx::format("{}", sa) + ") & " + ext_mask(rd + 1));
        case Op::Ins: {
            const auto size = rd + 1 - sa;
            const auto mask = (size >= 32 ? 0xFFFFFFFFu : ((1u << size) - 1)) << sa;
            return assign(rt, "(" + R(rt) + " & " + hex(~mask) + ") | ((" + R(rs) + " << " + nx::format("{}", sa) + ") & " + hex(mask) + ")");
        }
        case Op::Wsbh: return assign(rd, "cpu_wsbh(" + R(rt) + ")");
        case Op::Wsbw: return assign(rd, "cpu_wsbw(" + R(rt) + ")");
        case Op::Seb: return assign(rd, "(uint32_t)(int32_t)(int8_t)" + R(rt));
        case Op::Seh: return assign(rd, "(uint32_t)(int32_t)(int16_t)" + R(rt));
        case Op::Bitrev: return assign(rd, "cpu_bitrev(" + R(rt) + ")");
        case Op::Lb: return assign(rt, "mem_lb" + addr(in));
        case Op::Lbu: return assign(rt, "mem_lbu" + addr(in));
        case Op::Lh: return assign(rt, "mem_lh" + addr(in));
        case Op::Lhu: return assign(rt, "mem_lhu" + addr(in));
        case Op::Lw: case Op::Ll: return assign(rt, "mem_lw" + addr(in));
        case Op::Lwl: return assign(rt, "mem_lwl(" + addr(in) + ", " + R(rt) + ")");
        case Op::Lwr: return assign(rt, "mem_lwr(" + addr(in) + ", " + R(rt) + ")");
        case Op::Sb: return "mem_sb(" + addr(in) + ", " + R(rt) + ");";
        case Op::Sh: return "mem_sh(" + addr(in) + ", " + R(rt) + ");";
        case Op::Sw: return "mem_sw(" + addr(in) + ", " + R(rt) + ");";
        case Op::Sc: return "mem_sw(" + addr(in) + ", " + R(rt) + "); " + assign(rt, "1u");
        case Op::Swl: return "mem_swl(" + addr(in) + ", " + R(rt) + ");";
        case Op::Swr: return "mem_swr(" + addr(in) + ", " + R(rt) + ");";
        case Op::Cache: case Op::Sync: case Op::Vnop: case Op::Vsync: case Op::Vflush: return "";
        case Op::Mfic: return assign(rt, "recomp_mfic(ctx)");
        case Op::Mtic: return "recomp_mtic(ctx, " + R(rt) + ");";

        // FPU
        case Op::Lwc1: return FU(in.ft()) + " = mem_lw" + addr(in) + ";";
        case Op::Swc1: return "mem_sw(" + addr(in) + ", " + FU(in.ft()) + ");";
        case Op::Mfc1: return assign(rt, FU(in.fs()));
        case Op::Mtc1: return FU(in.fs()) + " = " + R(rt) + ";";
        case Op::Cfc1: return assign(rt, in.fs() == 31 ? nx::string("ctx->fcr31") : nx::string("0x00003351u"));
        case Op::Ctc1: return in.fs() == 31 ? "ctx->fcr31 = " + R(rt) + ";" : "";
        case Op::AddS: return F(in.fd()) + " = " + F(in.fs()) + " + " + F(in.ft()) + ";";
        case Op::SubS: return F(in.fd()) + " = " + F(in.fs()) + " - " + F(in.ft()) + ";";
        case Op::MulS: return F(in.fd()) + " = " + F(in.fs()) + " * " + F(in.ft()) + ";";
        case Op::DivS: return F(in.fd()) + " = " + F(in.fs()) + " / " + F(in.ft()) + ";";
        case Op::SqrtS: return F(in.fd()) + " = sqrtf(" + F(in.fs()) + ");";
        case Op::AbsS: return F(in.fd()) + " = fabsf(" + F(in.fs()) + ");";
        case Op::MovS: return F(in.fd()) + " = " + F(in.fs()) + ";";
        case Op::NegS: return F(in.fd()) + " = -" + F(in.fs()) + ";";
        case Op::CvtSW: return F(in.fd()) + " = (float)(int32_t)" + FU(in.fs()) + ";";
        case Op::CvtWS: return FU(in.fd()) + " = fpu_to_int(" + F(in.fs()) + ", (int)(ctx->fcr31 & 3));";
        case Op::RoundWS: return FU(in.fd()) + " = fpu_to_int(" + F(in.fs()) + ", 0);";
        case Op::TruncWS: return FU(in.fd()) + " = fpu_to_int(" + F(in.fs()) + ", 1);";
        case Op::CeilWS: return FU(in.fd()) + " = fpu_to_int(" + F(in.fs()) + ", 2);";
        case Op::FloorWS: return FU(in.fd()) + " = fpu_to_int(" + F(in.fs()) + ", 3);";
        case Op::CCondS: return "fpu_compare(ctx, " + nx::format("{}", w & 7) + ", " + F(in.fs()) + ", " + F(in.ft()) + ");";

        // VFPU
        case Op::LvS: case Op::LvQ: case Op::LvlQ: case Op::LvrQ:
            return "vfpu_load(ctx, " + hex(w) + ", " + R(rs) + " + " + hex(static_cast<std::uint32_t>(static_cast<std::int16_t>(w & 0xFFFC))) + ");";
        case Op::SvS: case Op::SvQ: case Op::SvlQ: case Op::SvrQ:
            return "vfpu_store(ctx, " + hex(w) + ", " + R(rs) + " + " + hex(static_cast<std::uint32_t>(static_cast<std::int16_t>(w & 0xFFFC))) + ");";

        case Op::Invalid: case Op::Syscall: case Op::Break: case Op::Halt: case Op::Mfc0: case Op::Mtc0: case Op::Eret:
            return "recomp_trap(ctx, " + hex(pc) + ", " + hex(w) + ");";
        default:
            if (in.is_vfpu()) return "vfpu_op(ctx, " + hex(w) + ");";
            return "recomp_trap(ctx, " + hex(pc) + ", " + hex(w) + "); /* unhandled */";
    }
}

}  // namespace

Generator::Generator(const analysis::Program& program, nx::vector<std::byte> image,
                     nx::vector<analysis::Function> functions, Options options)
    : program_(program), image_(std::move(image)), functions_(std::move(functions)), options_(options) {
    // Analysis and the image are indexed by absolute address (load_image builds the image that way, for psprx
    // too); functions keep those addresses. (Adding options_.base here would double them for modules linked at
    // their load address, like BOOT.BIN.)
    for (std::size_t i = 0; i < functions_.size(); ++i) by_start_[functions_[i].start] = i;
    for (const auto& imp : program_.module().imports)
        for (const auto& f : imp.functions) import_nids_[f.stub] = f.nid;
    for (const auto& [jr, jt] : analysis::find_jump_tables(program_)) {
        auto& targets = jump_tables_[jr];
        for (auto t : jt.targets) targets.push_back(t);
    }
}

std::uint32_t Generator::word(std::uint32_t abs) const {
    if (abs + 4 > image_.size()) throw FormatError("address outside the module image");
    std::uint32_t w;
    std::memcpy(&w, image_.data() + abs, 4);
    return w;
}

nx::string Generator::name_of(std::uint32_t abs) const {
    char buf[16];
    std::snprintf(buf, sizeof buf, "f_%08X", abs);
    return buf;
}

bool Generator::is_function_start(std::uint32_t abs) const { return by_start_.contains(abs); }

nx::string Generator::function_source(const analysis::Function& fn) {
    const std::uint32_t begin = fn.start, end = fn.end;
    ++stats_.functions;
    if (const auto stub = import_nids_.find(begin); stub != import_nids_.end())  // reached via recomp_call
        return "\nvoid " + name_of(begin) + "(Context* ctx) { hle_call(ctx, " + hex(stub->second) + "); } /* " +
               fn.name + " */\n";
    const auto decode = [&](std::uint32_t pc) { return mips::decode(word(pc), pc); };
    const auto inside = [&](std::uint32_t a) { return a >= begin && a < end; };
    const auto warn = [&](std::uint32_t pc, const nx::string& what) {
        char buf[24];
        std::snprintf(buf, sizeof buf, "%08X: ", pc);
        stats_.warnings.push_back(buf + what);
    };

    // Pass 1: labels and jump tables.
    nx::set<std::uint32_t> labels;
    nx::map<std::uint32_t, nx::vector<std::uint32_t>> tables;
    for (std::uint32_t pc = begin; pc < end; pc += 4) {
        const auto in = decode(pc);
        if (in.is_branch() && inside(in.branch_target())) labels.insert(in.branch_target());
        if (in.op == Op::J && inside(in.jump_target())) labels.insert(in.jump_target());
        if (in.op == Op::Jr && in.rs() != kRa) {
            if (const auto jt = jump_tables_.find(pc); jt != jump_tables_.end()) {
                nx::vector<std::uint32_t> t;
                for (auto target : jt->second)
                    if (inside(target)) t.push_back(target);
                    else warn(pc, "jump table entry outside the function");
                labels.insert(t.begin(), t.end());
                tables[pc] = std::move(t);
            }
        }
    }
    // A label on a delay slot needs a label after it (see the duplication below).
    for (std::uint32_t pc = begin; pc + 4 < end; pc += 4)
        if (is_control(decode(pc)) && labels.contains(pc + 4)) labels.insert(pc + 8);

    nx::string out;
    char buf[160];
    const bool overridden = options_.overrides.contains(begin);
    if (overridden) ++stats_.overrides;
    std::snprintf(buf, sizeof buf, "\n/* %s, %u instructions%s */\nvoid %s%s(Context* ctx) {\n",
                  fn.name.empty() ? "unnamed" : fn.name.c_str(), (end - begin) / 4,
                  overridden ? "; replaced by game/" : "", name_of(begin).c_str(), overridden ? "_orig" : "");
    out += buf;

    const auto line = [&](std::uint32_t pc, const nx::string& code, int indent = 1) {
        if (code.empty()) return;
        out.append(static_cast<std::size_t>(indent) * 4, ' ');
        out += code;
        out += '\n';
        (void)pc;
    };
    const auto call = [&](std::uint32_t pc, std::uint32_t target) -> nx::string {
        if (const auto it = import_nids_.find(target); it != import_nids_.end()) return "hle_call(ctx, " + hex(it->second) + ");";
        if (is_function_start(target)) return name_of(target) + "(ctx);";
        if (target < options_.base) {  // unlinked weak symbol (jal 0x0); guarded by an address check
            ++stats_.null_calls;
            return "recomp_trap(ctx, " + hex(pc) + ", " + hex(word(pc)) + "); /* call to unlinked weak symbol */";
        }
        warn(pc, "call into the middle of a function");
        return "recomp_call(ctx, " + hex(target) + ");";
    };
    const auto goto_or_tail = [&](std::uint32_t pc, std::uint32_t target) -> nx::string {
        if (inside(target)) return "goto " + label(target) + ";";
        if (is_function_start(target)) {
            ++stats_.tail_calls;
            return name_of(target) + "(ctx); return;";
        }
        warn(pc, "branch outside the function");
        return "recomp_trap(ctx, " + hex(pc) + ", " + hex(word(pc)) + "); return;";
    };

    for (std::uint32_t pc = begin; pc < end; pc += 4) {
        if (labels.contains(pc)) out += label(pc) + ":;\n";
        const auto in = decode(pc);
        ++stats_.instructions;
        std::snprintf(buf, sizeof buf, "    /* %08X: %s */\n", pc, mips::format(in).c_str());
        out += buf;
        if (!is_control(in)) {
            line(pc, simple(in, pc));
            continue;
        }

        // Control transfer with its delay slot.
        const std::uint32_t ds_pc = pc + 4;
        if (ds_pc >= end) warn(pc, "delay slot past the end of the function");
        const auto ds_in = decode(ds_pc);
        if (is_control(ds_in)) warn(ds_pc, "control instruction in a delay slot");
        std::snprintf(buf, sizeof buf, "/* %08X: %s */ ", ds_pc, mips::format(ds_in).c_str());
        const nx::string delay = buf + simple(ds_in, ds_pc);

        nx::string code;
        switch (in.op) {
            case Op::J:
                code = delay + "\n        " + goto_or_tail(pc, in.jump_target());
                break;
            case Op::Jal:
                code = delay + "\n        ctx->r[31] = " + hex(pc + 8) + "; " + call(pc, in.jump_target());
                break;
            case Op::Jr:
                if (in.rs() == kRa) {
                    code = delay + "\n        return;";
                } else if (const auto t = tables.find(pc); t != tables.end()) {
                    ++stats_.jump_tables;
                    code = "const uint32_t target = " + R(in.rs()) + ";\n        " + delay + "\n        switch (target) {\n";
                    nx::set<std::uint32_t> seen;
                    for (auto target : t->second)
                        if (seen.insert(target).second) code += "        case " + hex(target) + ": goto " + label(target) + ";\n";
                    code += "        default: recomp_trap(ctx, " + hex(pc) + ", target); return;\n        }";
                } else {
                    warn(pc, "computed jump without a jump table (treated as tail call)");
                    code = "const uint32_t target = " + R(in.rs()) + ";\n        " + delay +
                           "\n        recomp_call(ctx, target); return;";
                }
                break;
            case Op::Jalr:
                code = "const uint32_t target = " + R(in.rs()) + ";\n        " + delay + "\n        " +
                       (in.rd() ? R(in.rd()) + " = " + hex(pc + 8) + "; " : "") + "recomp_call(ctx, target);";
                break;
            default: {  // conditional / unconditional branches
                const auto cond = branch_condition(in);
                const bool link = in.op == Op::Bltzal || in.op == Op::Bgezal || in.op == Op::Bltzall || in.op == Op::Bgezall;
                const nx::string link_code = link ? "ctx->r[31] = " + hex(pc + 8) + "; " : "";
                const nx::string take = link ? call(pc, in.branch_target()) : goto_or_tail(pc, in.branch_target());
                if (cond.empty()) {
                    code = delay + "\n        " + link_code + take;
                } else if (in.is_likely()) {
                    code = "if (" + cond + ") {\n            " + delay + "\n            " + link_code + take + "\n        }";
                } else {
                    code = "const int taken = " + cond + ";\n        " + delay + "\n        " + link_code + "if (taken) { " + take + " }";
                }
                break;
            }
        }
        out += "    {\n        " + code + "\n    }\n";
        if (labels.contains(ds_pc)) {  // something jumps straight to the delay slot: give it its own copy
            out += "    goto " + label(pc + 8) + ";\n" + label(ds_pc) + ":;\n";
            line(ds_pc, simple(ds_in, ds_pc));
        }
        pc += 4;  // delay slot consumed
    }

    // Falling off the end continues into the next function (tiny split functions have no epilogue to read).
    const auto last = end - begin >= 8 ? decode(end - 8) : mips::Instruction{};
    const bool ends_with_exit = (last.op == Op::Jr) || last.op == Op::J ||
                                (last.op == Op::Beq && last.rs() == 0 && last.rt() == 0);
    if (labels.contains(end)) out += label(end) + ":;\n";
    if (!ends_with_exit || labels.contains(end)) {
        if (is_function_start(end)) {
            out += "    " + name_of(end) + "(ctx); /* falls through */\n";
        } else {
            out += "    recomp_trap(ctx, " + hex(end) + ", 0u); /* fell off the end of .text */\n";
        }
    }
    out += "}\n";
    return out;
}

nx::string Generator::header() const {
    nx::string out = "/* Generated by psprecomp. Do not edit. */\n#pragma once\n#include \"psprecore/recomp/recomp.h\"\n\n";
    for (const auto& fn : functions_) {
        const std::uint32_t abs = fn.start;
        out += "void " + name_of(abs) + "(Context* ctx);";
        if (!fn.name.empty()) out += " /* " + fn.name + " */";
        out += '\n';
        if (options_.overrides.contains(abs)) out += "void " + name_of(abs) + "_orig(Context* ctx); /* recompiled original */\n";
    }
    return out;
}

nx::string Generator::lookup_source() const {
    nx::string out = "/* Generated by psprecomp. Do not edit. */\n#include \"funcs.h\"\n\n"
                      "typedef struct { uint32_t address; RecompFunc func; } LookupEntry;\n\n"
                      "static const LookupEntry kFunctions[] = {\n";
    for (const auto& fn : functions_) {
        const auto abs = fn.start;
        out += "    {" + hex(abs) + ", " + name_of(abs) + "},\n";
    }
    out += "};\n\n"
           "RecompFunc recomp_lookup(uint32_t address) {\n"
           "    size_t lo = 0, hi = sizeof kFunctions / sizeof kFunctions[0];\n"
           "    while (lo < hi) {\n"
           "        const size_t mid = (lo + hi) / 2;\n"
           "        if (kFunctions[mid].address < address) lo = mid + 1; else hi = mid;\n"
           "    }\n"
           "    return lo < sizeof kFunctions / sizeof kFunctions[0] && kFunctions[lo].address == address ? kFunctions[lo].func : 0;\n"
           "}\n";
    return out;
}

}  // namespace psprecore::recomp
