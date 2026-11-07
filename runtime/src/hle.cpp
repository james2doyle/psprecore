#include "psprecore/runtime/hle.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>

#include "psprecore/runtime/kernel.hpp"
#include "core/foundation/containers/map.h"
#include "core/foundation/containers/hash_map.h"
#include "core/foundation/threading/sync.h"

namespace psprecore::rt {

namespace {

// Guest code runs one thread at a time (and renderers run inside sceGe calls), so no locking is needed.
constexpr int kPageBits = 16;
std::array<std::uint64_t, (0x20000000u >> kPageBits)> g_page_marks{};
std::uint64_t g_write_counter = 1, g_all_marked = 1;

std::uint32_t physical(std::uint32_t address) {
    address &= 0x1FFFFFFF;  // cached / uncached / kernel mirrors
    if ((address & 0x1F000000) == 0x04000000) address = 0x04000000 | (address & 0x1FFFFF);  // VRAM mirrors
    return address;
}

}  // namespace

void mark_guest_written(std::uint32_t address, std::uint32_t size) {
    if (!size) return;
    const std::uint32_t first = physical(address), last = std::min<std::uint32_t>(first + size - 1, 0x1FFFFFFF);
    ++g_write_counter;
    for (std::uint32_t p = first >> kPageBits; p <= (last >> kPageBits); ++p) g_page_marks[p] = g_write_counter;
}

void mark_guest_written_all() { g_all_marked = ++g_write_counter; }

std::uint64_t guest_write_counter() { return g_write_counter; }

namespace {
std::atomic<bool> g_timing_stats{false};
}  // namespace

void set_timing_stats(bool on) { g_timing_stats.store(on, std::memory_order_relaxed); }
bool timing_stats() { return g_timing_stats.load(std::memory_order_relaxed); }

std::uint64_t guest_written_since(std::uint32_t address, std::uint32_t size) {
    return std::max(g_all_marked, guest_marked_since(address, size));
}

std::uint64_t guest_marked_since(std::uint32_t address, std::uint32_t size) {
    std::uint64_t newest = 0;
    if (!size) return newest;
    const std::uint32_t first = physical(address), last = std::min<std::uint32_t>(first + size - 1, 0x1FFFFFFF);
    for (std::uint32_t p = first >> kPageBits; p <= (last >> kPageBits); ++p) newest = std::max(newest, g_page_marks[p]);
    return newest;
}

namespace {

struct Entry {
    const HleImport* import = nullptr;
    HleFunction fn = nullptr;
};

nx::hash_map<std::uint32_t, Entry>& table() {
    static nx::hash_map<std::uint32_t, Entry> t;
    return t;
}

}  // namespace

void hle_bind(const HleBinding* bindings, std::size_t count) {
    auto& t = table();
    if (t.empty())
        for (const auto& imp : hle_imports()) t[imp.nid].import = &imp;
    for (std::size_t i = 0; i < count; ++i) {
        bool found = false;
        for (auto& [nid, e] : t) {
            if (std::strcmp(e.import->name, bindings[i].name) == 0) {
                e.fn = bindings[i].fn;
                found = true;
            }
        }
        if (!found) std::fprintf(stderr, "hle_bind: %s is not imported by the game\n", bindings[i].name);
    }
}

void hle_install() {
    hle_install_threadman();
    hle_install_sysmem();
    hle_install_io();
    hle_install_display();
    hle_install_audio();
    hle_install_mpeg();
    hle_install_psmf();
    hle_install_atrac();
    hle_install_savedata();
    hle_install_font();
}

}  // namespace psprecore::rt

extern "C" void hle_call(Context* ctx, uint32_t nid) {
    using namespace psprecore::rt;
    const auto it = table().find(nid);
    const Entry* e = it == table().end() ? nullptr : &it->second;
    Kernel& k = kernel();
    if (k.trace_hle && e)
        std::fprintf(stderr, "[%8llu us] %-16s %s(%08x, %08x, %08x, %08x) ra=%08x\n",
                     static_cast<unsigned long long>(k.now()), k.current().name.c_str(), e->import->name,
                     ctx->r[4], ctx->r[5], ctx->r[6], ctx->r[7], ctx->r[31]);
    if (e && e->fn) {
        e->fn(*ctx);
        if (k.trace_hle) std::fprintf(stderr, "             -> %08x\n", ctx->r[2]);
        return;
    }
    static nx::set<std::uint32_t> reported;
    static nx::mutex mutex;
    {
        nx::scoped_lock lock(mutex);
        if (reported.insert(nid).second)
            std::fprintf(stderr, "HLE: unimplemented %s::%s (NID %08X), returning 0\n",
                         e ? e->import->library : "?", e ? e->import->name : "?", nid);
    }
    ctx->r[2] = 0;
}
