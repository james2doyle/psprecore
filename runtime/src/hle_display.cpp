// sceDisplay, sceGe_user, sceCtrl. Display lists are drawn by the software reference renderer.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>

#include "psprecore/runtime/ge.hpp"
#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "gpu/soft_renderer.hpp"
#include "core/foundation/containers/map.h"
#include "core/foundation/containers/small_vector.h"
#include "core/foundation/core/foundation.h"

namespace psprecore::rt {

namespace {

GuestMemory& mem() { return kernel().memory(); }

struct DisplayState {
    std::uint32_t framebuffer = 0, stride = 512, format = 3;
    std::uint64_t lists = 0, frames = 0;
    int next_callback_id = 1;
    nx::unique_ptr<ge::Processor> ge;
    std::uint64_t dump_frame = ~0ull;
    std::FILE* dump = nullptr;
    struct Callback {
        std::uint32_t signal_func, signal_arg, finish_func, finish_arg;
    };
    nx::map<int, Callback> callbacks;  // sceGeSetCallback
    GeDrawListener listener;           // set_ge_draw_listener
} g_display;

// In the dump: where a 2D (through-mode) draw lands on the screen and what it samples.
void dump_draw(const ge::Draw& d) {
    const ge::VertexFormat& f = d.state->format;
    if (!f.through || !d.count || !d.vertices) return;
    std::uint32_t count = d.count;  // indexed: as many vertices as the largest index reaches
    if (const int size = f.index_size()) {
        std::uint32_t top = 0;
        for (std::uint32_t i = 0; i < d.count; ++i) {
            const std::uint8_t* p = mem().ptr(d.indices + i * static_cast<std::uint32_t>(size));
            const std::uint32_t v = size == 1 ? p[0] : size == 2 ? p[0] | p[1] << 8 : p[0] | p[1] << 8 | p[2] << 16;
            top = std::max(top, v);
        }
        count = top + 1;
    }
    static const float kOnes[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    nx::vector<ge::Vertex> v;
    ge::decode_vertices(f, kOnes, mem().ptr(d.vertices), count, v);
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    for (const ge::Vertex& p : v) {
        x0 = std::min(x0, p.pos[0]), x1 = std::max(x1, p.pos[0]);
        y0 = std::min(y0, p.pos[1]), y1 = std::max(y1, p.pos[1]);
    }
    const auto& r = d.state->regs;
    int horizontal = 0, vertical = 0;
    char anchor[32] = "";
    if (anchor_2d_of(d.vertices, horizontal, vertical))
        std::snprintf(anchor, sizeof anchor, "  anchored %d %d", horizontal, vertical);
    std::fprintf(g_display.dump, "    2D x %.0f..%.0f y %.0f..%.0f  texture %06X%s  vertices %08X%s\n", x0, x1, y0, y1,
                 r[ge::TEXADDR0] & 0xFFFFFF, (r[ge::TEXTUREMAPENABLE] & 1) ? "" : " (off)", d.vertices, anchor);
}

GeRendererFactory g_renderer_factory;
FlipHandler g_flip_handler;

struct Anchor {
    std::uint32_t begin, end;
    int horizontal, vertical;
};
nx::vector<Anchor> g_anchors;  // anchor_2d
struct AnchorRegion {
    float x0, y0, x1, y1;
    int horizontal, vertical;
};
nx::vector<AnchorRegion> g_anchor_regions;  // anchor_2d_region

std::uint32_t g_pad_buttons = 0;
std::uint8_t g_pad_lx = 128, g_pad_ly = 128;
struct {
    float yaw = 0, pitch = 0;
    int age = 0;  // vblanks since game code last took the turn
} g_look;

using Callback = decltype(g_display.callbacks)::mapped_type;

nx::small_vector<Callback, 4> registered_callbacks() {
    nx::small_vector<Callback, 4> out;
    for (const auto& [_, cb] : g_display.callbacks) out.push_back(cb);
    return out;
}

ge::Processor& gpu() {
    if (!g_display.ge) {
        g_display.ge = nx::make_unique<ge::Processor>(mem());
        g_display.ge->set_renderer(g_renderer_factory ? g_renderer_factory(mem()) : gpu::make_soft_renderer(mem()));
        g_display.ge->on_command = ge::Processor::CommandListener([](std::uint32_t pc, std::uint32_t word) {
            if (g_display.dump && g_display.frames == g_display.dump_frame)
                std::fprintf(g_display.dump, "%08X: %08X  %s\n", pc, word, ge::describe(word).c_str());
        });
        g_display.ge->on_draw = ge::Processor::DrawListener([](const ge::Draw& d) {
            if (g_display.dump && g_display.frames == g_display.dump_frame) dump_draw(d);
            if (g_display.listener) g_display.listener(d);
        });
        // GE interrupts: the handlers run as guest calls inside the sceGe call that is executing the list. They are
        // copied first, since a handler may (un)register callbacks, which invalidates iterators into the map.
        g_display.ge->on_signal = ge::Processor::IdHandler([](std::uint32_t id) {
            const nx::small_vector<Callback, 4> handlers = registered_callbacks();
            for (const Callback& cb : handlers)
                if (cb.signal_func) kernel().call_guest(cb.signal_func, {id, cb.signal_arg});
        });
        g_display.ge->on_finish = ge::Processor::IdHandler([](std::uint32_t id) {
            const nx::small_vector<Callback, 4> handlers = registered_callbacks();
            for (const Callback& cb : handlers)
                if (cb.finish_func) kernel().call_guest(cb.finish_func, {id, cb.finish_arg});
        });
    }
    return *g_display.ge;
}

// ---- sceDisplay --------------------------------------------------------------------------------

void SetMode(Context& c) { ret(c, 0); }
void SetFrameBuf(Context& c) {
    mark_guest_written_all();  // once per frame: catches CPU writes through uncached addresses
    if (g_display.ge) g_display.ge->flush();  // the frame being shown is complete
    if (g_display.dump && g_display.frames == g_display.dump_frame) {
        const auto& st = gpu().stats();
        std::fprintf(g_display.dump, "--- end of frame %llu (totals: %llu lists, %llu draws, %llu vertices)\n",
                     static_cast<unsigned long long>(g_display.frames), static_cast<unsigned long long>(st.lists),
                     static_cast<unsigned long long>(st.draws), static_cast<unsigned long long>(st.vertices));
        std::fclose(g_display.dump);
        g_display.dump = nullptr;
        std::fprintf(stderr, "GE dump of frame %llu written\n", static_cast<unsigned long long>(g_display.dump_frame));
    }
    ++g_display.frames;
    g_display.framebuffer = arg(c, 0);
    g_display.stride = arg(c, 1);
    g_display.format = arg(c, 2);
    if (g_flip_handler && g_display.framebuffer) g_flip_handler(g_display.framebuffer, g_display.stride, g_display.format);
    ret(c, 0);
}
void GetFrameBuf(Context& c) {
    if (arg(c, 0)) mem().write32(arg(c, 0), g_display.framebuffer);
    if (arg(c, 1)) mem().write32(arg(c, 1), g_display.stride);
    if (arg(c, 2)) mem().write32(arg(c, 2), g_display.format);
    ret(c, 0);
}
void GetVcount(Context& c) { ret(c, static_cast<std::uint32_t>(kernel().vblank_count())); }
void WaitVblankStartCB(Context& c) {
    kernel().wait(WaitType::Vblank, 0, 0, true);
    ret(c, 0);
}
void WaitVblankCB(Context& c) {  // same wait; the callback variant matters only for callback delivery
    kernel().wait(WaitType::Vblank, 0, 0, true);
    ret(c, 0);
}
void GetFramePerSec(Context& c) {
    float f = 59.94f;
    std::memcpy(&c.fu[0], &f, 4);
}

// ---- sceGe_user --------------------------------------------------------------------------------

void EdramGetAddr(Context& c) { ret(c, 0x04000000); }
void ListEnQueue(Context& c) {
    Kernel& k = kernel();
    if (g_display.lists++ == 0)
        std::fprintf(stderr, "*** first GE display list at 0x%08X (stall 0x%08X) after %.3f s of guest time, vblank %llu\n",
                     arg(c, 0), arg(c, 1), static_cast<double>(k.now()) / 1e6,
                     static_cast<unsigned long long>(k.vblank_count()));
    ret(c, static_cast<std::uint32_t>(gpu().enqueue(arg(c, 0), arg(c, 1), false)));
}
void UpdateStallAddr(Context& c) {
    gpu().update_stall(static_cast<int>(arg(c, 0)), arg(c, 1));
    ret(c, 0);
}
// Lists run to completion when enqueued; queued drawing completes here.
void flush_renderer() {
    if (g_display.ge) g_display.ge->flush();
}
void ListSync(Context& c) {
    flush_renderer();
    ret(c, 0);
}
void DrawSync(Context& c) {
    flush_renderer();
    ret(c, 0);
}
// sceGeSetCallback(SceGeCbParam*): {signal func, signal cookie, finish func, finish cookie}.
void SetCallback(Context& c) {
    const std::uint32_t p = arg(c, 0);
    const int id = g_display.next_callback_id++;
    g_display.callbacks[id] = {mem().read32(p), mem().read32(p + 4), mem().read32(p + 8), mem().read32(p + 12)};
    ret(c, static_cast<std::uint32_t>(id));
}
void UnsetCallback(Context& c) {
    g_display.callbacks.erase(static_cast<int>(arg(c, 0)));
    ret(c, 0);
}
void ReturnZero(Context& c) { ret(c, 0); }

// ---- sceCtrl -----------------------------------------------------------------------------------

// sceCtrlReadBufferPositive(data, count): blocks until the next sample (once per vblank).
void ReadBufferPositive(Context& c) {
    Kernel& k = kernel();
    k.wait(WaitType::Vblank, 0, 0, false);
    const std::uint32_t data = arg(c, 0), count = arg(c, 1);
    for (std::uint32_t i = 0; i < count; ++i) {  // SceCtrlData: timestamp, buttons, lx, ly, reserved[6]
        const std::uint32_t p = data + i * 16;
        mem().fill(p, 0, 16);
        mem().write32(p, static_cast<std::uint32_t>(k.now()));
        mem().write32(p + 4, g_pad_buttons);
        *mem().ptr(p + 8) = g_pad_lx;
        *mem().ptr(p + 9) = g_pad_ly;
    }
    ret(c, count);
}

constexpr HleBinding kBindings[] = {
    {"sceDisplaySetMode", SetMode},
    {"sceDisplaySetFrameBuf", SetFrameBuf},
    {"sceDisplayGetFrameBuf", GetFrameBuf},
    {"sceDisplayWaitVblankStartCB", WaitVblankStartCB},
    {"sceDisplayWaitVblankCB", WaitVblankCB},
    {"sceDisplayGetFramePerSec", GetFramePerSec},
    {"sceDisplayGetVcount", GetVcount},
    {"sceGeEdramGetAddr", EdramGetAddr},
    {"sceGeListEnQueue", ListEnQueue},
    {"sceGeListEnQueueHead", ListEnQueue},
    {"sceGeListUpdateStallAddr", UpdateStallAddr},
    {"sceGeListSync", ListSync},
    {"sceGeDrawSync", DrawSync},
    {"sceGeSetCallback", SetCallback},
    {"sceGeUnsetCallback", UnsetCallback},
    {"sceGeContinue", ReturnZero},
    {"sceGeBreak", ReturnZero},
    {"sceCtrlSetSamplingCycle", ReturnZero},
    {"sceCtrlSetSamplingMode", ReturnZero},
    {"sceCtrlReadBufferPositive", ReadBufferPositive},
};

}  // namespace

void hle_install_display() { hle_bind(kBindings); }

void set_ge_renderer_factory(GeRendererFactory factory) {
    g_renderer_factory = std::move(factory);
}

void set_ge_draw_listener(GeDrawListener listener) {
    g_display.listener = std::move(listener);
}

ge::Stats ge_stats() { return g_display.ge ? g_display.ge->stats() : ge::Stats{}; }

void reset_ge_renderer() {
    if (!g_display.ge) return;  // created on first use
    g_display.ge->set_renderer(nullptr);  // the old one goes first (GPU resources)
    g_display.ge->set_renderer(g_renderer_factory ? g_renderer_factory(mem()) : gpu::make_soft_renderer(mem()));
}

void set_flip_handler(FlipHandler handler) {
    g_flip_handler = std::move(handler);
}

void set_pad(std::uint32_t buttons, std::uint8_t lx, std::uint8_t ly) {
    g_pad_buttons = buttons;
    g_pad_lx = lx;
    g_pad_ly = ly;
}

void add_camera_look(float yaw, float pitch) {
    constexpr int kStale = 6;
    if (++g_look.age > kStale) g_look.yaw = g_look.pitch = 0;
    g_look.yaw += yaw;
    g_look.pitch += pitch;
}

bool take_camera_look(float& yaw, float& pitch) {
    yaw = g_look.yaw;
    pitch = g_look.pitch;
    g_look = {};
    return yaw != 0 || pitch != 0;
}

void clear_2d_anchors() {
    g_anchors.clear();
    g_anchor_regions.clear();
}

void anchor_2d(std::uint32_t begin, std::uint32_t end, int horizontal, int vertical) {
    if (begin < end) g_anchors.push_back({begin & 0x1FFFFFFF, end & 0x1FFFFFFF, horizontal, vertical});
}

void anchor_2d_region(float x0, float y0, float x1, float y1, int horizontal, int vertical) {
    if (x0 < x1 && y0 < y1 && (horizontal || vertical)) g_anchor_regions.push_back({x0, y0, x1, y1, horizontal, vertical});
}

bool anchor_2d_of(float x0, float y0, float x1, float y1, int& horizontal, int& vertical) {
    for (const AnchorRegion& r : g_anchor_regions)
        if (x0 >= r.x0 && x1 <= r.x1 && y0 >= r.y0 && y1 <= r.y1) {
            horizontal = r.horizontal;
            vertical = r.vertical;
            return true;
        }
    return false;
}

bool anchor_2d_of(std::uint32_t address, int& horizontal, int& vertical) {
    address &= 0x1FFFFFFF;
    for (const Anchor& a : g_anchors)
        if (address >= a.begin && address < a.end) {
            horizontal = a.horizontal;
            vertical = a.vertical;
            return true;
        }
    return false;
}

bool read_framebuffer(nx::vector<std::uint32_t>& rgba, int& width, int& height) {
    if (!g_display.framebuffer) return false;
    if (g_display.ge) g_display.ge->flush();
    width = 480;
    height = 272;
    rgba.resize(static_cast<std::size_t>(width) * height);
    const std::uint32_t format = g_display.format, stride = g_display.stride ? g_display.stride : 512;
    const std::uint32_t bpp = format == 3 ? 4 : 2;
    const auto expand = [](std::uint32_t v, int bits) { return (v << (8 - bits)) | (v >> (2 * bits - 8)); };
    for (int y = 0; y < height; ++y) {
        const std::uint8_t* row = mem().ptr(g_display.framebuffer + static_cast<std::uint32_t>(y) * stride * bpp);
        std::uint32_t* out = rgba.data() + static_cast<std::size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            if (bpp == 4) {  // 8888 is stored as R,G,B,A bytes, like VK_FORMAT_R8G8B8A8
                std::uint32_t p;
                std::memcpy(&p, row + x * 4, 4);
                out[x] = p | 0xFF000000u;
                continue;
            }
            const std::uint32_t p = static_cast<std::uint32_t>(row[x * 2] | (row[x * 2 + 1] << 8));
            std::uint32_t r, g, b;
            switch (format) {
                case 0: r = expand(p & 31, 5); g = expand((p >> 5) & 63, 6); b = expand(p >> 11, 5); break;            // 565
                case 1: r = expand(p & 31, 5); g = expand((p >> 5) & 31, 5); b = expand((p >> 10) & 31, 5); break;     // 5551
                default: r = (p & 15) * 17; g = ((p >> 4) & 15) * 17; b = ((p >> 8) & 15) * 17; break;               // 4444
            }
            out[x] = r | (g << 8) | (b << 16) | 0xFF000000u;
        }
    }
    return true;
}

void set_ge_dump(std::uint64_t frame, const char* path) {
    g_display.dump_frame = frame;
    g_display.dump = std::fopen(path, "w");
}

std::uint64_t display_frame_count() { return g_display.frames; }

}  // namespace psprecore::rt
