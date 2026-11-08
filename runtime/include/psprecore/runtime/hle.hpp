#pragma once

// High-level emulation of the PSP system libraries the game imports.
//
// hle_call(ctx, nid) (recomp.h) dispatches through a table built from hle_imports(); implementations are bound
// by function name. Arguments follow the PSP EABI: a0-a3, t0-t3; results go to v0 (and v1 for 64-bit values).
// Calls without an implementation are logged once and return 0.

#include <cstdint>
#include "core/foundation/platform/filesystem.h"
#include <memory>
#include <span>

#include "psprecore/recomp/recomp.h"
#include "core/foundation/core/callable.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::rt {

class GuestMemory;
namespace ge {
class Renderer;
struct Stats;
struct Draw;
}

struct HleImport {
    const char* library;
    std::uint32_t nid;
    const char* name;
};
const nx::vector<HleImport>& hle_imports();

using HleFunction = void (*)(Context& ctx);
struct HleBinding {
    const char* name;
    HleFunction fn;
};

void hle_bind(const HleBinding* bindings, std::size_t count);
template <std::size_t N>
void hle_bind(const HleBinding (&bindings)[N]) { hle_bind(bindings, N); }

// Prints the fault and a symbolised host stack on crashes (Windows).
void install_crash_handler();

// Writes every GE command executed during display frame `frame` to `path` (debugging).
void set_ge_dump(std::uint64_t frame, const char* path);
std::uint64_t display_frame_count();  // framebuffer flips so far (the frame numbering set_ge_dump uses)

// Write tracking for renderers that cache guest memory (textures). The PSP's GE only sees CPU writes after a data
// cache writeback, so writebacks, DMA-style writes by HLE functions (file reads, movie frames, sceDmacMemcpy) and
// GE block transfers mark what may have changed; frame flips mark everything (writes through uncached addresses
// cannot be seen). A cache entry checked at write_counter() is still valid while written_since(range) is not
// newer. Tracking is per 64 KiB page.
void mark_guest_written(std::uint32_t address, std::uint32_t size);
void mark_guest_written_all();
std::uint64_t guest_write_counter();
// Host-time measurements per draw for --perf / --ge-bench (off otherwise: reading the clock around every draw costs
// several percent of a phone's CPU).
void set_timing_stats(bool on);
bool timing_stats();
std::uint64_t guest_written_since(std::uint32_t address, std::uint32_t size);  // newest mark covering the range
// The same without mark_guest_written_all(): only writes known to touch the range (HLE copies, reads, decoders).
std::uint64_t guest_marked_since(std::uint32_t address, std::uint32_t size);

// Movies (sceMpeg): whether one is on screen, whether the player may skip it, and skipping it. A skipped movie is
// fast-forwarded (the game still plays it to the end, with black frames and no sound; movie_skipping() tells the
// front end to run unthrottled until then). Games list the movies that must not be skipped by the key the log
// prints ("sceMpeg: movie <key>"), e.g. one that is the background of a title screen.
bool movie_playing();
bool movie_skipping();
bool movie_skippable();
void skip_movie();
void set_unskippable_movies(std::span<const std::uint64_t> keys);

// GE renderer used for display lists (call before the game starts; default: the software renderer).
using GeRendererFactory = nx::unique_function<std::unique_ptr<ge::Renderer>(GuestMemory&)>;
void set_ge_renderer_factory(GeRendererFactory factory);
// Replaces the GE renderer with a new one from the factory (between frames; e.g. the renderer setting changed).
void reset_ge_renderer();
// GE counters since start (zero before the first display list).
ge::Stats ge_stats();
// Called for every draw before it is decoded (empty function: none).
using GeDrawListener = nx::unique_function<void(const ge::Draw&)>;
void set_ge_draw_listener(GeDrawListener listener);
// Called on every framebuffer flip (sceDisplaySetFrameBuf) with the address, stride and pixel format.
using FlipHandler = nx::unique_function<void(std::uint32_t address, std::uint32_t stride, std::uint32_t format)>;
void set_flip_handler(FlipHandler handler);

// Display output and controller input for the host frontend.
bool read_framebuffer(nx::vector<std::uint32_t>& rgba, int& width, int& height);  // current frame as RGBA8
void set_pad(std::uint32_t buttons, std::uint8_t lx, std::uint8_t ly);
// Camera look, which the PSP has no control for (a second stick, a drag on the touch screen): the turn wanted
// this vblank in radians, yaw positive to the right, pitch positive upwards. Game code that drives its camera
// takes the turns gathered since its previous frame; turns nobody takes are dropped after a few vblanks, so a
// camera that comes back to life does not jump.
void add_camera_look(float yaw, float pitch);
bool take_camera_look(float& yaw, float& pitch);  // false: no turn

// Interface on screens of another shape ("Fill the screen"): 2D drawn from vertices in [begin, end) (guest
// addresses) moves with its scissor to the screen's edges, horizontal -1 left / 1 right and vertical -1 top /
// 1 bottom (0: stays centred), by the margins around the PSP's picture. Game code that knows its interface
// elements registers them each frame before they are drawn, after clear_2d_anchors(). A range anchored 0, 0 stays
// put whatever region it crosses. Other 2D lying wholly within a region (screen pixels; a panel's text, gauges
// drawn by other code) moves as the region says.
void clear_2d_anchors();
void anchor_2d(std::uint32_t begin, std::uint32_t end, int horizontal, int vertical);
void anchor_2d_region(float x0, float y0, float x1, float y1, int horizontal, int vertical);
bool anchor_2d_of(std::uint32_t address, int& horizontal, int& vertical);  // by vertices: registered ranges
bool anchor_2d_of(float x0, float y0, float x1, float y1, int& horizontal, int& vertical);  // by box: regions
bool write_png(const nx::string& path, const std::uint32_t* rgba, int width, int height);
bool read_png(const nx::string& path, nx::vector<std::uint32_t>& rgba, int& width, int& height);  // RGBA8

// Save/load list dialogs (sceUtilitySavedata list modes). With a frontend UI enabled, a dialog stays open
// (status RUNNING) until the frontend answers it; otherwise a slot is chosen automatically. Both calls are made
// between guest time slices (e.g. from Kernel::on_vblank), never concurrently with guest code.
struct SaveSlotInfo {
    nx::string name;    // save name suffix (e.g. "0001")
    bool exists = false;
    nx::string title;   // SFO savedata title and detail text written by the game (UTF-8)
    nx::string detail;
    nx::string icon;  // ICON0.PNG path, if any
    std::int64_t time = 0;       // last write, seconds since the Unix epoch
};
struct SaveDialog {
    bool save = false;  // save (any slot) or load (existing slots only)
    nx::vector<SaveSlotInfo> slots;
    int focus = 0;      // slot the PSP dialog would highlight first
};
void set_savedata_ui(bool enabled);
const SaveDialog* pending_save_dialog();  // nullptr when no dialog is open (or its answer is queued)
void answer_save_dialog(int slot);        // index into SaveDialog::slots, or -1 to cancel
// Deferred mode queues answers until apply_save_dialog_answer() (called once per vblank), so recordings and
// replays apply them at the same point of game time. Returns true and the slot when an answer was applied.
void set_save_dialog_deferred(bool deferred);
bool apply_save_dialog_answer(int& slot);

// The guest's wall clock (sceKernelLibcTime, gettimeofday) is this Unix time plus elapsed virtual time, so runs
// are reproducible: a recording stores the base it was made with. Default: the host time at startup.
void set_wall_clock_base(std::int64_t unix_seconds);
std::int64_t wall_clock_base();

// Directory with cooked media (cccook output); sceAtrac/sceMpeg look up replacements there.
void set_cooked_root(nx::string_view root);
const nx::string& cooked_root();

// Registers every HLE module (call once at startup).
void hle_install();

inline std::uint32_t arg(const Context& c, int i) {
    // First four go in a0-a3, the rest in t0-t3 (this toolchain passes the 5th/6th syscall
    // argument in t0/t1; PPSSPP's PARAM(n) reads the same registers).
    static constexpr int kRegs[8] = {4, 5, 6, 7, 8, 9, 10, 11};
    return c.r[kRegs[i]];
}
inline void ret(Context& c, std::uint32_t v) { c.r[2] = v; }
inline void ret64(Context& c, std::uint64_t v) {
    c.r[2] = static_cast<std::uint32_t>(v);
    c.r[3] = static_cast<std::uint32_t>(v >> 32);
}

// Per-module installers (one per hle_*.cpp).
void hle_install_threadman();
void hle_install_sysmem();
void hle_install_io();
void hle_install_display();
void hle_install_audio();
void hle_install_mpeg();
void hle_install_psmf();
void hle_install_atrac();
void hle_install_savedata();
void hle_install_font();

}  // namespace psprecore::rt
