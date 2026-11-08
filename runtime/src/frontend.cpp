// The front end (psprecore/platform/frontend.hpp): runs the recompiled game in a window (or headless, for testing).
//
//   <game> <extracted disc folder> [options]     (--help lists them; on Android they come from the launch intent)
//
// Options that act at a given vblank take VBLANK:VALUE, e.g. --screenshot 600:shot.png, --press 900:start.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <memory>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <SDL3/SDL_keycode.h>
#ifdef __ANDROID__
#include <SDL3/SDL_messagebox.h>
#include <SDL3/SDL_system.h>
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "psprecore/formats/elf.hpp"
#include "psprecore/formats/prx.hpp"
#include "psprecore/io/file.hpp"
#include "psprecore/io/path.hpp"
#include "psprecore/text.hpp"
#include "psprecore/platform/audio.hpp"
#include "psprecore/platform/display.hpp"
#include "psprecore/platform/frontend.hpp"
#include "platform/gpu_vulkan.hpp"
#include "platform/overlay.hpp"
#include "platform/settings.hpp"
#include "psprecore/gpu/button_icons.hpp"
#include "gpu/hash.hpp"
#include "gpu/soft_renderer.hpp"
#include "gpu/texture.hpp"
#include "gpu/vertex_pipeline.hpp"
#include "platform/memory_watch.hpp"
#include "psprecore/platform/mods.hpp"
#include "platform/sampling_profiler.hpp"
#include "psprecore/gpu/texture_collection.hpp"
#include "psprecore/gpu/texture_replacement.hpp"
#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/io.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "psprecore/runtime/memory.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/time.h"
#include "core/foundation/containers/hash_map.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/strings/format.h"
#include "core/foundation/serialization/arguments.h"
#include "core/foundation/threading/thread.h"


namespace {

// Records every audio source to <prefix>_<source>.wav (and forwards to the device, if any). For testing.
class WavRecorder final : public psprecore::rt::AudioSink {
public:
    WavRecorder(nx::string prefix, psprecore::rt::AudioSink* next) : prefix_(std::move(prefix)), next_(next) {}
    ~WavRecorder() override { flush(); }

    void submit(int source, const std::int16_t* frames, std::size_t count) override {
        if (source >= 0 && source < kSources) samples_[source].insert(samples_[source].end(), frames, frames + count * 2);
        if (next_) next_->submit(source, frames, count);
    }
    void flush() {
        for (int s = 0; s < kSources; ++s) {
            const auto& pcm = samples_[s];
            if (pcm.empty()) continue;
            const nx::string path = prefix_ + "_" + nx::format("{}", s) + ".wav";
            std::FILE* f = psprecore::io::open_c_file(path, "wb");
            if (!f) continue;
            const auto put32 = [f](std::uint32_t v) { std::fwrite(&v, 4, 1, f); };
            const auto put16 = [f](std::uint16_t v) { std::fwrite(&v, 2, 1, f); };
            const auto bytes = static_cast<std::uint32_t>(pcm.size() * 2);
            std::fwrite("RIFF", 1, 4, f);
            put32(36 + bytes);
            std::fwrite("WAVEfmt ", 1, 8, f);
            put32(16);
            put16(1);  // PCM
            put16(2);
            put32(44100);
            put32(44100 * 4);
            put16(4);
            put16(16);
            std::fwrite("data", 1, 4, f);
            put32(bytes);
            std::fwrite(pcm.data(), 2, pcm.size(), f);
            std::fclose(f);
        }
    }

private:
    nx::string prefix_;
    psprecore::rt::AudioSink* next_;
    nx::vector<std::int16_t> samples_[kSources];
};

// Sends every display list to two renderers (--renderer compare).
class TeeRenderer final : public psprecore::rt::ge::Renderer {
public:
    TeeRenderer(std::unique_ptr<Renderer> a, std::unique_ptr<Renderer> b) : a_(std::move(a)), b_(std::move(b)) {}
    void draw(const psprecore::rt::ge::State& s, psprecore::rt::ge::Prim prim, const nx::vector<psprecore::rt::ge::Vertex>& v) override {
        a_->draw(s, prim, v);
        b_->draw(s, prim, v);
    }
    void transfer(const psprecore::rt::ge::State& s) override {
        a_->transfer(s);
        b_->transfer(s);
    }
    void flush() override {
        a_->flush();
        b_->flush();
    }

private:
    std::unique_ptr<Renderer> a_, b_;
};

// Reference frame (w x h) next to the GPU frame (box-filtered down from `scale` times the size) and their
// difference (x4). Returns the composite; `mean` is the mean per-pixel max-channel difference and `bad` the
// share of pixels differing by more than 32.
nx::vector<std::uint32_t> compare_frames(const nx::vector<std::uint32_t>& ref, const nx::vector<std::uint32_t>& gpu, int w,
                                          int h, int scale, double& mean, double& bad) {
    nx::vector<std::uint32_t> out(static_cast<std::size_t>(w) * 3 * h, 0xFF000000u);
    double total = 0;
    std::size_t over = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int acc[3] = {0, 0, 0};
            for (int sy = 0; sy < scale; ++sy)
                for (int sx = 0; sx < scale; ++sx) {
                    const std::uint32_t px = gpu[static_cast<std::size_t>(y * scale + sy) * (w * scale) + (x * scale + sx)];
                    for (int c = 0; c < 3; ++c) acc[c] += static_cast<int>((px >> (c * 8)) & 0xFF);
                }
            const std::uint32_t r = ref[static_cast<std::size_t>(y) * w + x];
            std::uint32_t g = 0xFF000000u;
            int diff = 0;
            for (int c = 0; c < 3; ++c) {
                const int gv = acc[c] / (scale * scale);
                g |= static_cast<std::uint32_t>(gv) << (c * 8);
                diff = std::max(diff, std::abs(gv - static_cast<int>((r >> (c * 8)) & 0xFF)));
            }
            total += diff;
            if (diff > 32) ++over;
            const std::size_t row = static_cast<std::size_t>(y) * w * 3;
            out[row + x] = r | 0xFF000000u;
            out[row + w + x] = g;
            const std::uint32_t d = static_cast<std::uint32_t>(std::min(255, diff * 4));
            out[row + 2 * w + x] = 0xFF000000u | d | (d << 8) | (d << 16);
        }
    mean = total / (static_cast<double>(w) * h);
    bad = 100.0 * static_cast<double>(over) / (static_cast<double>(w) * h);
    return out;
}

// Input recordings (--record / --replay), a text file with one line per change:
//     psprecore-input 1 <wall clock base>    (older recordings: <game>-input)
//     <vblank> pad <buttons, hex> <lx> <ly>
//     <vblank> dialog <slot>            (save picker answer; -1 = cancel)
//     <vblank> skip                     (the playing movie was skipped)
//     <vblank> look <yaw> <pitch>       (camera turn this vblank, radians)
// With the same game data, saves and clock base the run is reproduced exactly (virtual time is deterministic).
class InputRecorder {
public:
    InputRecorder(nx::fs::path_view path, std::int64_t clock_base) : f_(psprecore::io::open_c_file(path, "w")) {
        if (!f_) throw psprecore::Error(nx::string("cannot write ") + path.native());
        std::fprintf(f_, "psprecore-input 1 %lld\n", static_cast<long long>(clock_base));
    }
    ~InputRecorder() {
        if (f_) std::fclose(f_);
    }
    InputRecorder(const InputRecorder&) = delete;
    InputRecorder& operator=(const InputRecorder&) = delete;

    void pad(std::uint64_t vblank, std::uint32_t buttons, std::uint8_t lx, std::uint8_t ly) {
        if (buttons == buttons_ && lx == lx_ && ly == ly_) return;
        buttons_ = buttons;
        lx_ = lx;
        ly_ = ly;
        std::fprintf(f_, "%llu pad %X %u %u\n", static_cast<unsigned long long>(vblank), buttons, lx, ly);
    }
    void dialog(std::uint64_t vblank, int slot) {
        std::fprintf(f_, "%llu dialog %d\n", static_cast<unsigned long long>(vblank), slot);
        std::fflush(f_);
    }
    void skip(std::uint64_t vblank) {
        std::fprintf(f_, "%llu skip\n", static_cast<unsigned long long>(vblank));
        std::fflush(f_);
    }
    void look(std::uint64_t vblank, float yaw, float pitch) {
        if (yaw != 0 || pitch != 0)
            std::fprintf(f_, "%llu look %.9g %.9g\n", static_cast<unsigned long long>(vblank), yaw, pitch);
    }
    void flush() { std::fflush(f_); }

private:
    std::FILE* f_;
    std::uint32_t buttons_ = 0;
    std::uint8_t lx_ = 128, ly_ = 128;
};

class InputReplay {
public:
    explicit InputReplay(nx::fs::path_view path) {
        std::FILE* f = psprecore::io::open_c_file(path, "r");
        if (!f) throw psprecore::Error(nx::string("cannot read ") + path.native());
        long long base = 0;
        char magic[64];
        const auto ends_with = [](nx::string_view s, nx::string_view tail) {
            return s.size() >= tail.size() && s.substr(s.size() - tail.size()) == tail;
        };
        // "psprecore-input"; any "<name>-input" header from the runtime's earlier, per-game names is read the same.
        if (std::fscanf(f, "%63s 1 %lld", magic, &base) != 2 || !ends_with(magic, "-input")) {
            std::fclose(f);
            throw psprecore::Error(nx::string(path.native()) + " is not an input recording");
        }
        clock_base_ = base;
        char kind[16];
        unsigned long long vblank = 0;
        while (std::fscanf(f, "%llu %15s", &vblank, kind) == 2) {
            Event e{vblank};
            if (nx::string_view(kind) == "pad") {
                unsigned b = 0, lx = 128, ly = 128;
                if (std::fscanf(f, "%x %u %u", &b, &lx, &ly) != 3) break;
                e.buttons = b;
                e.lx = static_cast<std::uint8_t>(lx);
                e.ly = static_cast<std::uint8_t>(ly);
            } else if (nx::string_view(kind) == "dialog") {
                if (std::fscanf(f, "%d", &e.dialog) != 1) break;
                e.is_dialog = true;
            } else if (nx::string_view(kind) == "skip") {
                e.is_skip = true;
            } else if (nx::string_view(kind) == "look") {
                if (std::fscanf(f, "%f %f", &e.yaw, &e.pitch) != 2) break;
                e.is_look = true;
            } else {
                break;
            }
            events_.push_back(e);
        }
        std::fclose(f);
    }

    std::int64_t clock_base() const { return clock_base_; }
    bool finished() const { return next_ >= events_.size(); }
    std::uint64_t last_vblank() const { return events_.empty() ? 0 : events_.back().vblank; }

    // Advances to `vblank`: updates the pad state and returns save picker answers due now (-2: none); `skip`
    // tells whether the movie was skipped now.
    int advance(std::uint64_t vblank, bool& skip) {
        int dialog = -2;
        skip = false;
        yaw_ = pitch_ = 0;
        for (; next_ < events_.size() && events_[next_].vblank <= vblank; ++next_) {
            const Event& e = events_[next_];
            if (e.is_look) {
                yaw_ += e.yaw;
                pitch_ += e.pitch;
            } else if (e.is_dialog) {
                dialog = e.dialog;
            } else if (e.is_skip) {
                skip = true;
            } else {
                buttons_ = e.buttons;
                lx_ = e.lx;
                ly_ = e.ly;
            }
        }
        return dialog;
    }
    std::uint32_t buttons() const { return buttons_; }
    std::uint8_t lx() const { return lx_; }
    std::uint8_t ly() const { return ly_; }
    float look_yaw() const { return yaw_; }  // this vblank's camera turn
    float look_pitch() const { return pitch_; }

private:
    struct Event {
        std::uint64_t vblank;
        std::uint32_t buttons = 0;
        std::uint8_t lx = 128, ly = 128;
        float yaw = 0, pitch = 0;
        bool is_dialog = false, is_skip = false, is_look = false;
        int dialog = -1;
    };
    nx::vector<Event> events_;
    std::size_t next_ = 0;
    std::int64_t clock_base_ = 0;
    std::uint32_t buttons_ = 0;
    std::uint8_t lx_ = 128, ly_ = 128;
    float yaw_ = 0, pitch_ = 0;
};

struct ScriptedPress {
    std::uint64_t vblank;
    std::uint32_t buttons;
};

struct ScriptedLook {  // a camera turn per vblank over [from, to)
    std::uint64_t from, to;
    float yaw, pitch;  // radians
};

struct ScriptedUi {
    std::uint64_t vblank;
    nx::string action;
};

std::uint32_t button_from_name(nx::string_view name) {
    using namespace psprecore::platform;
    static constexpr std::pair<nx::string_view, std::uint32_t> kNames[] = {
        {"select", kSelect}, {"start", kStart},   {"up", kUp},       {"right", kRight},
        {"down", kDown},     {"left", kLeft},     {"l", kLTrigger},  {"r", kRTrigger},
        {"triangle", kTriangle}, {"circle", kCircle}, {"cross", kCross}, {"square", kSquare},
    };
    for (const auto& [n, b] : kNames)
        if (n == name) return b;
    throw psprecore::Error("unknown button: " + nx::string(name));
}

std::uint32_t scripted_buttons(const nx::vector<ScriptedPress>& presses, std::uint64_t vblank) {
    constexpr std::uint64_t kHold = 6;
    std::uint32_t buttons = 0;
    for (const auto& p : presses)
        if (vblank >= p.vblank && vblank < p.vblank + kHold) buttons |= p.buttons;
    return buttons;
}

void scripted_look(const nx::vector<ScriptedLook>& looks, std::uint64_t vblank, float& yaw, float& pitch) {
    for (const auto& l : looks)
        if (vblank >= l.from && vblank < l.to) {
            yaw += l.yaw;
            pitch += l.pitch;
        }
}

}  // namespace
using namespace psprecore;

namespace {


#ifdef __ANDROID__
// The runtime logs to stderr; on Android forward it line by line to logcat (tagged with the game's short name) and
// to `log_path` (logcat's ring buffer is small and shared with the whole system).
const char* g_log_tag = "psprecore";
nx::thread g_stderr_reader;
int g_stderr_pipe = -1;  // write end

void redirect_stderr_to_logcat(nx::fs::path_view log_path) {
    static int fds[2];
    static std::FILE* log = nullptr;
    if (pipe(fds) != 0) return;
    log = io::open_c_file(log_path, "w");
    dup2(fds[1], STDERR_FILENO);
    g_stderr_pipe = fds[1];
    std::setvbuf(stderr, nullptr, _IOLBF, 0);
    g_stderr_reader = nx::thread([] {
        char buf[1024];
        nx::string line;
        for (ssize_t n; (n = read(fds[0], buf, sizeof buf)) > 0;)
            for (ssize_t i = 0; i < n; ++i) {
                if (buf[i] != '\n') {
                    line += buf[i];
                    continue;
                }
                __android_log_write(ANDROID_LOG_INFO, g_log_tag, line.c_str());
                if (log) {
                    std::fputs(line.c_str(), log);
                    std::fputc('\n', log);
                    std::fflush(log);
                }
                line.clear();
            }
    });
}
#endif

// Before the process ends: on Android waits until the last lines reached logcat and the log file (the reader
// thread would otherwise die with the process, losing e.g. the error that ended it).
void finish_stderr() {
    std::fflush(stderr);
#ifdef __ANDROID__
    if (!g_stderr_reader.joinable()) return;
    if (const int null_fd = open("/dev/null", O_WRONLY); null_fd >= 0) dup2(null_fd, STDERR_FILENO);
    close(g_stderr_pipe);  // the reader sees the end of the pipe once both write ends are closed
    g_stderr_reader.join();
#endif
}

}  // namespace

namespace {

struct Options {
    int exit_code = -1;  // >= 0: stop here (help shown, or a usage error)
    nx::string game_dir;
    bool trace = false, headless = false, no_audio = false, unthrottled = false, perf = false, mute = false;
#ifdef __ANDROID__
    bool touch = true;  // shown until a gamepad is used
#else
    bool touch = false;
#endif
    std::uint64_t screenshot_vblank = 0, screenshot_every = 0;
    nx::string screenshot_path, audio_dump, cooked_dir, renderer, settings, window;
    int scale = 0, soft_threads = 0;
    std::uint64_t vblanks = 0, ge_dump = 0, ge_dump_at = 0, ge_bench = 0, profile = 0, state_hash = 0;
    std::uint64_t tex_dump_vblank = 0, ram_dump_vblank = 0, window_shot_vblank = 0, watch_vblank = 0;
    int window_shots = 1;
    std::uint32_t watch_address = 0, watch_size = 0;
    nx::string tex_dump_dir, ram_dump_path, window_shot_path, tex_collect;
    nx::vector<nx::string> pokes;  // VBLANK:ADDR:VALUE
    nx::string record, replay, saves, mods;
    std::int64_t clock = 0;
    bool clock_given = false;
    nx::vector<ScriptedPress> presses;
    nx::vector<ScriptedLook> looks;
    nx::vector<ScriptedUi> ui_script;
};

// VBLANK:VALUE (options acting at a vblank) -> vblank and value.
std::pair<std::uint64_t, nx::string> at_vblank(nx::string_view option, nx::string_view arg) {
    const auto colon = arg.find(':');
    if (colon == nx::string_view::npos || colon == 0)
        throw Error(nx::format("--{} takes VBLANK:VALUE, not '{}'", option, arg));
    return {parse_u64(arg.substr(0, colon)), nx::string(arg.substr(colon + 1))};
}

Options parse_options(const platform::GameInfo& game, int argc, char** argv, const nx::string& data_dir) {
    Options o;
    nx::string screenshot, screenshot_every, window_shot, tex_dump, ram_dump, clock, watch;
    nx::vector<nx::string> presses, looks, ui;
    nx::cli::Parser cli(game.name, nx::format("{} on the psprecore runtime.", game.title));
    nx::cli::Command& c = cli.root();
#ifdef __ANDROID__
    c.positional("disc", o.game_dir, "extracted disc folder (with PSP_GAME); default: the app's files folder",
                 {.required = false});
#else
    c.positional("disc", o.game_dir, "extracted disc folder (with PSP_GAME inside)");
#endif
    // Playing.
    c.option("renderer", o.renderer, "vulkan, soft, or compare (both; screenshots: reference | Vulkan | difference); "
                                     "default: the setting",
             {.value_name = "NAME", .choices = {"vulkan", "soft", "compare"}});
    c.option("scale", o.scale, "Vulkan internal resolution, multiples of 480x272 (default: the setting)",
             {.value_name = "N"});
    c.option("window", o.window, "window size in pixels (default: 960x544)", {.value_name = "WxH"});
    c.option("settings", o.settings, "settings file, written when changed in the pause menu (default: <game>.ini)",
             {.value_name = "FILE"});
    c.option("saves", o.saves, "save data folder (default: saves/ next to the settings)", {.value_name = "DIR"});
    c.option("mods", o.mods, "mod folders, for the game's mount hook (default: mods/)", {.value_name = "DIR"});
    c.option("cooked", o.cooked_dir, "cooked media (default: cooked/)", {.value_name = "DIR"});
    c.flag("touch", o.touch, "show the on-screen touch controls from the start (default on Android)");
    c.option("soft-threads", o.soft_threads, "software renderer threads (default: one per core, at most 8)",
             {.value_name = "N"});
    // Recording and replay.
    c.option("record", o.record, "record controller input and save picker answers; snapshots the saves",
             {.value_name = "FILE"});
    c.option("replay", o.replay, "replay a recording (on a copy of its saves), then live input takes over",
             {.value_name = "FILE"});
    c.option("clock", clock, "guest wall clock at boot, UNIX time (default: now, or the recording's)",
             {.value_name = "UNIX"});
    // Testing.
    c.flag("headless", o.headless, "no window or audio; software renderer");
    c.option("vblanks", o.vblanks, "stop after N vblanks", {.value_name = "N"});
    c.flag("no-audio", o.no_audio, "no sound device");
    c.flag("mute", o.mute, "open the sound device but play silence");
    c.flag("unthrottled", o.unthrottled, "run as fast as possible");
    c.flag("perf", o.perf, "print host time per 300 vblanks and display frames");
    c.option("screenshot", screenshot, "write a PNG of the game at VBLANK", {.value_name = "VBLANK:FILE"});
    c.option("screenshot-every", screenshot_every, "write PREFIX_<vblank>.png every N vblanks",
             {.value_name = "N:PREFIX"});
    c.option("window-shots", o.window_shots,
             "with --window-shot: the next N presented frames, one file each (FILE gets _<vblank> before .png)",
             {.value_name = "N"});
    c.option("window-shot", window_shot, "capture the window, overlay included (a pause menu opened by --ui closes "
                                         "after it)", {.value_name = "VBLANK:FILE"});
    c.option_many("press", presses, "hold BUTTON for 6 vblanks: start, select, cross, circle, square, triangle, up, "
                                    "down, left, right, l, r", {.value_name = "VBLANK:BUTTON"});
    c.option_many("look", looks, "turn the camera by YAW and PITCH degrees per vblank from FROM to TO (as a second "
                                 "stick or a drag on the touch screen does)", {.value_name = "FROM-TO:YAW:PITCH"});
    c.option_many("ui", ui, "overlay input: menu, enter, escape, up, down, space, minimize, restore, skip-movie, tap:X,Y "
                            "(window-normalised), renderer:vulkan, renderer:software", {.value_name = "VBLANK:ACTION"});
    c.option("state-hash", o.state_hash, "print a hash of user RAM every N vblanks (determinism checks)",
             {.value_name = "N"});
    c.option("ram-dump", ram_dump, "write user RAM (0x08800000-0x0A000000) at VBLANK", {.value_name = "VBLANK:FILE"});
    c.option("audio-dump", o.audio_dump, "write PREFIX_<source>.wav per audio source at exit",
             {.value_name = "PREFIX"});
    c.flag("trace", o.trace, "log every HLE call");
    c.option("ge-dump", o.ge_dump, "write ge_dump.txt for display frame N", {.value_name = "N"});
    c.option("ge-dump-at", o.ge_dump_at, "write ge_dump.txt for the first full frame after VBLANK",
             {.value_name = "VBLANK"});
    c.option("ge-bench", o.ge_bench, "time vertex decoding and transform on the draws after VBLANK, then quit",
             {.value_name = "VBLANK"});
    c.option("tex-dump", tex_dump, "write the textures drawn at VBLANK (named by their texel hash)",
             {.value_name = "VBLANK:DIR"});
    c.option("tex-collect", o.tex_collect,
             "Vulkan renderer: add every texture the game uploads to DIR (for baking HD versions; see "
             "psprecore/gpu/texture_collection.hpp)",
             {.value_name = "DIR"});
    c.option("profile", o.profile, "Windows: sample where the host spends its time from VBLANK on",
             {.value_name = "VBLANK"});
    c.option_many("poke", o.pokes, "write the 32-bit VALUE to guest memory ADDR at VBLANK (hex address and value)",
                  {.value_name = "VBLANK:ADDR:VALUE"});
    c.option("watch", watch, "Windows: from VBLANK on, list the functions accessing guest memory ADDR..ADDR+SIZE (hex)",
             {.value_name = "VBLANK:ADDR:SIZE"});

    const nx::cli::ParseResult parsed = cli.parse(argc, argv);
    if (parsed.status == nx::cli::ParseStatus::Help) {
        std::fputs(cli.help().c_str(), stderr);
        o.exit_code = 0;
        return o;
    }
    if (!parsed.parsed()) {
        std::fprintf(stderr, "%s: %s\n\n%s", game.name, parsed.message.c_str(), cli.help().c_str());
        o.exit_code = 2;
        return o;
    }
    if (o.game_dir.empty()) o.game_dir = path_join(data_dir, game.folder);
    try {
        const auto split = [](nx::string_view option, const nx::string& arg, std::uint64_t& vblank, nx::string& value) {
            if (!arg.empty()) std::tie(vblank, value) = at_vblank(option, arg);
        };
        split("screenshot", screenshot, o.screenshot_vblank, o.screenshot_path);
        split("screenshot-every", screenshot_every, o.screenshot_every, o.screenshot_path);
        split("window-shot", window_shot, o.window_shot_vblank, o.window_shot_path);
        split("tex-dump", tex_dump, o.tex_dump_vblank, o.tex_dump_dir);
        split("ram-dump", ram_dump, o.ram_dump_vblank, o.ram_dump_path);
        if (!watch.empty()) {
            nx::string range;
            split("watch", watch, o.watch_vblank, range);
            const auto colon = range.find(':');
            if (colon == nx::string::npos) throw std::runtime_error("--watch: expected VBLANK:ADDR:SIZE");
            o.watch_address = static_cast<std::uint32_t>(std::strtoul(range.substr(0, colon).c_str(), nullptr, 16));
            o.watch_size = static_cast<std::uint32_t>(std::strtoul(range.substr(colon + 1).c_str(), nullptr, 16));
        }
        if (!clock.empty()) {
            o.clock = parse_i64(clock);
            o.clock_given = true;
        }
        for (const auto& p : presses) {
            const auto [at, button] = at_vblank("press", p);
            o.presses.push_back({at, button_from_name(button)});
        }
        for (const auto& l : looks) {  // FROM-TO:YAW:PITCH
            unsigned long long from = 0, to = 0;
            float yaw = 0, pitch = 0;
            if (std::sscanf(l.c_str(), "%llu-%llu:%f:%f", &from, &to, &yaw, &pitch) != 4)
                throw Error(nx::format("--look takes FROM-TO:YAW:PITCH, not '{}'", l));
            constexpr float kRadians = 3.14159265f / 180.0f;
            o.looks.push_back({from, to, yaw * kRadians, pitch * kRadians});
        }
        for (const auto& u : ui) {
            auto [at, action] = at_vblank("ui", u);
            o.ui_script.push_back({at, std::move(action)});
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", game.name, e.what());
        o.exit_code = 2;
    }
    if (o.scale < 0) o.scale = 1;
    return o;
}

}  // namespace

int psprecore::platform::run_game(const GameInfo& game, int argc, char** argv) {
    const std::uint32_t kLoadBase = game.load_base;  // must match the base psprecomp translated for
    // Saves, settings and cooked media live in the working directory on desktop and in the app's external
    // files directory (Android/data/<package>/files) on Android, next to the extracted game folder.
    nx::string data_dir = ".";
#ifdef __ANDROID__
    if (const char* ext = SDL_GetAndroidExternalStoragePath()) data_dir = ext;
    g_log_tag = game.name;
    redirect_stderr_to_logcat(path_join(data_dir, nx::format("{}.log", game.name)));
    if (argc > 1) {  // launch-intent options (testing)
        std::fprintf(stderr, "arguments:");
        for (int i = 1; i < argc; ++i) std::fprintf(stderr, " [%s]", argv[i]);
        std::fprintf(stderr, "\n");
    }
    const Options o = parse_options(game, argc, argv, data_dir);
    if (o.exit_code >= 0) {
        finish_stderr();
        return o.exit_code;
    }
    const nx::string& game_dir = o.game_dir;
    // A folder created by `adb push` before the app's first start belongs to the shell user, and the app is then
    // denied access to it.
    const auto probe = nx::fs::open_file(path_join(game_dir, "PSP_GAME", "SYSDIR", "BOOT.BIN"));
    if (!probe.has_value()) {
        const bool denied = probe.error().kind == nx::fs::io_error::PermissionDenied;
        const nx::string message =
            denied ? "The app is not allowed to read:\n\n" + game_dir +
                         "\n\nThe folder was probably created before the app's first start (for example by adb push), so it "
                         "belongs to another user. Uninstall the app (this deletes its files folder), install it again, start "
                         "it once, then copy the game files. scripts/build-android.ps1 -InstallData does this in the right order."
                   : "Copy your extracted game folder (with PSP_GAME inside) and the cooked folder to:\n\n" + data_dir +
                         "\n\nExpected: " + game_dir +
                         "\n\nscripts/build-android.ps1 -InstallData can copy them from your PC.";
        std::fprintf(stderr, "%s\n", message.c_str());
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, denied ? "Game files not readable" : "Game files not found",
                                 message.c_str(), nullptr);
        finish_stderr();
        return 1;
    }
#else
    const Options o = parse_options(game, argc, argv, data_dir);
    if (o.exit_code >= 0) return o.exit_code;
    const nx::string& game_dir = o.game_dir;
#endif
    try {
        const bool trace = o.trace, headless = o.headless, no_audio = o.no_audio, unthrottled = o.unthrottled;
        const bool perf = o.perf, mute = o.mute, touch = o.touch;
        rt::set_timing_stats(perf || o.ge_bench);
        const std::uint64_t screenshot_vblank = o.screenshot_vblank, screenshot_every = o.screenshot_every;
        const nx::string& screenshot_path = o.screenshot_path;
        const nx::string& audio_dump = o.audio_dump;
        nx::string cooked_dir = o.cooked_dir;  // default below
        nx::string renderer = o.renderer;  // empty: the setting (Vulkan or software, switchable in the pause menu)
        const int scale = o.scale;         // 0: from the settings file
        const nx::string settings_path =
            o.settings.empty() ? path_join(data_dir, nx::format("{}.ini", game.name)) : o.settings;
        const std::uint64_t max_vblanks = o.vblanks, ge_dump_vblank = o.ge_dump_at, ge_bench_vblank = o.ge_bench;
        const std::uint64_t profile_vblank = o.profile, tex_dump_vblank = o.tex_dump_vblank;
        const nx::string& tex_dump_dir = o.tex_dump_dir;
        const std::uint64_t state_hash_every = o.state_hash, ram_dump_vblank = o.ram_dump_vblank;
        const nx::string& ram_dump_path = o.ram_dump_path;
        const nx::string &record_path = o.record, &replay_path = o.replay, &saves_dir = o.saves, &mods_dir = o.mods;
        std::int64_t clock_base = o.clock_given ? o.clock : static_cast<std::int64_t>(std::time(nullptr));
        bool clock_given = o.clock_given;
        const nx::vector<ScriptedPress>& presses = o.presses;
        const nx::vector<ScriptedUi>& ui_script = o.ui_script;
        const std::uint64_t window_shot_vblank = o.window_shot_vblank;
        const int window_shots = std::max(1, o.window_shots);
        const nx::string& window_shot_path = o.window_shot_path;
        if (o.ge_dump) rt::set_ge_dump(o.ge_dump, "ge_dump.txt");
        if (o.soft_threads) rt::gpu::set_soft_renderer_threads(o.soft_threads);

        rt::install_crash_handler();
        rt::GuestMemory memory;
        const auto boot = io::read_all(path_join(game_dir, "PSP_GAME", "SYSDIR", "BOOT.BIN"));
        const elf::File file(boot);
        const auto module = prx::parse(file);
        const auto image = prx::load_image(file, module, kLoadBase);
        // load_image is indexed by absolute address (the recompiler and `psprx image` need that); the guest
        // only needs the module itself, so the leading zeros below the load base are skipped. Writing the
        // whole vector at kLoadBase would place every byte 0x08804000 too high and overflow guest RAM.
        const std::span<const std::byte> body(image.data() + kLoadBase, image.size() - kLoadBase);
        memory.write(kLoadBase, body);

        rt::Kernel kernel(memory);
        rt::set_kernel(&kernel);
        kernel.trace_hle = trace;
        kernel.max_vblanks = max_vblanks;
        kernel.user_heap().alloc_at(kLoadBase, static_cast<std::uint32_t>(body.size()));
        // Recordings are self-contained: --record snapshots the save folder next to the recording (<file>.saves),
        // and --replay runs on a fresh copy of that snapshot (<file>.saves-run), so the game can save freely.
        nx::string saves = saves_dir.empty() ? path_join(data_dir, "saves") : saves_dir;
        if (!replay_path.empty() && saves_dir.empty()) {
            const nx::string snapshot = replay_path + ".saves", run = replay_path + ".saves-run";
            (void)nx::fs::remove_all(run);
            (void)nx::fs::create_directories(run);
            if (nx::fs::exists(snapshot)) (void)nx::fs::copy(snapshot, run, true);
            saves = run;
        }
        if (!record_path.empty()) {
            const nx::string snapshot = record_path + ".saves";
            (void)nx::fs::remove_all(snapshot);
            (void)nx::fs::create_directories(snapshot);
            if (nx::fs::exists(saves)) (void)nx::fs::copy(saves, snapshot, true);
        }
        rt::set_storage(game_dir, saves);
        // Mods: every folder, except language packs other than the chosen one (applied at start).
        const auto mods = platform::find_mods(mods_dir.empty() ? path_join(data_dir, "mods") : mods_dir);
        const nx::string language = platform::Settings::load(settings_path).language;
        const nx::vector<nx::string> active_mods = platform::active_mods(mods, language);
        if (!language.empty()) std::fprintf(stderr, "mods: language %s\n", language.c_str());
        if (game.mount) game.mount(active_mods);
        rt::gpu::scan_texture_replacements(active_mods);
        if (!o.tex_collect.empty()) {
            if (o.headless) std::fprintf(stderr, "--tex-collect needs the Vulkan renderer (not --headless)\n");
            else rt::gpu::start_texture_collection(o.tex_collect);
        }

        // Input recording / replay. A replay also fixes the guest clock to the recording's.
        nx::unique_ptr<InputReplay> replay;
        nx::unique_ptr<InputRecorder> recorder_in;
        if (!replay_path.empty()) {
            replay = nx::make_unique<InputReplay>(replay_path);
            if (!clock_given) clock_base = replay->clock_base();
            std::fprintf(stderr, "replaying %s (until vblank %llu)\n", replay_path.c_str(),
                         static_cast<unsigned long long>(replay->last_vblank()));
        }
        rt::set_wall_clock_base(clock_base);
        if (!record_path.empty()) recorder_in = nx::make_unique<InputRecorder>(record_path, clock_base);
        const bool deterministic_dialogs = replay || recorder_in;
        if (deterministic_dialogs) {
            rt::set_savedata_ui(true);
            rt::set_save_dialog_deferred(true);  // answers apply at the next vblank (same as in recordings)  // list dialogs wait for recorded / picked answers
            rt::set_save_dialog_deferred(true);
        }
        // Per vblank: applies due save picker answers and replayed input; records what the game sees.
        // `skip`: the player skipped the movie this vblank (ignored, like live input, while a replay runs).
        const auto input_step = [&](std::uint32_t& buttons, std::uint8_t& lx, std::uint8_t& ly, float& yaw,
                                    float& pitch, bool skip) {
            const std::uint64_t vblank = kernel.vblank_count();
            if (!ram_dump_path.empty() && vblank == ram_dump_vblank) {  // debugging: user RAM to a file
                std::FILE* f = io::open_c_file(ram_dump_path, "wb");
                if (f) {
                    std::fwrite(memory.ptr(0x08800000), 1, 0x01800000, f);
                    std::fclose(f);
                }
            }
            if (state_hash_every && vblank % state_hash_every == 0) {  // determinism checks: user RAM (24 MiB)
                std::uint64_t h = 0xCBF29CE484222325ull;
                const auto* p = reinterpret_cast<const std::uint64_t*>(memory.ptr(0x08800000));
                for (std::size_t i = 0; i < 0x01800000 / 8; ++i) h = (h ^ p[i]) * 0x100000001B3ull;
                std::fprintf(stderr, "state vblank %llu: %016llx\n", static_cast<unsigned long long>(vblank),
                             static_cast<unsigned long long>(h));
            }
            if (replay && !replay->finished()) {
                const int dialog = replay->advance(vblank, skip);
                if (dialog != -2) rt::answer_save_dialog(dialog);
                buttons = replay->buttons();
                lx = replay->lx();
                ly = replay->ly();
                yaw = replay->look_yaw();
                pitch = replay->look_pitch();
                if (replay->finished()) std::fprintf(stderr, "replay finished at vblank %llu\n", static_cast<unsigned long long>(vblank));
            }
            // Headless runs have no picker: answer with the slot the PSP dialog would focus (recorded like a pick).
            if (headless && !(replay && !replay->finished()))
                if (const rt::SaveDialog* d = rt::pending_save_dialog()) rt::answer_save_dialog(d->slots.empty() ? -1 : d->focus);
            if (skip && rt::movie_playing()) {
                rt::skip_movie();
                if (recorder_in) recorder_in->skip(vblank);
            }
            int slot = 0;
            if (rt::apply_save_dialog_answer(slot) && recorder_in) recorder_in->dialog(vblank, slot);
            if (recorder_in) {
                recorder_in->pad(vblank, buttons, lx, ly);
                recorder_in->look(vblank, yaw, pitch);
            }
        };
        if (cooked_dir.empty()) {
            cooked_dir = path_join(data_dir, "cooked");
            if (!nx::fs::exists(cooked_dir)) cooked_dir = path_join(parent_path(argv[0]), "cooked");
        }
        rt::set_cooked_root(cooked_dir);
        rt::gpu::scan_baked_textures(path_join(cooked_dir, "textures"));
        rt::hle_install();

        // module_start(argc = strlen(path) + 1, argv = path), as the PSP loader does.
        const nx::string boot_path = "disc0:/PSP_GAME/SYSDIR/EBOOT.BIN";
        const std::uint32_t args = kernel.user_heap().alloc(static_cast<std::uint32_t>(boot_path.size() + 1));
        memory.write_string(args, boot_path);

        // Window + presentation + input, unless running headless.
        nx::unique_ptr<platform::Display> display;
        nx::vector<std::uint32_t> frame;
        bool screenshot_pending = false;  // GPU renderer: capture at the next flip
        std::uint64_t screenshot_at = 0;
        const auto screenshot = [&](const nx::vector<std::uint32_t>& rgba, int w, int h) {
            const std::uint64_t n = screenshot_pending ? screenshot_at : kernel.vblank_count();
            if (screenshot_every && n % screenshot_every == 0) {
                char name[32];
                std::snprintf(name, sizeof name, "_%06llu.png", static_cast<unsigned long long>(n));
                rt::write_png(screenshot_path + name, rgba.data(), w, h);
            } else if (!screenshot_every && n == screenshot_vblank) {
                rt::write_png(screenshot_path, rgba.data(), w, h);
            }
        };
        // --ge-bench: the non-indexed draws of a few vblanks, copied out, then decoded and transformed repeatedly.
        struct BenchDraw {
            rt::ge::State state;
            float morph[8];
            std::uint32_t count = 0;
            nx::vector<std::uint8_t> bytes;
        };
        nx::vector<BenchDraw> bench_draws;
        nx::unique_ptr<platform::SamplingProfiler> profiler;
        nx::unique_ptr<platform::MemoryWatch> memory_watch;
        // --tex-dump: every texture drawn during two vblanks as DIR/ADDRESS_fFORMAT_WxH_HASH.png, and per draw the
        // texture and the texel rectangle it uses (finding the textures of a scene, e.g. for replacement).
        nx::hash_map<std::uint64_t, int> dumped_textures;
        const auto tex_dump_tick = [&] {
            if (!tex_dump_vblank) return;
            const std::uint64_t vblank = kernel.vblank_count();
            if (vblank == tex_dump_vblank + 2) rt::set_ge_draw_listener({});  // two vblanks: 30 fps scenes draw every other one
            if (vblank != tex_dump_vblank) return;
            (void)nx::fs::create_directories(tex_dump_dir);
            rt::set_ge_draw_listener(rt::GeDrawListener([&](const rt::ge::Draw& d) {
                const rt::ge::State& s = *d.state;
                if (!s.on(rt::ge::TEXTUREMAPENABLE) || !d.count) return;
                rt::gpu::TextureSampler t;
                t.setup(s, memory);
                nx::vector<std::uint32_t> rgba(static_cast<std::size_t>(t.width()) * static_cast<std::size_t>(t.height()));
                t.decode(0, rgba.data());
                const std::uint64_t h = rt::gpu::hash_bytes(rgba.data(), rgba.size() * 4);
                const std::uint32_t address = rt::ge::vram_canonical(((s.param(rt::ge::TEXBUFWIDTH0) & 0x0F0000) << 8) |
                                                                     (s.param(rt::ge::TEXADDR0) & 0xFFFFF0));
                // Named by the content hash texture replacement matches (a dumped file can be edited and put in a
                // mod's textures/ folder as it is), then address, format, size and the texel hash the button icon
                // table uses (gpu/button_icons.cpp).
                const std::uint64_t texels = rt::gpu::icon_texture_hash(t.level_data(0), t.level_size(0));
                const nx::string name = nx::format("{:016X}_{:08X}_f{}_{}x{}_{:016X}", rt::gpu::texture_content_hash(s, t),
                                                   address, t.format(), t.width(), t.height(), texels);
                if (dumped_textures.emplace(h, 0).second) {
                    rt::write_png(path_join(tex_dump_dir, name + ".png"), rgba.data(), t.width(), t.height());
                    // Palette textures also as index maps (indexed/, named by the texels alone), for replacements
                    // that keep every palette (psprecore/gpu/texture_replacement.hpp).
                    nx::vector<std::uint8_t> indices(rgba.size());
                    if (t.decode_indices(0, indices.data())) {
                        const std::uint32_t scale = t.format() == 4 ? 17 : 1;
                        nx::vector<std::uint32_t> grey(indices.size());
                        for (std::size_t i = 0; i < indices.size(); ++i) {
                            const std::uint32_t g = indices[i] * scale;
                            grey[i] = g | (g << 8) | (g << 16) | 0xFF000000u;
                        }
                        const nx::string dir = path_join(tex_dump_dir, "indexed");
                        (void)nx::fs::create_directories(dir);
                        rt::write_png(path_join(dir, nx::format("{:016X}_{:08X}_f{}_{}x{}.png", rt::gpu::texture_texels_hash(t),
                                                               address, t.format(), t.width(), t.height())),
                                      grey.data(), t.width(), t.height());
                    }
                }
                if (d.indices) {
                    std::fprintf(stderr, "tex-dump: prim %u x%u indexed %s\n", d.prim, d.count, name.c_str());
                    return;
                }
                float morph[8];
                for (int i = 0; i < 8; ++i) morph[i] = s.flt(static_cast<std::uint8_t>(rt::ge::MORPHWEIGHT0 + i));
                nx::vector<rt::ge::Vertex> v;
                rt::ge::decode_vertices(s.format, morph, memory.ptr(rt::ge::vram_canonical(d.vertices)), d.count, v);
                float u0 = 1e9f, v0 = 1e9f, u1 = -1e9f, v1 = -1e9f, x0 = 1e9f, y0 = 1e9f;
                for (const auto& p : v) {
                    u0 = std::min(u0, p.uv[0]);
                    v0 = std::min(v0, p.uv[1]);
                    u1 = std::max(u1, p.uv[0]);
                    v1 = std::max(v1, p.uv[1]);
                    x0 = std::min(x0, p.pos[0]);
                    y0 = std::min(y0, p.pos[1]);
                }
                std::fprintf(stderr, "tex-dump: prim %u x%u %s at (%.0f, %.0f) uv %.0f,%.0f-%.0f,%.0f %s\n", d.prim, d.count,
                             s.format.through ? "2d" : "3d", x0, y0, u0, v0, u1, v1, name.c_str());
            }));
        };
        const auto bench_tick = [&] {
            tex_dump_tick();
            if (profile_vblank && kernel.vblank_count() == profile_vblank) profiler = nx::make_unique<platform::SamplingProfiler>();
            for (const nx::string& poke : o.pokes) {
                const auto a = poke.find(':'), b = poke.find(':', a == nx::string::npos ? a : a + 1);
                if (b == nx::string::npos ||
                    std::strtoull(poke.substr(0, a).c_str(), nullptr, 10) != kernel.vblank_count())
                    continue;
                const auto addr = static_cast<std::uint32_t>(std::strtoul(poke.substr(a + 1, b - a - 1).c_str(), nullptr, 16));
                const auto value = static_cast<std::uint32_t>(std::strtoul(poke.substr(b + 1).c_str(), nullptr, 16));
                memory.write32(addr, value);
                std::fprintf(stderr, "poke: 0x%08X = 0x%08X at vblank %s\n", addr, value, poke.substr(0, a).c_str());
            }
            if (o.watch_vblank && kernel.vblank_count() == o.watch_vblank)
                memory_watch = nx::make_unique<platform::MemoryWatch>(o.watch_address, o.watch_size);
            if (!ge_bench_vblank) return;
            const std::uint64_t vblank = kernel.vblank_count();
            if (vblank == ge_bench_vblank) {
                rt::set_ge_draw_listener(rt::GeDrawListener([&](const rt::ge::Draw& d) {
                    if (d.indices || !d.count) return;
                    BenchDraw b;
                    b.state = *d.state;
                    for (int i = 0; i < 8; ++i) b.morph[i] = d.state->flt(static_cast<std::uint8_t>(rt::ge::MORPHWEIGHT0 + i));
                    b.count = d.count;
                    const auto size = static_cast<std::size_t>(d.count) * static_cast<std::size_t>(d.state->format.stride());
                    const std::uint8_t* p = memory.ptr(rt::ge::vram_canonical(d.vertices));
                    b.bytes.assign(p, p + size);
                    bench_draws.push_back(std::move(b));
                }));
                return;
            }
            if (vblank != ge_bench_vblank + 4) return;
            rt::set_ge_draw_listener({});
            std::size_t vertices = 0;
            for (const auto& b : bench_draws) vertices += b.count;
            nx::vector<nx::vector<rt::ge::Vertex>> decoded(bench_draws.size());
            rt::gpu::VertexPipeline vp;
            double checksum = 0;
            constexpr int kRounds = 200;
            const auto t0 = nx::timestamp::now();
            for (int round = 0; round < kRounds; ++round)
                for (std::size_t i = 0; i < bench_draws.size(); ++i) {
                    const auto& b = bench_draws[i];
                    rt::ge::decode_vertices(b.state.format, b.morph, b.bytes.data(), b.count, decoded[i]);
                    checksum += decoded[i][0].pos[0];
                }
            const auto t1 = nx::timestamp::now();
            for (int round = 0; round < kRounds; ++round)
                for (std::size_t i = 0; i < bench_draws.size(); ++i) {
                    const auto& b = bench_draws[i];
                    vp.configure(b.state, b.state.on(rt::ge::TEXTUREMAPENABLE), 256, 256, b.state.on(rt::ge::FOGENABLE));
                    for (const auto& v : decoded[i]) checksum += vp.transform(b.state, v).clip[3];
                }
            const auto t2 = nx::timestamp::now();
            const double n = static_cast<double>(vertices) * kRounds;
            {  // per format and lighting: where the transform time goes
                nx::map<std::pair<std::uint32_t, bool>, std::pair<double, std::size_t>> by_format;
                for (std::size_t i = 0; i < bench_draws.size(); ++i) {
                    const auto& b = bench_draws[i];
                    const auto s0 = nx::timestamp::now();
                    for (int round = 0; round < 20; ++round) {
                        vp.configure(b.state, b.state.on(rt::ge::TEXTUREMAPENABLE), 256, 256, b.state.on(rt::ge::FOGENABLE));
                        for (const auto& v : decoded[i]) checksum += vp.transform(b.state, v).clip[3];
                    }
                    auto& e = by_format[{b.state.format.raw & 0xFFFFFF, b.state.on(rt::ge::LIGHTINGENABLE)}];
                    e.first += static_cast<double>((nx::timestamp::now() - s0).nanos()) / 20;
                    e.second += b.count;
                }
                for (const auto& [k, e] : by_format)
                    std::fprintf(stderr, "  format %06X lighting %d: %zu vertices, %.1f ns/vertex, %.0f us per capture\n", k.first,
                                 k.second ? 1 : 0, e.second, e.first / static_cast<double>(e.second), e.first / 1000.0);
            }
            std::fprintf(stderr, "ge bench: %zu draws, %zu vertices; decode %.2f ns/vertex, transform %.2f ns/vertex (%g)\n",
                         bench_draws.size(), vertices, static_cast<double>((t1 - t0).nanos()) / n,
                         static_cast<double>((t2 - t1).nanos()) / n, checksum);
            kernel.request_quit();
        };
        platform::VulkanGeRenderer* vulkan_renderer = nullptr;
        platform::AudioDevice* audio_device = nullptr;
        auto perf_start = nx::timestamp::now(), perf_last = perf_start;
        std::uint64_t perf_frames = 0;
        double perf_worst_ms = 0.0;  // longest host time between two vblanks
        const auto perf_tick = [&] {
            if (!perf) return;
            const auto now = nx::timestamp::now();
            perf_worst_ms = std::max(perf_worst_ms, (now - perf_last).millis_f());
            perf_last = now;
            if (kernel.vblank_count() % 300 != 0) return;
            const double ms = (now - perf_start).millis_f();
            const std::uint64_t frames = rt::display_frame_count();
            std::fprintf(stderr, "perf vblank %llu: %.1f ms per vblank (%.2fx real time, worst %.1f ms), %llu frames\n",
                         static_cast<unsigned long long>(kernel.vblank_count()), ms / 300.0, 300.0 * 16.683 / ms,
                         perf_worst_ms, static_cast<unsigned long long>(frames - perf_frames));
            perf_worst_ms = 0.0;
            {
                static rt::ge::Stats last;
                const rt::ge::Stats g = rt::ge_stats();
                const auto per_vblank = [&](std::uint64_t now_ns, std::uint64_t then_ns) {
                    return static_cast<double>(now_ns - then_ns) / 1e6 / 300.0;
                };
                std::fprintf(stderr, "    ge: %.2f ms/vblank in lists (decode %.2f, renderer %.2f), %.0f draws, %.0f vertices per vblank\n",
                             per_vblank(g.run_ns, last.run_ns), per_vblank(g.decode_ns, last.decode_ns),
                             per_vblank(g.render_ns, last.render_ns), static_cast<double>(g.draws - last.draws) / 300.0,
                             static_cast<double>(g.vertices - last.vertices) / 300.0);
                last = g;
            }
            if (vulkan_renderer) {
                const auto st = vulkan_renderer->take_stats();
                std::fprintf(stderr, "    vulkan draw: transform %.2f ms/vblank, textures %.2f ms/vblank\n",
                             static_cast<double>(st.transform_ns) / 1e6 / 300.0, static_cast<double>(st.texture_ns) / 1e6 / 300.0);
                std::fprintf(stderr, "    vulkan: %.1f MB texture hashing, %llu uploads (%.2f Mtexels), %llu pipelines (%.1f ms, worst %.1f)\n",
                             static_cast<double>(st.hashed_bytes) / 1e6,
                             static_cast<unsigned long long>(st.uploads), static_cast<double>(st.uploaded_texels) / 1e6,
                             static_cast<unsigned long long>(st.pipelines), st.pipeline_ms, st.worst_pipeline_ms);
            }
            if (audio_device) {
                const auto st = audio_device->take_stats();
                std::fprintf(stderr, "    audio: %llu underruns, %.1f ms skipped, %.1f ms dropped, queue %.1f..%.1f ms, %llu fell behind\n",
                             static_cast<unsigned long long>(st.underruns), st.skipped_frames / 44.1,
                             st.dropped_frames / 44.1, st.min_queued / 44.1, st.max_queued / 44.1,
                             static_cast<unsigned long long>(kernel.fell_behind_count()));
            }
            perf_start = now;
            perf_frames = frames;
        };
        if (headless)
            kernel.on_vblank = nx::unique_function<void()>([&] {
                perf_tick();
                bench_tick();
                if (ge_dump_vblank && kernel.vblank_count() == ge_dump_vblank)
                    rt::set_ge_dump(rt::display_frame_count() + 1, "ge_dump.txt");
                std::uint32_t buttons = scripted_buttons(presses, kernel.vblank_count());
                std::uint8_t lx = 128, ly = 128;
                float yaw = 0, pitch = 0;
                scripted_look(o.looks, kernel.vblank_count(), yaw, pitch);
                input_step(buttons, lx, ly, yaw, pitch, false);
                rt::set_pad(buttons, lx, ly);
                rt::add_camera_look(yaw, pitch);
                int w = 0, h = 0;
                if (!screenshot_path.empty() && rt::read_framebuffer(frame, w, h)) screenshot(frame, w, h);
            });
        platform::Settings settings = platform::Settings::load(settings_path);
        if (scale) settings.scale = scale;
        nx::unique_ptr<platform::Overlay> overlay;
        nx::unique_ptr<platform::AudioDevice> audio;
        nx::unique_ptr<WavRecorder> recorder;
        if (!headless && !no_audio) audio = nx::make_unique<platform::AudioDevice>();
        audio_device = audio.get();
        if (!audio_dump.empty()) recorder = nx::make_unique<WavRecorder>(audio_dump, audio.get());
        rt::AudioSink* const audio_sink = recorder ? static_cast<rt::AudioSink*>(recorder.get()) : audio.get();
        rt::set_audio_sink(audio_sink);
        // Skipping a movie fast-forwards through it (rt::movie_skipping): unthrottled, silent, a few presents a second.
        bool fast_forward = false, frame_drawn = true;
        int skipped_frames = 0;  // frames not drawn in a row (frame skipping)
        // Smooth motion (frame interpolation): frames are shown at every vblank instead of when the game finishes
        // them; between two of the game's frames, ones drawn part of the way from the earlier to the later.
        bool smooth = false, new_frame = false;
        int frame_period = 1, between_step = 0;  // vblanks between the game's last two frames; steps shown since
        std::uint64_t last_flip_vblank = 0;
        platform::VulkanGeRenderer::Output finished{};  // the game's last finished frame
        nx::timestamp fast_forward_present = nx::timestamp::now();
        if (!headless) {
            int window_w = 960, window_h = 544;
            if (const auto x = o.window.find('x'); x != nx::string::npos) {
                window_w = std::max(64, std::atoi(o.window.substr(0, x).c_str()));
                window_h = std::max(64, std::atoi(o.window.substr(x + 1).c_str()));
            }
            display = nx::make_unique<platform::Display>(game.title, window_w, window_h);
            display->set_touch_controls(touch);
            // Android blocks the game inside poll_events() while the app is in the background; the sound device
            // would play on.
            display->set_background_handler(nx::unique_function<void(bool)>([&](bool background) {
                if (audio) audio->set_paused(background);
            }));
            // The renderer can change at run time (pause menu) unless the command line fixed it.
            const bool renderer_choice = renderer.empty();
            if (renderer_choice) renderer = settings.software_renderer ? "soft" : "vulkan";
            bool gpu = renderer != "soft";
            bool switch_renderer = false;
            const auto apply = [&](const platform::Settings& s) {
                if (vulkan_renderer) {
                    vulkan_renderer->set_scale(s.scale);
                    vulkan_renderer->set_msaa(s.msaa);
                }
                if (display->fullscreen() != s.fullscreen) display->set_fullscreen(s.fullscreen);
                display->set_vsync(s.vsync);
                display->set_refresh_rate(s.refresh_60hz ? 60.0f : 0.0f);
                display->set_button_glyphs(s.button_glyphs);
                if (audio) audio->set_volume(mute ? 0.0f : static_cast<float>(s.volume) / 100.0f);
                kernel.realtime = s.frame_limit && !unthrottled && !fast_forward;
                if (renderer_choice) switch_renderer = s.software_renderer == gpu;  // applied when the menu closes
                if (rt::gpu::enable_baked_textures(s.hd_textures) && vulkan_renderer) vulkan_renderer->reload_textures();
            };
            apply(settings);
            platform::Overlay::Hooks hooks;
            hooks.apply = nx::unique_function<void(const platform::Settings&)>(apply);  // a copy: also called below
            hooks.quit = nx::unique_function<void()>([&] { kernel.request_quit(); });
            hooks.renderer_choice = renderer_choice;
            hooks.gpu_renderer = gpu;
            hooks.languages = platform::mod_languages(mods);
            hooks.hd_textures = rt::gpu::have_baked_textures();
            hooks.active_language = language;
            overlay = nx::make_unique<platform::Overlay>(*display, settings, std::move(hooks));
            display->set_ui(nx::unique_function<void()>([&] { overlay->draw(); }));
            rt::set_savedata_ui(true);
            // The pause menu stops the game inside on_vblank (the host main thread) until it is closed.
            bool window_shot_armed = false, window_shot_taken = false;
            const auto arm_window_shot = [&] {
                window_shot_armed = false;
                using Capture = platform::Display::CaptureHandler;
                nx::string path = window_shot_path;
                if (window_shots > 1 && path.ends_with(".png"))  // a series: name_<vblank>.png
                    path = nx::format("{}_{}.png", nx::string_view(path).substr(0, path.size() - 4), kernel.vblank_count());
                display->capture_next_present(Capture([&, path](const std::uint32_t* rgba, int w, int h) {
                    rt::write_png(path, rgba, w, h);
                    window_shot_taken = true;
                }));
            };
            // Skipping a movie: any press shows "Hold <Circle> to skip"; holding Circle (or a finger on the picture)
            // fills its ring, and a full ring skips. Circle does not reach the game while the prompt is up.
            constexpr float kHoldSeconds = 0.8f, kShowSeconds = 3.0f, kVblank = 1.0f / 59.94f;
            float skip_shown = 0.0f, skip_progress = 0.0f;
            std::uint32_t skip_previous = 0;
            const auto skip_prompt_step = [&](std::uint32_t& buttons) {
                const std::uint32_t pressed = buttons & ~skip_previous;
                skip_previous = buttons;
                bool skip = false;
                if (!rt::movie_skippable() || overlay->wants_input()) {
                    skip_shown = skip_progress = 0.0f;
                } else {
                    const bool held = (buttons & platform::kCircle) || display->screen_held();
                    if (pressed || held) skip_shown = kShowSeconds;
                    skip_progress = held ? skip_progress + kVblank / kHoldSeconds
                                         : std::max(0.0f, skip_progress - 2.0f * kVblank / kHoldSeconds);
                    if (skip_progress >= 1.0f) {
                        skip = true;
                        skip_shown = skip_progress = 0.0f;
                    }
                    if (skip_shown > 0.0f) buttons &= ~platform::kCircle;
                    skip_shown = std::max(0.0f, skip_shown - kVblank);
                }
                overlay->set_skip_prompt(std::min(1.0f, skip_shown / 0.3f), std::min(1.0f, skip_progress));
                return skip;
            };
            const auto pause = [&] {
                const platform::Settings before = settings;
                display->set_ui_modal(true);
                for (int frame = 0; overlay->menu_open() && !kernel.quit_requested(); ++frame) {
                    if (window_shot_taken && !ui_script.empty()) overlay->close_menu();  // scripted test run
                    if (window_shot_armed && frame == 5) arm_window_shot();  // auto-sized windows show after a frame
                    if (!display->poll_events()) kernel.request_quit();
                    if (display->take_menu_request()) overlay->handle_menu_request();
                    if (!display->present_again()) nx::this_thread::sleep_for(std::chrono::milliseconds(16));
                }
                display->set_ui_modal(overlay->wants_input());
                if (settings != before && !settings.save(settings_path))
                    std::fprintf(stderr, "could not write %s\n", settings_path.c_str());
            };
            {
                rt::set_ge_renderer_factory(rt::GeRendererFactory([&](rt::GuestMemory& mem) {
                    vulkan_renderer = nullptr;
                    if (!gpu) return rt::gpu::make_soft_renderer(mem);
                    auto r = std::make_unique<platform::VulkanGeRenderer>(display->vulkan(), mem, settings.scale,
                                                                          path_join(data_dir, "vulkan_pipelines.bin"));
                    r->set_msaa(settings.msaa);
                    vulkan_renderer = r.get();
                    if (renderer != "compare") return std::unique_ptr<rt::ge::Renderer>(std::move(r));
                    r->set_guest_copies(false);  // the software renderer already copies in guest memory
                    return std::unique_ptr<rt::ge::Renderer>(
                        std::make_unique<TeeRenderer>(rt::gpu::make_soft_renderer(mem), std::move(r)));
                }));
                rt::set_flip_handler(rt::FlipHandler([&](std::uint32_t address, std::uint32_t stride,
                                                         std::uint32_t format) {
                    if (!vulkan_renderer) return;
                    const bool capture = screenshot_pending;
                    const auto out = vulkan_renderer->finish_frame(address, stride, static_cast<int>(format), capture ? &frame : nullptr);
                    const std::uint64_t now_vblank = kernel.vblank_count();
                    frame_period = static_cast<int>(std::clamp<std::uint64_t>(now_vblank - last_flip_vblank, 1, 8));
                    last_flip_vblank = now_vblank;
                    // Smooth motion: shown from the next vblank on (on_vblank). Fast-forwarding: only every frame
                    // due for a present (10 a second) is drawn and shown.
                    if (smooth && out.image && frame_drawn && !fast_forward) {
                        finished = out;
                        new_frame = true;
                    } else if (out.image && frame_drawn) {
                        display->present_image(out.image, out.width, out.height, out.x, out.y);
                        fast_forward_present = nx::timestamp::now();
                    }
                    // Frame skipping: when the game runs behind real time, the next frame is not drawn (a few in a
                    // row at most, so the picture keeps moving), so the game keeps its speed instead of slowing down.
                    constexpr std::int64_t kSkipBehindUs = 20'000;
                    constexpr int kMaxSkipped = 3;
                    if (fast_forward)
                        frame_drawn = nx::timestamp::now() - fast_forward_present >= nx::duration::from_millis(100);
                    else
                        frame_drawn = !(settings.frame_skip && kernel.realtime && kernel.behind_us() > kSkipBehindUs &&
                                        skipped_frames < kMaxSkipped);
                    skipped_frames = frame_drawn ? 0 : skipped_frames + 1;
                    vulkan_renderer->set_skip_draws(!frame_drawn);
                    if (capture && renderer == "compare") {
                        nx::vector<std::uint32_t> ref;
                        int w = 0, h = 0;
                        if (rt::read_framebuffer(ref, w, h) && w * vulkan_renderer->scale() == out.width) {
                            double mean = 0, bad = 0;
                            const auto composite = compare_frames(ref, frame, w, h, vulkan_renderer->scale(), mean, bad);
                            std::fprintf(stderr, "compare vblank %llu: mean difference %.2f, %.2f%% of pixels over 32\n",
                                         static_cast<unsigned long long>(screenshot_at), mean, bad);
                            screenshot(composite, w * 3, h);
                        }
                        screenshot_pending = false;
                    } else if (capture) {
                        screenshot(frame, out.width, out.height);
                        screenshot_pending = false;
                    }
                }));
            }
            // Host time between the end of one vblank's handling and the start of the next (the game running and
            // rendering): long gaps are logged, so freezes reported from devices leave a trace in the log.
            auto vblank_done = nx::timestamp::now();
            kernel.on_vblank = nx::unique_function<void()>([&, pause, arm_window_shot] {
                const auto vblank_start = nx::timestamp::now();
                if (const nx::duration gap = vblank_start - vblank_done;
                    gap.millis() >= 1000 && !unthrottled)
                    std::fprintf(stderr, "stall: %lld ms before vblank %llu\n", static_cast<long long>(gap.millis()),
                                 static_cast<unsigned long long>(kernel.vblank_count()));
                perf_tick();
                bench_tick();
                const std::uint64_t vblank = kernel.vblank_count();
                if (ge_dump_vblank && vblank == ge_dump_vblank) rt::set_ge_dump(rt::display_frame_count() + 1, "ge_dump.txt");
                for (const auto& u : ui_script) {
                    if (u.vblank != vblank) continue;
                    if (u.action == "menu") overlay->handle_menu_request();
                    else if (u.action == "enter") display->inject_key(SDLK_RETURN);
                    else if (u.action == "escape") display->inject_key(SDLK_ESCAPE);
                    else if (u.action == "up") display->inject_key(SDLK_UP);
                    else if (u.action == "down") display->inject_key(SDLK_DOWN);
                    else if (u.action == "space") display->inject_key(SDLK_SPACE);
                    else if (u.action == "minimize" || u.action == "restore") display->set_minimized(u.action == "minimize");
                    else if (u.action == "skip-movie") rt::skip_movie();
                    else if (u.action == "renderer:vulkan" || u.action == "renderer:software") {
                        settings.software_renderer = u.action == "renderer:software";
                        apply(settings);
                    }
                    else if (u.action.starts_with("tap:")) {  // tap:X,Y in window-normalised coordinates
                        const auto comma = u.action.find(',');
                        if (comma != nx::string::npos)
                            display->inject_finger(static_cast<float>(parse_double(u.action.substr(4, comma - 4))), static_cast<float>(parse_double(u.action.substr(comma + 1))), 8);
                    }
                }
                if (!window_shot_path.empty() && vblank >= window_shot_vblank &&
                    vblank < window_shot_vblank + static_cast<std::uint64_t>(window_shots))
                    window_shot_armed = true;
                if (!display->poll_events()) kernel.request_quit();
                if (vulkan_renderer) vulkan_renderer->set_fill_aspect(settings.fill_screen ? display->aspect() : 0.0f);
                // Smooth motion: one frame per vblank. After the game finishes a frame, the frames between it and the
                // one before (1/period, 2/period... of the way), then the frame itself; at 60 frames a second or
                // more, just the frame.
                smooth = vulkan_renderer && settings.smooth_motion && !fast_forward;
                if (vulkan_renderer) vulkan_renderer->set_interpolation(smooth);
                if (smooth && finished.image) {
                    between_step = new_frame ? 1 : between_step + 1;
                    new_frame = false;
                    if (between_step < frame_period && frame_period <= 4) {
                        if (const auto mid = vulkan_renderer->render_between(static_cast<float>(between_step) / frame_period);
                            mid.image)
                            display->present_image(mid.image, mid.width, mid.height, mid.x, mid.y);
                    } else if (between_step == frame_period || (between_step == 1 && frame_period > 4)) {
                        display->present_image(finished.image, finished.width, finished.height, finished.x, finished.y);
                    }
                } else if (!smooth) {
                    finished = {};
                }
                if (vulkan_renderer) {  // follows the setting and, for Auto, the gamepad plugged in
                    const platform::ButtonGlyphs g = display->button_glyphs();
                    vulkan_renderer->set_button_icons(g == platform::ButtonGlyphs::Xbox       ? rt::gpu::IconStyle::Xbox
                                                      : g == platform::ButtonGlyphs::Nintendo ? rt::gpu::IconStyle::Nintendo
                                                      : g == platform::ButtonGlyphs::Keyboard ? rt::gpu::IconStyle::Keyboard
                                                                                              : rt::gpu::IconStyle::PlayStation);
                }
                if (display->in_background()) {  // Android: hold the game until the app returns
                    if (audio) audio->set_paused(true);
                    while (display->in_background() && !kernel.quit_requested()) {
                        nx::this_thread::sleep_for(std::chrono::milliseconds(50));
                        if (!display->poll_events()) kernel.request_quit();
                    }
                    if (audio) audio->set_paused(false);
                }
                if (display->take_menu_request()) overlay->handle_menu_request();
                if (overlay->menu_open()) pause();
                if (switch_renderer) {
                    switch_renderer = false;
                    gpu = !gpu;
                    renderer = gpu ? "vulkan" : "soft";
                    display->forget_last_image();
                    rt::reset_ge_renderer();
                    std::fprintf(stderr, "renderer: %s\n", gpu ? "Vulkan" : "software");
                }
                if (window_shot_armed) arm_window_shot();
                display->set_ui_modal(overlay->wants_input());  // e.g. the save dialog
                const auto pad = overlay->wants_input() ? platform::PadState{} : display->pad();
                std::uint32_t buttons = pad.buttons | scripted_buttons(presses, kernel.vblank_count());
                std::uint8_t lx = pad.lx, ly = pad.ly;
                float yaw = pad.look_yaw, pitch = pad.look_pitch;
                scripted_look(o.looks, kernel.vblank_count(), yaw, pitch);
                const bool skip = skip_prompt_step(buttons);
                input_step(buttons, lx, ly, yaw, pitch, skip);
                rt::set_pad(buttons, lx, ly);
                rt::add_camera_look(yaw, pitch);
                if (rt::movie_skipping() != fast_forward) {
                    fast_forward = !fast_forward;
                    std::fprintf(stderr, "fast-forward %s at vblank %llu\n", fast_forward ? "on" : "off",
                                 static_cast<unsigned long long>(vblank));
                    kernel.realtime = settings.frame_limit && !unthrottled && !fast_forward;
                    rt::set_audio_sink(fast_forward ? nullptr : audio_sink);
                    overlay->set_skipping(fast_forward);
                    if (!fast_forward && vulkan_renderer) {  // the frame in progress is drawn and shown again
                        frame_drawn = true;
                        vulkan_renderer->set_skip_draws(false);
                    }
                }
                if (gpu && !screenshot_path.empty()) {  // captured at the next flip
                    const std::uint64_t n = kernel.vblank_count();
                    if ((screenshot_every && n % screenshot_every == 0) || (!screenshot_every && n == screenshot_vblank)) {
                        screenshot_pending = true;
                        screenshot_at = n;
                    }
                }
                int w = 0, h = 0;
                if (!gpu && rt::read_framebuffer(frame, w, h)) {
                    display->present(frame.data(), w, h);
                    if (!screenshot_path.empty()) screenshot(frame, w, h);
                }
                vblank_done = nx::timestamp::now();
            });
        }

        std::fprintf(stderr, "%s loaded at 0x%08X (%zu bytes), entry 0x%08X\n", module.name.c_str(), kLoadBase,
                     body.size(), module.entry);
        if (game.loaded) game.loaded();
        const int code = kernel.run(module.entry, static_cast<std::uint32_t>(boot_path.size() + 1), args);
        std::fprintf(stderr, "stopped after %.3f s of guest time (%llu vblanks)\n",
                     static_cast<double>(kernel.now()) / 1e6, static_cast<unsigned long long>(kernel.vblank_count()));
        if (profiler) profiler->report(stderr);
        if (memory_watch) memory_watch->report(stderr);
        if (recorder) recorder->flush();
        if (recorder_in) recorder_in->flush();
        if (trace || headless) kernel.dump_threads(stderr);
        finish_stderr();
        std::_Exit(code);  // guest threads may still be parked on their batons
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", game.name, e.what());
        finish_stderr();
        return 1;
    }
}
