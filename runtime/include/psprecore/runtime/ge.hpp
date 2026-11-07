#pragma once

// GE (PSP GPU) command processor. Display lists are sequences of 32-bit words: command in bits 24-31,
// parameter in bits 0-23. This module walks lists in guest memory (jumps, calls, BASE/OFFSET addressing,
// stall addresses), keeps the register/matrix/CLUT state, decodes vertices and hands draws and block
// transfers to a Renderer (the software reference renderer now, the Vulkan renderer later).

#include <array>
#include <bit>
#include <cstdint>
#include <memory>

#include "core/foundation/core/callable.h"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::rt {

class GuestMemory;

namespace ge {

enum Cmd : std::uint8_t {
    NOP = 0x00, VADDR = 0x01, IADDR = 0x02, PRIM = 0x04, BEZIER = 0x05, SPLINE = 0x06, BOUNDINGBOX = 0x07,
    JUMP = 0x08, BJUMP = 0x09, CALL = 0x0A, RET = 0x0B, END = 0x0C, SIGNAL = 0x0E, FINISH = 0x0F,
    BASE = 0x10, VERTEXTYPE = 0x12, OFFSETADDR = 0x13, ORIGIN = 0x14, REGION1 = 0x15, REGION2 = 0x16,
    LIGHTINGENABLE = 0x17, LIGHTENABLE0 = 0x18, DEPTHCLAMPENABLE = 0x1C, CULLFACEENABLE = 0x1D,
    TEXTUREMAPENABLE = 0x1E, FOGENABLE = 0x1F, DITHERENABLE = 0x20, ALPHABLENDENABLE = 0x21,
    ALPHATESTENABLE = 0x22, ZTESTENABLE = 0x23, STENCILTESTENABLE = 0x24, ANTIALIASENABLE = 0x25,
    PATCHCULLENABLE = 0x26, COLORTESTENABLE = 0x27, LOGICOPENABLE = 0x28,
    BONEMATRIXNUMBER = 0x2A, BONEMATRIXDATA = 0x2B, MORPHWEIGHT0 = 0x2C,
    PATCHDIVISION = 0x36, PATCHPRIMITIVE = 0x37, PATCHFACING = 0x38,
    WORLDMATRIXNUMBER = 0x3A, WORLDMATRIXDATA = 0x3B, VIEWMATRIXNUMBER = 0x3C, VIEWMATRIXDATA = 0x3D,
    PROJMATRIXNUMBER = 0x3E, PROJMATRIXDATA = 0x3F, TGENMATRIXNUMBER = 0x40, TGENMATRIXDATA = 0x41,
    VIEWPORTXSCALE = 0x42, VIEWPORTYSCALE = 0x43, VIEWPORTZSCALE = 0x44,
    VIEWPORTXCENTER = 0x45, VIEWPORTYCENTER = 0x46, VIEWPORTZCENTER = 0x47,
    TEXSCALEU = 0x48, TEXSCALEV = 0x49, TEXOFFSETU = 0x4A, TEXOFFSETV = 0x4B, OFFSETX = 0x4C, OFFSETY = 0x4D,
    SHADEMODE = 0x50, REVERSENORMAL = 0x51, MATERIALUPDATE = 0x53, MATERIALEMISSIVE = 0x54,
    MATERIALAMBIENT = 0x55, MATERIALDIFFUSE = 0x56, MATERIALSPECULAR = 0x57, MATERIALALPHA = 0x58,
    MATERIALSPECULARCOEF = 0x5B, AMBIENTCOLOR = 0x5C, AMBIENTALPHA = 0x5D, LIGHTMODE = 0x5E, LIGHTTYPE0 = 0x5F,
    LIGHTPOSITION0 = 0x63, LIGHTDIRECTION0 = 0x6F, LIGHTATTENUATION0 = 0x7B, LIGHTSPOTEXPONENT0 = 0x87,
    LIGHTSPOTCUTOFF0 = 0x8B, LIGHTCOLOR0 = 0x8F, CULL = 0x9B, FRAMEBUFPTR = 0x9C, FRAMEBUFWIDTH = 0x9D,
    ZBUFPTR = 0x9E, ZBUFWIDTH = 0x9F, TEXADDR0 = 0xA0, TEXBUFWIDTH0 = 0xA8, CLUTADDR = 0xB0, CLUTADDRUPPER = 0xB1,
    TRANSFERSRC = 0xB2, TRANSFERSRCW = 0xB3, TRANSFERDST = 0xB4, TRANSFERDSTW = 0xB5, TEXSIZE0 = 0xB8,
    TEXMAPMODE = 0xC0, TEXSHADELS = 0xC1, TEXMODE = 0xC2, TEXFORMAT = 0xC3, LOADCLUT = 0xC4, CLUTFORMAT = 0xC5,
    TEXFILTER = 0xC6, TEXWRAP = 0xC7, TEXLEVEL = 0xC8, TEXFUNC = 0xC9, TEXENVCOLOR = 0xCA, TEXFLUSH = 0xCB,
    TEXSYNC = 0xCC, FOG1 = 0xCD, FOG2 = 0xCE, FOGCOLOR = 0xCF, TEXLODSLOPE = 0xD0, FRAMEBUFPIXFORMAT = 0xD2,
    CLEARMODE = 0xD3, SCISSOR1 = 0xD4, SCISSOR2 = 0xD5, MINZ = 0xD6, MAXZ = 0xD7, COLORTEST = 0xD8,
    COLORREF = 0xD9, COLORTESTMASK = 0xDA, ALPHATEST = 0xDB, STENCILTEST = 0xDC, STENCILOP = 0xDD, ZTEST = 0xDE,
    BLENDMODE = 0xDF, BLENDFIXEDA = 0xE0, BLENDFIXEDB = 0xE1, DITHER0 = 0xE2, LOGICOP = 0xE6,
    ZWRITEDISABLE = 0xE7, MASKRGB = 0xE8, MASKALPHA = 0xE9, TRANSFERSTART = 0xEA, TRANSFERSRCPOS = 0xEB,
    TRANSFERDSTPOS = 0xEC, TRANSFERSIZE = 0xEE,
};

enum class Prim : std::uint8_t { Points, Lines, LineStrip, Triangles, TriangleStrip, TriangleFan, Sprites };

// GE 24-bit float: the parameter is the top 24 bits of an IEEE single.
inline float f24(std::uint32_t param) { return std::bit_cast<float>(param << 8); }

// Decoded VERTEXTYPE. Component order in memory: weights, texcoord, color, normal, position; each aligned to its
// own size, the vertex to its largest component. With morphing, `morphs` such vertices follow each other.
struct VertexFormat {
    std::uint32_t raw = 0;
    int tc = 0, color = 0, normal = 0, pos = 0, weight = 0, index = 0;  // field encodings from VERTEXTYPE
    int weights = 1, morphs = 1;
    bool through = false;
    int tc_off = 0, color_off = 0, normal_off = 0, pos_off = 0;
    int size = 0;  // bytes per morph target
    int stride() const { return size * morphs; }
    int index_size() const { return index == 1 ? 1 : index == 2 ? 2 : index == 3 ? 4 : 0; }
    static VertexFormat decode(std::uint32_t vertex_type);
};

// A decoded vertex (morph targets already blended). Non-through attributes are normalized (s16 / 32768 etc.);
// through-mode positions are screen pixels and texcoords texels, as the hardware interprets them.
struct Vertex {
    float pos[3]{};
    float uv[2]{};
    float color[4]{1.0f, 1.0f, 1.0f, 1.0f};  // 0..1
    float normal[3]{};
    float weights[8]{};
};

// Everything a renderer needs: raw registers plus the state uploaded through data ports.
struct State {
    std::array<std::uint32_t, 256> regs{};
    float world[12]{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};  // 4x3, column major: x' = m0*x + m3*y + m6*z + m9
    float view[12]{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    float proj[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    float tgen[12]{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    float bones[8][12]{};
    std::array<std::uint8_t, 1024> clut{};
    VertexFormat format;
    std::uint32_t vertex_address = 0;  // where the draw's vertices start (guest address)

    std::uint32_t param(std::uint8_t cmd) const { return regs[cmd] & 0xFFFFFF; }
    float flt(std::uint8_t cmd) const { return f24(param(cmd)); }
    bool on(std::uint8_t cmd) const { return (regs[cmd] & 1) != 0; }
};

class Renderer {
public:
    virtual ~Renderer() = default;
    virtual void draw(const State& state, Prim prim, const nx::vector<Vertex>& vertices) = 0;
    virtual void transfer(const State& state) = 0;  // TRANSFERSTART block copy
    // Completes queued drawing. A renderer may defer work (the GE runs asynchronously on the PSP too); it is
    // flushed wherever the game can observe the results: a list's end, signal/finish handlers, sceGeDrawSync /
    // sceGeListSync, the display, and GE reads of VRAM the queued work may write.
    virtual void flush() {}
};

// Normalizes VRAM mirror addresses (0x04200000 and up) to the 2 MiB of real VRAM.
inline std::uint32_t vram_canonical(std::uint32_t addr) {
    addr &= 0x0FFFFFFF;
    return (addr & 0x0F000000) == 0x04000000 ? 0x04000000 | (addr & 0x1FFFFF) : addr;
}

nx::string_view command_name(std::uint8_t cmd);

// Decodes `count` consecutive vertices of format `f` at `data`, as draws do (benchmarks, tools).
void decode_vertices(const VertexFormat& f, const float* morph_weights, const std::uint8_t* data, std::uint32_t count,
                     nx::vector<Vertex>& out);

struct Draw {
    std::uint32_t list_pc = 0;  // address of the PRIM command
    std::uint32_t prim = 0;     // GE primitive type (0 points .. 6 sprites)
    std::uint32_t count = 0;
    std::uint32_t vertex_type = 0;
    std::uint32_t vertices = 0, indices = 0;  // guest addresses
    const State* state = nullptr;             // the state the draw uses (valid during the callback)
};

struct Stats {
    std::uint64_t lists = 0, commands = 0, draws = 0, vertices = 0, transfers = 0;
    // Host time (ns): list processing in total (signal handlers included), vertex decoding, the renderer's draws.
    std::uint64_t run_ns = 0, decode_ns = 0, render_ns = 0;
};

class Processor {
public:
    explicit Processor(GuestMemory& memory) : memory_(memory) {}

    // sceGeListEnQueue(Head): runs the list until it ends or reaches the stall address. Returns a list id.
    int enqueue(std::uint32_t list, std::uint32_t stall, bool head);
    // sceGeListUpdateStallAddr: resumes a stalled list.
    void update_stall(int id, std::uint32_t stall);
    bool list_done(int id) const;

    const std::array<std::uint32_t, 256>& registers() const { return state_.regs; }
    const State& state() const { return state_; }
    const Stats& stats() const { return stats_; }
    void set_renderer(std::unique_ptr<Renderer> renderer) { renderer_ = std::move(renderer); }
    void flush() {
        if (renderer_) renderer_->flush();
    }

    // Optional listeners.
    using DrawListener = nx::unique_function<void(const Draw&)>;
    using CommandListener = nx::unique_function<void(std::uint32_t pc, std::uint32_t word)>;
    using IdHandler = nx::unique_function<void(std::uint32_t id)>;
    DrawListener on_draw;
    CommandListener on_command;  // every executed command (dumps)
    // SIGNAL + END: the game's signal handler (sceGeSetCallback) runs with the signal id, then the list goes on.
    IdHandler on_signal;
    // FINISH + END: the list is complete; the finish handler runs with the FINISH id.
    IdHandler on_finish;

private:
    struct List {
        std::uint32_t pc = 0, stall = 0;
        bool done = false;
        nx::vector<std::uint32_t> call_stack;
        std::uint32_t signal = 0;  // parameter of a SIGNAL awaiting its END
        bool signal_pending = false;
        std::uint32_t finish = 0;
        bool finish_pending = false;
    };
    void run(List& list);
    void draw(std::uint32_t param);
    void patch(std::uint32_t param, bool bezier);
    std::uint32_t address(std::uint32_t param) const { return ((base_ | param) + offset_) & 0x0FFFFFFF; }

    GuestMemory& memory_;
    State state_;
    std::unique_ptr<Renderer> renderer_;
    nx::vector<Vertex> vertices_;
    std::uint32_t base_ = 0, offset_ = 0, vaddr_ = 0, iaddr_ = 0;
    std::uint32_t world_index_ = 0, view_index_ = 0, proj_index_ = 0, tgen_index_ = 0, bone_index_ = 0;
    bool bbox_visible_ = true;
    bool warned_patches_ = false;
    bool warned_patch_prim_ = false;  // non-triangle PATCHPRIMITIVE already reported
    std::uint32_t warned_signals_ = 0;  // bit per unsupported signal behaviour already reported
    // Lists stay where they are while running: a signal handler may enqueue more (nx::map moves its elements).
    nx::map<int, nx::unique_ptr<List>> lists_;
    int next_id_ = 1;
    Stats stats_;
};

// One-line human-readable decoding of a command word (used by --ge-dump).
nx::string describe(std::uint32_t word);

}  // namespace ge
}  // namespace psprecore::rt
