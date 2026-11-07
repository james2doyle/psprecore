#include "psprecore/runtime/ge.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/memory.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/core/time.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::rt::ge {

namespace {

// GE command names (public knowledge of the command set, as used by the homebrew PSPSDK's libgu).
constexpr std::array<nx::string_view, 256> kNames = [] {
    std::array<nx::string_view, 256> n{};
    const auto set = [&](int i, nx::string_view s) { n[static_cast<std::size_t>(i)] = s; };
    set(0x00, "NOP"); set(0x01, "VADDR"); set(0x02, "IADDR"); set(0x04, "PRIM"); set(0x05, "BEZIER");
    set(0x06, "SPLINE"); set(0x07, "BOUNDINGBOX"); set(0x08, "JUMP"); set(0x09, "BJUMP"); set(0x0A, "CALL");
    set(0x0B, "RET"); set(0x0C, "END"); set(0x0E, "SIGNAL"); set(0x0F, "FINISH"); set(0x10, "BASE");
    set(0x12, "VERTEXTYPE"); set(0x13, "OFFSETADDR"); set(0x14, "ORIGIN"); set(0x15, "REGION1"); set(0x16, "REGION2");
    set(0x17, "LIGHTINGENABLE"); set(0x18, "LIGHTENABLE0"); set(0x19, "LIGHTENABLE1"); set(0x1A, "LIGHTENABLE2");
    set(0x1B, "LIGHTENABLE3"); set(0x1C, "DEPTHCLAMPENABLE"); set(0x1D, "CULLFACEENABLE"); set(0x1E, "TEXTUREMAPENABLE");
    set(0x1F, "FOGENABLE"); set(0x20, "DITHERENABLE"); set(0x21, "ALPHABLENDENABLE"); set(0x22, "ALPHATESTENABLE");
    set(0x23, "ZTESTENABLE"); set(0x24, "STENCILTESTENABLE"); set(0x25, "ANTIALIASENABLE"); set(0x26, "PATCHCULLENABLE");
    set(0x27, "COLORTESTENABLE"); set(0x28, "LOGICOPENABLE"); set(0x2A, "BONEMATRIXNUMBER"); set(0x2B, "BONEMATRIXDATA");
    for (int i = 0; i < 8; ++i) n[0x2C + i] = "MORPHWEIGHT";
    set(0x36, "PATCHDIVISION"); set(0x37, "PATCHPRIMITIVE"); set(0x38, "PATCHFACING");
    set(0x3A, "WORLDMATRIXNUMBER"); set(0x3B, "WORLDMATRIXDATA"); set(0x3C, "VIEWMATRIXNUMBER"); set(0x3D, "VIEWMATRIXDATA");
    set(0x3E, "PROJMATRIXNUMBER"); set(0x3F, "PROJMATRIXDATA"); set(0x40, "TGENMATRIXNUMBER"); set(0x41, "TGENMATRIXDATA");
    set(0x42, "VIEWPORTXSCALE"); set(0x43, "VIEWPORTYSCALE"); set(0x44, "VIEWPORTZSCALE"); set(0x45, "VIEWPORTXCENTER");
    set(0x46, "VIEWPORTYCENTER"); set(0x47, "VIEWPORTZCENTER"); set(0x48, "TEXSCALEU"); set(0x49, "TEXSCALEV");
    set(0x4A, "TEXOFFSETU"); set(0x4B, "TEXOFFSETV"); set(0x4C, "OFFSETX"); set(0x4D, "OFFSETY");
    set(0x50, "SHADEMODE"); set(0x51, "REVERSENORMAL"); set(0x53, "MATERIALUPDATE"); set(0x54, "MATERIALEMISSIVE");
    set(0x55, "MATERIALAMBIENT"); set(0x56, "MATERIALDIFFUSE"); set(0x57, "MATERIALSPECULAR"); set(0x58, "MATERIALALPHA");
    set(0x5B, "MATERIALSPECULARCOEF"); set(0x5C, "AMBIENTCOLOR"); set(0x5D, "AMBIENTALPHA"); set(0x5E, "LIGHTMODE");
    for (int i = 0; i < 4; ++i) n[0x5F + i] = "LIGHTTYPE";
    for (int i = 0x63; i <= 0x6E; ++i) n[i] = "LIGHTPOSITION";
    for (int i = 0x6F; i <= 0x7A; ++i) n[i] = "LIGHTDIRECTION";
    for (int i = 0x7B; i <= 0x86; ++i) n[i] = "LIGHTATTENUATION";
    for (int i = 0x87; i <= 0x8A; ++i) n[i] = "LIGHTSPOTEXPONENT";
    for (int i = 0x8B; i <= 0x8E; ++i) n[i] = "LIGHTSPOTCUTOFF";
    for (int i = 0x8F; i <= 0x9A; ++i) n[i] = "LIGHTCOLOR";
    set(0x9B, "CULL"); set(0x9C, "FRAMEBUFPTR"); set(0x9D, "FRAMEBUFWIDTH"); set(0x9E, "ZBUFPTR"); set(0x9F, "ZBUFWIDTH");
    for (int i = 0; i < 8; ++i) { n[0xA0 + i] = "TEXADDR"; n[0xA8 + i] = "TEXBUFWIDTH"; n[0xB8 + i] = "TEXSIZE"; }
    set(0xB0, "CLUTADDR"); set(0xB1, "CLUTADDRUPPER"); set(0xB2, "TRANSFERSRC"); set(0xB3, "TRANSFERSRCW");
    set(0xB4, "TRANSFERDST"); set(0xB5, "TRANSFERDSTW"); set(0xC0, "TEXMAPMODE"); set(0xC1, "TEXSHADELS");
    set(0xC2, "TEXMODE"); set(0xC3, "TEXFORMAT"); set(0xC4, "LOADCLUT"); set(0xC5, "CLUTFORMAT"); set(0xC6, "TEXFILTER");
    set(0xC7, "TEXWRAP"); set(0xC8, "TEXLEVEL"); set(0xC9, "TEXFUNC"); set(0xCA, "TEXENVCOLOR"); set(0xCB, "TEXFLUSH");
    set(0xCC, "TEXSYNC"); set(0xCD, "FOG1"); set(0xCE, "FOG2"); set(0xCF, "FOGCOLOR"); set(0xD0, "TEXLODSLOPE");
    set(0xD2, "FRAMEBUFPIXFORMAT"); set(0xD3, "CLEARMODE"); set(0xD4, "SCISSOR1"); set(0xD5, "SCISSOR2");
    set(0xD6, "MINZ"); set(0xD7, "MAXZ"); set(0xD8, "COLORTEST"); set(0xD9, "COLORREF"); set(0xDA, "COLORTESTMASK");
    set(0xDB, "ALPHATEST"); set(0xDC, "STENCILTEST"); set(0xDD, "STENCILOP"); set(0xDE, "ZTEST"); set(0xDF, "BLENDMODE");
    set(0xE0, "BLENDFIXEDA"); set(0xE1, "BLENDFIXEDB"); for (int i = 0; i < 4; ++i) n[0xE2 + i] = "DITHER";
    set(0xE6, "LOGICOP"); set(0xE7, "ZWRITEDISABLE"); set(0xE8, "MASKRGB"); set(0xE9, "MASKALPHA");
    set(0xEA, "TRANSFERSTART"); set(0xEB, "TRANSFERSRCPOS"); set(0xEC, "TRANSFERDSTPOS"); set(0xEE, "TRANSFERSIZE");
    return n;
}();

constexpr const char* kPrims[8] = {"points", "lines", "line_strip", "triangles", "triangle_strip", "triangle_fan",
                                   "sprites", "?"};

}  // namespace

nx::string_view command_name(std::uint8_t cmd) {
    const auto n = kNames[cmd];
    return n.empty() ? nx::string_view("UNKNOWN") : n;
}

nx::string describe(std::uint32_t word) {
    const std::uint8_t cmd = static_cast<std::uint8_t>(word >> 24);
    const std::uint32_t p = word & 0xFFFFFF;
    char buf[96];
    if (cmd == PRIM)
        std::snprintf(buf, sizeof buf, "PRIM %s x%u", kPrims[(p >> 16) & 7], p & 0xFFFF);
    else
        std::snprintf(buf, sizeof buf, "%s 0x%06X", nx::string(command_name(cmd)).c_str(), p);
    return buf;
}

VertexFormat VertexFormat::decode(std::uint32_t vt) {
    VertexFormat f;
    f.raw = vt;
    f.tc = vt & 3;
    f.color = (vt >> 2) & 7;
    f.normal = (vt >> 5) & 3;
    f.pos = (vt >> 7) & 3;
    f.weight = (vt >> 9) & 3;
    f.index = (vt >> 11) & 3;
    f.weights = static_cast<int>((vt >> 14) & 7) + 1;
    f.morphs = static_cast<int>((vt >> 18) & 7) + 1;
    f.through = (vt >> 23) & 1;
    int offset = 0, max_align = 1;
    const auto place = [&](int size, int align) {
        offset = (offset + align - 1) & ~(align - 1);
        const int at = offset;
        offset += size;
        max_align = std::max(max_align, align);
        return at;
    };
    static constexpr int kScalar[4] = {0, 1, 2, 4};  // none, 8-bit, 16-bit, float
    if (f.weight) place(kScalar[f.weight] * f.weights, kScalar[f.weight]);
    if (f.tc) f.tc_off = place(kScalar[f.tc] * 2, kScalar[f.tc]);
    if (f.color >= 4) f.color_off = place(f.color == 7 ? 4 : 2, f.color == 7 ? 4 : 2);
    if (f.normal) f.normal_off = place(kScalar[f.normal] * 3, kScalar[f.normal]);
    f.pos_off = place(kScalar[f.pos ? f.pos : 3] * 3, kScalar[f.pos ? f.pos : 3]);
    f.size = (offset + max_align - 1) & ~(max_align - 1);
    return f;
}

namespace {

// Reads component `i` of an attribute with encoding `kind` (1: 8-bit, 2: 16-bit, 3: float).
float component(const std::uint8_t* p, int kind, int i, bool is_signed, float scale) {
    switch (kind) {
        case 1: return (is_signed ? static_cast<float>(static_cast<std::int8_t>(p[i])) : static_cast<float>(p[i])) * scale;
        case 2: {
            std::uint16_t v;
            std::memcpy(&v, p + i * 2, 2);
            return (is_signed ? static_cast<float>(static_cast<std::int16_t>(v)) : static_cast<float>(v)) * scale;
        }
        case 3: {
            float v;
            std::memcpy(&v, p + i * 4, 4);
            return v;
        }
        default: return 0.0f;
    }
}

void decode_color(const std::uint8_t* p, int kind, float out[4]) {
    std::uint32_t c = 0;
    std::memcpy(&c, p, kind == 7 ? 4 : 2);
    switch (kind) {
        case 4:  // 5650
            out[0] = static_cast<float>(c & 31) / 31.0f;
            out[1] = static_cast<float>((c >> 5) & 63) / 63.0f;
            out[2] = static_cast<float>((c >> 11) & 31) / 31.0f;
            out[3] = 1.0f;
            break;
        case 5:  // 5551
            out[0] = static_cast<float>(c & 31) / 31.0f;
            out[1] = static_cast<float>((c >> 5) & 31) / 31.0f;
            out[2] = static_cast<float>((c >> 10) & 31) / 31.0f;
            out[3] = static_cast<float>((c >> 15) & 1);
            break;
        case 6:  // 4444
            for (int i = 0; i < 4; ++i) out[i] = static_cast<float>((c >> (i * 4)) & 15) / 15.0f;
            break;
        default:  // 8888
            for (int i = 0; i < 4; ++i) out[i] = static_cast<float>((c >> (i * 8)) & 255) / 255.0f;
            break;
    }
}

// Decodes one vertex at `p`, blending morph targets with `morph_weights`. A vertex without a color component
// gets color[0] = -1 so the renderer substitutes the material color.
void decode_vertex(const VertexFormat& f, const std::uint8_t* p, const float* morph_weights, Vertex& v) {
    v = Vertex{};
    std::fill(std::begin(v.color), std::end(v.color), 0.0f);
    const bool through = f.through;
    const int morphs = through ? 1 : f.morphs;
    for (int m = 0; m < morphs; ++m) {
        const std::uint8_t* t = p + m * f.size;
        const float w = morphs == 1 ? 1.0f : morph_weights[m];
        if (f.weight)
            for (int i = 0; i < f.weights; ++i)
                v.weights[i] += w * component(t, f.weight, i, false, f.weight == 1 ? 1.0f / 128 : 1.0f / 32768);
        if (f.tc) {
            const float scale = through ? 1.0f : f.tc == 1 ? 1.0f / 128 : 1.0f / 32768;
            for (int i = 0; i < 2; ++i) v.uv[i] += w * component(t + f.tc_off, f.tc, i, false, scale);
        }
        if (f.color >= 4) {
            float c[4];
            decode_color(t + f.color_off, f.color, c);
            for (int i = 0; i < 4; ++i) v.color[i] += w * c[i];
        }
        if (f.normal)
            for (int i = 0; i < 3; ++i)
                v.normal[i] += w * component(t + f.normal_off, f.normal, i, true,
                                             f.normal == 1 ? 1.0f / 128 : 1.0f / 32768);
        const int pk = f.pos ? f.pos : 3;
        if (through) {  // screen coordinates; z is unsigned
            for (int i = 0; i < 3; ++i) v.pos[i] = component(t + f.pos_off, pk, i, i < 2, 1.0f);
        } else {
            for (int i = 0; i < 3; ++i)
                v.pos[i] += w * component(t + f.pos_off, pk, i, true, pk == 1 ? 1.0f / 128 : 1.0f / 32768);
        }
    }
    if (f.color < 4) v.color[0] = -1.0f;
}

// The same decoding specialised for one format without morph targets (the common case: the format is fixed at
// compile time, so the per-component switches go away). Results are identical to decode_vertex.
template <int TC, int COLOR, int NORMAL, int POS, int WEIGHT, bool THROUGH>
void decode_fixed(const VertexFormat& f, const std::uint8_t* t, const float*, Vertex& v) {
    v = Vertex{};
    if constexpr (WEIGHT != 0)
        for (int i = 0; i < f.weights; ++i)
            v.weights[i] = component(t, WEIGHT, i, false, WEIGHT == 1 ? 1.0f / 128 : 1.0f / 32768);
    if constexpr (TC != 0) {
        constexpr float scale = THROUGH ? 1.0f : TC == 1 ? 1.0f / 128 : 1.0f / 32768;
        v.uv[0] = component(t + f.tc_off, TC, 0, false, scale);
        v.uv[1] = component(t + f.tc_off, TC, 1, false, scale);
    }
    if constexpr (COLOR >= 4) {
        decode_color(t + f.color_off, COLOR, v.color);
    } else {
        v.color[0] = -1.0f;
        v.color[1] = v.color[2] = v.color[3] = 0.0f;
    }
    if constexpr (NORMAL != 0)
        for (int i = 0; i < 3; ++i)
            v.normal[i] = component(t + f.normal_off, NORMAL, i, true, NORMAL == 1 ? 1.0f / 128 : 1.0f / 32768);
    constexpr int pk = POS ? POS : 3;
    for (int i = 0; i < 3; ++i) {
        if constexpr (THROUGH) v.pos[i] = component(t + f.pos_off, pk, i, i < 2, 1.0f);
        else v.pos[i] = component(t + f.pos_off, pk, i, true, pk == 1 ? 1.0f / 128 : 1.0f / 32768);
    }
}

using DecodeFn = void (*)(const VertexFormat&, const std::uint8_t*, const float*, Vertex&);

// Common vertex formats get specialised decoders (measured on a 3D action game: ~75% float UV + 8888 color + s16
// position in battle; skinned models with float UV, s8 normals, s16 positions and float weights; through-mode
// sprites); every other format uses decode_vertex. Add a game's frequent formats here.
DecodeFn select_decoder(const VertexFormat& f) {
    if (f.morphs != 1) return decode_vertex;
    const auto is = [&](int tc, int color, int normal, int pos, int weight, bool through) {
        return f.tc == tc && f.color == color && f.normal == normal && f.pos == pos && f.weight == weight &&
               f.through == through;
    };
    if (is(3, 7, 0, 2, 0, false)) return decode_fixed<3, 7, 0, 2, 0, false>;
    if (is(3, 0, 1, 2, 3, false)) return decode_fixed<3, 0, 1, 2, 3, false>;
    if (is(2, 7, 0, 2, 0, true)) return decode_fixed<2, 7, 0, 2, 0, true>;
    if (is(2, 7, 0, 2, 0, false)) return decode_fixed<2, 7, 0, 2, 0, false>;
    if (is(2, 0, 0, 2, 0, false)) return decode_fixed<2, 0, 0, 2, 0, false>;
    if (is(3, 7, 1, 2, 0, false)) return decode_fixed<3, 7, 1, 2, 0, false>;
    if (is(3, 0, 1, 2, 0, false)) return decode_fixed<3, 0, 1, 2, 0, false>;
    return decode_vertex;
}

}  // namespace

void decode_vertices(const VertexFormat& f, const float* morph_weights, const std::uint8_t* data, std::uint32_t count,
                     nx::vector<Vertex>& out) {
    const DecodeFn decode = select_decoder(f);
    out.resize(count);
    for (std::uint32_t i = 0; i < count; ++i)
        decode(f, data + static_cast<std::size_t>(i) * static_cast<std::size_t>(f.stride()), morph_weights, out[i]);
}

namespace {

std::uint64_t now_ns() {
    return timing_stats() ? static_cast<std::uint64_t>(nx::timestamp::now().nanos()) : 0;
}

}  // namespace

void Processor::patch(std::uint32_t p, bool bezier) {
    const int u_count = static_cast<int>(p & 0xFF);
    const int v_count = static_cast<int>((p >> 8) & 0xFF);
    if (u_count <= 0 || v_count <= 0 || u_count > 16 || v_count > 16) {
        if (!warned_patches_) std::fprintf(stderr, "GE: patch %dx%d out of range, skipping\n", u_count, v_count);
        warned_patches_ = true;
        return;
    }
    if (!bezier) {  // B-spline patches are not implemented; keep the old warning.
        if (!warned_patches_) std::fprintf(stderr, "GE: spline patches are not rendered yet\n");
        warned_patches_ = true;
        // Still consume the control points so the stream stays in sync.
        state_.format = VertexFormat::decode(state_.param(VERTEXTYPE));
        const int stride = state_.format.stride();
        if (state_.format.index_size())
            iaddr_ += static_cast<std::uint32_t>(u_count * v_count * state_.format.index_size());
        else
            vaddr_ += static_cast<std::uint32_t>(u_count * v_count * stride);
        return;
    }
    const std::uint32_t div = state_.param(PATCHDIVISION);
    int s_div = static_cast<int>(div & 0xFF);
    int t_div = static_cast<int>((div >> 8) & 0xFF);
    s_div = std::clamp(s_div, 1, 64);
    t_div = std::clamp(t_div, 1, 64);
    const std::uint32_t patch_prim = state_.param(PATCHPRIMITIVE) & 7;
    if (patch_prim != 0) {
        // Only triangle output is implemented so far; still render as triangles.
        if (!warned_patch_prim_) {
            std::fprintf(stderr, "GE: patch primitive %u treated as triangles\n", patch_prim);
            warned_patch_prim_ = true;
        }
    }

    state_.format = VertexFormat::decode(state_.param(VERTEXTYPE));
    state_.vertex_address = vaddr_;
    const VertexFormat& f = state_.format;
    float morph_weights[8];
    for (int i = 0; i < 8; ++i) morph_weights[i] = state_.flt(static_cast<std::uint8_t>(MORPHWEIGHT0 + i));
    const int index_size = f.index_size();
    const DecodeFn decode = select_decoder(f);
    const auto in_vram = [](std::uint32_t a) { return (vram_canonical(a) & 0x0F000000) == 0x04000000; };
    if (in_vram(vaddr_) || (index_size && in_vram(iaddr_))) flush();
    const std::uint64_t decode_start = now_ns();
    nx::vector<Vertex> ctrl(static_cast<std::size_t>(u_count * v_count));
    for (int i = 0; i < u_count * v_count; ++i) {
        std::uint32_t n = static_cast<std::uint32_t>(i);
        if (index_size == 1) n = *memory_.ptr(iaddr_ + static_cast<std::uint32_t>(i));
        else if (index_size == 2) {
            std::uint16_t x;
            std::memcpy(&x, memory_.ptr(iaddr_ + static_cast<std::uint32_t>(i) * 2), 2);
            n = x;
        } else if (index_size == 4) n = memory_.read32(iaddr_ + static_cast<std::uint32_t>(i) * 4);
        decode(f, memory_.ptr(vram_canonical(vaddr_ + n * static_cast<std::uint32_t>(f.stride()))), morph_weights,
               ctrl[static_cast<std::size_t>(i)]);
    }
    if (index_size)
        iaddr_ += static_cast<std::uint32_t>(u_count * v_count * index_size);
    else
        vaddr_ += static_cast<std::uint32_t>(u_count * v_count * f.stride());

    // Bernstein basis for degree n = count - 1.
    auto bernstein = [](int n, float t, float* out) {
        float inv = 1.0f - t;
        // Binomial coefficients via Pascal's triangle (n <= 15).
        float c = 1.0f;
        for (int i = 0; i <= n; ++i) {
            if (i > 0) c = c * static_cast<float>(n - i + 1) / static_cast<float>(i);
            float p = 1.0f;
            for (int k = 0; k < i; ++k) p *= t;
            for (int k = 0; k < n - i; ++k) p *= inv;
            out[i] = c * p;
        }
        if (n == 0) out[0] = 1.0f;
    };
    // Bernstein weights depend on one parameter each: tabulate them once per division line.
    nx::vector<float> bu_table(static_cast<std::size_t>((s_div + 1) * u_count));
    for (int is = 0; is <= s_div; ++is)
        bernstein(u_count - 1, static_cast<float>(is) / static_cast<float>(s_div),
                  &bu_table[static_cast<std::size_t>(is) * static_cast<std::size_t>(u_count)]);
    nx::vector<float> bv_table(static_cast<std::size_t>((t_div + 1) * v_count));
    for (int it = 0; it <= t_div; ++it)
        bernstein(v_count - 1, static_cast<float>(it) / static_cast<float>(t_div),
                  &bv_table[static_cast<std::size_t>(it) * static_cast<std::size_t>(v_count)]);
    // Patch UVs are the (u, v) parameters; the pipeline applies TEXSCALE/OFFSET afterwards (3D patches
    // sample the texture normally, through-mode patches address texels directly).
    float tex_w = 1.0f, tex_h = 1.0f;
    if (f.through) {
        const std::uint32_t sz = state_.param(ge::TEXSIZE0);
        tex_w = static_cast<float>(1 << std::min<std::uint32_t>(sz & 15, 10));
        tex_h = static_cast<float>(1 << std::min<std::uint32_t>((sz >> 8) & 15, 10));
    }
    nx::vector<Vertex> grid(static_cast<std::size_t>((s_div + 1) * (t_div + 1)));
    for (int it = 0; it <= t_div; ++it) {
        const float v = static_cast<float>(it) / static_cast<float>(t_div);
        const float* bvr = &bv_table[static_cast<std::size_t>(it) * static_cast<std::size_t>(v_count)];
        for (int is = 0; is <= s_div; ++is) {
            const float u = static_cast<float>(is) / static_cast<float>(s_div);
            const float* bur = &bu_table[static_cast<std::size_t>(is) * static_cast<std::size_t>(u_count)];
            Vertex vert{};
            vert.color[0] = -1.0f;
            bool has_color = f.color >= 4;
            float col[4] = {0, 0, 0, 0};
            for (int j = 0; j < v_count; ++j) {
                for (int i = 0; i < u_count; ++i) {
                    const float w = bur[i] * bvr[static_cast<std::size_t>(j)];
                    const Vertex& c = ctrl[static_cast<std::size_t>(j * u_count + i)];
                    for (int k = 0; k < 3; ++k) vert.pos[k] += w * c.pos[k];
                    for (int k = 0; k < 3; ++k) vert.normal[k] += w * c.normal[k];
                    for (int k = 0; k < f.weights; ++k) vert.weights[k] += w * c.weights[k];
                    if (has_color)
                        for (int k = 0; k < 4; ++k) col[k] += w * c.color[k];
                }
            }
            if (has_color)
                for (int k = 0; k < 4; ++k) vert.color[k] = col[k];
            else
                vert.color[0] = -1.0f;
            if (f.through) {
                vert.uv[0] = u * tex_w;
                vert.uv[1] = v * tex_h;
            } else {
                vert.uv[0] = u;
                vert.uv[1] = v;
            }
            grid[static_cast<std::size_t>(it * (s_div + 1) + is)] = vert;
        }
    }
    vertices_.clear();
    vertices_.reserve(static_cast<std::size_t>(s_div * t_div * 6));
    for (int it = 0; it < t_div; ++it) {
        for (int is = 0; is < s_div; ++is) {
            const std::size_t a = static_cast<std::size_t>(it * (s_div + 1) + is);
            const std::size_t b = a + 1, c = a + static_cast<std::size_t>(s_div + 1), d = c + 1;
            vertices_.push_back(grid[a]);
            vertices_.push_back(grid[b]);
            vertices_.push_back(grid[d]);
            vertices_.push_back(grid[a]);
            vertices_.push_back(grid[d]);
            vertices_.push_back(grid[c]);
        }
    }
    ++stats_.draws;
    stats_.vertices += static_cast<std::uint64_t>(vertices_.size());
    if (on_draw) on_draw(Draw{0, static_cast<std::uint32_t>(Prim::Triangles), static_cast<std::uint32_t>(vertices_.size()), f.raw, vaddr_, iaddr_, &state_});
    const std::uint64_t render_start = now_ns();
    stats_.decode_ns += render_start - decode_start;
    if (renderer_ && !vertices_.empty()) {
        // The tessellator emits a fixed winding, so draw double-sided: with the incoming cull state the
        // grids can come out back-facing (PATCHFACING handling and hw winding convention are follow-ups).
        // A patch with no texture bound (address 0) would sample invalid memory: draw it untextured.
        State draw_state = state_;
        draw_state.regs[ge::CULLFACEENABLE] = 0;
        const std::uint32_t texaddr =
            ((draw_state.param(ge::TEXBUFWIDTH0) & 0x0F0000) << 8) | (draw_state.param(ge::TEXADDR0) & 0xFFFFF0);
        if ((texaddr & 0xFFFFF0) == 0) draw_state.regs[ge::TEXTUREMAPENABLE] = 0;
        renderer_->draw(draw_state, Prim::Triangles, vertices_);
        stats_.render_ns += now_ns() - render_start;
    }
}

void Processor::draw(std::uint32_t p) {
    const auto prim = static_cast<Prim>((p >> 16) & 7);
    const std::uint32_t count = p & 0xFFFF;
    state_.format = VertexFormat::decode(state_.param(VERTEXTYPE));
    state_.vertex_address = vaddr_;
    const VertexFormat& f = state_.format;
    ++stats_.draws;
    stats_.vertices += count;
    if (on_draw) on_draw(Draw{0, static_cast<std::uint32_t>(prim), count, f.raw, vaddr_, iaddr_, &state_});

    const auto in_vram = [](std::uint32_t a) { return (vram_canonical(a) & 0x0F000000) == 0x04000000; };
    if (in_vram(vaddr_) || (f.index_size() && in_vram(iaddr_))) flush();  // vertices drawn into VRAM first
    const std::uint64_t decode_start = now_ns();
    float morph_weights[8];
    for (int i = 0; i < 8; ++i) morph_weights[i] = state_.flt(static_cast<std::uint8_t>(MORPHWEIGHT0 + i));
    const int index_size = f.index_size();
    const DecodeFn decode = select_decoder(f);
    vertices_.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t n = i;
        if (index_size == 1) {
            n = *memory_.ptr(iaddr_ + i);
        } else if (index_size == 2) {
            std::uint16_t x;
            std::memcpy(&x, memory_.ptr(iaddr_ + i * 2), 2);
            n = x;
        } else if (index_size == 4) {
            n = memory_.read32(iaddr_ + i * 4);
        }
        decode(f, memory_.ptr(vram_canonical(vaddr_ + n * static_cast<std::uint32_t>(f.stride()))), morph_weights,
               vertices_[i]);
    }
    // The GE leaves the pointer it consumed just past the data, so consecutive PRIMs can continue.
    if (index_size)
        iaddr_ += count * static_cast<std::uint32_t>(index_size);
    else
        vaddr_ += count * static_cast<std::uint32_t>(f.stride());

    const std::uint64_t render_start = now_ns();
    stats_.decode_ns += render_start - decode_start;
    if (renderer_ && prim <= Prim::Sprites && count) {
        renderer_->draw(state_, prim, vertices_);
        stats_.render_ns += now_ns() - render_start;
    }
}

int Processor::enqueue(std::uint32_t list, std::uint32_t stall, bool) {
    const int id = next_id_++;
    List& l = *(lists_[id] = nx::make_unique<List>());
    l.pc = list & 0x0FFFFFFF;
    l.stall = stall & 0x0FFFFFFF;
    ++stats_.lists;
    const std::uint64_t start = now_ns();
    run(l);
    stats_.run_ns += now_ns() - start;
    // Keep only unfinished lists (plus a little history for list_done()).
    for (auto it = lists_.begin(); it != lists_.end() && lists_.size() > 64;)
        it = it->second->done ? lists_.erase(it) : std::next(it);
    return id;
}

void Processor::update_stall(int id, std::uint32_t stall) {
    const auto it = lists_.find(id);
    if (it == lists_.end() || it->second->done) return;
    List& l = *it->second;
    l.stall = stall & 0x0FFFFFFF;
    const std::uint64_t start = now_ns();
    run(l);
    stats_.run_ns += now_ns() - start;
}

bool Processor::list_done(int id) const {
    const auto it = lists_.find(id);
    return it == lists_.end() || it->second->done;
}

void Processor::run(List& l) {
    for (int guard = 0; guard < 10'000'000; ++guard) {
        if (l.done || (l.stall && l.pc == l.stall)) return;
        const std::uint32_t pc = l.pc;
        const std::uint32_t word = memory_.read32(pc);
        const std::uint8_t cmd = static_cast<std::uint8_t>(word >> 24);
        const std::uint32_t p = word & 0xFFFFFF;
        l.pc += 4;
        ++stats_.commands;
        state_.regs[cmd] = word;
        if (on_command) on_command(pc, word);

        switch (cmd) {
            case BASE: base_ = (p << 8) & 0xFF000000; break;
            case OFFSETADDR: offset_ = p << 8; break;
            case ORIGIN: offset_ = pc; break;
            case VADDR: vaddr_ = address(p); break;
            case IADDR: iaddr_ = address(p); break;
            case JUMP: l.pc = address(p & ~3u); break;
            case CALL:
                l.call_stack.push_back(l.pc);
                l.call_stack.push_back(offset_);
                l.pc = address(p & ~3u);
                break;
            case RET:
                if (l.call_stack.size() >= 2) {
                    offset_ = l.call_stack.back();
                    l.call_stack.pop_back();
                    l.pc = l.call_stack.back();
                    l.call_stack.pop_back();
                }
                break;
            case SIGNAL:
                l.signal = p;
                l.signal_pending = true;
                break;
            case FINISH:
                l.finish = p;
                l.finish_pending = true;
                break;
            case END:
                if (l.signal_pending) {  // END completes the SIGNAL; the list continues (sceGuSignal modes)
                    l.signal_pending = false;
                    const std::uint32_t behaviour = (l.signal >> 16) & 0xFF, id = l.signal & 0xFFFF;
                    switch (behaviour) {
                        case 1:  // wait: the GE resumes once the handler returns
                        case 2:  // no wait
                            flush();
                            if (on_signal) on_signal(id);
                            break;
                        case 3:  // pause until sceGeContinue: not modelled, the list continues
                            flush();
                            if (on_signal) on_signal(id);
                            [[fallthrough]];
                        default:
                            if (behaviour < 32 && !(warned_signals_ & (1u << behaviour))) {
                                std::fprintf(stderr, "GE: signal behaviour %u (id 0x%X) is not supported; continuing\n",
                                             behaviour, id);
                                warned_signals_ |= 1u << behaviour;
                            }
                            break;
                    }
                    break;
                }
                l.done = true;
                flush();
                if (l.finish_pending) {
                    l.finish_pending = false;
                    if (on_finish) on_finish(l.finish & 0xFFFF);
                }
                return;
            case PRIM: draw(p); break;
            case BEZIER: patch(p, true); break;
            case SPLINE: patch(p, false); break;
            case BOUNDINGBOX: bbox_visible_ = true; break;  // conservative: always "maybe visible"
            case BJUMP:
                if (!bbox_visible_) l.pc = address(p & ~3u);
                break;
            case WORLDMATRIXNUMBER: world_index_ = p & 15; break;
            case WORLDMATRIXDATA: if (world_index_ < 12) state_.world[world_index_++] = f24(p); break;
            case VIEWMATRIXNUMBER: view_index_ = p & 15; break;
            case VIEWMATRIXDATA: if (view_index_ < 12) state_.view[view_index_++] = f24(p); break;
            case PROJMATRIXNUMBER: proj_index_ = p & 15; break;
            case PROJMATRIXDATA: if (proj_index_ < 16) state_.proj[proj_index_++] = f24(p); break;
            case TGENMATRIXNUMBER: tgen_index_ = p & 15; break;
            case TGENMATRIXDATA: if (tgen_index_ < 12) state_.tgen[tgen_index_++] = f24(p); break;
            case BONEMATRIXNUMBER: bone_index_ = p & 127; break;
            case BONEMATRIXDATA:
                if (bone_index_ < 96) {
                    state_.bones[bone_index_ / 12][bone_index_ % 12] = f24(p);
                    ++bone_index_;
                }
                break;
            case LOADCLUT: {
                const std::uint32_t addr = vram_canonical(((state_.param(CLUTADDRUPPER) & 0x0F0000) << 8) |
                                                          (state_.param(CLUTADDR) & 0xFFFFF0));
                const std::uint32_t bytes = std::min<std::uint32_t>((p & 0x3F) * 32, 1024);
                if ((addr & 0x0F000000) == 0x04000000) flush();  // VRAM: may be a render target
                if (addr) std::memcpy(state_.clut.data(), memory_.ptr(addr), bytes);
                break;
            }
            case TRANSFERSTART:
                ++stats_.transfers;
                if (renderer_) renderer_->transfer(state_);
                break;
            default: break;
        }
    }
    std::fprintf(stderr, "GE: list did not terminate (pc 0x%08X)\n", l.pc);
    l.done = true;
}

}  // namespace psprecore::rt::ge
