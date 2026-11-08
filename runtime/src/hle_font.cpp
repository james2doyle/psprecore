// sceLibFont: the game's UI text. The PSP's system fonts live in the console's flash, which a dumped
// disc does not contain, so glyphs come from a host TrueType/OpenType font rasterized with stb_truetype.
// The host font is PSPRECORE_FONT (a path, optionally path#face for collections); otherwise the first
// readable system font wins (DejaVu/Arial/Roboto/Noto, per OS). Without any host font the stub behaviour
// remains: fixed 16x16 metrics and blank glyphs, so layout proceeds and text draws blank.
//
// The game builds its own glyph cache from these answers: GetFontInfo sizes its atlas cells (16x16 here),
// GetCharInfo reports the cell and the real advance, GetCharGlyphImage_Clip fills an 8-bit alpha bitmap.
// Struct layouts follow the PSP SDK (pspsdk, BSD-licensed); behaviour here is original.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "core/foundation/core/foundation.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

namespace psprecore::rt {

namespace {

GuestMemory& mem() { return kernel().memory(); }

std::uint32_t g_next_lib = 1, g_next_font = 1;

void write32(std::uint32_t addr, std::uint32_t v) { mem().write32(addr, v); }
void write16(std::uint32_t addr, std::uint32_t v) {
    const std::uint16_t x = static_cast<std::uint16_t>(v);
    std::memcpy(mem().ptr(addr), &x, 2);
}
void write_float(std::uint32_t addr, float v) {
    std::uint32_t w;
    std::memcpy(&w, &v, 4);
    mem().write32(addr, w);
}

// Glyphs are rasterized into a fixed cell the game sizes its cache from.
constexpr int kCell = 16;

// PGFCharInfo (SceFontCharInfo, 60 bytes): bitmap dims, then glyph metrics in 26.6
// fixed point. Advances MUST be pixels<<6: libfont clients lay text out from sfp26AdvanceH,
// so a plain pixel value reads back 64x too small and every line collapses.
void write_charinfo(std::uint32_t addr, int advance26, int asc26, int desc26) {
    constexpr int cell26 = kCell << 6;
    write32(addr, kCell);       // bitmapWidth
    write32(addr + 4, kCell);   // bitmapHeight
    write32(addr + 8, 0);       // bitmapLeft
    write32(addr + 12, 0);      // bitmapTop
    write32(addr + 16, cell26);  // sfp26Width
    write32(addr + 20, cell26);  // sfp26Height
    write32(addr + 24, asc26);   // sfp26Ascender
    write32(addr + 28, desc26);  // sfp26Descender
    write32(addr + 32, 0);       // sfp26BearingHX
    write32(addr + 36, 0);       // sfp26BearingHY
    write32(addr + 40, 0);       // sfp26BearingVX
    write32(addr + 44, 0);       // sfp26BearingVY
    write32(addr + 48, advance26);  // sfp26AdvanceH
    write32(addr + 52, 0);          // sfp26AdvanceV
    write16(addr + 56, 0);  // shadowFlag
    write16(addr + 58, 0);  // shadowId
}

// A glyph rasterized into the fixed cell the game sizes its cache from.
struct Cell {
    std::uint8_t px[kCell * kCell]{};
    int advance = kCell;
};

struct HostFont {
    stbtt_fontinfo info{};
    nx::vector<unsigned char> data;
    float scale = 0;  // pixels per font unit
    int baseline = 0;  // cell row the baseline sits on
    float ascender = 0, descender = 0;  // pixels, for GetFontInfo
    bool ready = false;
    bool tried = false;
};

HostFont& host_font() {
    static HostFont f;
    return f;
}

bool load_from_file(const char* path, unsigned face) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0 || size > 64L * 1024 * 1024) {
        std::fclose(f);
        return false;
    }
    HostFont& hf = host_font();
    hf.data.resize(static_cast<std::size_t>(size));
    const bool ok = std::fread(hf.data.data(), 1, hf.data.size(), f) == hf.data.size();
    std::fclose(f);
    if (!ok) {
        hf.data.clear();
        return false;
    }
    const int offset = stbtt_GetFontOffsetForIndex(hf.data.data(), static_cast<int>(face));
    if (offset < 0 || !stbtt_InitFont(&hf.info, hf.data.data(), offset)) {
        hf.data.clear();
        return false;
    }
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&hf.info, &ascent, &descent, &line_gap);
    if (ascent <= descent) {
        hf.data.clear();
        return false;
    }
    hf.scale = (static_cast<float>(kCell) - 1.0f) / static_cast<float>(ascent - descent);
    hf.baseline = static_cast<int>(hf.scale * static_cast<float>(ascent) + 0.5f);
    hf.ascender = hf.scale * static_cast<float>(ascent);
    hf.descender = hf.scale * static_cast<float>(-descent);
    hf.ready = true;
    std::fprintf(stderr, "sceFont: host font %s (face %u)\n", path, face);
    return true;
}

void ensure_font() {
    HostFont& hf = host_font();
    if (hf.ready || hf.tried) return;
    hf.tried = true;
    if (const char* env = std::getenv("PSPRECORE_FONT")) {
        nx::string path = env;
        unsigned face = 0;
        if (const char* hash = std::strrchr(path.c_str(), '#')) {
            face = static_cast<unsigned>(std::strtoul(hash + 1, nullptr, 10));
            path.resize(static_cast<std::size_t>(hash - path.c_str()));
        }
        if (!path.empty() && load_from_file(path.c_str(), face)) return;
        std::fprintf(stderr, "sceFont: PSPRECORE_FONT=%s did not load\n", env);
    }
    static constexpr const char* kCandidates[] = {
#if defined(_WIN32)
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
#elif defined(__APPLE__)
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
#elif defined(__ANDROID__)
        "/system/fonts/Roboto-Regular.ttf",
        "/system/fonts/NotoSans-Regular.ttf",
#else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
#endif
        nullptr,
    };
    for (const char* const* c = kCandidates; *c; ++c)
        if (load_from_file(*c, 0)) return;
    std::fprintf(stderr, "sceFont: no host font found (set PSPRECORE_FONT); text is blank\n");
}

// Rasterizes a codepoint into the cell, clipped to it. Whitespace yields an empty cell with a real advance.
bool rasterize(std::uint32_t code, Cell& cell) {
    std::memset(cell.px, 0, sizeof(cell.px));
    cell.advance = kCell;
    if (code < 0x20) {  // control characters: blank with no advance, like a missing PGF glyph
        cell.advance = 0;
        return true;
    }
    ensure_font();
    HostFont& hf = host_font();
    if (!hf.ready) return false;
    const int glyph = stbtt_FindGlyphIndex(&hf.info, static_cast<int>(code));
    int advance;
    stbtt_GetGlyphHMetrics(&hf.info, glyph, &advance, nullptr);
    cell.advance = std::max(1, static_cast<int>(advance * hf.scale + 0.5f));
    int ix0, iy0, ix1, iy1;
    stbtt_GetGlyphBitmapBoxSubpixel(&hf.info, glyph, hf.scale, hf.scale, 0, 0, &ix0, &iy0, &ix1, &iy1);
    const int w = ix1 - ix0, h = iy1 - iy0;
    if (w <= 0 || h <= 0) return true;
    nx::vector<std::uint8_t> tmp(static_cast<std::size_t>(w) * static_cast<std::size_t>(h));
    stbtt_MakeGlyphBitmapSubpixel(&hf.info, tmp.data(), w, h, w, hf.scale, hf.scale, 0, 0, glyph);
    const int dst_x = ix0, dst_y = hf.baseline + iy0;
    for (int y = 0; y < h; ++y) {
        const int dy = dst_y + y;
        if (dy < 0 || dy >= kCell) continue;
        for (int x = 0; x < w; ++x) {
            const int dx = dst_x + x;
            if (dx < 0 || dx >= kCell) continue;
            cell.px[static_cast<std::size_t>(dy) * kCell + dx] = tmp[static_cast<std::size_t>(y) * w + x];
        }
    }
    return true;
}

// sceFontNewLib(params, errorCode) -> lib handle
void NewLib(Context& c) {
    const std::uint32_t err = arg(c, 1);
    if (err) mem().write32(err, 0);
    ret(c, g_next_lib++);
}

// sceFontDoneLib(lib) -> 0
void DoneLib(Context& c) {
    (void)c;
    ret(c, 0);
}

// sceFontOpenUserFile(lib, fileName, mode, errorCode) -> font handle
void OpenUserFile(Context& c) {
    const std::uint32_t err = arg(c, 3);
    if (err) mem().write32(err, 0);
    std::fprintf(stderr, "sceFontOpenUserFile(%s)\n", mem().string(arg(c, 1)).c_str());
    ret(c, g_next_font++);
}

// sceFontOpen(lib, index, mode, errorCode) -> font handle
void Open(Context& c) {
    const std::uint32_t err = arg(c, 3);
    if (err) mem().write32(err, 0);
    ret(c, g_next_font++);
}

// sceFontClose(font) -> 0
void Close(Context& c) {
    (void)c;
    ret(c, 0);
}

// sceFontGetNumFontList(lib, errorCode) -> count
void GetNumFontList(Context& c) {
    const std::uint32_t err = arg(c, 1);
    if (err) mem().write32(err, 0);
    ret(c, 1);
}

// sceFontFindOptimumFont(lib, style, errorCode) -> index
void FindOptimumFont(Context& c) {
    const std::uint32_t err = arg(c, 2);
    if (err) mem().write32(err, 0);
    ret(c, 0);
}

// sceFontGetFontInfo(font, info): PGFFontInfo. Field scales follow the hardware PGF
// (ltn5.pgf, the serif-bold Latin font the game opens): the I/F pairs carry 26.6
// fixed point / float pixels. In particular maxSize must be real (1509x1210 =
// 23.6x18.9 px): the game derives inter-word spacing from it, and the stub
// 16px cells collapse words. ascender/descender stay at host-pixel scale: the
// full 26.6 header values (863/-196) fault the boot text converter here,
// so the pre-existing host-derived pixels are kept (root cause still open).
void GetFontInfo(Context& c) {
    const std::uint32_t info = arg(c, 1);
    if (!info) return ret(c, 0x80000100);  // EINVAL-ish
    ensure_font();
    const HostFont& hf = host_font();
    const int asc = hf.ready ? static_cast<int>(hf.ascender + 0.5f) : 16;
    const int desc = hf.ready ? static_cast<int>(hf.descender + 0.5f) : 16;
    const int fixedi[10] = {1509, 1210, asc, desc, 0, 0, 0, 0, kCell, kCell};
    for (int i = 0; i < 10; ++i) write32(info + 4 * static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(fixedi[i]));
    const float fixedf[10] = {1509.0f / 64, 1210.0f / 64, (float)asc, (float)desc, 0, 0, 0, 0, (float)kCell, (float)kCell};
    for (int i = 0; i < 10; ++i) write_float(info + 40 + 4 * static_cast<std::uint32_t>(i), fixedf[i]);
    write16(info + 80, kCell);  // maxGlyphWidth
    write16(info + 82, kCell);  // maxGlyphHeight
    write32(info + 84, 256);    // charMapLength
    write32(info + 88, 0);      // shadowMapLength
    // fontStyle left zeroed; BPP left 0 (filled by every calloc'd caller anyway)
    ret(c, 0);
}

// sceFontGetCharInfo(font, charCode, charInfo): PGFCharInfo (60 bytes).
void GetCharInfo(Context& c) {
    const std::uint32_t code = arg(c, 1), info = arg(c, 2);
    if (!info) return ret(c, 0x80000100);
    Cell cell;
    rasterize(code, cell);
    const HostFont& hf = host_font();
    const int asc26 = hf.ready ? static_cast<int>(hf.ascender * 64.0f + 0.5f) : 16 << 6;
    const int desc26 = hf.ready ? static_cast<int>(hf.descender * 64.0f + 0.5f) : 16 << 6;
    write_charinfo(info, cell.advance << 6, asc26, desc26);
    ret(c, 0);
}

// sceFontGetCharGlyphImage_Clip(font, charCode, glyphImage, clipX, clipY, clipWidth, clipHeight)
//
// GlyphImage (PGF.h): u32 pixelFormat, s32 xPos64, s32 yPos64 (26.6 fixed point pen
// position), u16 width, height, bytesPerLine, u16 pad, u32 bufferPtr. The game owns
// the struct and its buffer (a cell of its own glyph cache); libfont rasterizes INTO
// it, clipped to the clip rectangle, adding ink with saturation (never erasing).
// Only 4bpp and 8bpp buffers draw anything, like the hardware. The x fraction of
// xPos64 blends horizontally; the y fraction is discarded.
void GetCharGlyphImageClip(Context& c) {
    const std::uint32_t code = arg(c, 1), img = arg(c, 2);
    if (!img) return ret(c, 0x80000100);
    Cell cell;
    rasterize(code, cell);
    const std::uint32_t format = mem().read32(img);
    if (format != 0 && format != 1 && format != 2) return ret(c, 0);  // HW draws nothing
    const std::int32_t xpos = static_cast<std::int32_t>(mem().read32(img + 4));
    const std::int32_t ypos = static_cast<std::int32_t>(mem().read32(img + 8));
    std::uint16_t buf_w = 0, buf_h = 0, bpl = 0;
    std::memcpy(&buf_w, mem().ptr(img + 12), 2);
    std::memcpy(&buf_h, mem().ptr(img + 14), 2);
    std::memcpy(&bpl, mem().ptr(img + 16), 2);
    const std::uint32_t buffer = mem().read32(img + 20);
    int clip_x = static_cast<int>(arg(c, 3)), clip_y = static_cast<int>(arg(c, 4));
    int clip_w = static_cast<int>(arg(c, 5)), clip_h = static_cast<int>(arg(c, 6));
    if (clip_x < 0) clip_x = 0;
    if (clip_y < 0) clip_y = 0;
    if (clip_w < 0) clip_w = 8192;
    if (clip_h < 0) clip_h = 8192;
    if (!buffer || !buf_w || !buf_h || !bpl) return ret(c, 0x80000100);
    const int x0 = xpos >> 6, y0 = ypos >> 6, frac = xpos & 63;
    const auto ink = [&](int gx, int gy) -> int {
        if (gx < 0 || gy < 0 || gx >= kCell || gy >= kCell) return 0;
        const int v = cell.px[static_cast<std::size_t>(gy) * kCell + gx];
        return format == 2 ? v : v >> 4;
    };
    for (int gy = 0; gy < kCell; ++gy) {
        const int dy = y0 + gy;
        if (dy < clip_y || dy >= clip_y + clip_h || dy < 0 || dy >= buf_h) continue;
        for (int gx = 0; gx < kCell; ++gx) {
            const int dx = x0 + gx;
            if (dx < clip_x || dx >= clip_x + clip_w || dx < 0 || dx >= buf_w) continue;
            const int a = ink(gx - 1, gy), b = ink(gx, gy);
            const int blended = (a * frac) / 64 + (b * (64 - frac) + 63) / 64;
            if (!blended) continue;
            if (format == 2) {
                uint8_t* dst = mem().ptr(buffer + static_cast<std::uint32_t>(dy) * bpl + static_cast<std::uint32_t>(dx));
                const int sum = *dst + blended;
                *dst = static_cast<std::uint8_t>(sum > 255 ? 255 : sum);
            } else {
                const std::uint32_t addr = buffer + static_cast<std::uint32_t>(dy) * bpl + static_cast<std::uint32_t>(dx) / 2;
                uint8_t* dst = mem().ptr(addr);
                const int shift = ((dx & 1) != static_cast<int>(format)) ? 4 : 0;
                const int old = (*dst >> shift) & 15;
                int sum = old + blended;
                if (sum > 15) sum = 15;
                *dst = static_cast<std::uint8_t>((*dst & ~(15 << shift)) | (sum << shift));
            }
        }
    }
    ret(c, 0);
}

constexpr HleBinding kBindings[] = {
    {"sceFontNewLib", NewLib},
    {"sceFontDoneLib", DoneLib},
    {"sceFontOpenUserFile", OpenUserFile},
    {"sceFontOpen", Open},
    {"sceFontClose", Close},
    {"sceFontGetNumFontList", GetNumFontList},
    {"sceFontFindOptimumFont", FindOptimumFont},
    {"sceFontGetFontInfo", GetFontInfo},
    {"sceFontGetCharInfo", GetCharInfo},
    {"sceFontGetCharGlyphImage_Clip", GetCharGlyphImageClip},
};

}  // namespace

void hle_install_font() { hle_bind(kBindings); }

}  // namespace psprecore::rt
