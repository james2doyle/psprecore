// sceLibFont: the game's UI text. A stub implementation with fixed 16x16 metrics and blank glyphs: enough
// for text layout and rendering to proceed (the game builds its own glyph cache from these answers). Real font
// data (glyph shapes from the game's font files) is future work; text draws as blank boxes until then.
//
// Struct layouts follow the PSP SDK (pspsdk, BSD-licensed); behaviour here is original.

#include <cstdio>
#include <cstring>

#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "core/foundation/core/foundation.h"

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

// SceFontCharacterData: width, height, top, bottom, shiftX, shiftY, shiftVertX, shiftVertY, hShift, vShift.
void write_chardata(std::uint32_t addr, std::uint32_t w, std::uint32_t h, std::uint32_t advance) {
    write32(addr, w);
    write32(addr + 4, h);
    write32(addr + 8, 0);   // top
    write32(addr + 12, 0);  // bottom
    write32(addr + 16, 0);  // shiftX
    write32(addr + 20, 0);  // shiftY
    write32(addr + 24, 0);  // shiftVertX
    write32(addr + 28, 0);  // shiftVertY
    write32(addr + 32, advance);
    write32(addr + 36, 0);  // vShift
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

// sceFontGetFontInfo(font, info)
void GetFontInfo(Context& c) {
    const std::uint32_t info = arg(c, 1);
    if (!info) return ret(c, 0x80000100);  // EINVAL-ish
    write_chardata(info, 16, 16, 16);     // maxFixed
    for (int i = 0; i < 10; ++i) write_float(info + 40 + 4 * static_cast<std::uint32_t>(i), i < 2 ? 16.0f : 0.0f);
    write16(info + 80, 16);  // maxWidth
    write16(info + 82, 16);  // maxHeight
    write32(info + 84, 256);  // characters
    write32(info + 88, 0);    // subCharacters
    // fontStyle left zeroed; depth 1 byte would go at +252 (leave 0: filled by every calloc'd caller anyway)
    ret(c, 0);
}

// sceFontGetCharInfo(font, charCode, charInfo)
void GetCharInfo(Context& c) {
    const std::uint32_t info = arg(c, 2);
    if (!info) return ret(c, 0x80000100);
    write32(info, 16);  // width
    write32(info + 4, 16);  // height
    write32(info + 8, 0);  // x
    write32(info + 12, 0);  // y
    write_chardata(info + 16, 16, 16, 16);
    write32(info + 56, 0);
    ret(c, 0);
}

// sceFontGetCharGlyphImage_Clip(font, charCode, glyphImage, clipX, clipY, clipWidth, clipHeight)
//
// GlyphImage (PGF.h): u32 pixelFormat, s32 xPos, s32 yPos, u16 width, u16 height, u16 bytesPerLine, u16 pad,
// u32 bufferPtr. The caller sets xPos/yPos; we fill the format/size/bitmap. The bitmap is a blank glyph for now.
void GetCharGlyphImageClip(Context& c) {
    const std::uint32_t img = arg(c, 2);
    if (!img) return ret(c, 0x80000100);
    constexpr std::uint32_t kWidth = 16, kHeight = 16;
    static std::uint32_t buffer = 0;
    if (!buffer) {
        buffer = kernel().user_heap().alloc(kWidth * kHeight, 64);
        if (!buffer) return ret(c, 0x80000100);
        mem().fill(buffer, 0, kWidth * kHeight);
    }
    mem().write32(img + 0, 2);             // pixelFormat
    write16(img + 12, kWidth);       // width
    write16(img + 14, kHeight);      // height
    write16(img + 16, kWidth);       // bytesPerLine
    write16(img + 18, 0);            // pad
    mem().write32(img + 20, buffer);       // bitmap
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
