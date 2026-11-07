// scePsmf: the PSMF container queries the game runs before playing a movie
// (sceMpegQueryStreamOffset opens the cooked media; this validates and describes).
// Semantics follow PPSSPP's scePsmf.cpp (offsets are big-endian in the header).

#include <cstring>

#include "psprecore/formats/psmf.hpp"
#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "core/foundation/core/foundation.h"

namespace psprecore::rt {

namespace {

GuestMemory& mem() { return kernel().memory(); }

// Video stream descriptors live at +0x82, 16 bytes each; dims are bytes 12/13 in 16px units (PPSSPP).
bool video_dims(const std::uint8_t* hdr, std::uint32_t& w, std::uint32_t& h) {
    const std::uint32_t n = (std::uint32_t{hdr[0x80]} << 8) | hdr[0x81];
    for (std::uint32_t i = 0; i < n && 0x82 + (i + 1) * 16 <= psmf::kPackSize; ++i) {
        const std::uint8_t* s = hdr + 0x82 + i * 16;
        if ((s[0] & 0xE0) == 0xE0) {  // video stream id
            w = s[12] * 16;
            h = s[13] * 16;
            return w && h;
        }
    }
    return false;
}

// scePsmfSetPsmf(psmf, buffer): validates the 2048-byte PSMF header and records it in the game's struct
// {version, headerSize, headerOffset, streamSize, ...} (PPSSPP's PsmfData layout).
void SetPsmf(Context& c) {
    const std::uint32_t psmf = arg(c, 0), data = arg(c, 1);
    if (!psmf || !data || !mem().valid(psmf, 32) || !mem().valid(data, 0x82)) return ret(c, 0x80020102);
    const std::uint8_t* p = mem().ptr(data);
    if (std::memcmp(p, "PSMF", 4) != 0) return ret(c, 0x80610001);
    const std::uint32_t version = std::uint32_t{p[4]} | (std::uint32_t{p[5]} << 8) | (std::uint32_t{p[6]} << 16) |
                                  (std::uint32_t{p[7]} << 24);
    const std::uint32_t stream_offset = (std::uint32_t{p[8]} << 24) | (std::uint32_t{p[9]} << 16) |
                                        (std::uint32_t{p[10]} << 8) | p[11];
    const std::uint32_t stream_size = (std::uint32_t{p[12]} << 24) | (std::uint32_t{p[13]} << 16) |
                                      (std::uint32_t{p[14]} << 8) | p[15];
    if (!version || !stream_offset) return ret(c, 0x80610002);
    for (int i = 0; i < 8; ++i) mem().write32(psmf + i * 4, 0);
    mem().write32(psmf, version);
    mem().write32(psmf + 4, 0x800);
    mem().write32(psmf + 8, data);
    mem().write32(psmf + 12, stream_size);
    ret(c, 0);
}

// scePsmfGetVideoInfo(psmf, info*): the current video stream's picture size (PPSSPP reads the
// header via the psmf struct's headerOffset and the stream descriptors at +0x82).
void GetVideoInfo(Context& c) {
    const std::uint32_t psmf = arg(c, 0), info = arg(c, 1);
    if (!psmf || !info || !mem().valid(psmf, 32) || !mem().valid(info, 8)) return ret(c, 0x80020102);
    const std::uint32_t header = mem().read32(psmf + 8);
    if (!header || !mem().valid(header, 0x92)) return ret(c, 0x80610002);
    const std::uint8_t* p = mem().ptr(header);
    if (std::memcmp(p, "PSMF", 4) != 0) return ret(c, 0x80610002);
    std::uint32_t w = 0, h = 0;
    if (!video_dims(p, w, h)) return ret(c, 0x80610002);
    mem().write32(info, w);
    mem().write32(info + 4, h);
    ret(c, 0);
}

constexpr HleBinding kBindings[] = {
    {"scePsmfSetPsmf", SetPsmf},
    {"scePsmfGetVideoInfo", GetVideoInfo},
};

}  // namespace

void hle_install_psmf() { hle_bind(kBindings); }

}  // namespace psprecore::rt
