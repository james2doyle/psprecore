// sceMpeg: PSMF movie playback (opening movie, cutscenes).
//
// The game streams a PSMF file (MPEG-PS, 2048-byte packs) into a ring buffer through a guest callback and pulls
// access units (AVC video, ATRAC audio) out of it. This module implements the ring buffer and a demuxer with
// real timestamps, so playback lasts as long as the movie and ends the way the game expects. Pictures and
// sound come from the cooked movie (AV1 + Vorbis, see cccook); without it frames are black and audio silent.
//
// Skipping (skip_movie) fast-forwards: the game plays the rest of the movie as usual, so its scripts see a normal
// playthrough (cutting the stream short can leave the scene that follows waiting for the movie forever), while the
// frames are no longer decoded (black) and the sound is silent. The front end runs the game unthrottled meanwhile.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

#include "audio/vorbis.hpp"
#include "psprecore/formats/cooked.hpp"
#include "psprecore/formats/psmf.hpp"
#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "media/av1_video.hpp"
#include "core/foundation/core/foundation.h"

namespace psprecore::rt {

namespace {

constexpr std::uint32_t kErrorNoData = 0x80618001;
constexpr std::uint32_t kPacketSize = 2048;
constexpr std::uint32_t kAtracOutputBytes = 8192;  // 2048 stereo s16 samples per ATRAC3+ frame
constexpr std::uint32_t kAtracEsSize = 2112;

GuestMemory& mem() { return kernel().memory(); }

// Guest SceMpegRingbuffer layout.
enum RingField : std::uint32_t {
    kPackets = 0, kPacketsRead = 4, kPacketsWritten = 8, kPacketsFree = 12, kPacketSizeField = 16, kData = 20,
    kCallback = 24, kCallbackArg = 28, kDataUpperBound = 32, kSema = 36, kMpeg = 40,
};

using psmf::AccessUnit;

enum StreamType { kAvc = psmf::kVideo, kAtrac = psmf::kAudio, kPcm = 2 };

struct Mpeg {
    std::uint32_t handle = 0, ringbuffer = 0;
    std::uint64_t written = 0, freed = 0;  // cumulative packet counts
    bool eof = false;
    std::uint64_t key = 0;               // the movie (cooked::fnv1a64 of its PSMF header)
    bool skipping = false;               // skip_movie: fast-forward to the end (black frames, silence)
    bool decoding = false;               // a frame was decoded since the video stream was registered
    std::uint64_t last_decode_vblank = 0;
    std::uint64_t stream_size = 0;        // PSMF payload bytes (from the header at QueryStreamOffset)
    std::uint64_t fed_bytes = 0;         // payload bytes handed to the demux so far
    std::uint64_t last_pts[2] = {};      // last access unit timestamp per stream
    psmf::Demuxer demux;
    bool registered[2] = {};
    int next_stream_id = 1;
    int stream_type[16] = {};
    std::uint32_t pixel_mode = 3;
    // Last vblank each stream's access units were pulled (GetAvcAu/GetAtracAu success).
    // A registered stream the game stops draining must not pin the ringbuffer forever
    // (this game's boot ident pulls one audio unit, then video only): streams idle for
    // more than a couple of seconds are ignored when freeing packets.
    std::uint64_t last_get[2] = {};
    // YCbCr direct-to-VRAM path (sceMpegAvcInitYCbCr / DecodeStopYCbCr / AvcCsc), used for cutscenes.
    std::uint32_t ycbcr_mode = 0, ycbcr_width = 0, ycbcr_height = 0, ycbcr_buffer = 0;
    // Cooked media for the current movie (psprecore/formats/cooked.hpp).
    nx::unique_ptr<media::Av1Video> video;
    nx::unique_ptr<audio::VorbisStream> audio;
    nx::vector<std::uint32_t> frame;
    int frame_width = 0;
    bool have_frame = false;  // the last next_frame() produced a real picture (else outputs stay black)
    std::uint64_t audio_base = 0;
    bool have_audio_base = false;
} g_mpeg;

void store16(std::uint8_t* p, std::uint32_t v) {
    const auto x = static_cast<std::uint16_t>(v);
    std::memcpy(p, &x, 2);
}

void sync_ring() {
    const std::uint32_t r = g_mpeg.ringbuffer;
    if (!r) return;
    const std::uint32_t packets = mem().read32(r + kPackets);
    mem().write32(r + kPacketsRead, static_cast<std::uint32_t>(g_mpeg.freed));
    mem().write32(r + kPacketsWritten, packets ? static_cast<std::uint32_t>(g_mpeg.written % packets) : 0);
    mem().write32(r + kPacketsFree, packets - static_cast<std::uint32_t>(g_mpeg.written - g_mpeg.freed));
}

// Frees packets every stream being drained has moved past (idle streams do not pin the buffer).
void release_packets() {
    std::uint64_t keep = g_mpeg.written;
    const std::uint64_t now = kernel().vblank_count();
    for (int s = 0; s < 2; ++s) {
        if (!g_mpeg.registered[s]) continue;
        if (g_mpeg.last_get[s] && now - g_mpeg.last_get[s] > 120) continue;  // idle: ignore
        const auto stream = static_cast<psmf::Stream>(s);
        const auto& units = g_mpeg.demux.units(stream);
        std::uint64_t pack = 0;
        if (!units.empty()) keep = std::min(keep, units.front().pack);
        else if (g_mpeg.demux.assembling(stream, pack)) keep = std::min(keep, pack);
    }
    g_mpeg.freed = std::max(g_mpeg.freed, keep);
    sync_ring();
}

// Pops the next complete unit of a stream, or returns false (no data yet / end of stream).
bool next_unit(int stream, AccessUnit& out) {
    auto& units = g_mpeg.demux.units(static_cast<psmf::Stream>(stream));
    if (units.empty() && g_mpeg.eof) g_mpeg.demux.finish();
    if (units.empty()) return false;
    out = std::move(units.front());
    units.pop_front();
    return true;
}

void write_au(std::uint32_t au, const AccessUnit& u) {
    mem().write32(au + 0, static_cast<std::uint32_t>(u.pts >> 32));
    mem().write32(au + 4, static_cast<std::uint32_t>(u.pts));
    mem().write32(au + 8, static_cast<std::uint32_t>(u.dts >> 32));
    mem().write32(au + 12, static_cast<std::uint32_t>(u.dts));
    mem().write32(au + 20, static_cast<std::uint32_t>(u.data.size()));
}

// ---- library and ring buffer ---------------------------------------------------------------------------

void Init(Context& c) { ret(c, 0); }
void Finish(Context& c) { ret(c, 0); }
void QueryMemSize(Context& c) { ret(c, 0x10000); }
void RingbufferQueryMemSize(Context& c) { ret(c, arg(c, 0) * (104 + kPacketSize)); }

// sceMpegRingbufferConstruct(ringbuffer, packets, data, size, callback, callback_arg)
void RingbufferConstruct(Context& c) {
    const std::uint32_t r = arg(c, 0), packets = arg(c, 1), data = arg(c, 2);
    mem().fill(r, 0, 48);
    mem().write32(r + kPackets, packets);
    mem().write32(r + kPacketsFree, packets);
    mem().write32(r + kPacketSizeField, kPacketSize);
    mem().write32(r + kData, data);
    mem().write32(r + kCallback, arg(c, 4));
    mem().write32(r + kCallbackArg, arg(c, 5));
    mem().write32(r + kDataUpperBound, data + packets * kPacketSize);
    mem().write32(r + kSema, 0xFFFFFFFF);
    ret(c, 0);
}
void RingbufferDestruct(Context& c) { ret(c, 0); }

// sceMpegCreate(mpeg, data, size, ringbuffer, frame_width, mode, ddr_top)
void Create(Context& c) {
    const std::uint32_t mpeg = arg(c, 0), data = arg(c, 1), ring = arg(c, 3);
    g_mpeg = Mpeg{};
    g_mpeg.handle = data + 0x30;
    g_mpeg.ringbuffer = ring;
    mem().write32(mpeg, g_mpeg.handle);
    mem().write_string(g_mpeg.handle, "LIBMPEG");
    mem().write_string(g_mpeg.handle + 8, "001");
    mem().write32(g_mpeg.handle + 12, 0xFFFFFFFF);
    if (ring) mem().write32(ring + kMpeg, mpeg);
    ret(c, 0);
}
void Delete(Context& c) {
    g_mpeg.skipping = false;
    g_mpeg.decoding = false;
    ret(c, 0);
}

// sceMpegRingbufferPut(ringbuffer, packets, available): fills packets through the guest callback.
void RingbufferPut(Context& c) {
    const std::uint32_t r = arg(c, 0);
    std::uint32_t wanted = std::min(arg(c, 1), arg(c, 2));
    const std::uint32_t packets = mem().read32(r + kPackets), data = mem().read32(r + kData);
    const std::uint32_t callback = mem().read32(r + kCallback), callback_arg = mem().read32(r + kCallbackArg);
    g_mpeg.ringbuffer = r;
    std::uint32_t added = 0;
    while (wanted > 0 && packets && callback) {
        const std::uint32_t slot = static_cast<std::uint32_t>(g_mpeg.written % packets);
        const std::uint32_t chunk = std::min(wanted, packets - slot);
        const auto got = static_cast<std::int32_t>(kernel().call_guest(callback, {data + slot * kPacketSize, chunk, callback_arg}));
        if (got <= 0) {
            if (got == 0 && !g_mpeg.eof) {
                g_mpeg.eof = true;
                std::fprintf(stderr, "sceMpeg: end of stream at vblank %llu\n",
                             static_cast<unsigned long long>(kernel().vblank_count()));
            }
            break;
        }
        const std::uint32_t n = std::min(static_cast<std::uint32_t>(got), chunk);
        for (std::uint32_t i = 0; i < n; ++i)
            g_mpeg.demux.feed({reinterpret_cast<const std::byte*>(mem().ptr(data + (slot + i) * kPacketSize)), kPacketSize},
                              g_mpeg.written + i);
        g_mpeg.written += n;
        added += n;
        wanted -= n;
        if (n < chunk) break;
    }
    sync_ring();
    g_mpeg.fed_bytes += static_cast<std::uint64_t>(added) * kPacketSize;
    if (!g_mpeg.eof && g_mpeg.stream_size && g_mpeg.fed_bytes >= g_mpeg.stream_size) {
        g_mpeg.eof = true;  // the whole stream is in; the demux drains, then the movie is over
        std::fprintf(stderr, "sceMpeg: end of stream at vblank %llu\n",
                     static_cast<unsigned long long>(kernel().vblank_count()));
    }
    ret(c, added);
}

void RingbufferAvailableSize(Context& c) {
    const std::uint32_t r = arg(c, 0);
    ret(c, mem().read32(r + kPackets) - static_cast<std::uint32_t>(g_mpeg.written - g_mpeg.freed));
}

// ---- streams and access units --------------------------------------------------------------------------

// sceMpegQueryStreamOffset(mpeg, psmf_header, offset*): data starts at the big-endian offset at +8.
void QueryStreamOffset(Context& c) {
    psmf::Header header;
    if (!psmf::parse_header({reinterpret_cast<const std::byte*>(mem().ptr(arg(c, 1))), psmf::kPackSize}, header)) {
        ret(c, 0x80610002);  // invalid value
        return;
    }
    mem().write32(arg(c, 2), header.data_offset);

    // Identify the movie by its header and open the cooked streams.
    const std::uint64_t key = cooked::fnv1a64({reinterpret_cast<const std::byte*>(mem().ptr(arg(c, 1))), psmf::kPackSize});
    g_mpeg.video = media::Av1Video::open(cooked::movie_video_path(cooked_root(), key));
    g_mpeg.audio = audio::VorbisStream::open(cooked::movie_audio_path(cooked_root(), key));
    g_mpeg.frame.clear();
    g_mpeg.have_audio_base = false;
    g_mpeg.key = key;
    g_mpeg.skipping = false;
    g_mpeg.decoding = false;
    g_mpeg.stream_size = header.data_size;
    g_mpeg.fed_bytes = 0;
    g_mpeg.last_pts[0] = g_mpeg.last_pts[1] = 0;
    std::fprintf(stderr, "sceMpeg: movie %016llX at vblank %llu\n", static_cast<unsigned long long>(key),
                 static_cast<unsigned long long>(kernel().vblank_count()));
    if (!g_mpeg.video)
        std::fprintf(stderr, "sceMpeg: no cooked movie %s (run cccook); showing black\n",
                     cooked::movie_video_path(cooked_root(), key).string().c_str());
    ret(c, 0);
}

void RegistStream(Context& c) {
    const int type = static_cast<int>(arg(c, 1));
    const int id = g_mpeg.next_stream_id++;
    g_mpeg.stream_type[id & 15] = type;
    if (type == kAvc || type == kAtrac) g_mpeg.registered[type] = true;
    ret(c, static_cast<std::uint32_t>(id));
}
void UnRegistStream(Context& c) {
    const int type = g_mpeg.stream_type[arg(c, 1) & 15];
    if (type == kAvc || type == kAtrac) g_mpeg.registered[type] = false;
    if (type == kAvc) g_mpeg.decoding = g_mpeg.skipping = false;  // the game is done with the movie
    release_packets();
    ret(c, 0);
}
void FlushAllStream(Context& c) {
    g_mpeg.demux.reset();
    g_mpeg.freed = g_mpeg.written;
    g_mpeg.eof = false;
    sync_ring();
    ret(c, 0);
}

void MallocAvcEsBuf(Context& c) { ret(c, 1); }
void FreeAvcEsBuf(Context& c) { ret(c, 0); }

// sceMpegQueryAtracEsSize(mpeg, es_size*, out_size*)
void QueryAtracEsSize(Context& c) {
    if (arg(c, 1)) mem().write32(arg(c, 1), kAtracEsSize);
    if (arg(c, 2)) mem().write32(arg(c, 2), kAtracOutputBytes);
    ret(c, 0);
}

// sceMpegInitAu(mpeg, es_buffer, au)
void InitAu(Context& c) {
    const std::uint32_t au = arg(c, 2);
    for (std::uint32_t i = 0; i < 16; i += 4) mem().write32(au + i, 0xFFFFFFFF);
    mem().write32(au + 16, arg(c, 1));
    mem().write32(au + 20, 0);
    ret(c, 0);
}

// sceMpegGetAvcAu / sceMpegGetAtracAu(mpeg, stream, au, attr*)
void get_au(Context& c, int stream) {
    AccessUnit u;
    if (!next_unit(stream, u)) {
        if (g_mpeg.eof) {  // drained to the end (PPSSPP: pts kept, dts -1, NO_DATA)
            mem().write32(arg(c, 2), static_cast<std::uint32_t>(g_mpeg.last_pts[stream] >> 32));
            mem().write32(arg(c, 2) + 4, static_cast<std::uint32_t>(g_mpeg.last_pts[stream]));
            mem().write32(arg(c, 2) + 8, 0xFFFFFFFF);
            mem().write32(arg(c, 2) + 12, 0xFFFFFFFF);
            mem().write32(arg(c, 2) + 20, 0);
        }
        ret(c, kErrorNoData);
        return;
    }
    write_au(arg(c, 2), u);
    g_mpeg.last_pts[stream] = u.pts;
    g_mpeg.last_get[stream] = kernel().vblank_count();
    if (arg(c, 3)) mem().write32(arg(c, 3), stream == kAvc ? 1 : 0);
    release_packets();
    ret(c, 0);
}
void GetAvcAu(Context& c) { get_au(c, kAvc); }
void GetAtracAu(Context& c) { get_au(c, kAtrac); }
void GetPcmAu(Context& c) { ret(c, kErrorNoData); }

// sceMpegAvcDecodeMode(mpeg, mode*): mode = {unknown, pixel format}
void AvcDecodeMode(Context& c) {
    if (arg(c, 1)) g_mpeg.pixel_mode = mem().read32(arg(c, 1) + 4);
    ret(c, 0);
}

// Advances the cooked movie one picture (shared by the RGB and YCbCr decode paths).
// True when a real picture is available; otherwise the caller draws black.
bool next_frame() {
    g_mpeg.decoding = true;
    g_mpeg.last_decode_vblank = kernel().vblank_count();
    g_mpeg.have_frame = false;
    if (!g_mpeg.skipping && g_mpeg.video && g_mpeg.video->next(g_mpeg.frame)) {
        g_mpeg.frame_width = g_mpeg.video->width();
        g_mpeg.have_frame = !g_mpeg.frame.empty() && g_mpeg.frame_width > 0;
    }
    return g_mpeg.have_frame;
}

// BT.601 full-range RGBA -> YCbCr planar 4:2:0 (the inverse of Av1Video's convert,
// so AvcCsc below reproduces the cooked picture).
void rgba_to_ycbcr(const nx::vector<std::uint32_t>& rgba, int w, int h, std::uint8_t* y, std::uint8_t* u,
                   std::uint8_t* v, int stride) {
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const std::uint32_t p = rgba[static_cast<std::size_t>(j) * static_cast<std::size_t>(w) + i];
            const int r = p & 0xFF, g = (p >> 8) & 0xFF, b = (p >> 16) & 0xFF;
            y[j * stride + i] = static_cast<std::uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
        }
    }
    for (int j = 0; j < h / 2; ++j) {
        for (int i = 0; i < w / 2; ++i) {
            int r = 0, g = 0, b = 0;
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) {
                    const std::uint32_t p =
                        rgba[static_cast<std::size_t>(2 * j + dj) * static_cast<std::size_t>(w) + (2 * i + di)];
                    r += p & 0xFF;
                    g += (p >> 8) & 0xFF;
                    b += (p >> 16) & 0xFF;
                }
            r /= 4;
            g /= 4;
            b /= 4;
            u[j * (stride / 2) + i] = static_cast<std::uint8_t>((-43 * r - 84 * g + 127 * b) / 256 + 128);
            v[j * (stride / 2) + i] = static_cast<std::uint8_t>((127 * r - 106 * g - 21 * b) / 256 + 128);
        }
    }
}

void write_ycbcr(std::uint32_t buffer, int fw, int fh, const nx::vector<std::uint32_t>& frame, int w) {
    std::uint8_t* base = mem().ptr(buffer);
    rgba_to_ycbcr(frame, w, 272, base, base + fw * fh, base + fw * fh + (fw / 2) * (fh / 2), fw);
}

// True black in planar 4:2:0 (Y=0, neutral chroma); plain zeros would convert to dark green.
void fill_black_ycbcr(std::uint32_t buffer, int fw, int fh) {
    mem().fill(buffer, 0, static_cast<std::uint32_t>(fw * fh));
    std::memset(mem().ptr(buffer + fw * fh), 128, static_cast<std::size_t>(fw * fh / 2));
}

// BT.601 full-range YCbCr planar 4:2:0 -> RGBA8888 (mirrors Av1Video's convert).
// range* is {x, y, width, height}: only that rectangle is converted (PPSSPP).
void AvcCsc(Context& c) {
    const std::uint32_t source = arg(c, 1), range = arg(c, 2), frame_width = arg(c, 3), dest = arg(c, 4);
    if (!dest) {
        static int zerodst = 0;
        if (++zerodst <= 3 || zerodst % 50 == 0)
            std::fprintf(stderr, "sceMpeg: AvcCsc dest=0 (vblank %llu) #%d\n",
                         (unsigned long long)kernel().vblank_count(), zerodst);
    }
    if (!source || !dest || !range || !mem().valid(range, 16)) return ret(c, 0xFFFFFFFF);
    const std::uint32_t fw = frame_width ? frame_width : 480;
    constexpr int kFh = 272;
    // Source YCbCr is packed at the InitYCbCr width; dest uses frameWidth as its stride (PPSSPP).
    const int sfw = g_mpeg.ycbcr_width ? static_cast<int>(g_mpeg.ycbcr_width) : 480;
    int x = static_cast<int>(mem().read32(range)), y = static_cast<int>(mem().read32(range + 4));
    int w = static_cast<int>(mem().read32(range + 8)), h = static_cast<int>(mem().read32(range + 12));
    x = std::clamp(x, 0, 480);
    y = std::clamp(y, 0, 272);
    w = std::clamp(w, 0, 480 - x);
    h = std::clamp(h, 0, 272 - y);
    if (!mem().valid(dest, fw * kFh * 4) || !mem().valid(source, static_cast<std::uint32_t>(sfw * kFh * 3 / 2)))
        return ret(c, 0xFFFFFFFF);
    mark_guest_written(dest, fw * kFh * 4);
    std::uint8_t* row = mem().ptr(dest);
    if (!g_mpeg.have_frame) {
        constexpr std::uint32_t kBlack = 0xFF000000u;
        for (int j = y; j < y + h; ++j)
            for (int i = x; i < x + w; ++i)
                std::memcpy(row + (j * fw + i) * 4, &kBlack, 4);
    } else {
        const std::uint8_t* yp = mem().ptr(source);
        const std::uint8_t* up = yp + sfw * kFh, * vp = up + (sfw / 2) * (kFh / 2);
        for (int j = y; j < y + h; ++j) {
            std::uint32_t* out = reinterpret_cast<std::uint32_t*>(row + j * fw * 4);
            for (int i = x; i < x + w; ++i) {
                const int yy = yp[j * sfw + i];
                const int uu = up[(j / 2) * (sfw / 2) + (i / 2)] - 128;
                const int vv = vp[(j / 2) * (sfw / 2) + (i / 2)] - 128;
                const int r = std::clamp((yy * 65536 + 91881 * vv + 32768) >> 16, 0, 255);
                const int g = std::clamp((yy * 65536 + 32768 - 22554 * uu - 46802 * vv) >> 16, 0, 255);
                const int b = std::clamp((yy * 65536 + 116130 * uu + 32768) >> 16, 0, 255);
                out[i] = static_cast<std::uint32_t>(r) | (static_cast<std::uint32_t>(g) << 8) |
                         (static_cast<std::uint32_t>(b) << 16) | 0xFF000000u;
            }
        }
    }
    ret(c, 0);
}

// sceMpegAvcDecode(mpeg, au, frame_width, buffer*, init*): the next frame of the cooked movie, in display order
// (black without cooked media). Writes 480x272 pixels in the pixel mode chosen by sceMpegAvcDecodeMode.
void AvcDecode(Context& c) {
    const bool have = next_frame();
    const std::uint32_t width = arg(c, 2), buffer = arg(c, 3) ? mem().read32(arg(c, 3)) : 0;
    if (buffer && width) {
        const std::uint32_t bpp = g_mpeg.pixel_mode == 3 ? 4 : 2;
        mark_guest_written(buffer, width * 272 * bpp);  // the decoded frame lands in the game's buffer
        const int fw = g_mpeg.frame_width;
        constexpr std::uint32_t kBlack = 0xFF000000u;
        const std::uint16_t black16 = g_mpeg.pixel_mode == 1 ? 0x8000 : g_mpeg.pixel_mode == 2 ? 0xF000 : 0;
        for (std::uint32_t y = 0; y < 272; ++y) {
            std::uint8_t* row = mem().ptr(buffer + y * width * bpp);
            if (!have) {  // black (no cooked movie, or skipping): no per-pixel conversion
                if (bpp == 4) {
                    for (std::uint32_t x = 0; x < 480; ++x) std::memcpy(row + x * 4, &kBlack, 4);
                } else {
                    for (std::uint32_t x = 0; x < 480; ++x) store16(row + x * 2, black16);
                }
                continue;
            }
            if (bpp == 4 && fw >= 480 && (y + 1) * static_cast<std::uint32_t>(fw) <= g_mpeg.frame.size()) {
                std::memcpy(row, g_mpeg.frame.data() + y * static_cast<std::uint32_t>(fw), 480 * 4);  // 8888 as decoded
                continue;
            }
            for (std::uint32_t x = 0; x < 480; ++x) {
                const bool inside = have && static_cast<int>(x) < fw &&
                                    (y + 1) * static_cast<std::uint32_t>(fw) <= g_mpeg.frame.size();
                const std::uint32_t p = inside ? g_mpeg.frame[y * static_cast<std::uint32_t>(fw) + x] : 0xFF000000u;
                const std::uint32_t r = p & 0xFF, g = (p >> 8) & 0xFF, b = (p >> 16) & 0xFF;
                switch (g_mpeg.pixel_mode) {
                    case 0: store16(row + x * 2, (r >> 3) | ((g >> 2) << 5) | ((b >> 3) << 11)); break;
                    case 1: store16(row + x * 2, (r >> 3) | ((g >> 3) << 5) | ((b >> 3) << 10) | 0x8000); break;
                    case 2: store16(row + x * 2, (r >> 4) | ((g >> 4) << 4) | ((b >> 4) << 8) | 0xF000); break;
                    default: std::memcpy(row + x * 4, &p, 4); break;
                }
            }
        }
    }
    if (arg(c, 4)) mem().write32(arg(c, 4), 1);
    ret(c, 0);
}
void AvcDecodeFlush(Context& c) { ret(c, 0); }

// sceMpegQueryStreamSize(header_buffer, size*, width*, height*): memory and picture size for the PSMF.
// sceMpegQueryStreamSize(header_buffer, size*): the PSMF stream size (PPSSPP takes only these two
// arguments; anything else in the registers is not an output).
void QueryStreamSize(Context& c) {
    psmf::Header header;
    if (!psmf::parse_header({reinterpret_cast<const std::byte*>(mem().ptr(arg(c, 0))), psmf::kPackSize}, header)) {
        if (arg(c, 1) && mem().valid(arg(c, 1), 4)) mem().write32(arg(c, 1), 0);
        return ret(c, 0x80610002);  // invalid value
    }
    if (arg(c, 1)) mem().write32(arg(c, 1), header.data_size);
    ret(c, 0);
}

// sceMpegAvcQueryYCbCrSize(mpeg, mode, width, height, size*): planar 4:2:0 buffer size.
void AvcQueryYCbCrSize(Context& c) {
    const std::uint32_t width = arg(c, 2), height = arg(c, 3);
    if (!width || !height || (width & 15) || (height & 15) || width > 480 || height > 272)
        return ret(c, 0x80610002);
    if (arg(c, 4)) mem().write32(arg(c, 4), (width / 2) * (height / 2) * 6 + 128);
    ret(c, 0);
}

// sceMpegAvcInitYCbCr(mpeg, mode, width, height, buffer): the game's YCbCr target for DecodeStopYCbCr.
void AvcInitYCbCr(Context& c) {
    g_mpeg.ycbcr_mode = arg(c, 1);
    g_mpeg.ycbcr_width = arg(c, 2);
    g_mpeg.ycbcr_height = arg(c, 3);
    g_mpeg.ycbcr_buffer = arg(c, 4);
    ret(c, 0);
}

// sceMpegAvcDecodeYCbCr(mpeg, au, buffer, init*): the next cooked picture as YCbCr planar 4:2:0
// (black without cooked media). Dimensions come from sceMpegAvcInitYCbCr; init is flagged complete.
void AvcDecodeYCbCr(Context& c) {
    const bool have = next_frame();
    // PPSSPP: bufferAddr points at the game's buffer pointer.
    const std::uint32_t buffer =
        (arg(c, 2) && mem().valid(arg(c, 2), 4)) ? mem().read32(arg(c, 2)) : arg(c, 2);
    const int fw = g_mpeg.ycbcr_width ? static_cast<int>(g_mpeg.ycbcr_width) : 480;
    const int fh = g_mpeg.ycbcr_height ? static_cast<int>(g_mpeg.ycbcr_height) : 272;
    if (buffer && fw > 0 && fh > 0 && mem().valid(buffer, static_cast<std::uint32_t>(fw * fh * 3 / 2))) {
        mark_guest_written(buffer, static_cast<std::uint32_t>(fw * fh * 3 / 2));
        if (have && g_mpeg.frame_width >= fw)
            write_ycbcr(buffer, fw, fh, g_mpeg.frame, g_mpeg.frame_width);
        else
            fill_black_ycbcr(buffer, fw, fh);
    }
    if (arg(c, 3) && mem().valid(arg(c, 3), 4)) mem().write32(arg(c, 3), 1);
    ret(c, 0);
}

// sceMpegAvcDecodeStopYCbCr(mpeg, buffer, status*): the next cooked picture as YCbCr planar 4:2:0
// (black without cooked media). Dimensions come from sceMpegAvcInitYCbCr; the status word is signaled
// on completion (PPSSPP's stub writes 0, but the game polls it, so a synchronous decode reports 1).
// Like AvcDecodeYCbCr, the game passes the address of its buffer-pointer slot: the real target is the
// pointer stored there (writing the frame to the slot itself would corrupt the movie struct).
void AvcDecodeStopYCbCr(Context& c) {
    const bool have = next_frame();
    const std::uint32_t slot = arg(c, 1);
    const std::uint32_t buffer =
        slot ? ((mem().valid(slot, 4) && mem().read32(slot)) ? mem().read32(slot) : slot) : g_mpeg.ycbcr_buffer;
    const int fw = g_mpeg.ycbcr_width ? static_cast<int>(g_mpeg.ycbcr_width) : 480;
    const int fh = g_mpeg.ycbcr_height ? static_cast<int>(g_mpeg.ycbcr_height) : 272;
    if (buffer && fw > 0 && fh > 0 && mem().valid(buffer, static_cast<std::uint32_t>(fw * fh * 3 / 2))) {
        mark_guest_written(buffer, static_cast<std::uint32_t>(fw * fh * 3 / 2));
        if (have && g_mpeg.frame_width >= fw)
            write_ycbcr(buffer, fw, fh, g_mpeg.frame, g_mpeg.frame_width);
        else
            fill_black_ycbcr(buffer, fw, fh);
    }
    if (arg(c, 2) && mem().valid(arg(c, 2), 4)) mem().write32(arg(c, 2), 1);
    ret(c, 0);
}

// sceMpegAtracDecode(mpeg, au, buffer, init): 2048 stereo samples of the cooked movie audio at the unit's
// timestamp (silence without cooked media).
void AtracDecode(Context& c) {
    const std::uint32_t au = arg(c, 1), buffer = arg(c, 2);
    if (!buffer) return ret(c, 0);
    mem().fill(buffer, 0, kAtracOutputBytes);
    if (g_mpeg.audio && au && !g_mpeg.skipping) {
        const std::uint64_t pts = (std::uint64_t{mem().read32(au)} << 32) | mem().read32(au + 4);
        if (!g_mpeg.have_audio_base) {
            g_mpeg.audio_base = pts;
            g_mpeg.have_audio_base = true;
        }
        const std::uint64_t ticks = pts >= g_mpeg.audio_base ? pts - g_mpeg.audio_base : 0;
        const std::uint64_t frame = (ticks * 44100 / 90000 + 1024) / 2048;  // nearest ATRAC frame
        auto* out = reinterpret_cast<std::int16_t*>(mem().ptr(buffer));
        if (g_mpeg.audio->channels() == 2) {
            g_mpeg.audio->read(frame * 2048, out, 2048);
        } else {
            std::int16_t mono[2048] = {};
            const std::uint32_t got = g_mpeg.audio->read(frame * 2048, mono, 2048);
            for (std::uint32_t i = 0; i < got; ++i) out[2 * i] = out[2 * i + 1] = mono[i];
        }
    }
    ret(c, 0);
}

constexpr HleBinding kBindings[] = {
    {"sceMpegInit", Init},
    {"sceMpegFinish", Finish},
    {"sceMpegQueryMemSize", QueryMemSize},
    {"sceMpegRingbufferQueryMemSize", RingbufferQueryMemSize},
    {"sceMpegRingbufferConstruct", RingbufferConstruct},
    {"sceMpegRingbufferDestruct", RingbufferDestruct},
    {"sceMpegRingbufferPut", RingbufferPut},
    {"sceMpegRingbufferAvailableSize", RingbufferAvailableSize},
    {"sceMpegCreate", Create},
    {"sceMpegDelete", Delete},
    {"sceMpegQueryStreamOffset", QueryStreamOffset},
    {"sceMpegRegistStream", RegistStream},
    {"sceMpegUnRegistStream", UnRegistStream},
    {"sceMpegFlushAllStream", FlushAllStream},
    {"sceMpegMallocAvcEsBuf", MallocAvcEsBuf},
    {"sceMpegFreeAvcEsBuf", FreeAvcEsBuf},
    {"sceMpegQueryAtracEsSize", QueryAtracEsSize},
    {"sceMpegInitAu", InitAu},
    {"sceMpegGetAvcAu", GetAvcAu},
    {"sceMpegGetAtracAu", GetAtracAu},
    {"sceMpegGetPcmAu", GetPcmAu},
    {"sceMpegAvcDecodeMode", AvcDecodeMode},
    {"sceMpegAvcDecode", AvcDecode},
    {"sceMpegAvcDecodeFlush", AvcDecodeFlush},
    {"sceMpegAvcDecodeYCbCr", AvcDecodeYCbCr},
    {"sceMpegQueryStreamSize", QueryStreamSize},
    {"sceMpegAvcQueryYCbCrSize", AvcQueryYCbCrSize},
    {"sceMpegAvcInitYCbCr", AvcInitYCbCr},
    {"sceMpegAvcDecodeStopYCbCr", AvcDecodeStopYCbCr},
    {"sceMpegAvcCsc", AvcCsc},
    {"sceMpegAtracDecode", AtracDecode},
};

}  // namespace

void hle_install_mpeg() { hle_bind(kBindings); }

namespace {
nx::vector<std::uint64_t> g_unskippable;  // set_unskippable_movies
}  // namespace

bool movie_playing() {
    // A movie is on screen while the game decodes its frames (it stops at the end without telling sceMpeg).
    return g_mpeg.registered[kAvc] && g_mpeg.decoding && kernel().vblank_count() - g_mpeg.last_decode_vblank < 30;
}

bool movie_skipping() { return g_mpeg.skipping && movie_playing(); }

bool movie_skippable() {
    return movie_playing() && !g_mpeg.skipping &&
           std::find(g_unskippable.begin(), g_unskippable.end(), g_mpeg.key) == g_unskippable.end();
}

void set_unskippable_movies(std::span<const std::uint64_t> keys) { g_unskippable.assign(keys.begin(), keys.end()); }

void skip_movie() {
    if (!movie_playing() || g_mpeg.skipping) return;
    std::fprintf(stderr, "sceMpeg: skipping the movie from vblank %llu\n",
                 static_cast<unsigned long long>(kernel().vblank_count()));
    g_mpeg.skipping = true;
    g_mpeg.frame.clear();  // black from now on
}

}  // namespace psprecore::rt
