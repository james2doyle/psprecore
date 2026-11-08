// IoFileMgrForUser and sceUmdUser.
//
// disc0:/... maps to the extracted disc folder, ms0:/... to the host save folder. Games that read an archive
// through the raw UMD block devices (umd0:/umd1:: seeks are absolute sectors, reads are sector counts) get the
// file mapped with map_disc_file at its LBN. A DiscOverlay (mods) can change those sectors and whole files.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>

#include "psprecore/io/file.hpp"
#include "psprecore/io/path.hpp"
#include "psprecore/text.hpp"
#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/io.hpp"
#include "psprecore/runtime/kernel.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/platform/filesystem.h"
#include "core/foundation/strings/format.h"
#include "core/foundation/strings/utf8_string.h"

namespace psprecore::rt {

namespace {

constexpr std::uint32_t kSector = 2048;
constexpr std::uint32_t kErrorNotFound = 0x80010002, kErrorBadFile = 0x80020323;

nx::string g_game_dir, g_save_dir;

GuestMemory& mem() { return kernel().memory(); }

struct OpenFile {
    nx::fs::stream_ptr in;                     // files opened for reading
    // Files opened for writing are kept in memory and written back whole after every write (the game writes
    // little; save data goes through the savedata utility).
    bool writable = false;
    nx::vector<char> contents;
    nx::string host;
    const nx::vector<char>* memory = nullptr;  // served from memory (a file replaced by the disc overlay)
    std::uint64_t memory_pos = 0;               // position in `contents` or `memory`
    bool block = false;        // umd0:/umd1:
    std::uint64_t sector = 0;  // block devices: current absolute LBN
    bool lbn_bound = false;    // disc0:/sce_lbn<LBN>_size<SIZE>: reads/seeks stay in [lbn_start, lbn_start + lbn_size)
    std::uint64_t lbn_start = 0, lbn_size = 0;  // host offsets of the range
    std::int64_t async_result = 0;
    bool async_pending = false;
    nx::string path;
};
// Open files stay where they are (nx::map moves its elements).
nx::map<int, nx::unique_ptr<OpenFile>> g_files;
int g_next_fd = 3;

struct OpenDir {
    nx::vector<nx::fs::directory_entry> entries;
    std::size_t next = 0;
};
nx::map<int, OpenDir> g_dirs;

// The file read through the block devices (map_disc_file).
nx::string g_mapped_path;  // relative to the disc folder, lower case
std::uint32_t g_mapped_lbn = 0;
nx::fs::stream_ptr g_mapped;
nx::vector<DiscFile> g_disc_files;  // lbn.tsv: every file's start sector (for sce_lbn opens)
std::unique_ptr<DiscOverlay> g_overlay;

nx::string lower(nx::string s) {
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Guest path on the disc -> path relative to the disc folder, lower case with '/' (empty for other devices).
nx::string disc_relative(const nx::string& guest) {
    const auto colon = guest.find(':');
    if (colon == nx::string::npos) return {};
    const nx::string device = lower(guest.substr(0, colon));
    if (device != "disc0" && device != "umd0" && device != "umd1") return {};
    nx::string rest = lower(guest.substr(colon + 1));
    for (auto& ch : rest)
        if (ch == '\\') ch = '/';
    while (!rest.empty() && rest.front() == '/') rest.erase(0, 1);
    return rest;
}

// Guest path -> host path (empty if the device is not backed by files).
nx::string host_path(const nx::string& guest) {
    const auto colon = guest.find(':');
    if (colon == nx::string::npos) return {};
    const nx::string device = lower(guest.substr(0, colon));
    nx::string rest = guest.substr(colon + 1);
    while (!rest.empty() && (rest.front() == '/' || rest.front() == '\\')) rest.erase(0, 1);
    if (device == "disc0" || device == "umd0" || device == "umd1") return rest.empty() ? nx::string{} : path_join(g_game_dir, rest);
    if (device == "ms0" || device == "fatms0") return path_join(g_save_dir, rest);
    return {};
}

bool is_block_device(const nx::string& guest) {
    const nx::string l = lower(guest);
    return l == "umd0:" || l == "umd1:" || l == "umd0:/" || l == "umd1:/";
}

// disc0:/sce_lbn<LBN>_size<SIZE> (lower-cased, no leading slash) -> the sector range, or nullopt.
std::optional<std::pair<std::uint32_t, std::uint32_t>> parse_lbn(nx::string_view rel) {
    unsigned lbn = 0, size = 0;
    if (std::sscanf(nx::string(rel).c_str(), "sce_lbn%x_size%x", &lbn, &size) == 2 && size != 0)
        return std::make_pair(static_cast<std::uint32_t>(lbn), static_cast<std::uint32_t>(size));
    return std::nullopt;
}

std::int64_t block_read(OpenFile& f, std::uint32_t buffer, std::uint32_t sectors) {
    nx::vector<char> data(static_cast<std::size_t>(sectors) * kSector, 0);
    if (f.sector >= g_mapped_lbn && !g_mapped_path.empty()) {
        if (g_overlay) g_overlay->read_sectors(f.sector - g_mapped_lbn, sectors, data.data());
        else read_disc_sectors(f.sector - g_mapped_lbn, sectors, data.data());
    }
    std::memcpy(mem().ptr(buffer), data.data(), data.size());
    f.sector += sectors;
    return sectors;
}

std::int64_t do_read(int fd, std::uint32_t buffer, std::uint32_t size) {
    const auto it = g_files.find(fd);
    if (it == g_files.end()) return static_cast<std::int32_t>(kErrorBadFile);
    OpenFile& f = *it->second;
    if (f.block) return block_read(f, buffer, size);
    if (f.memory || f.writable) {
        const nx::vector<char>& data = f.writable ? f.contents : *f.memory;
        const std::uint64_t n = std::min<std::uint64_t>(size, data.size() - std::min<std::uint64_t>(f.memory_pos, data.size()));
        std::memcpy(mem().ptr(buffer), data.data() + f.memory_pos, n);
        f.memory_pos += n;
        return static_cast<std::int64_t>(n);
    }
    if (!f.in) return -1;
    if (f.lbn_bound) {
        const std::uint64_t pos = f.in->tell();
        const std::uint64_t left = pos < f.lbn_start + f.lbn_size ? f.lbn_start + f.lbn_size - pos : 0;
        size = static_cast<std::uint32_t>(std::min<std::uint64_t>(size, left));
    }
    return f.in ? static_cast<std::int64_t>(f.in->read(mem().ptr(buffer), size)) : 0;
}

std::int64_t do_seek(int fd, std::int64_t offset, int whence) {
    const auto it = g_files.find(fd);
    if (it == g_files.end()) return static_cast<std::int32_t>(kErrorBadFile);
    OpenFile& f = *it->second;
    if (f.block) {
        f.sector = whence == 0 ? static_cast<std::uint64_t>(offset) : f.sector + offset;
        return static_cast<std::int64_t>(f.sector);
    }
    if (f.memory || f.writable) {
        const std::size_t size = f.writable ? f.contents.size() : f.memory->size();
        const std::int64_t base = whence == 0 ? 0 : whence == 1 ? static_cast<std::int64_t>(f.memory_pos)
                                                                : static_cast<std::int64_t>(size);
        f.memory_pos = static_cast<std::uint64_t>(std::max<std::int64_t>(0, base + offset));
        return static_cast<std::int64_t>(f.memory_pos);
    }
    if (!f.in) return -1;
    if (f.lbn_bound) {
        // Seeks stay inside the opened range; positions are range-relative, like the PSP reports them.
        const std::int64_t base = whence == 0 ? static_cast<std::int64_t>(f.lbn_start)
                                : whence == 1 ? static_cast<std::int64_t>(f.in->tell())
                                              : static_cast<std::int64_t>(f.lbn_start + f.lbn_size);
        const std::int64_t pos = std::clamp(base + offset, static_cast<std::int64_t>(f.lbn_start),
                                            static_cast<std::int64_t>(f.lbn_start + f.lbn_size));
        f.in->seek(pos, nx::fs::seek_origin::begin);
        return pos - static_cast<std::int64_t>(f.lbn_start);
    }
    f.in->seek(offset, whence == 0 ? nx::fs::seek_origin::begin
                       : whence == 1 ? nx::fs::seek_origin::current
                                     : nx::fs::seek_origin::end);
    return f.in->tell();
}

int do_open(const nx::string& path, std::uint32_t flags) {
    OpenFile f;
    f.path = path;
    if (is_block_device(path)) {
        f.block = true;
    } else if (const auto lbn = parse_lbn(disc_relative(path))) {
        // disc0:/sce_lbn<LBN>_size<SIZE>: a sector range served from the extracted file with the exact
        // same start (a directory extent like SYSDIR/EBOOT.BIN aliases many files; only the exact file's
        // bytes match what the UMD would return).
        const DiscFile* cover = nullptr;
        for (const auto& e : g_disc_files)
            if (lbn->first == e.lbn && lbn->second <= e.size)
                cover = &e;
        if (!cover)
            for (const auto& e : g_disc_files)
                if (lbn->first >= e.lbn && lbn->first + lbn->second <= e.lbn + e.size &&
                    (!cover || e.lbn > cover->lbn))
                    cover = &e;
        if (!cover) {
            std::fprintf(stderr, "sce_lbn: no file covers LBN 0x%x size 0x%x\n", lbn->first, lbn->second);
            return static_cast<int>(kErrorNotFound);
        }
        const nx::string host = path_join(g_game_dir, cover->path);
        auto in = nx::fs::open_file(host);
        if (!in.has_value()) return static_cast<int>(kErrorNotFound);
        const std::uint64_t start = static_cast<std::uint64_t>(lbn->first - cover->lbn) * kSector;
        if (!in.value()->seek(static_cast<int64_t>(start))) return static_cast<int>(kErrorNotFound);
        f.in = std::move(in.value());
        f.lbn_bound = true;
        f.lbn_start = start;
        f.lbn_size = lbn->second;
    } else {
        const nx::string host = host_path(path);
        if (host.empty()) return static_cast<int>(kErrorNotFound);
        if (const nx::vector<char>* replaced = g_overlay ? g_overlay->file(disc_relative(path)) : nullptr) {
            f.memory = replaced;
            const int fd = g_next_fd++;
            g_files.emplace(fd, nx::make_unique<OpenFile>(std::move(f)));
            return fd;
        }
        const bool write = flags & 0x0002, create = flags & 0x0200, truncate = flags & 0x0400;  // PSP_O_WRONLY/CREAT/TRUNC
        if (create && !nx::fs::exists(host)) {
            (void)nx::fs::create_directories(parent_path(host));
            if (!nx::fs::file_write(host, {}).has_value()) return static_cast<int>(kErrorNotFound);
        }
        if (write) {
            f.writable = true;
            f.host = host;
            if (truncate) {
                if (!nx::fs::file_write(host, {}).has_value()) return static_cast<int>(kErrorNotFound);
            } else {
                const auto data = nx::fs::file_read(host);
                if (!data.has_value()) return static_cast<int>(kErrorNotFound);
                const auto* bytes = reinterpret_cast<const char*>(data.value().data());
                f.contents.assign(bytes, bytes + data.value().size());
            }
        } else {
            auto in = nx::fs::open_file(host);
            if (!in.has_value()) return static_cast<int>(kErrorNotFound);
            f.in = std::move(in.value());
        }
    }
    const int fd = g_next_fd++;
    g_files.emplace(fd, nx::make_unique<OpenFile>(std::move(f)));
    return fd;
}

void write64(std::uint32_t addr, std::int64_t v) {
    mem().write32(addr, static_cast<std::uint32_t>(v));
    mem().write32(addr + 4, static_cast<std::uint32_t>(static_cast<std::uint64_t>(v) >> 32));
}

// SceIoStat: mode, attr, size (u64), 3 × ScePspDateTime (16 bytes), private[6].
void write_stat(std::uint32_t addr, nx::fs::path_view host, std::uint32_t lbn) {
    mem().fill(addr, 0, 88);
    const nx::fs::file_stat st = nx::fs::stat_file(host);
    const bool dir = st.type == nx::fs::file_type::directory;
    const std::uint64_t size = dir ? 0 : st.size;
    mem().write32(addr, dir ? 0x11FF : 0x21FF);
    mem().write32(addr + 4, dir ? 0x10 : 0x20);
    write64(addr + 8, static_cast<std::int64_t>(size));
    mem().write32(addr + 64, lbn);
}

// ---- IoFileMgrForUser --------------------------------------------------------------------------

void Open(Context& c) {
    const nx::string path = mem().string(arg(c, 0));
    const int fd = do_open(path, arg(c, 1));
    if (kernel().trace_hle) std::fprintf(stderr, "  sceIoOpen(%s) -> %d\n", path.c_str(), fd);
    ret(c, static_cast<std::uint32_t>(fd));
}
void OpenAsync(Context& c) {
    const int fd = do_open(mem().string(arg(c, 0)), arg(c, 1));
    if (fd > 0) g_files[fd]->async_result = fd;
    ret(c, static_cast<std::uint32_t>(fd));
}
void Close(Context& c) { ret(c, g_files.erase(static_cast<int>(arg(c, 0))) ? 0 : kErrorBadFile); }
void CloseAsync(Context& c) {
    const int fd = static_cast<int>(arg(c, 0));
    if (!g_files.contains(fd)) return ret(c, kErrorBadFile);
    g_files[fd]->async_result = 0;
    g_files[fd]->async_pending = true;  // closed after the result is collected
    ret(c, 0);
}
void Read(Context& c) {
    mark_guest_written(arg(c, 1), arg(c, 2));
    ret(c, static_cast<std::uint32_t>(do_read(static_cast<int>(arg(c, 0)), arg(c, 1), arg(c, 2))));
}
void ReadAsync(Context& c) {
    mark_guest_written(arg(c, 1), arg(c, 2));
    const int fd = static_cast<int>(arg(c, 0));
    if (!g_files.contains(fd)) return ret(c, kErrorBadFile);
    g_files[fd]->async_result = do_read(fd, arg(c, 1), arg(c, 2));
    ret(c, 0);
}
// 64-bit offset arrives in a2/a3 (aligned register pair); whence in t0.
void Lseek(Context& c) {
    const std::int64_t off = static_cast<std::int64_t>(static_cast<std::uint64_t>(arg(c, 2)) | (static_cast<std::uint64_t>(arg(c, 3)) << 32));
    ret64(c, static_cast<std::uint64_t>(do_seek(static_cast<int>(arg(c, 0)), off, static_cast<int>(arg(c, 4)))));
}
void LseekAsync(Context& c) {
    const int fd = static_cast<int>(arg(c, 0));
    if (!g_files.contains(fd)) return ret(c, kErrorBadFile);
    const std::int64_t off = static_cast<std::int64_t>(static_cast<std::uint64_t>(arg(c, 2)) | (static_cast<std::uint64_t>(arg(c, 3)) << 32));
    g_files[fd]->async_result = do_seek(fd, off, static_cast<int>(arg(c, 4)));
    ret(c, 0);
}
void WaitAsync(Context& c) {
    const int fd = static_cast<int>(arg(c, 0));
    const auto it = g_files.find(fd);
    if (it == g_files.end()) return ret(c, kErrorBadFile);
    if (arg(c, 1)) write64(arg(c, 1), it->second->async_result);
    if (it->second->async_pending) g_files.erase(it);  // pending close
    ret(c, 0);
}
// Our async operations complete inline, so polling always finds the result ready.
void PollAsync(Context& c) {
    const int fd = static_cast<int>(arg(c, 0));
    const auto it = g_files.find(fd);
    if (it == g_files.end()) return ret(c, kErrorBadFile);
    if (arg(c, 1)) write64(arg(c, 1), it->second->async_result);
    ret(c, 0);
}
void Write(Context& c) {
    const int fd = static_cast<int>(arg(c, 0));
    const std::uint32_t size = arg(c, 2);
    if (fd == 1 || fd == 2) {
        std::fwrite(mem().ptr(arg(c, 1)), 1, size, stderr);
        return ret(c, size);
    }
    const auto it = g_files.find(fd);
    if (it == g_files.end()) return ret(c, kErrorBadFile);
    OpenFile& f = *it->second;
    if (f.writable) {
        const auto* src = reinterpret_cast<const char*>(mem().ptr(arg(c, 1)));
        if (f.contents.size() < f.memory_pos + size) f.contents.resize(f.memory_pos + size);
        std::memcpy(f.contents.data() + f.memory_pos, src, size);
        f.memory_pos += size;
        (void)nx::fs::file_write(f.host, std::span(reinterpret_cast<const std::uint8_t*>(f.contents.data()), f.contents.size()));
    }
    ret(c, size);
}
void Getstat(Context& c) {
    const nx::string path = mem().string(arg(c, 0));
    if (const auto lbn = parse_lbn(disc_relative(path))) {
        // The range's size is known without touching the disc; report it like a regular file at its LBN.
        mem().fill(arg(c, 1), 0, 88);
        mem().write32(arg(c, 1), 0x21FF);
        mem().write32(arg(c, 1) + 4, 0x20);
        write64(arg(c, 1) + 8, static_cast<std::int64_t>(lbn->second));
        mem().write32(arg(c, 1) + 64, lbn->first);
        return ret(c, 0);
    }
    const nx::string host = host_path(path);
    if (host.empty() || !nx::fs::exists(host)) return ret(c, kErrorNotFound);
    write_stat(arg(c, 1), host, !g_mapped_path.empty() && disc_relative(path) == g_mapped_path ? g_mapped_lbn : 0);
    ret(c, 0);
}
void Mkdir(Context& c) {
    const nx::string host = host_path(mem().string(arg(c, 0)));
    if (!host.empty()) (void)nx::fs::create_directories(host);
    ret(c, 0);
}
void Rename(Context& c) {
    const bool ok = nx::fs::rename(host_path(mem().string(arg(c, 0))), host_path(mem().string(arg(c, 1)))).has_value();
    ret(c, ok ? 0 : kErrorNotFound);
}
void Dopen(Context& c) {
    const nx::string host = host_path(mem().string(arg(c, 0)));
    if (host.empty() || !is_directory(host)) return ret(c, kErrorNotFound);
    OpenDir d;
    if (auto list = nx::fs::directory_entries(host); list.has_value()) d.entries = std::move(list.value());
    // By name, the same on every platform (the host's listing order is not).
    std::sort(d.entries.begin(), d.entries.end(), [](const auto& a, const auto& b) { return a.path < b.path; });
    const int fd = g_next_fd++;
    g_dirs.emplace(fd, std::move(d));
    ret(c, static_cast<std::uint32_t>(fd));
}
// SceIoDirent: SceIoStat (88 bytes), name[256], private pointer, dummy.
void Dread(Context& c) {
    const auto it = g_dirs.find(static_cast<int>(arg(c, 0)));
    if (it == g_dirs.end()) return ret(c, kErrorBadFile);
    OpenDir& d = it->second;
    if (d.next >= d.entries.size()) return ret(c, 0);
    const auto& e = d.entries[d.next++];
    const std::uint32_t out = arg(c, 1);
    write_stat(out, e.path, 0);
    const nx::string name = file_name(e.path).substr(0, 255);
    mem().write_string(out + 88, name);
    ret(c, 1);
}
void Dclose(Context& c) { ret(c, g_dirs.erase(static_cast<int>(arg(c, 0))) ? 0 : kErrorBadFile); }
void Devctl(Context& c) {
    if (kernel().trace_hle)
        std::fprintf(stderr, "  sceIoDevctl(%s, %08x)\n", mem().string(arg(c, 0)).c_str(), arg(c, 1));
    ret(c, 0);
}

// ---- sceUmdUser --------------------------------------------------------------------------------

void ReturnZero(Context& c) { ret(c, 0); }
void ReturnOne(Context& c) { ret(c, 1); }
constexpr std::uint32_t kUmdReady = 0x32;  // present | inited | ready
int g_umd_callback = 0;
void UmdGetDriveStat(Context& c) { ret(c, kUmdReady); }
// The game learns that the disc is readable from its UMD callback (it gates movie streaming on it).
void UmdRegisterCallback(Context& c) {
    g_umd_callback = static_cast<int>(arg(c, 0));
    kernel().notify_callback(g_umd_callback, kUmdReady);
    ret(c, 0);
}
void UmdUnRegisterCallback(Context& c) {
    g_umd_callback = 0;
    ret(c, 0);
}
void UmdActivate(Context& c) {
    if (g_umd_callback) kernel().notify_callback(g_umd_callback, kUmdReady);
    ret(c, 0);
}

constexpr HleBinding kBindings[] = {
    {"sceIoOpen", Open},           {"sceIoOpenAsync", OpenAsync},     {"sceIoClose", Close},
    {"sceIoCloseAsync", CloseAsync}, {"sceIoRead", Read},             {"sceIoReadAsync", ReadAsync},
    {"sceIoLseek", Lseek},         {"sceIoLseekAsync", LseekAsync},   {"sceIoWaitAsync", WaitAsync},
    {"sceIoPollAsync", PollAsync},
    {"sceIoWaitAsyncCB", WaitAsync}, {"sceIoWrite", Write},           {"sceIoGetstat", Getstat},
    {"sceIoMkdir", Mkdir},         {"sceIoRename", Rename},           {"sceIoDopen", Dopen},
    {"sceIoDread", Dread},         {"sceIoDclose", Dclose},           {"sceIoDevctl", Devctl},
    {"sceIoIoctl", ReturnZero},
    {"sceUmdCheckMedium", ReturnOne}, {"sceUmdActivate", UmdActivate}, {"sceUmdGetDriveStat", UmdGetDriveStat},
    {"sceUmdWaitDriveStatCB", ReturnZero}, {"sceUmdWaitDriveStatWithTimer", ReturnZero},
    {"sceUmdRegisterUMDCallBack", UmdRegisterCallback}, {"sceUmdUnRegisterUMDCallBack", UmdUnRegisterCallback},
    {"sceUmdGetErrorStat", ReturnZero},
};

}  // namespace

void set_storage(nx::string_view game_dir, nx::string_view save_dir) {
    g_game_dir = game_dir;
    g_save_dir = save_dir;
}

void map_disc_file(nx::string_view path, std::uint32_t lbn) {
    g_mapped_path = disc_relative(nx::format("disc0:/{}", path));
    g_mapped_lbn = lbn;
    const nx::string host = path_join(g_game_dir, path);
    auto opened = nx::fs::open_file(host);
    g_mapped = opened.has_value() ? std::move(opened.value()) : nx::fs::stream_ptr{};
    if (!g_mapped) std::fprintf(stderr, "warning: %s not found\n", host.c_str());
}

void read_disc_sectors(std::uint64_t first, std::uint64_t count, char* out) {
    std::memset(out, 0, static_cast<std::size_t>(count) * kSector);
    if (g_mapped && g_mapped->seek(static_cast<std::int64_t>(first * kSector)))
        (void)g_mapped->read(out, static_cast<std::size_t>(count) * kSector);  // short at the end of the file
}

void set_disc_overlay(std::unique_ptr<DiscOverlay> overlay) { g_overlay = std::move(overlay); }

void map_disc_files(nx::vector<DiscFile> files) { g_disc_files = std::move(files); }

const nx::string& game_dir() { return g_game_dir; }
const nx::string& save_dir() { return g_save_dir; }

void hle_install_io() { hle_bind(kBindings); }

}  // namespace psprecore::rt
