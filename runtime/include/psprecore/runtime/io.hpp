#pragma once

#include <cstdint>
#include <memory>

#include "core/foundation/core/foundation.h"
#include "core/foundation/platform/filesystem.h"

namespace psprecore::rt {

// Where disc0:/ (the extracted disc: folder containing PSP_GAME) and ms0:/ (saves) point on the host.
void set_storage(nx::string_view game_dir, nx::string_view save_dir);
const nx::string& game_dir();
const nx::string& save_dir();

// Raw disc reads: games that read an archive through the UMD block devices (umd0:/umd1:: absolute sector seeks,
// sector-count reads) instead of by name. `path` (relative to the disc folder) is the file starting at sector
// `lbn`; sceIoGetstat reports it there. Other sectors read as zeros. Call after set_storage().
void map_disc_file(nx::string_view path, std::uint32_t lbn);
// Sectors of the mapped file as they are on the disc, `first` relative to its start (for overlays).
void read_disc_sectors(std::uint64_t first, std::uint64_t count, char* out);

// Files placed on the disc for sce_lbn opens (disc0:/sce_lbn<LBN>_size<SIZE>): the game reads an index file for
// LBNs and opens sector ranges directly. Served from the extracted files: each entry names the disc-relative
// file containing LBN `lbn` (the file's own start sector) and its size. Call after set_storage().
struct DiscFile {
    std::uint32_t lbn = 0;  // the file's start sector on the disc
    std::uint32_t size = 0;
    nx::string path;  // disc-relative, '/' separators (matched case-insensitively)
};
void map_disc_files(nx::vector<DiscFile> files);

// What the game reads from the disc, changed (mods): the mapped file's sectors and whole files by name.
class DiscOverlay {
public:
    virtual ~DiscOverlay() = default;
    // Fills `count` sectors of the mapped file from `first` (relative to its start).
    virtual void read_sectors(std::uint64_t first, std::uint64_t count, char* out) = 0;
    // Contents served instead of the disc file `path` (relative to the disc folder, lower case, '/'), or null.
    virtual const nx::vector<char>* file(nx::string_view path) = 0;
};
void set_disc_overlay(std::unique_ptr<DiscOverlay> overlay);

}  // namespace psprecore::rt
