#include "platform/gpu_vulkan.hpp"

#include <vk_mem_alloc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <type_traits>

#include "psprecore/gpu/texture_collection.hpp"
#include "psprecore/gpu/texture_replacement.hpp"
#include "psprecore/formats/ktx2.hpp"
#include "psprecore/gpu/ui_font.hpp"
#include "psprecore/runtime/hle.hpp"
#include "psprecore/runtime/memory.hpp"
#include "copy_frag.h"
#include "copy_vert.h"
#include "ge_frag.h"
#include "ge_vert.h"
#include "gpu/hash.hpp"
#include "gpu/texture.hpp"
#include "gpu/vertex_pipeline.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/core/time.h"
#include "core/foundation/containers/hash_map.h"
#include "core/foundation/containers/hash_set.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/strings/format.h"

namespace psprecore::platform {

namespace {

namespace ge = rt::ge;
using rt::gpu::ClipVertex;
using rt::gpu::hash_bytes;

constexpr int kFrames = 2;
constexpr VkDeviceSize kVertexBytes = 16u << 20;
constexpr VkDeviceSize kStagingBytes = 32u << 20;
constexpr int kTargetHeight = 512;  // render targets cover the GE's addressable 512 lines
constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(nx::format("Vulkan GE renderer: {} failed ({})", what, static_cast<int>(r)).c_str());
}


struct GpuVertex {
    float pos[4];
    float uvq[3];
    std::uint32_t color;
    std::uint32_t spec_fog;
};
static_assert(sizeof(GpuVertex) == 36);

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    std::uint8_t* ptr = nullptr;
};

struct Image {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation memory = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    int width = 0, height = 0;
    std::uint32_t mips = 1;
    bool replacement = false;  // a replacement picture: sampled with linear minification and its mipmaps
};

struct Target {
    std::uint32_t address = 0, stride = 0;
    int format = 3;
    int used_height = 272;  // lines the game has drawn to (scissor extent); textures past it are not ours
    // Filling the screen: guest pixels added on each side (guest x, y map to host x + margin_x, y + margin_y),
    // and whether 3D was drawn since the target was last shown (else it is shown without the margins).
    int margin_x = 0, margin_y = 0;
    bool filled = false;
    Image color;  // single-sample: what textures, the display and readbacks use
    // With MSAA: drawing goes to the multisampled image, resolved into `color` when a rendering pass ends. After
    // the CPU writes into `color` (uploads), the samples are refreshed from it before the next drawing.
    Image msaa;
    bool msaa_stale = false;
};

struct Texture {
    Image image;
    std::uint64_t content = 0;
    std::uint64_t last_frame = 0;
    std::uint64_t checked = 0;  // rt::guest_write_counter() when the data was last verified
    std::uint64_t sample = 0;   // hash of a sample of the texels at the last full check
    std::uint8_t quick_checks = 0;  // checks by sample only since the last full one
    std::uint64_t uploaded_frame = 0;  // frame_number when the current content was uploaded
    bool collected = false;            // handed to texture collection (--tex-collect)
    std::uint8_t replacing = 0;        // a replacement picture still loading: kReplacementLoading, kIndexMapLoading
    std::uint64_t texels = 0;          // texture_texels_hash, while an index map loads
};

// Everything that selects a VkPipeline. All fields are bytes so the struct hashes and compares as memory.
struct PipelineKey {
    std::uint8_t topology = 0;
    std::uint8_t blend = 0, src = 0, dst = 0, op = 0;
    std::uint8_t write_mask = 0xF;
    std::uint8_t depth_test = 0, depth_func = 0, depth_write = 0;
    std::uint8_t stencil = 0, stencil_func = 0, stencil_fail = 0, stencil_zfail = 0, stencil_pass = 0;
    std::uint8_t logic = 0, logic_op = 0;
    // Fragment specialization constants (shaders/ge.slang).
    std::uint8_t texture = 0, tex_func = 0, tex_alpha = 0, color_double = 0, fog = 0, alpha_test = 1, color_test = 1,
                 clear = 0, src_factor = 0, double_alpha = 0;
    std::uint8_t samples = 1;  // VkSampleCountFlagBits
    bool operator==(const PipelineKey& o) const { return std::memcmp(this, &o, sizeof *this) == 0; }
};
static_assert(std::is_trivially_copyable_v<PipelineKey>);
constexpr char kPipelineKeysMagic[8] = {'F', 'F', '7', 'P', 'K', 'E', 'Y', '1'};
struct PipelineKeyHash {
    std::size_t operator()(const PipelineKey& k) const { return static_cast<std::size_t>(hash_bytes(&k, sizeof k)); }
};

struct FragmentConstants {
    float env[4];
    float fog[4];
    float fixed_a[4];
    std::uint32_t alpha_ref, alpha_mask, color_ref, color_mask;
};
static_assert(sizeof(FragmentConstants) == 64);

struct Frame {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer upload = VK_NULL_HANDLE, draw = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    Buffer vertices, staging;
    VkDeviceSize vertex_used = 0, staging_used = 0;
    nx::vector<Image> garbage;
    nx::map<std::pair<VkImageView, VkSampler>, VkDescriptorSet> sets;
};

VkCompareOp compare_op(std::uint32_t f) {
    static constexpr VkCompareOp kOps[8] = {VK_COMPARE_OP_NEVER, VK_COMPARE_OP_ALWAYS, VK_COMPARE_OP_EQUAL,
                                            VK_COMPARE_OP_NOT_EQUAL, VK_COMPARE_OP_LESS, VK_COMPARE_OP_LESS_OR_EQUAL,
                                            VK_COMPARE_OP_GREATER, VK_COMPARE_OP_GREATER_OR_EQUAL};
    return kOps[f & 7];
}
VkStencilOp stencil_op(std::uint32_t op) {
    static constexpr VkStencilOp kOps[8] = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_ZERO, VK_STENCIL_OP_REPLACE,
                                            VK_STENCIL_OP_INVERT, VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                                            VK_STENCIL_OP_DECREMENT_AND_CLAMP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP};
    return kOps[op & 7];
}

std::uint32_t pack_color(const float c[4]) {
    std::uint32_t out = 0;
    for (int i = 0; i < 4; ++i)
        out |= static_cast<std::uint32_t>(std::clamp(static_cast<int>(c[i] + 0.5f), 0, 255)) << (i * 8);
    return out;
}

}  // namespace

struct VulkanGeRenderer::Impl {
    VulkanContext vk;
    rt::GuestMemory& memory;
    int scale;
    int pending_scale;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT, pending_samples = VK_SAMPLE_COUNT_1_BIT;
    VkShaderModule copy_vs = VK_NULL_HANDLE, copy_fs = VK_NULL_HANDLE;
    std::array<VkPipeline, 4> copy_pipelines{};  // by log2(samples) - 1
    bool guest_copies = true;
    bool skip_draws = false;
    int fill_x = 0, fill_y = 0;                   // set_fill_aspect: margins for display framebuffers
    float fill_aspect = 0;                        // and the aspect ratio shown (the margins may be a little larger)

    // Frame interpolation (set_interpolation): the draws into displayed framebuffers, kept for the last two frames
    // shown, are drawn again into `replay` with their matrices blended (render_between).
    struct Record {
        std::uint32_t target = 0;  // framebuffer address
        std::uint64_t key = 0;     // what identifies the draw from frame to frame (draw_key)
        ge::Prim prim{};
        ge::State state;
        nx::vector<ge::Vertex> vertices;
    };
    bool interpolation = false, replaying = false;
    nx::vector<Record> recorded, current_draws, previous_draws;
    Target replay;       // frames drawn between two of the game's, with their own depth buffer
    Image replay_depth;
    nx::hash_set<std::uint32_t> display_targets;  // framebuffers that have been displayed

    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkShaderModule vert = VK_NULL_HANDLE, frag = VK_NULL_HANDLE;
    nx::hash_map<PipelineKey, VkPipeline, PipelineKeyHash> pipelines;
    nx::string cache_path;
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::uint64_t cache_dirty_frame = 0;  // frame of the last pipeline created since the cache was saved (0: saved)
    nx::map<std::uint32_t, VkSampler> samplers;
    std::array<Frame, kFrames> frames;
    int current = 0;
    bool recording = false;
    std::uint64_t frame_number = 1;
    VulkanGeRenderer::Stats stats;

    // Targets and depth buffers stay where they are (bound_target/bound_depth and returned references point at them;
    // nx::map moves its elements on insert).
    nx::map<std::uint32_t, nx::unique_ptr<Target>> targets;  // by VRAM address
    nx::map<std::uint64_t, nx::unique_ptr<Image>> depths;    // by (address, width)
    nx::hash_map<std::uint64_t, Texture> textures;
    rt::gpu::IconStyle icon_style = rt::gpu::IconStyle::PlayStation;
    Image white;  // bound when texturing is off (the descriptor is always declared)
    Buffer readback;
    VkDeviceSize readback_size = 0;

    // Active rendering.
    Target* bound_target = nullptr;
    Image* bound_depth = nullptr;
    bool rendering = false;
    VkPipeline bound_pipeline = VK_NULL_HANDLE;

    rt::gpu::VertexPipeline vp;
    nx::vector<ClipVertex> clip;
    nx::vector<GpuVertex> out;

    Impl(const VulkanContext& context, rt::GuestMemory& mem, int s, nx::string cache_file)
        : vk(context), memory(mem), scale(std::max(1, s)), pending_scale(scale), cache_path(std::move(cache_file)) {}

    Frame& frame() { return frames[static_cast<std::size_t>(current)]; }

    // ---- resources -------------------------------------------------------------------------------------

    Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool readback = false) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = (readback ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT) |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        Buffer b;
        VmaAllocationInfo info;
        check(vmaCreateBuffer(vk.allocator, &bci, &aci, &b.buffer, &b.memory, &info), "buffer");
        b.ptr = static_cast<std::uint8_t*>(info.pMappedData);
        return b;
    }

    Image make_image(int w, int h, VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, std::uint32_t mips = 1,
                     VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT) {
        Image img;
        img.width = w;
        img.height = h;
        img.aspect = aspect;
        img.mips = mips;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = format;
        ici.extent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
        ici.mipLevels = mips;
        ici.arrayLayers = 1;
        ici.samples = sample_count;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(vk.allocator, &ici, &aci, &img.image, &img.memory, nullptr), "image");
        if (!(usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
            return img;  // transfer-only staging image: no view
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = img.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = format;
        vci.subresourceRange = {aspect, 0, mips, 0, 1};
        check(vkCreateImageView(vk.device, &vci, nullptr, &img.view), "image view");
        return img;
    }

    void destroy(Image& img) {
        if (img.view) vkDestroyImageView(vk.device, img.view, nullptr);
        if (img.image) vmaDestroyImage(vk.allocator, img.image, img.memory);
        img = Image{};
    }

    // Conservative full barrier: correct for every transition this renderer makes.
    void transition(VkCommandBuffer cmd, Image& img, VkImageLayout to) {
        if (img.layout == to) return;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = img.layout;
        b.newLayout = to;
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img.image;
        b.subresourceRange = {img.aspect, 0, img.mips, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
        img.layout = to;
    }

    // Reserves staging memory for this frame; null when the frame's staging buffer is full.
    std::uint8_t* staging(VkDeviceSize size, VkDeviceSize& offset) {
        Frame& f = frame();
        const VkDeviceSize aligned = (f.staging_used + 15) & ~VkDeviceSize{15};
        if (aligned + size > kStagingBytes) return nullptr;
        offset = aligned;
        f.staging_used = aligned + size;
        return f.staging.ptr + aligned;
    }

    void init() {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.bindingCount = 1;
        dl.pBindings = &binding;
        check(vkCreateDescriptorSetLayout(vk.device, &dl, nullptr, &set_layout), "descriptor set layout");
        VkPushConstantRange push{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(FragmentConstants)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &set_layout;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &push;
        check(vkCreatePipelineLayout(vk.device, &pl, nullptr, &layout), "pipeline layout");

        const auto module = [&](const unsigned char* code, std::size_t size) {
            VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            ci.codeSize = size;
            ci.pCode = reinterpret_cast<const std::uint32_t*>(code);
            VkShaderModule m;
            check(vkCreateShaderModule(vk.device, &ci, nullptr, &m), "shader module");
            return m;
        };
        vert = module(ge_vert, sizeof ge_vert);
        frag = module(ge_frag, sizeof ge_frag);
        copy_vs = module(copy_vert, sizeof copy_vert);
        copy_fs = module(copy_frag, sizeof copy_frag);
        load_pipeline_cache();

        for (Frame& f : frames) {
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.queueFamilyIndex = vk.queue_family;
            check(vkCreateCommandPool(vk.device, &pci, nullptr, &f.pool), "command pool");
            VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cai.commandPool = f.pool;
            cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cai.commandBufferCount = 2;
            VkCommandBuffer cmds[2];
            check(vkAllocateCommandBuffers(vk.device, &cai, cmds), "command buffers");
            f.upload = cmds[0];
            f.draw = cmds[1];
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            check(vkCreateFence(vk.device, &fci, nullptr, &f.fence), "fence");
            const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8192};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            dpi.maxSets = 8192;
            dpi.poolSizeCount = 1;
            dpi.pPoolSizes = &size;
            check(vkCreateDescriptorPool(vk.device, &dpi, nullptr, &f.descriptors), "descriptor pool");
            f.vertices = make_buffer(kVertexBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            f.staging = make_buffer(kStagingBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }

        begin_frame();
        white = make_image(1, 1, kColorFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        transition(frame().draw, white, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const VkClearColorValue one{{1, 1, 1, 1}};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(frame().draw, white.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &one, 1, &range);
        transition(frame().draw, white, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    static nx::vector<char> read_file(nx::fs::path_view path) {
        auto data = nx::fs::file_read(path);
        if (!data.has_value()) return {};
        const auto& bytes = data.value();
        return nx::vector<char>(reinterpret_cast<const char*>(bytes.data()), reinterpret_cast<const char*>(bytes.data()) + bytes.size());
    }
    static void write_file(nx::fs::path_view path, const void* data, std::size_t size) {
        (void)nx::fs::file_write_atomic(path, std::span(static_cast<const std::uint8_t*>(data), size));
    }
    nx::string keys_path() const { return cache_path + ".keys"; }

    // Pipelines are created when a draw first needs them, which takes milliseconds each and stalls the game (and
    // its audio) when an effect brings several at once. So the driver's pipeline cache is kept on disk (the driver
    // checks its header and ignores data from another GPU or driver), and every pipeline used before is created
    // again at start from the saved key list.
    void load_pipeline_cache() {
        if (cache_path.empty()) {
            VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
            check(vkCreatePipelineCache(vk.device, &ci, nullptr, &cache), "pipeline cache");
            return;
        }
        const nx::vector<char> data = read_file(cache_path);
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = data.size();
        ci.pInitialData = data.empty() ? nullptr : data.data();
        if (vkCreatePipelineCache(vk.device, &ci, nullptr, &cache) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            check(vkCreatePipelineCache(vk.device, &ci, nullptr, &cache), "pipeline cache");
        }
        const nx::vector<char> keys = read_file(keys_path());
        constexpr std::size_t header = sizeof kPipelineKeysMagic + 4;
        std::uint32_t key_size = 0;
        if (keys.size() >= header) std::memcpy(&key_size, keys.data() + sizeof kPipelineKeysMagic, 4);
        if (keys.size() < header || std::memcmp(keys.data(), kPipelineKeysMagic, sizeof kPipelineKeysMagic) != 0 ||
            key_size != sizeof(PipelineKey))
            return;
        const auto start = nx::timestamp::now();
        std::size_t count = 0;
        for (std::size_t at = header; at + sizeof(PipelineKey) <= keys.size(); at += sizeof(PipelineKey), ++count) {
            PipelineKey k;
            std::memcpy(&k, keys.data() + at, sizeof k);
            pipeline(k);
        }
        cache_dirty_frame = 0;
        stats = {};
        std::fprintf(stderr, "Vulkan: %zu pipelines prepared in %.0f ms\n", count,
                     (nx::timestamp::now() - start).millis_f());
    }

    void save_pipeline_cache() {
        cache_dirty_frame = 0;
        if (cache_path.empty()) return;
        std::size_t size = 0;
        if (vkGetPipelineCacheData(vk.device, cache, &size, nullptr) != VK_SUCCESS || !size) return;
        nx::vector<char> data(size);
        if (vkGetPipelineCacheData(vk.device, cache, &size, data.data()) != VK_SUCCESS) return;
        write_file(cache_path, data.data(), size);
        nx::vector<char> keys(kPipelineKeysMagic, kPipelineKeysMagic + sizeof kPipelineKeysMagic);
        const std::uint32_t key_size = sizeof(PipelineKey);
        keys.insert(keys.end(), reinterpret_cast<const char*>(&key_size), reinterpret_cast<const char*>(&key_size) + 4);
        for (const auto& [k, _] : pipelines)
            keys.insert(keys.end(), reinterpret_cast<const char*>(&k), reinterpret_cast<const char*>(&k) + sizeof k);
        write_file(keys_path(), keys.data(), keys.size());
    }

    void shutdown() {
        vkDeviceWaitIdle(vk.device);
        if (cache_dirty_frame) save_pipeline_cache();
        vkDestroyPipelineCache(vk.device, cache, nullptr);
        for (Frame& f : frames) {
            for (Image& img : f.garbage) destroy(img);
            vmaDestroyBuffer(vk.allocator, f.vertices.buffer, f.vertices.memory);
            vmaDestroyBuffer(vk.allocator, f.staging.buffer, f.staging.memory);
            vkDestroyDescriptorPool(vk.device, f.descriptors, nullptr);
            vkDestroyFence(vk.device, f.fence, nullptr);
            vkDestroyCommandPool(vk.device, f.pool, nullptr);
        }
        for (auto& [_, t] : targets) {
                destroy(t->color);
                destroy(t->msaa);
            }
        for (auto& [_, d] : depths) destroy(*d);
        destroy(replay.color);
        destroy(replay.msaa);
        destroy(replay_depth);
        for (auto& [_, t] : textures) destroy(t.image);
        destroy(white);
        if (readback.buffer) vmaDestroyBuffer(vk.allocator, readback.buffer, readback.memory);
        for (auto& [_, p] : pipelines) vkDestroyPipeline(vk.device, p, nullptr);
        for (auto& [_, s] : samplers) vkDestroySampler(vk.device, s, nullptr);
        for (VkPipeline p : copy_pipelines)
            if (p) vkDestroyPipeline(vk.device, p, nullptr);
        vkDestroyShaderModule(vk.device, vert, nullptr);
        vkDestroyShaderModule(vk.device, frag, nullptr);
        vkDestroyShaderModule(vk.device, copy_vs, nullptr);
        vkDestroyShaderModule(vk.device, copy_fs, nullptr);
        vkDestroyPipelineLayout(vk.device, layout, nullptr);
        vkDestroyDescriptorSetLayout(vk.device, set_layout, nullptr);
    }

    // ---- frames ----------------------------------------------------------------------------------------

    void begin_frame() {
        if (recording) return;
        if (pending_scale != scale || pending_samples != samples) {
            // Render targets are rebuilt at the new size and re-seeded from guest VRAM on first use.
            vkDeviceWaitIdle(vk.device);
            for (auto& [_, t] : targets) {
                destroy(t->color);
                destroy(t->msaa);
            }
            for (auto& [_, d] : depths) destroy(*d);
            targets.clear();
            depths.clear();
            destroy(replay.color);
            destroy(replay.msaa);
            destroy(replay_depth);
            replay = Target{};
            scale = pending_scale;
            samples = pending_samples;
        }
        Frame& f = frame();
        vkWaitForFences(vk.device, 1, &f.fence, VK_TRUE, UINT64_MAX);
        for (Image& img : f.garbage) destroy(img);
        f.garbage.clear();
        f.sets.clear();
        vkResetDescriptorPool(vk.device, f.descriptors, 0);
        vkResetCommandPool(vk.device, f.pool, 0);
        f.vertex_used = f.staging_used = 0;
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(f.upload, &bi);
        vkBeginCommandBuffer(f.draw, &bi);
        recording = true;
        bound_pipeline = VK_NULL_HANDLE;
    }

    void submit_frame() {
        Frame& f = frame();
        end_rendering();
        vkEndCommandBuffer(f.upload);
        vkEndCommandBuffer(f.draw);
        // VMA_AUTO may select non-coherent memory, notably on Android. Make CPU uploads visible to the GPU.
        check(vmaFlushAllocation(vk.allocator, f.vertices.memory, 0, VK_WHOLE_SIZE), "flush vertex buffer");
        check(vmaFlushAllocation(vk.allocator, f.staging.memory, 0, VK_WHOLE_SIZE), "flush staging buffer");
        vkResetFences(vk.device, 1, &f.fence);
        const VkCommandBuffer cmds[2] = {f.upload, f.draw};
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 2;
        si.pCommandBuffers = cmds;
        check(vkQueueSubmit(vk.queue, 1, &si, f.fence), "vkQueueSubmit");
        recording = false;
        current = (current + 1) % kFrames;
        ++frame_number;
        evict_textures();
    }

    void evict_textures() {
        for (auto it = textures.begin(); it != textures.end();) {
            if (frame_number - it->second.last_frame > 240) {
                frame().garbage.push_back(it->second.image);  // destroyed once this frame slot is reused
                it = textures.erase(it);
            } else {
                ++it;
            }
        }
    }

    // ---- render targets ----------------------------------------------------------------------------------

    static int bytes_per_pixel(int format) { return format == 3 ? 4 : 2; }

    // Uploads guest VRAM pixels (x, y, w, h in PSP pixels) into a render target, scaled.
    void upload_region(Target& t, int x, int y, int w, int h) {
        x = std::clamp(x, 0, static_cast<int>(t.stride));
        y = std::clamp(y, 0, kTargetHeight);
        w = std::min(w, static_cast<int>(t.stride) - x);
        h = std::min(h, kTargetHeight - y);
        if (w <= 0 || h <= 0) return;
        VkDeviceSize offset = 0;
        auto* dst = reinterpret_cast<std::uint32_t*>(staging(static_cast<VkDeviceSize>(w) * h * 4, offset));
        if (!dst) return;
        const int bpp = bytes_per_pixel(t.format);
        for (int row = 0; row < h; ++row) {
            const std::uint8_t* src = memory.ptr(t.address + static_cast<std::uint32_t>(((y + row) * static_cast<int>(t.stride) + x) * bpp));
            for (int col = 0; col < w; ++col) {
                std::uint32_t p;
                if (bpp == 4) {
                    std::memcpy(&p, src + col * 4, 4);
                } else {
                    const std::uint32_t v = static_cast<std::uint32_t>(src[col * 2] | (src[col * 2 + 1] << 8));
                    p = t.format == 0 ? rt::gpu::decode_5650(v) : t.format == 1 ? rt::gpu::decode_5551(v) : rt::gpu::decode_4444(v);
                }
                dst[row * w + col] = p;
            }
        }
        end_rendering();
        const VkCommandBuffer cmd = frame().draw;
        Image tmp = make_image(w, h, kColorFormat, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT);
        transition(cmd, tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
        vkCmdCopyBufferToImage(cmd, frame().staging.buffer, tmp.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        transition(cmd, tmp, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        transition(cmd, t.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {w, h, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstOffsets[0] = {(x + t.margin_x) * scale, (y + t.margin_y) * scale, 0};
        blit.dstOffsets[1] = {(x + w + t.margin_x) * scale, (y + h + t.margin_y) * scale, 1};
        vkCmdBlitImage(cmd, tmp.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.color.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &blit, VK_FILTER_NEAREST);
        frame().garbage.push_back(tmp);
        t.msaa_stale = t.msaa.image != VK_NULL_HANDLE;
    }

    Target& target(std::uint32_t address, std::uint32_t stride, int format) {
        address = 0x04000000 | (address & 0x1FFFF0);
        stride = std::clamp<std::uint32_t>(stride, 64, 1024);
        // Off-screen passes can share guest depth with a displayed framebuffer. Identical host margins preserve
        // that sharing (and the depth texel mapping) instead of partitioning depth by presentation shape.
        const int margin_x = fill_x, margin_y = fill_y;
        auto it = targets.find(address);
        if (it != targets.end() && it->second->stride == stride && it->second->margin_x == margin_x &&
            it->second->margin_y == margin_y) {
            it->second->format = format;
            return *it->second;
        }
        if (it != targets.end()) {
            if (bound_target == it->second.get()) end_rendering();
            frame().garbage.push_back(it->second->color);
            if (it->second->msaa.image) frame().garbage.push_back(it->second->msaa);
            targets.erase(it);
        }
        Target& t = *(targets[address] = nx::make_unique<Target>());
        t.address = address;
        t.stride = stride;
        t.format = format;
        t.margin_x = margin_x;
        t.margin_y = margin_y;
        t.color = make_image((static_cast<int>(stride) + 2 * margin_x) * scale, (kTargetHeight + 2 * margin_y) * scale,
                             kColorFormat,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT);
        if (samples != VK_SAMPLE_COUNT_1_BIT)
            t.msaa = make_image(t.color.width, t.color.height, kColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT, 1, samples);
        // Start from what the CPU left in VRAM.
        upload_region(t, 0, 0, static_cast<int>(stride), 272);
        return t;
    }

    Image& depth(std::uint32_t address, const Target& t) {
        const std::uint64_t key = (std::uint64_t{0x04000000u | (address & 0x1FFFF0)} << 16) | t.stride |
                                  std::uint64_t(t.margin_x / 8) << 44 | std::uint64_t(t.margin_y / 8) << 54;
        auto it = depths.find(key);
        if (it != depths.end()) return *it->second;
        Image& d = *(depths[key] = nx::make_unique<Image>());
        const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        d = make_image(t.color.width, t.color.height, vk.depth_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT, aspect, 1, samples);
        const VkCommandBuffer cmd = frame().draw;
        end_rendering();
        transition(cmd, d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const VkClearDepthStencilValue zero{0.0f, 0};
        const VkImageSubresourceRange range{aspect, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, d.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        return d;
    }

    void begin_rendering(Target& t, Image& d) {
        if (rendering && bound_target == &t && bound_depth == &d) return;
        end_rendering();
        if (t.msaa_stale) refresh_msaa(t);
        const VkCommandBuffer cmd = frame().draw;
        transition(cmd, t.color, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        transition(cmd, d, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = t.color.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        if (t.msaa.image) {  // draw the samples, resolve into `color` when the pass ends
            transition(cmd, t.msaa, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            color.imageView = t.msaa.view;
            color.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            color.resolveImageView = t.color.view;
            color.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        VkRenderingAttachmentInfo ds{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ds.imageView = d.view;
        ds.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        ds.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ds.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {static_cast<std::uint32_t>(t.color.width), static_cast<std::uint32_t>(t.color.height)}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        ri.pDepthAttachment = &ds;
        ri.pStencilAttachment = &ds;
        vk.begin_rendering(cmd, &ri);
        const VkViewport viewport{0, 0, static_cast<float>(t.color.width), static_cast<float>(t.color.height), 0, 1};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        rendering = true;
        bound_target = &t;
        bound_depth = &d;
        bound_pipeline = VK_NULL_HANDLE;
    }

    void end_rendering() {
        if (!rendering) return;
        vk.end_rendering(frame().draw);
        rendering = false;
    }

    // Every sample of the multisampled image gets the texel of `color` (after the CPU wrote into `color`).
    void refresh_msaa(Target& t) {
        t.msaa_stale = false;
        const VkCommandBuffer cmd = frame().draw;
        transition(cmd, t.color, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        transition(cmd, t.msaa, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        const VkDescriptorSet set = descriptor(t.color.view, sampler(false, false, true, true, false, 1));
        if (!set) return;
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = t.msaa.view;
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        const VkExtent2D extent{static_cast<std::uint32_t>(t.msaa.width), static_cast<std::uint32_t>(t.msaa.height)};
        ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        vk.begin_rendering(cmd, &ri);
        const VkViewport viewport{0, 0, static_cast<float>(extent.width), static_cast<float>(extent.height), 0, 1};
        const VkRect2D scissor{{0, 0}, extent};
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, copy_pipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vk.end_rendering(cmd);
        bound_pipeline = VK_NULL_HANDLE;
    }

    VkPipeline copy_pipeline() {
        int index = 0;
        while ((1 << (index + 1)) < static_cast<int>(samples)) ++index;
        VkPipeline& p = copy_pipelines[static_cast<std::size_t>(index)];
        if (p) return p;
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, copy_vs, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, copy_fs, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp_state.viewportCount = 1;
        vp_state.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = samples;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState blend{};
        blend.colorWriteMask = 0xF;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;
        const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dynamic;
        VkPipelineRenderingCreateInfo rendering_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        const VkFormat color_format = kColorFormat;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachmentFormats = &color_format;
        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &rendering_info;
        pci.stageCount = 2;
        pci.pStages = stages;
        pci.pVertexInputState = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp_state;
        pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb;
        pci.pDynamicState = &dyn;
        pci.layout = layout;
        check(vkCreateGraphicsPipelines(vk.device, cache, 1, &pci, nullptr, &p), "copy pipeline");
        return p;
    }

    // ---- textures ----------------------------------------------------------------------------------------

    VkSampler sampler(bool min_linear, bool mag_linear, bool clamp_u, bool clamp_v, bool mipmap, std::uint32_t levels,
                      bool trilinear = false) {
        const std::uint32_t key = (min_linear ? 1u : 0u) | (mag_linear ? 2u : 0u) | (clamp_u ? 4u : 0u) | (clamp_v ? 8u : 0u) |
                                  (mipmap ? 16u : 0u) | (trilinear ? 32u : 0u) | (levels << 8);
        auto& s = samplers[key];
        if (s) return s;
        VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        ci.magFilter = mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.minFilter = min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        ci.mipmapMode = trilinear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (trilinear && vk.sampler_anisotropy) {  // keeps surfaces seen at a slant sharp
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(vk.gpu, &props);
            ci.anisotropyEnable = VK_TRUE;
            ci.maxAnisotropy = std::min(8.0f, props.limits.maxSamplerAnisotropy);
        }
        ci.addressModeU = clamp_u ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeV = clamp_v ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        ci.maxLod = mipmap ? static_cast<float>(levels - 1) : 0.0f;
        check(vkCreateSampler(vk.device, &ci, nullptr, &s), "sampler");
        return s;
    }

    // A texture still valid in the cache. Collection (--tex-collect) takes it once its content has lasted 30 frames,
    // which leaves out streamed textures (movie frames and the like, new every frame).
    Image* cached(Texture& t, const ge::State& s, const rt::gpu::TextureSampler& ts) {
        if (t.replacing) replace(t, ts);
        if (!t.collected && rt::gpu::collecting_textures() && frame_number - t.uploaded_frame >= 30) {
            t.collected = true;
            if (!rt::gpu::find_icon_atlas(ts.format(), ts.width(0), ts.height(0), ts.level_data(0), ts.level_size(0))) {
                nx::vector<std::uint32_t> rgba(static_cast<std::size_t>(ts.width(0)) * static_cast<std::size_t>(ts.height(0)));
                ts.decode(0, rgba.data());
                const std::uint32_t wrap = s.param(ge::TEXWRAP);
                rt::gpu::collect_texture(t.content, rgba.data(), ts.width(0), ts.height(0), ts.format(), s.format.through,
                                         (wrap & 1) == 0, (wrap & 0x100) == 0,
                                         {ts.level_data(0), static_cast<std::size_t>(ts.level_size(0))});
            }
        }
        return &t.image;
    }

    std::uint64_t sample_texels(const rt::gpu::TextureSampler& ts) {
        std::uint64_t h = 0;
        for (int l = 0; l < ts.levels(); ++l) {
            const std::uint8_t* data = ts.level_data(l);
            const std::size_t size = ts.level_size(l);
            if (size <= 4096) {
                h = hash_bytes(data, size, h);
                stats.hashed_bytes += size;
                continue;
            }
            for (std::size_t at = 0; at < size; at += 2048) h = hash_bytes(data + at, std::min<std::size_t>(64, size - at), h);
            stats.hashed_bytes += size / 32;
        }
        return h;
    }

    // A replacement picture in place of a texture (psprecore/gpu/texture_replacement.hpp), with its mipmaps; the
    // image t had (the game's texture while the picture loaded) goes. With `palette`, the picture is an index map
    // coloured with the palette in use.
    bool upload_replacement(Texture& t, const rt::gpu::ReplacementImage& r, const rt::gpu::TextureSampler* palette = nullptr) {
        using rt::gpu::Compression;
        VkDeviceSize offset = 0;
        const VkDeviceSize bytes = r.compression == Compression::None ? static_cast<VkDeviceSize>(r.rgba.size()) * 4
                                                                       : static_cast<VkDeviceSize>(r.blocks.size());
        auto* dst = staging(bytes, offset);
        if (!dst) return false;  // staging full this frame: the game's own texture for now
        if (r.compression != Compression::None) {
            std::memcpy(dst, r.blocks.data(), bytes);
        } else if (palette) {
            auto* out = reinterpret_cast<std::uint32_t*>(dst);
            const bool four_bit = palette->format() == 4;
            for (std::size_t i = 0; i < r.rgba.size(); ++i) {
                const std::uint32_t grey = r.rgba[i] & 0xFF;
                out[i] = palette->palette_color(four_bit ? (grey + 8) / 17 : grey);
            }
        } else {
            std::memcpy(dst, r.rgba.data(), bytes);
        }
        if (t.image.image) frame().garbage.push_back(t.image);  // destroyed once this frame slot is reused
        const VkFormat format = r.compression == Compression::BC7       ? VK_FORMAT_BC7_UNORM_BLOCK
                                : r.compression == Compression::ASTC4x4 ? VK_FORMAT_ASTC_4x4_UNORM_BLOCK
                                                                        : kColorFormat;
        t.image = make_image(r.width, r.height, format, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                             VK_IMAGE_ASPECT_COLOR_BIT, r.levels);
        t.image.replacement = true;
        const VkCommandBuffer cmd = frame().upload;
        transition(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        nx::vector<VkBufferImageCopy> copies(r.levels);
        for (std::uint32_t l = 0; l < r.levels; ++l) {
            const std::uint32_t w = std::max(1u, static_cast<std::uint32_t>(r.width) >> l);
            const std::uint32_t h = std::max(1u, static_cast<std::uint32_t>(r.height) >> l);
            copies[l].bufferOffset = offset;
            copies[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            copies[l].imageExtent = {w, h, 1};
            offset += r.compression == Compression::None ? static_cast<VkDeviceSize>(w) * h * 4
                                                          : ktx2::level_size(static_cast<int>(w), static_cast<int>(h));
        }
        vkCmdCopyBufferToImage(cmd, frame().staging.buffer, t.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                               r.levels, copies.data());
        transition(cmd, t.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return true;
    }

    static constexpr std::uint8_t kReplacementLoading = 1, kIndexMapLoading = 2;

    // Puts the replacement picture for t's content in place, if there is one: true once uploaded. While it loads
    // (or the staging buffer is full), t.replacing has the next draws ask again.
    bool replace(Texture& t, const rt::gpu::TextureSampler& ts) {
        using rt::gpu::Lookup;
        if (t.replacing != kIndexMapLoading) {
            const auto found = rt::gpu::texture_replacement(t.content);
            if (found.state == Lookup::Ready && upload_replacement(t, *found.image)) {
                t.replacing = 0;
                return true;
            }
            if (found.state != Lookup::None) {
                t.replacing = kReplacementLoading;
                return false;
            }
        }
        if ((ts.format() == 4 || ts.format() == 5) && rt::gpu::have_indexed_replacements()) {
            if (t.replacing != kIndexMapLoading) t.texels = rt::gpu::texture_texels_hash(ts);
            const auto found = rt::gpu::indexed_replacement(t.texels);
            if (found.state == Lookup::Ready && upload_replacement(t, *found.image, &ts)) {
                t.replacing = 0;
                return true;
            }
            if (found.state != Lookup::None) {
                t.replacing = kIndexMapLoading;
                return false;
            }
        }
        t.replacing = 0;
        return false;
    }

    // Texture from guest memory, decoded and uploaded once per content.
    Image* texture(const ge::State& s, const rt::gpu::TextureSampler& ts, std::uint32_t address) {
        std::uint64_t key = hash_bytes(&address, 4);
        const std::uint32_t params[] = {s.param(ge::TEXFORMAT), s.param(ge::TEXMODE), s.param(ge::TEXSIZE0),
                                        s.param(ge::TEXBUFWIDTH0), static_cast<std::uint32_t>(ts.levels())};
        key = hash_bytes(params, sizeof params, key);
        // The CLUT can change between draws (palette swaps); texture data only between batches, so its (much
        // larger) hash is computed once per batch.
        std::uint64_t clut = 0;
        if (ts.uses_clut()) {
            clut = hash_bytes(s.clut.data(), s.clut.size() * sizeof(s.clut[0]));
            const std::uint32_t clut_format = s.param(ge::CLUTFORMAT);
            clut = hash_bytes(&clut_format, 4, clut);
        }
        // Each palette gets its own cache entry: games draw the same texels with different CLUTs (palette swaps,
        // colour cycling), which would otherwise re-decode the texture on every switch.
        key = hash_bytes(&clut, sizeof clut, key);
        Texture& t = textures[key];
        t.last_frame = frame_number;
        // Re-hash the texels only when guest memory under them may have changed since the last check.
        std::uint64_t written = 0, marked = 0;
        for (int l = 0; l < ts.levels(); ++l) {
            const std::uint32_t level_address = ((s.param(static_cast<std::uint8_t>(ge::TEXBUFWIDTH0 + l)) & 0x0F0000) << 8) |
                                                (s.param(static_cast<std::uint8_t>(ge::TEXADDR0 + l)) & 0xFFFFF0);
            const auto bytes = static_cast<std::uint32_t>(ts.level_size(l));
            written = std::max(written, rt::guest_written_since(level_address, bytes));
            marked = std::max(marked, rt::guest_marked_since(level_address, bytes));
        }
        if (t.image.image && written <= t.checked) return cached(t, s, ts);
        // The memory may have changed: the game writes back its whole data cache every frame, so most re-checks find
        // nothing. Unless something known wrote to these texels (marked: an HLE copy, a file read, a movie frame), a
        // sample (64 bytes of every 2 KiB) decides whether the full hash runs; every 16th check runs it anyway, so
        // a change the sample misses still shows within a fraction of a second.
        const std::uint64_t sample = sample_texels(ts);
        if (t.image.image && marked <= t.checked && sample == t.sample && ++t.quick_checks < 16) {
            t.checked = rt::guest_write_counter();
            return cached(t, s, ts);
        }
        t.quick_checks = 0;
        t.sample = sample;
        const std::uint64_t content = rt::gpu::texture_content_hash(s, ts);
        for (int l = 0; l < ts.levels(); ++l) stats.hashed_bytes += ts.level_size(l);
        t.checked = rt::guest_write_counter();
        if (t.image.image && t.content == content) return cached(t, s, ts);
        ++stats.uploads;
        stats.uploaded_texels += static_cast<std::uint64_t>(ts.width(0)) * static_cast<std::uint64_t>(ts.height(0));
        if (t.image.image) frame().garbage.push_back(t.image);
        t.image = {};
        t.content = content;
        t.uploaded_frame = frame_number;
        t.collected = false;
        t.replacing = 0;
        if (replace(t, ts)) return &t.image;

        const std::uint32_t levels = static_cast<std::uint32_t>(ts.levels());
        t.image = make_image(ts.width(0), ts.height(0), kColorFormat,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT, levels);
        const VkCommandBuffer cmd = frame().upload;
        transition(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        for (std::uint32_t l = 0; l < levels; ++l) {
            const int w = ts.width(static_cast<int>(l)), h = ts.height(static_cast<int>(l));
            VkDeviceSize offset = 0;
            auto* dst = reinterpret_cast<std::uint32_t*>(staging(static_cast<VkDeviceSize>(w) * h * 4, offset));
            if (!dst) break;  // staging full: the level stays undefined this frame (rare)
            std::uint32_t* decoded = dst;
            ts.decode(static_cast<int>(l), decoded);
            const bool keep_font = rt::gpu::ui_font_wanted();
            if (l == 0 && (icon_style != rt::gpu::IconStyle::PlayStation || keep_font))
                if (const auto* atlas = rt::gpu::find_icon_atlas(ts.format(), w, h, ts.level_data(0), ts.level_size(0))) {
                    rt::gpu::draw_button_icons(*atlas, decoded, w, icon_style);
                    if (keep_font && atlas->hash == rt::gpu::ui_font()->atlas_hash)
                        rt::gpu::keep_ui_font_sheet(decoded, w, h);
                }
            VkBufferImageCopy copy{};
            copy.bufferOffset = offset;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
            copy.imageExtent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
            vkCmdCopyBufferToImage(cmd, frame().staging.buffer, t.image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        }
        transition(cmd, t.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return &t.image;
    }

    // A texture inside a render target: sample a scaled copy of that region (render-to-texture effects).
    Image* target_texture(Target& t, std::uint32_t address, int w, int h) {
        const int bpp = bytes_per_pixel(t.format);
        const std::uint32_t offset = (address - t.address) / static_cast<std::uint32_t>(bpp);
        const int x = static_cast<int>(offset % t.stride), y = static_cast<int>(offset / t.stride);
        const int cw = std::min(w, static_cast<int>(t.stride) - x), ch = std::min(h, kTargetHeight - y);
        if (cw <= 0 || ch <= 0) return nullptr;
        end_rendering();
        const VkCommandBuffer cmd = frame().draw;
        Image copy = make_image(w * scale, h * scale, kColorFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                VK_IMAGE_ASPECT_COLOR_BIT);
        transition(cmd, copy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        transition(cmd, t.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkImageCopy region{};
        region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.srcOffset = {(x + t.margin_x) * scale, (y + t.margin_y) * scale, 0};
        region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.extent = {static_cast<std::uint32_t>(cw * scale), static_cast<std::uint32_t>(ch * scale), 1};
        vkCmdCopyImage(cmd, t.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region);
        transition(cmd, copy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        frame().garbage.push_back(copy);
        return &frame().garbage.back();
    }

    Target* target_containing(std::uint32_t address) {
        address = rt::ge::vram_canonical(address);
        for (auto& [base, t] : targets) {
            const std::uint32_t size = t->stride * static_cast<std::uint32_t>(t->used_height * bytes_per_pixel(t->format));
            if (address >= base && address < base + size) return t.get();
        }
        return nullptr;
    }

    VkDescriptorSet descriptor(VkImageView view, VkSampler smp) {
        Frame& f = frame();
        auto& set = f.sets[{view, smp}];
        if (set) return set;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = f.descriptors;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &set_layout;
        if (vkAllocateDescriptorSets(vk.device, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
        VkDescriptorImageInfo info{smp, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(vk.device, 1, &w, 0, nullptr);
        return set;
    }

    // ---- pipelines ---------------------------------------------------------------------------------------

    VkPipeline pipeline(const PipelineKey& k) {
        auto& p = pipelines[k];
        if (p) return p;
        const std::uint32_t spec_data[10] = {k.texture, k.tex_func, k.tex_alpha, k.color_double, k.fog,
                                             k.alpha_test, k.color_test, k.clear, k.src_factor, k.double_alpha};
        VkSpecializationMapEntry entries[10];
        for (std::uint32_t i = 0; i < 10; ++i) entries[i] = {i, i * 4, 4};
        VkSpecializationInfo spec{10, entries, sizeof spec_data, spec_data};
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert, "main", nullptr};
        stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag, "main", &spec};

        const VkVertexInputBindingDescription binding{0, sizeof(GpuVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription attributes[4] = {
            {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 16},
            {2, 0, VK_FORMAT_R8G8B8A8_UNORM, 28},
            {3, 0, VK_FORMAT_R8G8B8A8_UNORM, 32},
        };
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 4;
        vi.pVertexAttributeDescriptions = attributes;
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = k.topology == 0 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
                      : k.topology == 1 ? VK_PRIMITIVE_TOPOLOGY_LINE_LIST
                                        : VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        VkPipelineViewportStateCreateInfo vp_state{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp_state.viewportCount = 1;
        vp_state.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.depthClampEnable = vk.depth_clamp ? VK_TRUE : VK_FALSE;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;  // culled on the CPU with the PSP's screen-space rule
        rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = static_cast<VkSampleCountFlagBits>(k.samples);
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = k.depth_test;
        ds.depthWriteEnable = k.depth_write;
        ds.depthCompareOp = compare_op(k.depth_func);
        ds.stencilTestEnable = k.stencil;
        ds.front.compareOp = compare_op(k.stencil_func);
        ds.front.failOp = stencil_op(k.stencil_fail);
        ds.front.depthFailOp = stencil_op(k.stencil_zfail);
        ds.front.passOp = stencil_op(k.stencil_pass);
        ds.back = ds.front;
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = k.blend;
        blend.srcColorBlendFactor = static_cast<VkBlendFactor>(k.src);
        blend.dstColorBlendFactor = static_cast<VkBlendFactor>(k.dst);
        blend.colorBlendOp = static_cast<VkBlendOp>(k.op);
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        blend.colorWriteMask = k.write_mask;
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.logicOpEnable = k.logic;
        cb.logicOp = static_cast<VkLogicOp>(k.logic_op);
        cb.attachmentCount = 1;
        cb.pAttachments = &blend;
        const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                                          VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                          VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = static_cast<std::uint32_t>(std::size(dynamic));
        dyn.pDynamicStates = dynamic;
        VkPipelineRenderingCreateInfo rendering_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        const VkFormat color_format = kColorFormat;
        rendering_info.colorAttachmentCount = 1;
        rendering_info.pColorAttachmentFormats = &color_format;
        rendering_info.depthAttachmentFormat = vk.depth_format;
        rendering_info.stencilAttachmentFormat = vk.depth_format;

        VkGraphicsPipelineCreateInfo pci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pci.pNext = &rendering_info;
        pci.stageCount = 2;
        pci.pStages = stages;
        pci.pVertexInputState = &vi;
        pci.pInputAssemblyState = &ia;
        pci.pViewportState = &vp_state;
        pci.pRasterizationState = &rs;
        pci.pMultisampleState = &ms;
        pci.pDepthStencilState = &ds;
        pci.pColorBlendState = &cb;
        pci.pDynamicState = &dyn;
        pci.layout = layout;
        const auto start = nx::timestamp::now();
        check(vkCreateGraphicsPipelines(vk.device, cache, 1, &pci, nullptr, &p), "graphics pipeline");
        cache_dirty_frame = frame_number;
        const double took = (nx::timestamp::now() - start).millis_f();
        ++stats.pipelines;
        stats.pipeline_ms += took;
        stats.worst_pipeline_ms = std::max(stats.worst_pipeline_ms, took);
        return p;
    }

    // ---- vertex assembly ---------------------------------------------------------------------------------

    float target_w = 512, target_h = kTargetHeight;
    float margin_x = 0, margin_y = 0;  // margins of the target drawn to (guest pixels)
    float shift_x = 0, shift_y = 0;    // 2D moved to the screen's edges (rt::anchor_2d), guest pixels
    // Filling the screen: sprites reaching an edge of the PSP's screen are stretched over the margins beyond it.
    // 1: each such edge (clears, drawn in strips); 2: only sprites covering the whole screen (solid fills: fades).
    int stretch_edges = 0;
    bool expand_overlay_x = false, expand_overlay_y = false;

    // Drawing-space pixels (already projected) -> Vulkan clip space with w = 1.
    GpuVertex screen_vertex(float x, float y, float z, float s, float t, float q, const float* color, const float* spec, float fog) const {
        GpuVertex v{};
        v.pos[0] = 2.0f * (x + margin_x + shift_x) / target_w - 1.0f;
        v.pos[1] = 2.0f * (y + margin_y + shift_y) / target_h - 1.0f;
        v.pos[2] = std::clamp(z, 0.0f, 65535.0f) / 65535.0f;
        v.pos[3] = 1.0f;
        v.uvq[0] = s;
        v.uvq[1] = t;
        v.uvq[2] = q;
        const float sf[4] = {spec[0], spec[1], spec[2], fog * 255.0f};
        v.color = pack_color(color);
        v.spec_fog = pack_color(sf);
        return v;
    }

    // Transformed vertex -> Vulkan clip space, folding in the GE viewport so the GPU interpolates perspective-correct.
    GpuVertex clip_vertex(const ClipVertex& c, const float* color) const {
        GpuVertex v{};
        const float w = c.clip[3];
        v.pos[0] = c.clip[0] * (2.0f * vp.vp_scale[0] / target_w) + w * (2.0f * (vp.vp_center[0] - vp.offset_x + margin_x) / target_w - 1.0f);
        v.pos[1] = c.clip[1] * (2.0f * vp.vp_scale[1] / target_h) + w * (2.0f * (vp.vp_center[1] - vp.offset_y + margin_y) / target_h - 1.0f);
        v.pos[2] = c.clip[2] * (vp.vp_scale[2] / 65535.0f) + w * (vp.vp_center[2] / 65535.0f);
        v.pos[3] = w;
        v.uvq[0] = c.s;
        v.uvq[1] = c.t;
        v.uvq[2] = c.q;
        const float sf[4] = {c.spec[0], c.spec[1], c.spec[2], c.fog * 255.0f};
        v.color = pack_color(color);
        v.spec_fog = pack_color(sf);
        return v;
    }

    void screen_xy(const ClipVertex& c, float& x, float& y, float& z) const {
        if (vp.through) {
            x = c.clip[0];
            y = c.clip[1];
            z = c.clip[2];
            return;
        }
        const float iw = 1.0f / c.clip[3];
        x = c.clip[0] * iw * vp.vp_scale[0] + vp.vp_center[0] - vp.offset_x;
        y = c.clip[1] * iw * vp.vp_scale[1] + vp.vp_center[1] - vp.offset_y;
        z = c.clip[2] * iw * vp.vp_scale[2] + vp.vp_center[2];
    }

    bool cull = false, flat = false;
    int cull_mode = 0;

    void triangle(const ClipVertex& a, const ClipVertex& b, const ClipVertex& c) {
        const float* provoking = c.color;
        if (vp.through) {
            for (const ClipVertex* p : {&a, &b, &c})
                out.push_back(screen_vertex(p->clip[0], p->clip[1], p->clip[2], p->s, p->t, p->q, flat ? provoking : p->color, p->spec, p->fog));
            return;
        }
        if (cull && a.clip[3] > 0 && b.clip[3] > 0 && c.clip[3] > 0) {
            float ax, ay, az, bx, by, bz, cx, cy, cz;
            screen_xy(a, ax, ay, az);
            screen_xy(b, bx, by, bz);
            screen_xy(c, cx, cy, cz);
            const float area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
            if (cull_mode == 1 ? area < 0.0f : area > 0.0f) return;
        }
        if (expand_overlay_x || expand_overlay_y) {
            for (const ClipVertex* p : {&a, &b, &c}) {
                float x, y, z;
                screen_xy(*p, x, y, z);
                if (expand_overlay_x) x = (x - 240.0f) * (480.0f + 2.0f * margin_x) / 480.0f + 240.0f;
                if (expand_overlay_y) y = (y - 136.0f) * (272.0f + 2.0f * margin_y) / 272.0f + 136.0f;
                out.push_back(screen_vertex(x, y, z, p->s / p->q, p->t / p->q, 1, flat ? provoking : p->color,
                                            p->spec, p->fog));
            }
            return;
        }
        // Near-plane clipping (z >= -w), as the software renderer does; the GPU clips the rest.
        ClipVertex in[8] = {a, b, c}, clipped[8];
        int n = 0;
        for (int i = 0; i < 3; ++i) {
            const ClipVertex& p = in[i];
            const ClipVertex& q = in[(i + 1) % 3];
            const float dp = p.clip[2] + p.clip[3], dq = q.clip[2] + q.clip[3];
            if (dp >= 0.0f) clipped[n++] = p;
            if ((dp >= 0.0f) != (dq >= 0.0f)) {
                const float t = dp / (dp - dq);
                ClipVertex r;
                const auto lerp = [t](float x, float y) { return x + (y - x) * t; };
                for (int k = 0; k < 4; ++k) r.clip[k] = lerp(p.clip[k], q.clip[k]);
                r.s = lerp(p.s, q.s);
                r.t = lerp(p.t, q.t);
                r.q = lerp(p.q, q.q);
                for (int k = 0; k < 4; ++k) r.color[k] = lerp(p.color[k], q.color[k]);
                for (int k = 0; k < 3; ++k) r.spec[k] = lerp(p.spec[k], q.spec[k]);
                r.fog = lerp(p.fog, q.fog);
                clipped[n++] = r;
            }
        }
        for (int i = 1; i + 1 < n; ++i)
            for (const ClipVertex* p : {&clipped[0], &clipped[i], &clipped[i + 1]})
                out.push_back(clip_vertex(*p, flat ? provoking : p->color));
    }

    void sprite(const ClipVertex& a, const ClipVertex& b) {
        float x0, y0, z0, x1, y1, z1;
        screen_xy(a, x0, y0, z0);
        screen_xy(b, x1, y1, z1);
        if (stretch_edges) {
            float& left = x0 < x1 ? x0 : x1;
            float& right = x0 < x1 ? x1 : x0;
            float& top = y0 < y1 ? y0 : y1;
            float& bottom = y0 < y1 ? y1 : y0;
            if (stretch_edges == 1 || (left <= 0.0f && right >= 480.0f && top <= 0.0f && bottom >= 272.0f)) {
                if (left <= 0.0f) left = -margin_x;
                if (right >= 480.0f) right = 480.0f + margin_x;
                if (top <= 0.0f) top = -margin_y;
                if (bottom >= 272.0f) bottom = 272.0f + margin_y;
            }
        }
        const float u0 = a.s / a.q, v0 = a.t / a.q, u1 = b.s / b.q, v1 = b.t / b.q;
        const GpuVertex tl = screen_vertex(x0, y0, z1, u0, v0, 1, b.color, b.spec, b.fog);
        const GpuVertex tr = screen_vertex(x1, y0, z1, u1, v0, 1, b.color, b.spec, b.fog);
        const GpuVertex bl = screen_vertex(x0, y1, z1, u0, v1, 1, b.color, b.spec, b.fog);
        const GpuVertex br = screen_vertex(x1, y1, z1, u1, v1, 1, b.color, b.spec, b.fog);
        out.insert(out.end(), {tl, tr, bl, tr, br, bl});
    }

    void point_vertex(const ClipVertex& c) {
        float x, y, z;
        if (!vp.through && c.clip[3] <= 0) return;
        screen_xy(c, x, y, z);
        out.push_back(screen_vertex(x + 0.5f, y + 0.5f, z, c.s / c.q, c.t / c.q, 1, c.color, c.spec, c.fog));
    }

    // ---- draw --------------------------------------------------------------------------------------------

    void draw(const ge::State& s, ge::Prim prim, const nx::vector<ge::Vertex>& vertices) {
        begin_frame();
        const std::uint32_t clear_bits = s.param(ge::CLEARMODE);
        const bool clear = clear_bits & 1;
        const bool through = s.format.through;
        PipelineKey key;
        FragmentConstants pc{};

        // Texture.
        rt::gpu::TextureSampler ts;
        const bool texturing = s.on(ge::TEXTUREMAPENABLE) && !clear;
        if (texturing) ts.setup(s, memory);
        const bool fog = s.on(ge::FOGENABLE) && !through && !clear;
        vp.configure(s, texturing, texturing ? ts.width() : 1, texturing ? ts.height() : 1, fog);
        flat = !(s.param(ge::SHADEMODE) & 1);
        cull = s.on(ge::CULLFACEENABLE) && !through;
        cull_mode = static_cast<int>(s.param(ge::CULL) & 1);

        const bool timing = rt::timing_stats();
        const auto transform_start = timing ? nx::timestamp::now() : nx::timestamp{};
        clip.resize(vertices.size());
        for (std::size_t i = 0; i < vertices.size(); ++i) clip[i] = vp.transform(s, vertices[i]);

        Target& t = replaying ? replay
                              : target(s.param(ge::FRAMEBUFPTR), s.param(ge::FRAMEBUFWIDTH) & 0x7FC,
                                       static_cast<int>(s.param(ge::FRAMEBUFPIXFORMAT) & 3));
        if (interpolation && !replaying && display_targets.contains(t.address))
            recorded.push_back({t.address, draw_key(s, prim, vertices.size()), prim, s, vertices});
        const bool margins = t.margin_x || t.margin_y;
        target_w = static_cast<float>(static_cast<int>(t.stride) + 2 * t.margin_x);
        target_h = static_cast<float>(kTargetHeight + 2 * t.margin_y);
        margin_x = static_cast<float>(t.margin_x);
        margin_y = static_cast<float>(t.margin_y);
        stretch_edges = !margins ? 0 : clear ? 1 : through && !texturing ? 2 : 0;
        shift_x = shift_y = 0;
        if (margins && through && t.filled && !vertices.empty()) {  // (frames without 3D are shown without margins)
            int horizontal = 0, vertical = 0;
            bool anchored = rt::anchor_2d_of(s.vertex_address, horizontal, vertical);
            if (!anchored) {
                float x0 = vertices[0].pos[0], x1 = x0, y0 = vertices[0].pos[1], y1 = y0;
                for (const ge::Vertex& v : vertices) {
                    x0 = std::min(x0, v.pos[0]), x1 = std::max(x1, v.pos[0]);
                    y0 = std::min(y0, v.pos[1]), y1 = std::max(y1, v.pos[1]);
                }
                anchored = rt::anchor_2d_of(x0, y0, x1, y1, horizontal, vertical);
            }
            if (anchored) {
                shift_x = static_cast<float>(horizontal * t.margin_x);
                shift_y = static_cast<float>(vertical * t.margin_y);
                stretch_edges = 0;
            }
        }
        // Some fades and dialogs are projected geometry rather than through-mode sprites. Extend only axes that
        // span the PSP screen: a full-width, partial-height dialog keeps its height and vertical placement. Scene
        // composites also use direct-color textures; admit those only for a small solid-color overlay texture. Only
        // expand simple panels: larger projected meshes are positioned artwork and must retain their proportions.
        bool overlay_texture = !texturing || ts.format() >= 4;
        if (texturing && ts.format() < 4 && ts.levels() == 1 && ts.width() * ts.height() <= 16) {
            std::uint32_t pixels[16];
            ts.decode(0, pixels);
            overlay_texture = std::all_of(pixels, pixels + ts.width() * ts.height(),
                                          [first = pixels[0]](std::uint32_t pixel) { return pixel == first; });
        }
        expand_overlay_x = expand_overlay_y = margins && !through && !clear && t.filled && clip.size() >= 4 &&
                                               clip.size() <= 6 && overlay_texture && s.on(ge::ALPHABLENDENABLE) &&
                                               !s.on(ge::ZTESTENABLE) && s.on(ge::ZWRITEDISABLE);
        if (expand_overlay_x || expand_overlay_y) {
            float left = 1e9f, top = 1e9f, right = -1e9f, bottom = -1e9f, plane = 0;
            for (std::size_t i = 0; i < clip.size(); ++i) {
                const ClipVertex& c = clip[i];
                if (c.clip[3] <= 0 || std::abs(c.clip[3] - clip[0].clip[3]) > 0.0001f) {
                    expand_overlay_x = expand_overlay_y = false;
                    break;
                }
                float x, y, z;
                screen_xy(c, x, y, z);
                if (i == 0) plane = z;
                if (std::abs(z - plane) > 0.01f) {
                    expand_overlay_x = expand_overlay_y = false;
                    break;
                }
                left = std::min(left, x), right = std::max(right, x);
                top = std::min(top, y), bottom = std::max(bottom, y);
            }
            expand_overlay_x = expand_overlay_x && left <= 0.01f && right >= 479.99f;
            expand_overlay_y = expand_overlay_y && top <= 0.01f && bottom >= 271.99f;
        }
        if (margins && !through && !clear) t.filled = true;

        out.clear();
        const std::size_t n = clip.size();
        switch (prim) {
            case ge::Prim::Points:
                key.topology = 2;
                for (std::size_t i = 0; i < n; ++i) point_vertex(clip[i]);
                break;
            case ge::Prim::Lines:
            case ge::Prim::LineStrip:
                key.topology = 1;
                for (std::size_t i = 0; i + 1 < n; i += prim == ge::Prim::Lines ? 2 : 1) {
                    point_vertex(clip[i]);
                    point_vertex(clip[i + 1]);
                }
                break;
            case ge::Prim::Triangles:
                for (std::size_t i = 0; i + 2 < n; i += 3) triangle(clip[i], clip[i + 1], clip[i + 2]);
                break;
            case ge::Prim::TriangleStrip:
                for (std::size_t i = 0; i + 2 < n; ++i) {
                    if (i & 1) triangle(clip[i + 1], clip[i], clip[i + 2]);
                    else triangle(clip[i], clip[i + 1], clip[i + 2]);
                }
                break;
            case ge::Prim::TriangleFan:
                for (std::size_t i = 1; i + 1 < n; ++i) triangle(clip[0], clip[i], clip[i + 1]);
                break;
            case ge::Prim::Sprites:
                for (std::size_t i = 0; i + 1 < n; i += 2) sprite(clip[i], clip[i + 1]);
                break;
        }
        if (timing) stats.transform_ns += static_cast<std::uint64_t>((nx::timestamp::now() - transform_start).nanos());
        if (out.empty()) return;

        // Fixed-function state.
        std::uint32_t stencil_ref = 0, stencil_mask = 0xFF;
        float blend_constants[4] = {0, 0, 0, 0};
        if (clear) {
            key.clear = 1;
            key.write_mask = static_cast<std::uint8_t>(((clear_bits >> 8) & 1 ? 0x7 : 0) | ((clear_bits >> 9) & 1 ? 0x8 : 0));
            key.depth_test = 1;
            key.depth_func = 1;  // always
            key.depth_write = (clear_bits >> 10) & 1;
            if ((clear_bits >> 9) & 1) {
                key.stencil = 1;
                key.stencil_func = 1;
                key.stencil_fail = key.stencil_zfail = key.stencil_pass = 2;  // replace
                stencil_ref = out[0].color >> 24;
            }
        } else {
            if (texturing) {
                const std::uint32_t func = s.param(ge::TEXFUNC);
                key.texture = 1;
                key.tex_func = static_cast<std::uint8_t>(std::min<std::uint32_t>(func & 7, 4));
                key.tex_alpha = (func >> 8) & 1;
                key.color_double = (func >> 16) & 1;
                const std::uint32_t env = s.param(ge::TEXENVCOLOR);
                for (int i = 0; i < 3; ++i) pc.env[i] = static_cast<float>((env >> (i * 8)) & 0xFF) / 255.0f;
            }
            key.fog = fog;
            const std::uint32_t fog_color = s.param(ge::FOGCOLOR);
            for (int i = 0; i < 3; ++i) pc.fog[i] = static_cast<float>((fog_color >> (i * 8)) & 0xFF) / 255.0f;
            if (s.on(ge::ALPHATESTENABLE)) {
                const std::uint32_t at = s.param(ge::ALPHATEST);
                key.alpha_test = static_cast<std::uint8_t>(at & 7);
                pc.alpha_ref = (at >> 8) & 0xFF;
                pc.alpha_mask = (at >> 16) & 0xFF;
            }
            if (s.on(ge::COLORTESTENABLE)) {
                key.color_test = static_cast<std::uint8_t>(s.param(ge::COLORTEST) & 3);
                pc.color_ref = s.param(ge::COLORREF);
                pc.color_mask = s.param(ge::COLORTESTMASK);
            }
            if (s.on(ge::ZTESTENABLE)) {
                key.depth_test = 1;
                key.depth_func = static_cast<std::uint8_t>(s.param(ge::ZTEST) & 7);
                key.depth_write = !s.on(ge::ZWRITEDISABLE);
            }
            const bool stencil = s.on(ge::STENCILTESTENABLE);
            if (stencil) {
                const std::uint32_t st = s.param(ge::STENCILTEST), so = s.param(ge::STENCILOP);
                key.stencil = 1;
                key.stencil_func = static_cast<std::uint8_t>(st & 7);
                key.stencil_fail = static_cast<std::uint8_t>(so & 7);
                key.stencil_zfail = static_cast<std::uint8_t>((so >> 8) & 7);
                key.stencil_pass = static_cast<std::uint8_t>((so >> 16) & 7);
                stencil_ref = (st >> 8) & 0xFF;
                stencil_mask = (st >> 16) & 0xFF;
            }
            // Per-channel write masks; the alpha channel holds stencil and is only written with the stencil test.
            const std::uint32_t mask_rgb = s.param(ge::MASKRGB), mask_a = s.param(ge::MASKALPHA) & 0xFF;
            key.write_mask = 0;
            for (int i = 0; i < 3; ++i)
                if (((mask_rgb >> (i * 8)) & 0xFF) != 0xFF) key.write_mask |= static_cast<std::uint8_t>(1 << i);
            if (stencil && mask_a != 0xFF) key.write_mask |= 0x8;
            if (s.on(ge::ALPHABLENDENABLE)) {
                const std::uint32_t bm = s.param(ge::BLENDMODE);
                const std::uint32_t a = bm & 15, b = (bm >> 4) & 15, eq = (bm >> 8) & 7;
                const std::uint32_t fa = s.param(ge::BLENDFIXEDA), fb = s.param(ge::BLENDFIXEDB);
                key.blend = 1;
                static constexpr VkBlendFactor kSrc[10] = {
                    VK_BLEND_FACTOR_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA,
                    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                    VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA,
                    VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA};
                static constexpr VkBlendFactor kDst[10] = {
                    VK_BLEND_FACTOR_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA,
                    VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                    VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_DST_ALPHA,
                    VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA};
                // Doubled source alpha (codes 6/7) has no Vulkan factor: the shader applies the source factor and,
                // for a doubled destination factor, outputs min(2a, 1) as alpha. Doubled destination alpha
                // (codes 8/9) is approximated with plain destination alpha.
                const bool dst_double = b == 6 || b == 7;
                if (a == 6 || a == 7) {
                    key.src = VK_BLEND_FACTOR_ONE;
                    key.src_factor = a == 6 ? 2 : 3;
                } else if (dst_double && (a == 2 || a == 3)) {
                    key.src = VK_BLEND_FACTOR_ONE;
                    key.src_factor = a == 2 ? 4 : 5;
                } else if (a < 10) {
                    key.src = static_cast<std::uint8_t>(kSrc[a]);
                } else if ((fa & 0xFFFFFF) == 0xFFFFFF) {
                    key.src = VK_BLEND_FACTOR_ONE;
                } else if ((fa & 0xFFFFFF) == 0) {
                    key.src = VK_BLEND_FACTOR_ZERO;
                } else {  // Vulkan has one constant color: apply BLENDFIXEDA in the shader
                    key.src = VK_BLEND_FACTOR_ONE;
                    key.src_factor = 1;
                    for (int i = 0; i < 3; ++i) pc.fixed_a[i] = static_cast<float>((fa >> (i * 8)) & 0xFF) / 255.0f;
                }
                if (dst_double) {
                    key.dst = b == 6 ? VK_BLEND_FACTOR_SRC_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
                    key.double_alpha = 1;
                } else if (b < 10) {
                    key.dst = static_cast<std::uint8_t>(kDst[b]);
                } else if ((fb & 0xFFFFFF) == 0xFFFFFF) {
                    key.dst = VK_BLEND_FACTOR_ONE;
                } else if ((fb & 0xFFFFFF) == 0) {
                    key.dst = VK_BLEND_FACTOR_ZERO;
                } else {
                    key.dst = VK_BLEND_FACTOR_CONSTANT_COLOR;
                    for (int i = 0; i < 3; ++i) blend_constants[i] = static_cast<float>((fb >> (i * 8)) & 0xFF) / 255.0f;
                }
                static constexpr VkBlendOp kOps[8] = {VK_BLEND_OP_ADD, VK_BLEND_OP_SUBTRACT, VK_BLEND_OP_REVERSE_SUBTRACT,
                                                      VK_BLEND_OP_MIN, VK_BLEND_OP_MAX, VK_BLEND_OP_ADD, VK_BLEND_OP_ADD,
                                                      VK_BLEND_OP_ADD};
                key.op = static_cast<std::uint8_t>(kOps[eq]);
            }
            if (s.on(ge::LOGICOPENABLE) && vk.logic_op) {
                key.logic = 1;
                key.logic_op = static_cast<std::uint8_t>(s.param(ge::LOGICOP) & 15);  // GE and Vulkan orders match
                key.blend = 0;
            }
        }

        // Resources (may end the current rendering for copies), then the render pass.
        VkImageView view = white.view;
        VkSampler smp = sampler(false, false, false, false, false, 1);
        if (texturing) {
            const std::uint32_t tex_addr = rt::ge::vram_canonical(((s.param(ge::TEXBUFWIDTH0) & 0x0F0000) << 8) |
                                                                  (s.param(ge::TEXADDR0) & 0xFFFFF0));
            const auto texture_start = timing ? nx::timestamp::now() : nx::timestamp{};
            Image* img = nullptr;
            Target* rt_tex = ts.format() < 4 ? target_containing(tex_addr) : nullptr;
            if (rt_tex) img = target_texture(*rt_tex, tex_addr, ts.width(), ts.height());
            if (!img) img = texture(s, ts, tex_addr);
            if (timing) stats.texture_ns += static_cast<std::uint64_t>((nx::timestamp::now() - texture_start).nanos());
            const std::uint32_t filter = s.param(ge::TEXFILTER), wrap = s.param(ge::TEXWRAP);
            if (img) {
                view = img->view;
                smp = img->replacement ? sampler(true, (filter >> 8) & 1, wrap & 1, (wrap >> 8) & 1, true, img->mips, true)
                                       : sampler(filter & 1, (filter >> 8) & 1, wrap & 1, (wrap >> 8) & 1, (filter & 4) != 0, img->mips);
            }
        }
        Image& d = replaying ? replay_depth : depth(s.param(ge::ZBUFPTR), t);
        begin_rendering(t, d);

        Frame& f = frame();
        const VkDeviceSize bytes = out.size() * sizeof(GpuVertex);
        if (f.vertex_used + bytes > kVertexBytes) return;  // out of vertex space this frame
        std::memcpy(f.vertices.ptr + f.vertex_used, out.data(), bytes);
        const VkCommandBuffer cmd = f.draw;
        key.samples = static_cast<std::uint8_t>(samples);
        const VkPipeline p = pipeline(key);
        if (p != bound_pipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
            bound_pipeline = p;
        }
        const VkDescriptorSet set = descriptor(view, smp);
        if (!set) return;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof pc, &pc);
        const std::uint32_t sc1 = s.param(ge::SCISSOR1), sc2 = s.param(ge::SCISSOR2);
        const int x1 = static_cast<int>(sc1 & 0x3FF), y1 = static_cast<int>((sc1 >> 10) & 0x3FF);
        const int x2 = std::min(static_cast<int>(sc2 & 0x3FF), static_cast<int>(t.stride) - 1);
        const int y2 = std::min(static_cast<int>((sc2 >> 10) & 0x3FF), kTargetHeight - 1);
        t.used_height = std::max(t.used_height, y2 + 1);
        // Anchored 2D keeps its scissor around it: whatever it hides off the PSP's screen stays hidden.
        const int sx = static_cast<int>(shift_x), sy = static_cast<int>(shift_y);
        VkRect2D scissor{{(x1 + t.margin_x + sx) * scale, (y1 + t.margin_y + sy) * scale}, {0, 0}};
        if (x2 >= x1 && y2 >= y1)
            scissor.extent = {static_cast<std::uint32_t>((x2 - x1 + 1) * scale), static_cast<std::uint32_t>((y2 - y1 + 1) * scale)};
        // Filling the screen: a full-screen scissor opens the margins to 3D, clears and stretched fills (other 2D
        // stays centred).
        if ((t.margin_x || t.margin_y) && x1 == 0 && y1 == 0 && x2 >= 479 && y2 >= 271 && (!through || stretch_edges)) {
            scissor.offset = {0, 0};
            scissor.extent = {static_cast<std::uint32_t>((480 + 2 * t.margin_x) * scale),
                              static_cast<std::uint32_t>((272 + 2 * t.margin_y) * scale)};
        }
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref);
        vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_mask);
        vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, clear ? 0xFF : ~s.param(ge::MASKALPHA) & 0xFF);
        vkCmdSetBlendConstants(cmd, blend_constants);
        const VkDeviceSize offset = f.vertex_used;
        vkCmdBindVertexBuffers(cmd, 0, 1, &f.vertices.buffer, &offset);
        vkCmdDraw(cmd, static_cast<std::uint32_t>(out.size()), 1, 0, 0);
        f.vertex_used += (bytes + 255) & ~VkDeviceSize{255};
    }

    void transfer(const ge::State& s) {
        // Copy in guest memory like the hardware, then refresh any render target the destination overlaps.
        const std::uint32_t src_w = s.param(ge::TRANSFERSRCW), dst_w = s.param(ge::TRANSFERDSTW);
        const std::uint32_t src = rt::ge::vram_canonical(((src_w & 0xFF0000) << 8) | (s.param(ge::TRANSFERSRC) & 0xFFFFF0));
        const std::uint32_t dst = rt::ge::vram_canonical(((dst_w & 0xFF0000) << 8) | (s.param(ge::TRANSFERDST) & 0xFFFFF0));
        const std::uint32_t src_stride = src_w & 0xFFF8, dst_stride = dst_w & 0xFFF8;
        const std::uint32_t sp = s.param(ge::TRANSFERSRCPOS), dp = s.param(ge::TRANSFERDSTPOS), sz = s.param(ge::TRANSFERSIZE);
        const std::uint32_t sx = sp & 0x3FF, sy = (sp >> 10) & 0x3FF, dx = dp & 0x3FF, dy = (dp >> 10) & 0x3FF;
        const std::uint32_t w = (sz & 0x3FF) + 1, h = ((sz >> 10) & 0x3FF) + 1;
        const std::uint32_t bpp = (s.param(ge::TRANSFERSTART) & 1) ? 4 : 2;
        rt::mark_guest_written(dst + (dy * dst_stride + dx) * bpp, ((h - 1) * dst_stride + w) * bpp);
        if (guest_copies)
            for (std::uint32_t y = 0; y < h; ++y)
                std::memmove(memory.ptr(dst + ((dy + y) * dst_stride + dx) * bpp), memory.ptr(src + ((sy + y) * src_stride + sx) * bpp), w * bpp);
        if (Target* t = target_containing(dst); t && t->address == dst && t->stride == dst_stride) {
            begin_frame();
            upload_region(*t, static_cast<int>(dx), static_cast<int>(dy), static_cast<int>(w), static_cast<int>(h));
        }
    }

    VulkanGeRenderer::Output finish_frame(std::uint32_t address, std::uint32_t stride, int format,
                                          nx::vector<std::uint32_t>* capture) {
        // Saved once no new pipeline has appeared for ~10 s (the app may be killed rather than closed on Android).
        if (cache_dirty_frame && frame_number - cache_dirty_frame > 300) save_pipeline_cache();
        // A frame whose draws were all skipped, with nothing else recorded: no GPU work at all (each submit costs
        // a render target transition and a fence wait, which on mobile GPUs adds up when fast-forwarding).
        if (skip_draws && !recording && !capture) return {};
        begin_frame();
        const std::uint32_t shown = 0x04000000 | (address & 0x1FFFF0);
        display_targets.insert(shown);  // gets margins when filling the screen, its draws are recorded
        Target& t = target(address, stride ? stride : 512, format);
        if (interpolation) {  // this frame's draws become the current ones
            previous_draws.swap(current_draws);
            current_draws.clear();
            for (Record& r : recorded)
                if (r.target == shown) current_draws.push_back(std::move(r));
            recorded.clear();
        }
        return show(t, capture);
    }

    // Ends the frame with `t` ready to be displayed (and read back into `capture`) and submits it.
    VulkanGeRenderer::Output show(Target& t, nx::vector<std::uint32_t>* capture) {
        end_rendering();
        transition(frame().draw, t.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        // With margins: all of them when 3D was drawn (the scene extends into them), else the centre only (movies,
        // menus and other 2D screens stay pillarboxed).
        const bool fill = (t.margin_x || t.margin_y) && t.filled && fill_aspect > 0;
        t.filled = false;
        VulkanGeRenderer::Output o{t.color.image, 480 * scale, 272 * scale, t.margin_x * scale, t.margin_y * scale};
        if (fill) {  // the display's aspect ratio exactly, from the middle of the margins (rounded up to 8 pixels)
            const int full_w = (480 + 2 * t.margin_x) * scale, full_h = (272 + 2 * t.margin_y) * scale;
            const float s = static_cast<float>(scale);
            if (t.margin_x) o.width = std::min(full_w, static_cast<int>(272.0f * s * fill_aspect + 0.5f));
            if (t.margin_y) o.height = std::min(full_h, static_cast<int>(480.0f * s / fill_aspect + 0.5f));
            o.x = (full_w - o.width) / 2;
            o.y = (full_h - o.height) / 2;
        }
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(o.width) * static_cast<VkDeviceSize>(o.height) * 4;
        if (capture) {
            if (readback_size < bytes) {
                if (readback.buffer) {
                    vkDeviceWaitIdle(vk.device);
                    vmaDestroyBuffer(vk.allocator, readback.buffer, readback.memory);
                }
                readback = make_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
                readback_size = bytes;
            }
            VkBufferImageCopy copy{};
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageOffset = {o.x, o.y, 0};
            copy.imageExtent = {static_cast<std::uint32_t>(o.width), static_cast<std::uint32_t>(o.height), 1};
            vkCmdCopyImageToBuffer(frame().draw, t.color.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &copy);
        }
        const VkFence fence = frame().fence;
        submit_frame();
        if (capture) {
            vkWaitForFences(vk.device, 1, &fence, VK_TRUE, UINT64_MAX);
            vmaInvalidateAllocation(vk.allocator, readback.memory, 0, bytes);
            capture->resize(static_cast<std::size_t>(o.width) * static_cast<std::size_t>(o.height));
            std::memcpy(capture->data(), readback.ptr, bytes);
            for (auto& p : *capture) p |= 0xFF000000u;  // alpha holds stencil
        }
        bound_target = nullptr;
        bound_depth = nullptr;
        return o;
    }

    // ---- frame interpolation -----------------------------------------------------------------------------

    // Identifies a draw across frames: its vertex data, format, count, primitive and texture.
    static std::uint64_t draw_key(const ge::State& s, ge::Prim prim, std::size_t count) {
        const std::uint32_t parts[] = {s.param(ge::VADDR),     s.param(ge::BASE),         s.param(ge::VERTEXTYPE),
                                       static_cast<std::uint32_t>(count), static_cast<std::uint32_t>(prim),
                                       s.param(ge::TEXADDR0), s.param(ge::TEXBUFWIDTH0)};
        return hash_bytes(parts, sizeof parts);
    }

    // The camera's position in the world (view = rotation R, translation t: eye = -R^T t).
    static std::array<float, 3> eye(const ge::State& s) {
        const float* v = s.view;
        return {-(v[0] * v[9] + v[1] * v[10] + v[2] * v[11]), -(v[3] * v[9] + v[4] * v[10] + v[5] * v[11]),
                -(v[6] * v[9] + v[7] * v[10] + v[8] * v[11])};
    }

    // Where a draw's object is in the world: its world matrix applied to its first bone's origin (skinned) or to
    // the model's origin.
    static std::array<float, 3> subject(const ge::State& s) {
        const float* w = s.world;
        const float* b = s.format.weight ? s.bones[0] : nullptr;
        const float x = b ? b[9] : 0.0f, y = b ? b[10] : 0.0f, z = b ? b[11] : 0.0f;
        return {w[0] * x + w[3] * y + w[6] * z + w[9], w[1] * x + w[4] * y + w[7] * z + w[10],
                w[2] * x + w[5] * y + w[8] * z + w[11]};
    }

    static float distance(const std::array<float, 3>& a, const std::array<float, 3>& b) {
        return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
    }

    // A camera cut between two frames, judged on their matched 3D draws: the view turns by more than ~15 degrees,
    // the field of view changes by more than 10%, or the camera moves more than a tenth of its (median) distance
    // to the objects drawn - far more than a camera moves in one frame.
    static bool camera_cut(const nx::vector<std::pair<const Record*, const Record*>>& matched) {
        if (matched.empty()) return false;
        const ge::State& a = matched.front().first->state;
        const ge::State& b = matched.front().second->state;
        float turn = 0;
        for (int i = 0; i < 9; ++i) turn = std::max(turn, std::abs(a.view[i] - b.view[i]));
        if (turn > 0.25f) return true;
        if (std::abs(a.proj[0] - b.proj[0]) > 0.1f * std::abs(b.proj[0]) ||
            std::abs(a.proj[5] - b.proj[5]) > 0.1f * std::abs(b.proj[5]))
            return true;
        const auto eye_b = eye(b);
        const float moved = distance(eye(a), eye_b);
        nx::vector<float> distances;
        for (std::size_t i = 0; i < matched.size() && distances.size() < 64; i += std::max<std::size_t>(1, matched.size() / 64))
            distances.push_back(distance(subject(matched[i].second->state), eye_b));
        std::nth_element(distances.begin(), distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2), distances.end());
        return moved > 0.1f * distances[distances.size() / 2];
    }

    // `out` (a copy of `b`) with its matrices `t` of the way from `a` to `b`.
    static void blend(ge::State& out, const ge::State& a, const ge::State& b, float t) {
        const auto mix = [t](float* o, const float* x, const float* y, int n) {
            for (int i = 0; i < n; ++i) o[i] = x[i] + (y[i] - x[i]) * t;
        };
        // 4x3 matrices: blended elementwise, then each axis given its blended length again (a plain blend of two
        // rotations shrinks what it turns).
        const auto mix43 = [&](float* o, const float* x, const float* y) {
            mix(o, x, y, 12);
            for (int axis = 0; axis < 9; axis += 3) {
                const auto length = [axis](const float* m) {
                    return std::sqrt(m[axis] * m[axis] + m[axis + 1] * m[axis + 1] + m[axis + 2] * m[axis + 2]);
                };
                const float now = length(o), want = length(x) + (length(y) - length(x)) * t;
                if (now > 1e-6f)
                    for (int i = 0; i < 3; ++i) o[axis + i] *= want / now;
            }
        };
        mix43(out.world, a.world, b.world);
        mix43(out.view, a.view, b.view);
        mix(out.proj, a.proj, b.proj, 16);
        mix43(out.tgen, a.tgen, b.tgen);
        for (int bone = 0; bone < 8; ++bone) mix43(out.bones[bone], a.bones[bone], b.bones[bone]);
    }

    // The replay target, sized like `like` (the framebuffer of the frames being blended).
    Target& replay_target(const Target& like) {
        if (replay.color.image && replay.color.width == like.color.width && replay.color.height == like.color.height &&
            replay.stride == like.stride && replay.margin_x == like.margin_x && replay.margin_y == like.margin_y)
            return replay;
        if (replay.color.image) {
            frame().garbage.push_back(replay.color);
            if (replay.msaa.image) frame().garbage.push_back(replay.msaa);
            frame().garbage.push_back(replay_depth);
        }
        replay = Target{};
        replay.address = like.address;
        replay.stride = like.stride;
        replay.format = like.format;
        replay.margin_x = like.margin_x;
        replay.margin_y = like.margin_y;
        replay.color = make_image(like.color.width, like.color.height, kColorFormat,
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                      VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                  VK_IMAGE_ASPECT_COLOR_BIT);
        if (samples != VK_SAMPLE_COUNT_1_BIT)
            replay.msaa = make_image(like.color.width, like.color.height, kColorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                                     VK_IMAGE_ASPECT_COLOR_BIT, 1, samples);
        const VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        replay_depth = make_image(like.color.width, like.color.height, vk.depth_format,
                                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, aspect, 1,
                                  samples);
        const VkCommandBuffer cmd = frame().draw;
        transition(cmd, replay_depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const VkClearDepthStencilValue zero{0.0f, 0};
        const VkImageSubresourceRange range{aspect, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, replay_depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &range);
        return replay;
    }

    VulkanGeRenderer::Output render_between(float t) {
        if (current_draws.empty() || previous_draws.empty()) return {};
        const auto real = targets.find(current_draws.front().target);
        if (real == targets.end()) return {};
        // Draws of the frame before, per key in order: the n-th draw with a key matches the n-th one before.
        nx::hash_map<std::uint64_t, nx::vector<const Record*>> before;
        for (const Record& r : previous_draws) before[r.key].push_back(&r);
        nx::hash_map<std::uint64_t, std::size_t> taken;
        begin_frame();
        end_rendering();
        replay_target(*real->second);
        // Each draw with its match (null: none, or not a 3D draw).
        nx::vector<const Record*> earlier(current_draws.size());
        nx::vector<std::pair<const Record*, const Record*>> matched;
        for (std::size_t i = 0; i < current_draws.size(); ++i) {
            const Record& r = current_draws[i];
            if (r.state.format.through || (r.state.param(ge::CLEARMODE) & 1)) continue;
            if (const auto it = before.find(r.key); it != before.end()) {
                std::size_t& n = taken[r.key];
                if (n < it->second.size()) earlier[i] = it->second[n++];
            }
            if (earlier[i]) matched.emplace_back(earlier[i], &r);
        }
        const bool cut = camera_cut(matched);
        replaying = true;
        ge::State s;
        for (std::size_t i = 0; i < current_draws.size(); ++i) {
            const Record& r = current_draws[i];
            if (!earlier[i] || cut) {
                draw(r.state, r.prim, r.vertices);
                continue;
            }
            s = r.state;
            blend(s, earlier[i]->state, r.state, t);
            draw(s, r.prim, r.vertices);
        }
        replaying = false;
        return show(replay, nullptr);
    }
};

VulkanGeRenderer::VulkanGeRenderer(const VulkanContext& vk, rt::GuestMemory& memory, int scale,
                                   nx::string pipeline_cache)
    : impl_(nx::make_unique<Impl>(vk, memory, scale, std::move(pipeline_cache))) {
    impl_->init();
    // Baked textures in block-compressed formats, where the GPU samples them (and copies into them).
    const auto usable = [&](VkFormat format) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(vk.gpu, format, &props);
        constexpr VkFormatFeatureFlags kNeeded = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                                                 VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        return (props.optimalTilingFeatures & kNeeded) == kNeeded;
    };
    rt::gpu::set_compressed_textures(vk.texture_bc && usable(VK_FORMAT_BC7_UNORM_BLOCK),
                                     vk.texture_astc && usable(VK_FORMAT_ASTC_4x4_UNORM_BLOCK));
}

VulkanGeRenderer::~VulkanGeRenderer() { impl_->shutdown(); }

void VulkanGeRenderer::draw(const rt::ge::State& state, rt::ge::Prim prim, const nx::vector<rt::ge::Vertex>& vertices) {
    if (!impl_->skip_draws) impl_->draw(state, prim, vertices);
}

void VulkanGeRenderer::transfer(const rt::ge::State& state) { impl_->transfer(state); }

VulkanGeRenderer::Stats VulkanGeRenderer::take_stats() { return std::exchange(impl_->stats, Stats{}); }

void VulkanGeRenderer::set_scale(int scale) { impl_->pending_scale = std::max(1, scale); }

void VulkanGeRenderer::set_button_icons(rt::gpu::IconStyle style) {
    if (impl_->icon_style == style) return;
    impl_->icon_style = style;
    reload_textures();
}

void VulkanGeRenderer::reload_textures() {
    for (auto& [_, t] : impl_->textures) impl_->frame().garbage.push_back(t.image);  // destroyed once the slot is reused
    impl_->textures.clear();
}

void VulkanGeRenderer::set_msaa(int samples) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(impl_->vk.gpu, &props);
    const VkSampleCountFlags supported = props.limits.framebufferColorSampleCounts & props.limits.framebufferDepthSampleCounts &
                                         props.limits.framebufferStencilSampleCounts;
    int n = 1;
    while (n * 2 <= std::min(samples, 8) && (supported & static_cast<VkSampleCountFlags>(n * 2))) n *= 2;
    impl_->pending_samples = static_cast<VkSampleCountFlagBits>(n);
}

int VulkanGeRenderer::msaa() const { return static_cast<int>(impl_->samples); }

int VulkanGeRenderer::scale() const { return impl_->scale; }

void VulkanGeRenderer::set_guest_copies(bool on) { impl_->guest_copies = on; }

void VulkanGeRenderer::set_skip_draws(bool skip) { impl_->skip_draws = skip; }

void VulkanGeRenderer::set_interpolation(bool on) {
    if (impl_->interpolation == on) return;
    impl_->interpolation = on;
    impl_->recorded.clear();
    impl_->current_draws.clear();
    impl_->previous_draws.clear();
}

VulkanGeRenderer::Output VulkanGeRenderer::render_between(float t) { return impl_->render_between(t); }

void VulkanGeRenderer::set_fill_aspect(float aspect) {
    // Margins rounded up to 8 pixels (window resizing does not rebuild the framebuffers at every step; the shown
    // picture is cut to the exact ratio): left/right for displays wider than 480:272, top/bottom for narrower
    // ones; no larger than the GPU's largest image.
    int x = 0, y = 0;
    if (aspect > 480.0f / 272.0f) x = (static_cast<int>(std::ceil((272.0f * aspect - 480.0f) / 2.0f)) + 7) / 8 * 8;
    else if (aspect > 0.0f) y = (static_cast<int>(std::ceil((480.0f / aspect - 272.0f) / 2.0f)) + 7) / 8 * 8;
    impl_->fill_aspect = x || y ? aspect : 0.0f;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(impl_->vk.gpu, &props);
    const int largest = static_cast<int>(props.limits.maxImageDimension2D) / std::max(1, impl_->pending_scale);
    impl_->fill_x = std::clamp(x, 0, std::max(0, (largest - 1024) / 2)) / 8 * 8;
    impl_->fill_y = std::clamp(y, 0, std::max(0, (largest - kTargetHeight) / 2)) / 8 * 8;
}

VulkanGeRenderer::Output VulkanGeRenderer::finish_frame(std::uint32_t address, std::uint32_t stride, int format,
                                                       nx::vector<std::uint32_t>* capture) {
    return impl_->finish_frame(address, stride, format, capture);
}

}  // namespace psprecore::platform
