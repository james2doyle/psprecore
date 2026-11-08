// SDL3 window + Vulkan presentation. A frame is either uploaded from the CPU (software renderer) or taken from a
// GPU image (the Vulkan GE renderer's render target) and blitted, scaled and aspect-preserving, into the
// swapchain; the Dear ImGui overlay is drawn on top. One frame in flight keeps this simple.

// Before the SDL/Vulkan headers, which pull in <windows.h>: its TRANSPARENT macro collides with a name in nx::map.
#include "core/foundation/containers/map.h"

#include <SDL3/SDL.h>
#ifdef __ANDROID__
#include <jni.h>
#endif
#include <SDL3/SDL_vulkan.h>
#ifndef _WIN32
#include <dlfcn.h>
#endif
#ifdef JDARC_HAVE_ADRENOTOOLS
#include <adrenotools/driver.h>
#endif
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <volk.h>

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "psprecore/platform/display.hpp"
#include "psprecore/io/path.hpp"
#include "platform/touch_controls.hpp"
#include "core/foundation/core/foundation.h"
#include "core/foundation/containers/map.h"
#include "core/foundation/strings/utf8_string.h"
#include "core/foundation/strings/format.h"

namespace psprecore::platform {

namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(nx::format("Vulkan: {} failed ({})", what, static_cast<int>(r)).c_str());
}

}  // namespace

struct Display::Impl {
    SDL_Window* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;
    VulkanContext ctx;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_UNORM;
    VkExtent2D extent{};
    nx::vector<VkImage> images;
    nx::vector<VkImageView> views;
    bool swapchain_dirty = false;
    bool vsync = true;
    float refresh_rate = 0.0f;  // Android: requested panel refresh rate (0: default)
    bool background = false;  // Android: the window surface is gone until the app returns
    bool surface_wanted = false;  // a surface to create as soon as the platform has a window for it
    nx::unique_function<void(bool)> background_handler;
    // SDL3 does not queue app lifecycle events: they reach event watches only, while they happen.
    static bool SDLCALL lifecycle_watch(void* self, SDL_Event* e);
    std::uint32_t present_errors = 0;  // logged ones (the first few)

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    nx::vector<VkSemaphore> rendered;  // one per swapchain image
    VkFence fence = VK_NULL_HANDLE;

    VkImage frame = VK_NULL_HANDLE;
    VmaAllocation frame_mem = VK_NULL_HANDLE;
    int frame_w = 0, frame_h = 0;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_mem = VK_NULL_HANDLE;
    void* staging_ptr = nullptr;

    PadState pad;
    std::uint32_t keyboard_buttons = 0, gamepad_buttons = 0;
    float key_x = 0, key_y = 0;
    SDL_Gamepad* gamepad = nullptr;
    ButtonGlyphs glyphs = ButtonGlyphs::Auto;             // the setting
    ButtonGlyphs pad_glyphs = ButtonGlyphs::PlayStation;  // the connected gamepad's
    enum class Device : std::uint8_t { None, Keyboard, Gamepad, Touch } last_device = Device::None;
    ButtonGlyphs resolved_glyphs() const {
        if (glyphs != ButtonGlyphs::Auto) return glyphs;
        switch (last_device) {
            case Device::Keyboard: return ButtonGlyphs::Keyboard;
            case Device::Gamepad: return pad_glyphs;
            case Device::Touch: return ButtonGlyphs::PlayStation;
            default: break;
        }
#ifdef __ANDROID__
        return gamepad ? pad_glyphs : ButtonGlyphs::PlayStation;
#else
        return gamepad ? pad_glyphs : ButtonGlyphs::Keyboard;
#endif
    }
    bool menu_request = false;
    bool stick_l = false, stick_r = false;
    TouchControls touch;
    bool touch_mode = false;  // the last input came from the touchscreen
    bool ui_modal = false;    // set_ui_modal
    struct InjectedFinger {
        SDL_FingerID id;
        int polls;
    };
    nx::vector<InjectedFinger> injected;

    bool imgui = false;
    float ui_dpi = 1.0f, ui_scale = 0.0f;  // the font is rasterised at the DPI scale; ui_scale follows the window
    ImGuiStyle ui_base_style;
    nx::unique_function<void()> ui;
    VkImage last_source = VK_NULL_HANDLE;  // the last presented frame, for present_again()
    int last_x = 0, last_y = 0, last_w = 0, last_h = 0;
    VkSampler ui_sampler = VK_NULL_HANDLE;
    struct UiTexture {
        VkImage image;
        VmaAllocation memory;
        VkImageView view;
    };
    nx::map<std::uint64_t, UiTexture> ui_textures;  // by descriptor set
    bool can_capture = false;  // swapchain images support TRANSFER_SRC
    CaptureHandler capture;

    void create_device();
    void init_imgui();
    void shutdown_imgui();
    void create_swapchain();
    void destroy_swapchain();
    void ensure_frame_image(int w, int h);
    void update_pad();
    bool begin_present();
    bool try_create_surface();
    void drop_surface();
    void present_failed(const char* what, VkResult r);
    void blit_and_present(VkImage source, int x, int y, int width, int height,
                          nx::function_ref<void(VkCommandBuffer)> prepare);
};

void Display::Impl::create_device() {
    // A whole Vulkan driver in a file (custom GPU driver packages, e.g. Turnip for Adreno): driven directly
    // through adrenotools instead of the system's libvulkan.so. The app's GameActivity installs the chosen
    // package into gpu_drivers/<name>/ (meta.json naming the library) and passes its path in
    // PSPRECORE_VULKAN_DRIVER; unset: the system driver as usual.
    bool custom_ready = false;
    void* custom_module = nullptr;
    if (const char* custom = std::getenv("PSPRECORE_VULKAN_DRIVER")) {
#if defined(__ANDROID__) && defined(JDARC_HAVE_ADRENOTOOLS)
        const std::string path(custom);
        const std::string::size_type slash = path.rfind('/');
        if (slash != std::string::npos) {
            // Hook libraries ship beside libmain.so (jniLibs -> nativeLibraryDir, where adrenotools
            // expects them); the driver package lives in the app's files dir. libmain.so's own location
            // (volk is statically linked into it) gives the hooks' directory.
            Dl_info info{};
            std::string hooks = ".";
            if (dladdr(reinterpret_cast<const void*>(&volkInitialize), &info) && info.dli_fname) {
                hooks = info.dli_fname;
                if (const auto s = hooks.rfind('/'); s != std::string::npos) hooks.resize(s);
            }
            custom_module = adrenotools_open_libvulkan(
                RTLD_NOW | RTLD_LOCAL, ADRENOTOOLS_DRIVER_CUSTOM, nullptr, hooks.c_str(),
                (path.substr(0, slash + 1)).c_str(), path.substr(slash + 1).c_str(), nullptr, nullptr);
            if (!custom_module) {
                std::fprintf(stderr, "jdarc: custom Vulkan driver failed to load, using the system one\n");
            }
        }
        if (custom_module) {
            const auto get_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                dlsym(custom_module, "vkGetInstanceProcAddr"));
            if (!get_proc) {
                std::fprintf(stderr, "jdarc: not a Vulkan driver: %s\n", custom);
            } else {
                volkInitializeCustom(get_proc);
                std::fprintf(stderr, "Vulkan driver: %s\n", custom);
                custom_ready = true;
            }
        }
#else
        std::fprintf(stderr, "jdarc: custom Vulkan driver ignored (no adrenotools): %s\n", custom);
#endif
    }
    if (!custom_ready) check(volkInitialize(), "volkInitialize");
    Uint32 ext_count = 0;
    const char* const* sdl_exts = SDL_Vulkan_GetInstanceExtensions(&ext_count);
    nx::vector<const char*> exts(sdl_exts, sdl_exts + ext_count);
    nx::vector<const char*> layers;
    if (std::getenv("PSPRECORE_VULKAN_VALIDATION")) layers.push_back("VK_LAYER_KHRONOS_validation");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = SDL_GetWindowTitle(window);
    app.pEngineName = "psprecore";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<std::uint32_t>(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    ici.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    check(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");
    volkLoadInstance(instance);
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface))
        throw std::runtime_error(nx::format("SDL_Vulkan_CreateSurface: {}", SDL_GetError()).c_str());

    std::uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    nx::vector<VkPhysicalDevice> gpus(count);
    vkEnumeratePhysicalDevices(instance, &count, gpus.data());
    int best_score = -1;
    for (VkPhysicalDevice g : gpus) {
        std::uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g, &qcount, nullptr);
        nx::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(g, &qcount, qprops.data());
        for (std::uint32_t i = 0; i < qcount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(g, i, surface, &present);
            if (!(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) || !present) continue;
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(g, &props);
            const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 2 : 1;
            if (score > best_score) {
                best_score = score;
                gpu = g;
                queue_family = i;
            }
            break;
        }
    }
    if (!gpu) throw std::runtime_error("no Vulkan device can present to the window");
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);
    std::fprintf(stderr, "Vulkan device: %s\n", props.deviceName);

    // Vulkan 1.3 where available; otherwise 1.1 + VK_KHR_dynamic_rendering (the GE renderer needs it).
    const std::uint32_t api = std::min<std::uint32_t>(props.apiVersion, VK_API_VERSION_1_3);
    std::uint32_t ext_total = 0;
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &ext_total, nullptr);
    nx::vector<VkExtensionProperties> available(ext_total);
    vkEnumerateDeviceExtensionProperties(gpu, nullptr, &ext_total, available.data());
    const auto has_ext = [&](const char* name) {
        return std::any_of(available.begin(), available.end(), [&](const auto& e) { return std::strcmp(e.extensionName, name) == 0; });
    };
    nx::vector<const char*> device_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    const bool core13 = api >= VK_API_VERSION_1_3;
    if (!core13) {
        if (!has_ext(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) throw std::runtime_error("Vulkan dynamic rendering is not supported");
        device_exts.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
        device_exts.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
        device_exts.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
    }
    VkPhysicalDeviceFeatures supported;
    vkGetPhysicalDeviceFeatures(gpu, &supported);
    VkPhysicalDeviceFeatures enabled{};
    enabled.logicOp = supported.logicOp;
    enabled.samplerAnisotropy = supported.samplerAnisotropy;
    enabled.depthClamp = supported.depthClamp;
    enabled.textureCompressionBC = supported.textureCompressionBC;
    enabled.textureCompressionASTC_LDR = supported.textureCompressionASTC_LDR;
    VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    features13.dynamicRendering = VK_TRUE;
    VkPhysicalDeviceDynamicRenderingFeaturesKHR dynamic_khr{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
    dynamic_khr.dynamicRendering = VK_TRUE;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = core13 ? static_cast<void*>(&features13) : static_cast<void*>(&dynamic_khr);
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<std::uint32_t>(device_exts.size());
    dci.ppEnabledExtensionNames = device_exts.data();
    dci.pEnabledFeatures = &enabled;
    check(vkCreateDevice(gpu, &dci, nullptr, &device), "vkCreateDevice");
    volkLoadDevice(device);
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VmaVulkanFunctions fns{};
    fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = gpu;
    aci.device = device;
    aci.instance = instance;
    aci.vulkanApiVersion = api;
    aci.pVulkanFunctions = &fns;
    check(vmaCreateAllocator(&aci, &allocator), "vmaCreateAllocator");

    ctx.instance = instance;
    ctx.gpu = gpu;
    ctx.device = device;
    ctx.queue = queue;
    ctx.queue_family = queue_family;
    ctx.allocator = allocator;
    ctx.api_version = api;
    ctx.logic_op = enabled.logicOp;
    ctx.sampler_anisotropy = enabled.samplerAnisotropy;
    ctx.depth_clamp = enabled.depthClamp;
    ctx.texture_bc = enabled.textureCompressionBC;
    ctx.texture_astc = enabled.textureCompressionASTC_LDR;
    ctx.begin_rendering = core13 ? vkCmdBeginRendering : vkCmdBeginRenderingKHR;
    ctx.end_rendering = core13 ? vkCmdEndRendering : vkCmdEndRenderingKHR;
    for (const VkFormat f : {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}) {
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(gpu, f, &fp);
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            ctx.depth_format = f;
            break;
        }
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family;
    check(vkCreateCommandPool(device, &pci, nullptr, &pool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device, &cai, &cmd), "vkAllocateCommandBuffers");
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(device, &sci, nullptr, &acquired), "vkCreateSemaphore");
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(device, &fci, nullptr, &fence), "vkCreateFence");
}

void Display::Impl::create_swapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface, &caps);
    int w = 0, h = 0;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    extent = caps.currentExtent.width != 0xFFFFFFFF
                 ? caps.currentExtent
                 : VkExtent2D{std::clamp<std::uint32_t>(static_cast<std::uint32_t>(w), caps.minImageExtent.width, caps.maxImageExtent.width),
                              std::clamp<std::uint32_t>(static_cast<std::uint32_t>(h), caps.minImageExtent.height, caps.maxImageExtent.height)};
    // We do not pre-rotate: on Android (landscape on a portrait panel) the swapchain uses the identity transform at
    // the surface's current size and the compositor rotates it.
    if (extent.width == 0 || extent.height == 0) return;  // minimised

    std::uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, nullptr);
    nx::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface, &count, formats.data());
    VkSurfaceFormatKHR chosen = formats[0];
    for (const auto& f : formats)
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) {
            chosen = f;
            break;
        }
    swapchain_format = chosen.format;

    VkSwapchainCreateInfoKHR sci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    sci.surface = surface;
    sci.minImageCount = std::max(caps.minImageCount, 2u);
    if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
    sci.imageFormat = chosen.format;
    sci.imageColorSpace = chosen.colorSpace;
    sci.imageExtent = extent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    can_capture = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    if (can_capture) sci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                                                                     : caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    for (const auto mode : {VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
                            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR})
        if (caps.supportedCompositeAlpha & mode) {
            sci.compositeAlpha = mode;
            break;
        }
    sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;  // vsync; always available
    if (!vsync) {
        std::uint32_t mode_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &mode_count, nullptr);
        nx::vector<VkPresentModeKHR> modes(mode_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &mode_count, modes.data());
        const auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
        if (has(VK_PRESENT_MODE_MAILBOX_KHR)) sci.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
        else if (has(VK_PRESENT_MODE_IMMEDIATE_KHR)) sci.presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    sci.clipped = VK_TRUE;
    check(vkCreateSwapchainKHR(device, &sci, nullptr, &swapchain), "vkCreateSwapchainKHR");
    static constexpr const char* kModes[] = {"immediate", "mailbox", "fifo (vsync)", "fifo relaxed"};
    std::fprintf(stderr, "Swapchain: %ux%u, %u images, %s\n", extent.width, extent.height, sci.minImageCount,
                 sci.presentMode < 4 ? kModes[sci.presentMode] : "other");
    vkGetSwapchainImagesKHR(device, swapchain, &count, nullptr);
    images.resize(count);
    vkGetSwapchainImagesKHR(device, swapchain, &count, images.data());
    views.resize(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = images[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = swapchain_format;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &vci, nullptr, &views[i]), "swapchain image view");
    }
    VkSemaphoreCreateInfo semci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    rendered.resize(count);
    for (auto& s : rendered) check(vkCreateSemaphore(device, &semci, nullptr, &s), "vkCreateSemaphore");
    swapchain_dirty = false;
}

void Display::Impl::destroy_swapchain() {
    if (!device) return;
    vkDeviceWaitIdle(device);
    for (auto s : rendered) vkDestroySemaphore(device, s, nullptr);
    rendered.clear();
    for (auto v : views) vkDestroyImageView(device, v, nullptr);
    views.clear();
    if (swapchain) vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
    images.clear();
}

void Display::Impl::ensure_frame_image(int w, int h) {
    if (frame && frame_w == w && frame_h == h) return;
    vkDeviceWaitIdle(device);
    if (frame) vmaDestroyImage(allocator, frame, frame_mem);
    if (staging) vmaDestroyBuffer(allocator, staging, staging_mem);
    frame_w = w;
    frame_h = h;

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    check(vmaCreateImage(allocator, &ici, &aci, &frame, &frame_mem, nullptr), "frame image");

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = static_cast<VkDeviceSize>(w) * h * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO;
    sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info;
    check(vmaCreateBuffer(allocator, &bci, &sai, &staging, &staging_mem, &info), "staging buffer");
    staging_ptr = info.pMappedData;
}

void Display::Impl::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;

    // A system UI font where one exists (it is not redistributed); otherwise ImGui's built-in font, enlarged.
    const float dpi = std::max(1.0f, SDL_GetWindowDisplayScale(window));
    ui_dpi = dpi;
    const float size = std::round(18.0f * dpi);
    nx::vector<nx::string> fonts;
    if (const char* windir = std::getenv("WINDIR")) fonts.push_back(path_join(windir, "Fonts", "segoeui.ttf"));
    fonts.push_back("/system/fonts/Roboto-Regular.ttf");
    fonts.push_back("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
    // Latin, Cyrillic and Greek (names of language packs and saves); the ranges must outlive the atlas build.
    static ImVector<ImWchar> ranges;
    if (ranges.empty()) {
        ImFontGlyphRangesBuilder builder;
        builder.AddRanges(io.Fonts->GetGlyphRangesCyrillic());  // with Basic Latin and Latin-1
        builder.AddRanges(io.Fonts->GetGlyphRangesGreek());
        builder.BuildRanges(&ranges);
    }
    bool loaded = false;
    for (const auto& f : fonts) {
        if (!nx::fs::exists(f)) continue;
        if (io.Fonts->AddFontFromFileTTF(f.c_str(), size, nullptr, ranges.Data)) {
            loaded = true;
            break;
        }
    }
    if (!loaded) {
        ImFontConfig config;
        config.SizePixels = std::round(13.0f * dpi) * 1.5f;
        io.Fonts->AddFontDefault(&config);
    }
    ui_base_style = style;

    ImGui_ImplSDL3_InitForVulkan(window);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = ctx.api_version;
    info.Instance = instance;
    info.PhysicalDevice = gpu;
    info.Device = device;
    info.QueueFamily = queue_family;
    info.Queue = queue;
    info.DescriptorPoolSize = 64;  // the font atlas plus overlay textures
    info.MinImageCount = 2;
    info.ImageCount = std::max<std::uint32_t>(2, static_cast<std::uint32_t>(images.size()));
    info.UseDynamicRendering = true;
    info.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchain_format;
    if (!ImGui_ImplVulkan_Init(&info)) throw std::runtime_error("ImGui Vulkan backend initialisation failed");

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod = 1.0f;
    check(vkCreateSampler(device, &sci, nullptr, &ui_sampler), "vkCreateSampler");
    imgui = true;
}

void Display::Impl::shutdown_imgui() {
    if (!imgui) return;
    for (auto& [set, t] : ui_textures) {
        ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(set));
        vkDestroyImageView(device, t.view, nullptr);
        vmaDestroyImage(allocator, t.image, t.memory);
    }
    ui_textures.clear();
    vkDestroySampler(device, ui_sampler, nullptr);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    imgui = false;
}

void Display::Impl::update_pad() {
    // Last-used device wins: keyboard and gamepad are combined, analog comes from whichever moved.
    std::uint32_t buttons = keyboard_buttons | gamepad_buttons;
    float x = key_x, y = key_y;
    if (gamepad) {
        const float gx = SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTX) / 32767.0f;
        const float gy = SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_LEFTY) / 32767.0f;
        if (std::abs(gx) > 0.2f || std::abs(gy) > 0.2f) {
            x = gx;
            y = gy;
        }
    }
    if (touch_mode) {
        buttons |= touch.buttons();
        float tx = 0, ty = 0;
        if (touch.stick(tx, ty)) {
            x = tx;
            y = ty;
        }
    }
    // Camera look: the right stick turns at up to kStickTurn per poll (one poll per vblank), a drag across the
    // picture by the screen height turns half way round.
    constexpr float kPi = 3.14159265f, kStickTurn = 150.0f / 60.0f * kPi / 180.0f, kDragTurn = kPi;
    pad.look_yaw = pad.look_pitch = 0;
    if (gamepad) {
        const auto curve = [](float v) {  // dead zone, then quadratic for fine control near the centre
            const float a = std::max(0.0f, (std::abs(v) - 0.15f) / 0.85f);
            return std::copysign(a * a, v);
        };
        pad.look_yaw = curve(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_RIGHTX) / 32767.0f) * kStickTurn;
        pad.look_pitch = -curve(SDL_GetGamepadAxis(gamepad, SDL_GAMEPAD_AXIS_RIGHTY) / 32767.0f) * kStickTurn;
    }
    float drag_x = 0, drag_y = 0;
    touch.take_drag(drag_x, drag_y);
    if (touch_mode && !ui_modal) {
        pad.look_yaw += drag_x * kDragTurn;
        pad.look_pitch -= drag_y * kDragTurn;  // like a mouse: dragging up looks up
    }
    pad.buttons = buttons;
    pad.lx = static_cast<std::uint8_t>(std::clamp(128.0f + x * 127.0f, 0.0f, 255.0f));
    pad.ly = static_cast<std::uint8_t>(std::clamp(128.0f + y * 127.0f, 0.0f, 255.0f));
}

// ---------------------------------------------------------------------------------------------

bool SDLCALL Display::Impl::lifecycle_watch(void* self, SDL_Event* e) {
    auto& d = *static_cast<Impl*>(self);
    switch (e->type) {
        case SDL_EVENT_WILL_ENTER_BACKGROUND:
            if (!d.background) {  // Android destroys the native window: drop everything presented on it
                std::fprintf(stderr, "app: background\n");
                d.background = true;
                d.surface_wanted = false;
                d.drop_surface();
                d.touch.release_all();
                if (d.background_handler) d.background_handler(true);
            }
            break;
        case SDL_EVENT_DID_ENTER_FOREGROUND:
            // The new native window may not exist yet (Android creates it after resuming): poll_events() creates
            // the surface once it does, the game staying paused until then.
            if (d.background) {
                std::fprintf(stderr, "app: foreground\n");
                d.surface_wanted = true;
            }
            break;
        default: break;
    }
    return true;
}

Display::Display(const char* title, int width, int height) : impl_(nx::make_unique<Impl>()) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) throw std::runtime_error(nx::format("SDL_Init: {}", SDL_GetError()).c_str());
    SDL_AddEventWatch(&Impl::lifecycle_watch, impl_.get());
    SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
#ifdef __ANDROID__
    flags |= SDL_WINDOW_FULLSCREEN;  // immersive: hides the status and navigation bars
    // SDL requests "any orientation" for resizable windows, overriding the manifest; the game is landscape only.
    SDL_SetHint(SDL_HINT_ORIENTATIONS, "LandscapeLeft LandscapeRight");
#endif
    impl_->window = SDL_CreateWindow(title, width, height, flags);
    if (!impl_->window) throw std::runtime_error(nx::format("SDL_CreateWindow: {}", SDL_GetError()).c_str());
    impl_->create_device();
    impl_->create_swapchain();
    impl_->init_imgui();
}

Display::~Display() {
    auto& d = *impl_;
    SDL_RemoveEventWatch(&Impl::lifecycle_watch, impl_.get());
    if (d.device) {
        vkDeviceWaitIdle(d.device);
        d.shutdown_imgui();
        d.destroy_swapchain();
        if (d.frame) vmaDestroyImage(d.allocator, d.frame, d.frame_mem);
        if (d.staging) vmaDestroyBuffer(d.allocator, d.staging, d.staging_mem);
        vmaDestroyAllocator(d.allocator);
        vkDestroyFence(d.device, d.fence, nullptr);
        vkDestroySemaphore(d.device, d.acquired, nullptr);
        vkDestroyCommandPool(d.device, d.pool, nullptr);
        vkDestroyDevice(d.device, nullptr);
    }
    if (d.surface) vkDestroySurfaceKHR(d.instance, d.surface, nullptr);
    if (d.instance) vkDestroyInstance(d.instance, nullptr);
    if (d.gamepad) SDL_CloseGamepad(d.gamepad);
    if (d.window) SDL_DestroyWindow(d.window);
    SDL_Quit();
}

bool Display::poll_events() {
    auto& d = *impl_;
    SDL_Event e;
    bool running = true;
    const auto key_bit = [](SDL_Keycode k) -> std::uint32_t {
        switch (k) {
            case SDLK_UP: return kUp;
            case SDLK_DOWN: return kDown;
            case SDLK_LEFT: return kLeft;
            case SDLK_RIGHT: return kRight;
            case SDLK_K: return kCross;
            case SDLK_L: return kCircle;
            case SDLK_J: return kSquare;
            case SDLK_I: return kTriangle;
            case SDLK_Q: return kLTrigger;
            case SDLK_E: return kRTrigger;
            case SDLK_RETURN: return kStart;
            case SDLK_BACKSPACE: return kSelect;
            default: return 0;
        }
    };
    const auto pad_bit = [](Uint8 b) -> std::uint32_t {
        switch (b) {
            case SDL_GAMEPAD_BUTTON_SOUTH: return kCross;
            case SDL_GAMEPAD_BUTTON_EAST: return kCircle;
            case SDL_GAMEPAD_BUTTON_WEST: return kSquare;
            case SDL_GAMEPAD_BUTTON_NORTH: return kTriangle;
            case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return kLTrigger;
            case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return kRTrigger;
            case SDL_GAMEPAD_BUTTON_START: return kStart;
            case SDL_GAMEPAD_BUTTON_BACK: return kSelect;
            case SDL_GAMEPAD_BUTTON_DPAD_UP: return kUp;
            case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return kDown;
            case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return kLeft;
            case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return kRight;
            default: return 0;
        }
    };
    const auto glyphs_of = [](SDL_Gamepad* pad) {
        switch (SDL_GetGamepadType(pad)) {
            case SDL_GAMEPAD_TYPE_PS3:
            case SDL_GAMEPAD_TYPE_PS4:
            case SDL_GAMEPAD_TYPE_PS5: return ButtonGlyphs::PlayStation;
            case SDL_GAMEPAD_TYPE_XBOX360:
            case SDL_GAMEPAD_TYPE_XBOXONE: return ButtonGlyphs::Xbox;
            case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_PRO:
            case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_LEFT:
            case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_RIGHT:
            case SDL_GAMEPAD_TYPE_NINTENDO_SWITCH_JOYCON_PAIR: return ButtonGlyphs::Nintendo;
            default: break;
        }
        // Other pads by the label of their bottom face button; most generic ones are labelled like Xbox pads.
        switch (SDL_GetGamepadButtonLabel(pad, SDL_GAMEPAD_BUTTON_SOUTH)) {
            case SDL_GAMEPAD_BUTTON_LABEL_CROSS: return ButtonGlyphs::PlayStation;
            case SDL_GAMEPAD_BUTTON_LABEL_B: return ButtonGlyphs::Nintendo;
            default: return ButtonGlyphs::Xbox;
        }
    };
    for (auto it = d.injected.begin(); it != d.injected.end();) {  // scripted test touches
        if (--it->polls > 0) {
            ++it;
            continue;
        }
        SDL_Event up{};
        up.type = SDL_EVENT_FINGER_UP;
        up.tfinger.fingerID = it->id;
        up.tfinger.windowID = SDL_GetWindowID(d.window);
        SDL_PushEvent(&up);
        it = d.injected.erase(it);
    }
    int win_w = 0, win_h = 0;
    SDL_GetWindowSize(d.window, &win_w, &win_h);
    d.touch.set_size(static_cast<float>(win_w), static_cast<float>(win_h));
    d.touch.tick();
    while (SDL_PollEvent(&e)) {
        if (d.imgui) ImGui_ImplSDL3_ProcessEvent(&e);
        switch (e.type) {
            case SDL_EVENT_FINGER_DOWN:
                d.touch_mode = true;
                d.last_device = Impl::Device::Touch;
                if (!d.ui_modal) d.touch.finger_down(e.tfinger.fingerID, e.tfinger.x, e.tfinger.y);
                break;
            case SDL_EVENT_FINGER_MOTION:
                if (!d.ui_modal) d.touch.finger_motion(e.tfinger.fingerID, e.tfinger.x, e.tfinger.y);
                break;
            case SDL_EVENT_FINGER_UP:
            case SDL_EVENT_FINGER_CANCELED: d.touch.finger_up(e.tfinger.fingerID); break;
            case SDL_EVENT_GAMEPAD_AXIS_MOTION:
                if (std::abs(e.gaxis.value) > 16000) {
                    d.touch_mode = false;
                    d.last_device = Impl::Device::Gamepad;
                }
                break;
            case SDL_EVENT_QUIT: running = false; break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED: d.swapchain_dirty = true; break;
            case SDL_EVENT_KEY_DOWN:
            case SDL_EVENT_KEY_UP: {
                const bool down = e.type == SDL_EVENT_KEY_DOWN;
                if (down && !e.key.repeat) {
                    if (e.key.key == SDLK_ESCAPE || e.key.key == SDLK_F1 || e.key.key == SDLK_AC_BACK) d.menu_request = true;
                    if (e.key.key != SDLK_AC_BACK) {  // Android's back button is not a keyboard
                        d.touch_mode = false;
                        d.touch.release_all();
                    }
                    if (e.key.key == SDLK_F11 || (e.key.key == SDLK_RETURN && (e.key.mod & SDL_KMOD_ALT))) {
                        set_fullscreen(!fullscreen());
                        break;  // Alt+Enter is not Start
                    }
                }
                if (const auto bit = key_bit(e.key.key)) {
                    d.keyboard_buttons = down ? d.keyboard_buttons | bit : d.keyboard_buttons & ~bit;
                    if (down) d.last_device = Impl::Device::Keyboard;
                }
                const float v = down ? 1.0f : 0.0f;
                if (e.key.key == SDLK_A) d.key_x = -v;
                if (e.key.key == SDLK_D) d.key_x = v;
                if (e.key.key == SDLK_W) d.key_y = -v;
                if (e.key.key == SDLK_S) d.key_y = v;
                break;
            }
            case SDL_EVENT_GAMEPAD_ADDED:
                if (!d.gamepad && (d.gamepad = SDL_OpenGamepad(e.gdevice.which))) d.pad_glyphs = glyphs_of(d.gamepad);
                break;
            case SDL_EVENT_GAMEPAD_REMOVED:
                if (d.gamepad && SDL_GetGamepadID(d.gamepad) == e.gdevice.which) {
                    SDL_CloseGamepad(d.gamepad);
                    d.gamepad = nullptr;
                    d.gamepad_buttons = 0;
                    d.pad_glyphs = ButtonGlyphs::PlayStation;
                    if (d.last_device == Impl::Device::Gamepad) d.last_device = Impl::Device::None;
                }
                break;
            case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
            case SDL_EVENT_GAMEPAD_BUTTON_UP: {
                const bool down = e.gbutton.down;
                if (down) {
                    d.touch_mode = false;
                    d.touch.release_all();
                }
                if (e.gbutton.button == SDL_GAMEPAD_BUTTON_GUIDE && down) d.menu_request = true;
                if (e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK) d.stick_l = down;
                if (e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) d.stick_r = down;
                if (down && d.stick_l && d.stick_r &&
                    (e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK || e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK))
                    d.menu_request = true;
                const auto bit = pad_bit(e.gbutton.button);
                d.gamepad_buttons = e.gbutton.down ? d.gamepad_buttons | bit : d.gamepad_buttons & ~bit;
                break;
            }
            default: break;
        }
    }
    if (d.background && d.surface_wanted && d.try_create_surface()) {
        d.background = false;
        d.surface_wanted = false;
        std::fprintf(stderr, "app: surface recreated\n");
        if (d.background_handler) d.background_handler(false);
    }
    d.update_pad();
    return running;
}

PadState Display::pad() const { return impl_->pad; }

bool Display::in_background() const { return impl_->background; }

void Display::set_background_handler(nx::unique_function<void(bool)> handler) {
    impl_->background_handler = std::move(handler);
}

const VulkanContext& Display::vulkan() const { return impl_->ctx; }

// A surface for the window, unless the platform has no window for it yet (Android between resuming and creating the
// new native window: SDL would hand Vulkan a null window, giving a surface that never has a size).
bool Display::Impl::try_create_surface() {
#ifdef __ANDROID__
    if (!SDL_GetPointerProperty(SDL_GetWindowProperties(window), SDL_PROP_WINDOW_ANDROID_WINDOW_POINTER, nullptr))
        return false;
#endif
    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        surface = VK_NULL_HANDLE;
        return false;
    }
    swapchain_dirty = true;
    return true;
}

void Display::Impl::drop_surface() {
    destroy_swapchain();
    if (surface) vkDestroySurfaceKHR(instance, surface, nullptr);
    surface = VK_NULL_HANDLE;
}

// After a failed acquire or present: recreate what is needed; the frame is dropped.
void Display::Impl::present_failed(const char* what, VkResult r) {
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        swapchain_dirty = true;
        return;
    }
    if (present_errors < 20) {
        ++present_errors;
        std::fprintf(stderr, "present: %s failed (VkResult %d)\n", what, static_cast<int>(r));
    }
    if (r == VK_TIMEOUT || r == VK_NOT_READY) return;  // still busy: try again next frame (recreating would wait)
    if (r == VK_ERROR_SURFACE_LOST_KHR) {
        drop_surface();
        surface_wanted = true;  // recreated by the next begin_present
    } else {
        swapchain_dirty = true;
    }
}

// Recreates the swapchain if needed and waits for the previous presentation; false while nothing can be shown
// (minimised, in the background, no surface yet, or the GPU still busy with the last frame after a second).
bool Display::Impl::begin_present() {
    if (background) return false;
    if (!surface && (!surface_wanted || !try_create_surface())) return false;
    surface_wanted = false;
    if (swapchain_dirty || !swapchain) {
        destroy_swapchain();
        create_swapchain();
        if (!swapchain) return false;
    }
    if (const VkResult r = vkWaitForFences(device, 1, &fence, VK_TRUE, 1'000'000'000ull); r != VK_SUCCESS) {
        present_failed("waiting for the previous frame", r);
        return false;
    }
    return true;
}

void Display::Impl::blit_and_present(VkImage source, int x, int y, int width, int height,
                                     nx::function_ref<void(VkCommandBuffer)> prepare) {
    std::uint32_t index = 0;
    // Anything but success leaves `acquired` unsignalled: submitting work that waits on it would never finish and
    // every later frame would wait for it (a frozen game).
    VkResult r = vkAcquireNextImageKHR(device, swapchain, 1'000'000'000ull, acquired, VK_NULL_HANDLE, &index);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
        present_failed("acquiring a swapchain image", r);
        return;
    }
    ImDrawData* overlay = nullptr;
    if (imgui) {
        // Scale the UI with the window height (a 544-line window gets about 16 px text), up to the DPI scale.
#ifdef __ANDROID__
        constexpr float kTouchFactor = 1.35f;  // finger-sized widgets
#else
        constexpr float kTouchFactor = 1.0f;
#endif
        const float scale = std::clamp(std::min(ui_dpi, static_cast<float>(extent.height) / 600.0f) * kTouchFactor, 0.75f,
                                       ui_dpi * kTouchFactor);
        if (scale != ui_scale) {
            ui_scale = scale;
            ImGui::GetStyle() = ui_base_style;
            ImGui::GetStyle().ScaleAllSizes(scale);
            ImGui::GetIO().FontGlobalScale = scale / ui_dpi;
        }
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();
        if (ui) ui();
        if (touch_mode && !ui_modal) touch.draw(ImGui::GetForegroundDrawList(), resolved_glyphs());
        ImGui::Render();
        overlay = ImGui::GetDrawData();
        if (overlay->TotalVtxCount == 0) overlay = nullptr;
    }
    last_source = source;
    last_x = x;
    last_y = y;
    last_w = width;
    last_h = height;
    vkResetFences(device, 1, &fence);
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    if (prepare) prepare(cmd);

    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImage target = images[index];
    b.image = target;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    // ALL_COMMANDS also orders the blit after earlier submissions writing `source` (e.g. the GE renderer).
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    const VkClearColorValue black{};
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    // Fit the frame into the window, keeping its aspect ratio.
    const double scale = std::min(static_cast<double>(extent.width) / width, static_cast<double>(extent.height) / height);
    const int dw = static_cast<int>(width * scale), dh = static_cast<int>(height * scale);
    const int dx = (static_cast<int>(extent.width) - dw) / 2, dy = (static_cast<int>(extent.height) - dh) / 2;
    VkImageBlit blit{};
    blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[0] = {x, y, 0};
    blit.srcOffsets[1] = {x + width, y + height, 1};
    blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.dstOffsets[0] = {dx, dy, 0};
    blit.dstOffsets[1] = {dx + dw, dy + dh, 1};
    vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                   VK_FILTER_LINEAR);
    if (overlay) {
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
        VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        color.imageView = views[index];
        color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &color;
        ctx.begin_rendering(cmd, &ri);
        ImGui_ImplVulkan_RenderDrawData(overlay, cmd);
        ctx.end_rendering(cmd);
        b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &b);
    } else {
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
    }
    VkBuffer readback = VK_NULL_HANDLE;
    VmaAllocation readback_mem = VK_NULL_HANDLE;
    void* readback_ptr = nullptr;
    if (capture && can_capture) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = static_cast<VkDeviceSize>(extent.width) * extent.height * 4;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info;
        check(vmaCreateBuffer(allocator, &bci, &aci, &readback, &readback_mem, &info), "capture buffer");
        readback_ptr = info.pMappedData;
        b.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                             1, &b);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(cmd, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, &copy);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.dstAccessMask = 0;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
    }
    vkEndCommandBuffer(cmd);

    const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &acquired;
    si.pWaitDstStageMask = &wait_stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &rendered[index];
    check(vkQueueSubmit(queue, 1, &si, fence), "vkQueueSubmit");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &rendered[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &index;
    r = vkQueuePresentKHR(queue, &pi);
    // SUBOPTIMAL is expected with the identity transform on a rotated display; size changes arrive as events.
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) present_failed("presenting", r);

    if (readback) {
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);  // testing only
        vmaInvalidateAllocation(allocator, readback_mem, 0, VK_WHOLE_SIZE);
        nx::vector<std::uint32_t> rgba(static_cast<std::size_t>(extent.width) * extent.height);
        std::memcpy(rgba.data(), readback_ptr, rgba.size() * 4);
        const bool bgra = swapchain_format == VK_FORMAT_B8G8R8A8_UNORM || swapchain_format == VK_FORMAT_B8G8R8A8_SRGB;
        for (auto& px : rgba) {
            if (bgra) px = (px & 0xFF00FF00u) | ((px >> 16) & 0xFFu) | ((px & 0xFFu) << 16);
            px |= 0xFF000000u;
        }
        vmaDestroyBuffer(allocator, readback, readback_mem);
        auto done = std::move(capture);
        capture.reset();
        done(rgba.data(), static_cast<int>(extent.width), static_cast<int>(extent.height));
    }
}

void Display::present(const std::uint32_t* rgba, int width, int height) {
    auto& d = *impl_;
    if (!d.begin_present()) return;
    d.ensure_frame_image(width, height);
    std::memcpy(d.staging_ptr, rgba, static_cast<std::size_t>(width) * height * 4);
    check(vmaFlushAllocation(d.allocator, d.staging_mem, 0, VK_WHOLE_SIZE), "flush display staging");
    d.blit_and_present(d.frame, 0, 0, width, height, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.image = d.frame;
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1};
        vkCmdCopyBufferToImage(cmd, d.staging, d.frame, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &b);
    });
}

void Display::present_image(VkImage image, int width, int height, int x, int y) {
    auto& d = *impl_;
    if (!d.begin_present()) return;
    d.blit_and_present(image, x, y, width, height, {});
}

float Display::aspect() const {
    const VkExtent2D e = impl_->extent;
    return e.width && e.height ? static_cast<float>(e.width) / static_cast<float>(e.height) : 0.0f;
}

void Display::forget_last_image() { impl_->last_source = VK_NULL_HANDLE; }

bool Display::present_again() {
    auto& d = *impl_;
    if (!d.last_source || !d.begin_present()) return false;
    d.blit_and_present(d.last_source, d.last_x, d.last_y, d.last_w, d.last_h, {});
    return true;
}

void Display::capture_next_present(CaptureHandler done) {
    impl_->capture = std::move(done);
}

void Display::inject_key(std::uint32_t keycode) {
    const SDL_Scancode scancode = SDL_GetScancodeFromKey(keycode, nullptr);
    for (const bool down : {true, false}) {
        SDL_Event e{};
        e.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
        e.key.windowID = SDL_GetWindowID(impl_->window);
        e.key.key = keycode;
        e.key.scancode = scancode;
        e.key.down = down;
        SDL_PushEvent(&e);
    }
}

void Display::set_ui(nx::unique_function<void()> ui) { impl_->ui = std::move(ui); }

bool Display::screen_held() const { return impl_->touch_mode && !impl_->ui_modal && impl_->touch.free_touch(); }

bool Display::take_menu_request() {
    const bool touch = impl_->touch.take_menu_request();
    return std::exchange(impl_->menu_request, false) || touch;
}

void Display::set_ui_modal(bool modal) {
    if (modal && !impl_->ui_modal) impl_->touch.release_all();
    impl_->ui_modal = modal;
}

void Display::set_touch_controls(bool visible) {
    impl_->touch_mode = visible;
    if (visible) impl_->last_device = Impl::Device::Touch;
}

bool Display::touch_controls() const { return impl_->touch_mode; }

bool Display::gamepad_connected() const { return impl_->gamepad != nullptr; }

void Display::set_button_glyphs(ButtonGlyphs glyphs) { impl_->glyphs = glyphs; }

ButtonGlyphs Display::button_glyphs() const { return impl_->resolved_glyphs(); }

void Display::set_minimized(bool minimized) {
    if (minimized) SDL_MinimizeWindow(impl_->window);
    else SDL_RestoreWindow(impl_->window);
}

void Display::inject_finger(float x, float y, int polls) {
    static SDL_FingerID next = 0x7F000000;
    SDL_Event e{};
    e.type = SDL_EVENT_FINGER_DOWN;
    e.tfinger.fingerID = ++next;
    e.tfinger.windowID = SDL_GetWindowID(impl_->window);
    e.tfinger.x = x;
    e.tfinger.y = y;
    SDL_PushEvent(&e);
    impl_->injected.push_back({next, std::max(1, polls)});
}

void Display::set_fullscreen(bool on) { SDL_SetWindowFullscreen(impl_->window, on); }

void Display::set_vsync(bool on) {
    if (impl_->vsync == on) return;
    impl_->vsync = on;
    impl_->swapchain_dirty = true;
}

void Display::set_refresh_rate(float hz) {
    if (impl_->refresh_rate == hz) return;
    impl_->refresh_rate = hz;
#ifdef __ANDROID__
    // The window's preferred refresh rate is set on the UI thread (GameActivity.setPreferredRefreshRate posts it):
    // asking from this thread while the UI thread waits on SDL's surface handling would deadlock.
    auto* env = static_cast<JNIEnv*>(SDL_GetAndroidJNIEnv());
    auto activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!env || !activity) return;
    jclass cls = env->GetObjectClass(activity);
    if (jmethodID method = env->GetMethodID(cls, "setPreferredRefreshRate", "(F)V"))
        env->CallVoidMethod(activity, method, static_cast<jfloat>(hz));
    if (env->ExceptionCheck()) env->ExceptionClear();
    env->DeleteLocalRef(cls);
    env->DeleteLocalRef(activity);
#endif
}

bool Display::fullscreen() const { return (SDL_GetWindowFlags(impl_->window) & SDL_WINDOW_FULLSCREEN) != 0; }

std::uint64_t Display::create_texture(const std::uint32_t* rgba, int width, int height) {
    auto& d = *impl_;
    if (!d.imgui || width <= 0 || height <= 0) return 0;
    Impl::UiTexture t{};
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = {static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    check(vmaCreateImage(d.allocator, &ici, &aci, &t.image, &t.memory, nullptr), "overlay texture");
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(d.device, &vci, nullptr, &t.view), "overlay texture view");

    // One-off upload through a temporary staging buffer; waits for the GPU (textures are created rarely).
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(height) * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO;
    sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_mem = VK_NULL_HANDLE;
    VmaAllocationInfo info;
    check(vmaCreateBuffer(d.allocator, &bci, &sai, &staging, &staging_mem, &info), "overlay staging buffer");
    std::memcpy(info.pMappedData, rgba, bytes);
    check(vmaFlushAllocation(d.allocator, staging_mem, 0, bytes), "flush overlay staging");

    vkWaitForFences(d.device, 1, &d.fence, VK_TRUE, UINT64_MAX);
    vkResetCommandBuffer(d.cmd, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(d.cmd, &bi);
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.image = t.image;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = ici.extent;
    vkCmdCopyBufferToImage(d.cmd, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(d.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr,
                         1, &b);
    vkEndCommandBuffer(d.cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &d.cmd;
    vkResetFences(d.device, 1, &d.fence);
    check(vkQueueSubmit(d.queue, 1, &si, d.fence), "vkQueueSubmit");
    vkWaitForFences(d.device, 1, &d.fence, VK_TRUE, UINT64_MAX);
    vmaDestroyBuffer(d.allocator, staging, staging_mem);

    const VkDescriptorSet set = ImGui_ImplVulkan_AddTexture(d.ui_sampler, t.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    const auto id = reinterpret_cast<std::uint64_t>(set);
    d.ui_textures.emplace(id, t);
    return id;
}

void Display::destroy_texture(std::uint64_t texture) {
    auto& d = *impl_;
    const auto it = d.ui_textures.find(texture);
    if (it == d.ui_textures.end()) return;
    vkDeviceWaitIdle(d.device);
    ImGui_ImplVulkan_RemoveTexture(reinterpret_cast<VkDescriptorSet>(texture));
    vkDestroyImageView(d.device, it->second.view, nullptr);
    vmaDestroyImage(d.allocator, it->second.image, it->second.memory);
    d.ui_textures.erase(it);
}

}  // namespace psprecore::platform
