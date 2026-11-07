# Runtime for recompiled code: guest memory, HLE kernel and system libraries.
add_library(psprecore_runtime STATIC
    src/recomp_runtime.c
    src/memory.cpp
    src/kernel.cpp
    src/hle.cpp
    src/hle_imports.cpp
    src/hle_threadman.cpp
    src/hle_sysmem.cpp
    src/hle_io.cpp
    src/hle_display.cpp
    src/hle_audio.cpp
    src/hle_mpeg.cpp
    src/hle_psmf.cpp
    src/hle_atrac.cpp
    src/hle_savedata.cpp
    src/hle_font.cpp
    src/crash_handler.cpp
    src/vfpu.cpp
    src/ge.cpp
    src/image_io.cpp
    src/gpu/button_icons.cpp
    src/gpu/ui_font.cpp
    src/gpu/texture_replacement.cpp
    src/gpu/texture_collection.cpp
    src/gpu/texture.cpp
    src/gpu/soft_renderer.cpp
    src/gpu/vertex_pipeline.cpp
    src/audio/sas.cpp
    src/audio/reverb.cpp
    src/audio/vorbis.cpp
    src/media/av1_video.cpp
)
if(WIN32)
    target_link_libraries(psprecore_runtime PRIVATE dbghelp)
endif()
add_library(psprecore::runtime ALIAS psprecore_runtime)
target_include_directories(psprecore_runtime PUBLIC include PRIVATE src)
target_link_libraries(psprecore_runtime PUBLIC psprecore::mips psprecore::formats nx::foundation PRIVATE psprecore_stb psprecore_dav1d)
psprecore_target_defaults(psprecore_runtime)

# The front end (psprecore/platform/frontend.hpp): window, audio, renderers, pause menu. A game's main() runs it;
# games build their executable with psprecore_add_game (cmake/PsprecoreGame.cmake).
if(PSPRECORE_BUILD_GAME)
    add_library(psprecore_frontend STATIC
        src/frontend.cpp
        src/platform/display_vulkan.cpp
        src/platform/audio_miniaudio.cpp
        src/platform/gpu_vulkan.cpp
        src/platform/sampling_profiler.cpp
        src/platform/memory_watch.cpp
        src/platform/mods.cpp
        src/platform/overlay.cpp
        src/platform/settings.cpp
        src/platform/touch_controls.cpp
        src/platform/button_glyphs.cpp
    )
    add_library(psprecore::frontend ALIAS psprecore_frontend)
    target_include_directories(psprecore_frontend PRIVATE src)
    psprecore_add_shader(psprecore_frontend ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ge.slang vertex_main vertex ge_vert)
    psprecore_add_shader(psprecore_frontend ${CMAKE_CURRENT_SOURCE_DIR}/shaders/ge.slang fragment_main fragment ge_frag)
    psprecore_add_shader(psprecore_frontend ${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy.slang vertex_main vertex copy_vert)
    psprecore_add_shader(psprecore_frontend ${CMAKE_CURRENT_SOURCE_DIR}/shaders/copy.slang fragment_main fragment copy_frag)
    target_link_libraries(psprecore_frontend PUBLIC psprecore::runtime psprecore::formats SDL3::SDL3-static psprecore_vulkan psprecore_imgui
                                               psprecore_miniaudio)
    if(ANDROID)
        target_link_libraries(psprecore_frontend PUBLIC android log)
    endif()
    psprecore_target_defaults(psprecore_frontend)
    # Global operator new/delete (mimalloc) belong in the executable itself, not in a library the linker may skip.
    set(PSPRECORE_ALLOCATOR_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/platform/allocator.cpp" CACHE INTERNAL "")
endif()
