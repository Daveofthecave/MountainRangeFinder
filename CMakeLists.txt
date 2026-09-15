# CMakeLists.txt
# Portable build for MountainRangeFinder (Linux + Windows). The Makefile stays
# the canonical Linux build; this file exists so Windows can build with
# MSVC + nvcc without Makefile gymnastics, and so IDEs can open the project.
#
#   Linux:   cmake -B build && cmake --build build -j
#   Windows: cmake -B build -G "Visual Studio 17 2022" -A x64
#            cmake --build build --config Release
#
# Options:
#   -DMRF_ARCHS="80;86;89;89-virtual"   CUDA architectures (Turing: "75")
#   -DMRF_FAST_MATH=ON                  add --use_fast_math (run selftest after)
#   -DMRF_STATIC_RUNTIME=ON             static CUDA/C++ runtimes (release builds)

cmake_minimum_required(VERSION 3.24)
project(MountainRangeFinder LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MRF_FAST_MATH "Enable --use_fast_math on the CUDA pipeline" OFF)
option(MRF_STATIC_RUNTIME "Static-link the CUDA and C++ runtimes" OFF)

# sm_80 (Ampere) through sm_89 (Ada) as native SASS, plus compute_89 PTX so
# newer cards (Blackwell) JIT-compile it on first launch. Mirrors the
# Makefile's -gencode list.
set(MRF_ARCHS "80;86;89;89-virtual" CACHE STRING "CUDA architectures to build for")

# --- cubiomes (C) ---------------------------------------------------------
file(GLOB CUBIOMES_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cubiomes/*.c")
list(FILTER CUBIOMES_SOURCES EXCLUDE REGEX "tests\\.c$")

add_library(cubiomes STATIC ${CUBIOMES_SOURCES})
target_include_directories(cubiomes PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/cubiomes")
if(MSVC)
  target_compile_options(cubiomes PRIVATE /wd4244 /wd4267 /wd4305 /wd4996)
else()
  target_compile_definitions(cubiomes PRIVATE _POSIX_C_SOURCE=199309L)
endif()

# --- the searcher ---------------------------------------------------------
add_executable(mountain_rangefinder main.cpp cpu.cpp probe.cpp gpu.cu)
target_include_directories(mountain_rangefinder PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cubiomes")
target_link_libraries(mountain_rangefinder PRIVATE cubiomes)
set_target_properties(mountain_rangefinder PROPERTIES CUDA_ARCHITECTURES "${MRF_ARCHS}")

find_package(Threads REQUIRED)
target_link_libraries(mountain_rangefinder PRIVATE Threads::Threads)
if(UNIX)
  target_link_libraries(mountain_rangefinder PRIVATE m)
endif()

if(MRF_FAST_MATH)
  target_compile_options(mountain_rangefinder PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

if(MRF_STATIC_RUNTIME)
  set_target_properties(mountain_rangefinder PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  if(MSVC)
    # Every target must share one CRT flavor: the static CUDA runtime carries
    # LIBCMT objects, so /MD objects anywhere in the graph re-trigger LNK4098.
    set_target_properties(mountain_rangefinder cubiomes PROPERTIES
      MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  elseif(UNIX)
    target_link_options(mountain_rangefinder PRIVATE -static-libstdc++ -static-libgcc)
  endif()
endif()

# --- selftest --------------------------------------------------------------
add_executable(selftest selftest.cu)
target_include_directories(selftest PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/cubiomes")
target_link_libraries(selftest PRIVATE cubiomes)
set_target_properties(selftest PROPERTIES CUDA_ARCHITECTURES "${MRF_ARCHS}")

# selftest links the /MT cubiomes objects, so it needs the same CRT flavor.
# This lives here (not in the block above) because selftest doesn't exist
# until this line; set_target_properties requires all targets to be defined.
if(MRF_STATIC_RUNTIME AND MSVC)
  set_target_properties(selftest PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

enable_testing()
add_test(NAME selftest COMMAND selftest)
