# mlx-rocm MVP — standalone CMake.
#
# Host-only build (e.g. host-a, no ROCm):
#   cmake -S . -B build && cmake --build build
#   ./build/gemm_smoke host
#
# ROCm build (e.g. guest-a, RX 9070 XT / gfx1201):
#   cmake -S . -B build -DMLX_ROCM_WITH_HIP=ON && cmake --build build
#   ./build/gemm_smoke hip
#
# MLX_ROCM_WITH_HIP: AUTO (default) | ON | OFF.

cmake_minimum_required(VERSION 3.20)
project(mlx_rocm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------- ROCm detection ------------------------------
# Manual find_path/find_library (distro-agnostic: Fedora/TheRock ship the
# hip CMake configs in non-standard places; the libs are always plain .so).
if(NOT DEFINED MLX_ROCM_WITH_HIP)
  set(MLX_ROCM_WITH_HIP AUTO)
endif()

find_path(MLX_ROCM_HIP_INCLUDE_DIR
          NAMES hip/hip_runtime.h
          PATHS /usr/include /opt/rocm/include /opt/rocm/hip/include)
find_path(MLX_ROCM_HIPBLASLT_INCLUDE_DIR
          NAMES hipblaslt/hipblaslt.h
          PATHS /usr/include /opt/rocm/include)
find_library(MLX_ROCM_HIP_LIBRARY NAMES amdhip64
             PATHS /usr/lib64 /usr/lib/x86_64-linux-gnu /opt/rocm/lib)
find_library(MLX_ROCM_HIPBLASLT_LIBRARY NAMES hipblaslt
             PATHS /usr/lib64 /usr/lib/x86_64-linux-gnu /opt/rocm/lib)

set(MLX_ROCM_HAVE_HIP FALSE)
if(MLX_ROCM_HIP_INCLUDE_DIR AND MLX_ROCM_HIPBLASLT_INCLUDE_DIR
   AND MLX_ROCM_HIP_LIBRARY AND MLX_ROCM_HIPBLASLT_LIBRARY)
  set(MLX_ROCM_HAVE_HIP TRUE)
endif()

string(TOUPPER "${MLX_ROCM_WITH_HIP}" MLX_ROCM_WITH_HIP_U)

if(MLX_ROCM_WITH_HIP_U STREQUAL "AUTO")
  set(MLX_ROCM_WITH_HIP "${MLX_ROCM_HAVE_HIP}")
endif()

set(MLX_ROCM_HIP_BUILT FALSE)
if(MLX_ROCM_WITH_HIP)
  if(NOT MLX_ROCM_HAVE_HIP)
    message(FATAL_ERROR
            "MLX_ROCM_WITH_HIP=ON but hip/hipBLASLt were not found "
            "(looked for hip/hip_runtime.h, hipblaslt/hipblaslt.h, "
            "libamdhip64, libhipblaslt).")
  endif()
  set(MLX_ROCM_HIP_BUILT TRUE)
endif()

message(STATUS "mlx-rocm: WITH_HIP=${MLX_ROCM_WITH_HIP} (found ROCm: ${MLX_ROCM_HAVE_HIP})")

# ------------------------------- targets ------------------------------------
add_library(mlx_rocm SHARED src/cpp/mlx_rocm/gemm/hipblaslt.cpp)
target_include_directories(mlx_rocm PUBLIC src/cpp)
set_target_properties(mlx_rocm PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MLX_ROCM_HIP_BUILT)
  target_compile_definitions(mlx_rocm PUBLIC MLX_ROCM_WITH_HIP=1)
  # host-compiler (g++) build of the HIP headers; hipcc defines this itself.
  target_compile_definitions(mlx_rocm PRIVATE __HIP_PLATFORM_AMD__=1)
  target_include_directories(mlx_rocm SYSTEM PRIVATE
                             ${MLX_ROCM_HIP_INCLUDE_DIR}
                             ${MLX_ROCM_HIPBLASLT_INCLUDE_DIR})
  target_link_libraries(mlx_rocm PRIVATE
                        ${MLX_ROCM_HIP_LIBRARY}
                        ${MLX_ROCM_HIPBLASLT_LIBRARY})
endif()

add_executable(gemm_smoke examples/gemm_smoke.cpp)
target_link_libraries(gemm_smoke PRIVATE mlx_rocm)
if(MLX_ROCM_HIP_BUILT)
  target_compile_definitions(gemm_smoke PRIVATE MLX_ROCM_GEMM_HIP_BUILT=1)
endif()

if(MLX_ROCM_HIP_BUILT)
  # Copy the lib next to the driver so the smoke run and the python ctypes
  # loader can find it without LD_LIBRARY_PATH acrobatics.
  add_custom_command(TARGET mlx_rocm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mlx_rocm>
            ${CMAKE_BINARY_DIR}/libmlx_rocm.so)
endif()
