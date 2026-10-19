find_package(Threads REQUIRED)

# Device runtime seams. Every consumer links ${NINFER_CUDART_TARGET} for the device runtime and
# ${NINFER_BLAS_TARGET} for the single dense-GEMM call site; the two backends resolve them to
# their own packages. NINFER_NVTX_TARGET and NINFER_DRIVER_TARGET exist only on CUDA today:
# ROCm has no NVTX package, and the HIP driver API is part of the HIP runtime library.
if(NINFER_BACKEND STREQUAL "hip")
  find_package(hip REQUIRED CONFIG)
  find_package(hipblas REQUIRED CONFIG)
  # hipCUB for the cub/ shadow headers. Not in the system search paths (unlike the ROCm
  # packages, which Debian installs under /usr/lib/*/cmake), so the build passes its location
  # through CMAKE_PREFIX_PATH.
  find_package(hipcub REQUIRED CONFIG)
  get_target_property(NINFER_HIPCUB_INCLUDE_DIR hip::hipcub INTERFACE_INCLUDE_DIRECTORIES)
  set(NINFER_CUDART_TARGET hip::host)
  set(NINFER_BLAS_TARGET roc::hipblas)
  add_library(ninfer_nvtx_stub INTERFACE)
  set(NINFER_NVTX_TARGET ninfer_nvtx_stub)
  set(NINFER_DRIVER_TARGET hip::host)
else()
  find_package(CUDAToolkit REQUIRED)
  # Windows links the static CUDA runtime so shipped binaries do not depend on a cudart DLL.
  if(WIN32)
    set(NINFER_CUDART_TARGET CUDA::cudart_static)
  else()
    set(NINFER_CUDART_TARGET CUDA::cudart)
  endif()
  set(NINFER_BLAS_TARGET CUDA::cublas)
  set(NINFER_NVTX_TARGET CUDA::nvtx3)
  set(NINFER_DRIVER_TARGET CUDA::cuda_driver)
endif()

# vcpkg (the Windows build) exposes FFmpeg and libcurl through CMake packages; Linux uses
# pkg-config. Both paths produce the same two consumer targets.
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(ninfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(ninfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(NINFER_FFMPEG_TARGET ninfer_ffmpeg_dependencies)
else()
  find_package(PkgConfig REQUIRED)
  # Keep the floors the previous root build required: src/media/decode/decode.cpp is written
  # against these ABIs, so an older installation must fail at configure rather than at link.
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(NINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
endif()

# Repository-pinned header dependencies. No configure-time downloads.
add_library(ninfer::json INTERFACE IMPORTED GLOBAL)
target_include_directories(ninfer::json INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party)

# Source base for the custom-template frontend; consumers will link it explicitly.
add_subdirectory(third_party/llama-jinja EXCLUDE_FROM_ALL)

# CPU grammar source base; consumers will link it explicitly.
add_subdirectory(third_party/xgrammar EXCLUDE_FROM_ALL)

if(NINFER_BUILD_PRODUCT_SUPPORT)
  # Media acquisition uses CURLOPT_PROTOCOLS_STR and CURLOPT_REDIR_PROTOCOLS_STR,
  # introduced in libcurl 7.85 (not merely the version of the maintainer environment).
  if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
    find_package(CURL 7.85 REQUIRED)
    set(NINFER_CURL_TARGET CURL::libcurl)
  else()
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(NINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
  add_library(ninfer::httplib INTERFACE IMPORTED GLOBAL)
  target_include_directories(ninfer::httplib INTERFACE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  add_subdirectory(third_party/spdlog)
endif()
