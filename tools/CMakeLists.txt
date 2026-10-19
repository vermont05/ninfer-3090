# The two multi-GPU/NCCL probes below are CUDA-only: they enumerate peer-capable devices and
# probe tensor-parallel all-reduce transports. The HIP backend is a single-eGPU target where
# neither exists, and their .cu sources configure to no link language there, so it skips them.
if(NOT NINFER_BACKEND STREQUAL "hip")

# Standalone CUDA enumeration probe, plus a real DeviceContext construction check.
#
# It deliberately does not link ninfer_core. The enumeration half needs to answer "which CUDA
# indices are the two cards, and can they reach each other" before the model loader is involved,
# on a machine where the answer is not obvious -- a display GPU commonly takes cuda=0 even when
# nvidia-smi lists it elsewhere, so the indices for --devices must be read from a CUDA-level
# enumeration rather than assumed.
#
# device.cu is compiled in directly rather than linked from the core. It only includes core/device.h
# and standard headers, so this costs a single translation unit and keeps the probe around 1 MB
# instead of the ~500 MB a core link would produce -- which matters because this binary gets
# uploaded to rented hardware to be run there.
add_executable(ninfer-multi-gpu-probe
  multi_gpu_probe.cu
  ${CMAKE_SOURCE_DIR}/src/core/device.cu)
target_include_directories(ninfer-multi-gpu-probe PRIVATE ${CMAKE_SOURCE_DIR}/src)
set_target_properties(ninfer-multi-gpu-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# Go/no-go probe for tensor-parallel all-reduce transports (peer push, mapped host, staged host,
# optionally NCCL): per-collective latency in graphs, a correctness soak, and the pinned-memory
# portability check. Standalone like the probe above, so it can be uploaded to rented hardware.
add_executable(ninfer-tp-probe tp_probe.cu)
set_target_properties(ninfer-tp-probe PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
endif()

# Measures what the dequantise-to-int8 prefill route would cost on the weight side: the group-scale
# spread within a row, and the relative L2 of reconstructing a weight as int8 with one scale per
# row. Reads the real artifact, because the answer depends on the actual scale distribution.
add_executable(ninfer-w4-row-scale-error w4_row_scale_error.cpp)
target_link_libraries(ninfer-w4-row-scale-error PRIVATE ninfer_artifact)
ninfer_internal_includes(ninfer-w4-row-scale-error)
set_target_properties(ninfer-w4-row-scale-error PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

# First device contact for the HIP backend's memory system: DeviceContext construction,
# cuMemGetAllocationGranularity plus a VmmRegion back/release/restore round-trip, an arena
# suballocation, and a graph capture/instantiate/launch. It runs no model -- it only proves the
# memory system is live on the real device (and that hipGraph capture is legal on this part)
# before anything is asked to compute there. Links ninfer_core, which carries the runtime and
# driver interfaces.
add_executable(ninfer-hip-memory-probe hip_memory_probe.cu)
target_link_libraries(ninfer-hip-memory-probe PRIVATE ninfer_core)
ninfer_internal_includes(ninfer-hip-memory-probe)
ninfer_hipify_cu(ninfer-hip-memory-probe)
set_target_properties(ninfer-hip-memory-probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")
