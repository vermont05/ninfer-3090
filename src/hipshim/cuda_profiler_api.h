#pragma once

// Shadow of <cuda_profiler_api.h>. Consumers are bench programs that wrap a profiled
// region in cudaProfilerStart/cudaProfilerStop for the NVIDIA tooling (the --profile
// flags). ROCm has no equivalent start/stop API -- rocprof attaches externally -- so the
// calls are inert here: benches still compile and run, and --profile simply does not
// gate the range on this backend.

#include "cuda_runtime.h"

inline cudaError_t cudaProfilerStart() { return cudaSuccess; }
inline cudaError_t cudaProfilerStop() { return cudaSuccess; }
