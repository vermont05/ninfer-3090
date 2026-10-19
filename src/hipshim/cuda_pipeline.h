#pragma once

// Shadow of <cuda_pipeline.h>. The tree's only consumer is ops/common/memory.cuh, whose
// pipe_copy / pipe_commit / pipe_wait call the three entry points CUDA declares there.
// HIP has no async-copy pipeline, so this shadow does the copy synchronously: the copy is
// finished when __pipeline_memcpy_async returns, commit and wait_prior are no-ops, and the
// order the pattern relies on (issue, commit, ..., wait_prior, read) still holds. What is
// lost is the overlap the async form buys, not correctness.

#include <cstddef>

__device__ __attribute__((always_inline)) inline void __pipeline_memcpy_async(void* dst,
                                                                                const void* src,
                                                                                std::size_t size) {
    auto* out = static_cast<unsigned char*>(dst);
    const auto* in = static_cast<const unsigned char*>(src);
    for (std::size_t i = 0; i < size; ++i) { out[i] = in[i]; }
}

__device__ __attribute__((always_inline)) inline void __pipeline_commit() {}

__device__ __attribute__((always_inline)) inline void __pipeline_wait_prior(int) {}
