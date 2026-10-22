// Unit tests for the ldmatrix helpers of ops/common/mma.cuh.
//
// The SIMT branch of that header re-expresses PTX ldmatrix fragment layouts as broadcasts,
// and its .trans path carried a wrong-addressing bug (value arithmetic where element
// indexing was meant) that no surrounding Op test can pin down on its own. This test
// feeds a matrix whose 256 elements are pairwise distinct — element (row, col) holds
// row * 8 + col, so a single read names both the element and its position — and checks
// every word of the fragment each helper must produce:
//
//   non-transposed: lane l's word of tile i is (row l >> 2, col (l & 3) * 2) of tile i, the
//     tile's rows addressed by provider lanes 8i + (l >> 2);
//   .transposed: lane l's word of tile i is provider rows 2 * (l & 3) and +1 of tile i read
//     at column l >> 2 — the same matrix seen transposed.
//
// A wrong element for any lane therefore shows up as a value that exists somewhere else
// in the matrix, which is exactly what the fixed bug produced. The .cu branch of mma.cuh
// is the one under test; on the CUDA build (inline-asm branch) this test skips.

#if !defined(__HIPCC__)
#include <cstdio>
int main() {
    std::printf("SKIP: mma ldmatrix reference test is for the SIMT branch\n");
    return 77;
}
#else

#include "ops/common/mma.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr int kRows = 32;
constexpr int kCols = 8;

__forceinline__ __host__ __device__ std::uint16_t element(int row, int col) {
    return static_cast<std::uint16_t>(row * kCols + col);
}

// The matrix fill is the call-site shape: each lane provides the address of its own row
// (32 rows for x4, the first 16 for x2), exactly as the grouped attention loaders build
// per-lane row addresses for the instruction.
__device__ __forceinline__ void fill_matrix(std::uint16_t (*smem)[kCols]) {
    for (int i = threadIdx.x; i < kRows * kCols; i += 32) {
        smem[i / kCols][i % kCols] = element(i / kCols, i % kCols);
    }
}

__device__ __forceinline__ std::uintptr_t row_addr(std::uint16_t (*smem)[kCols], int row) {
    return reinterpret_cast<std::uintptr_t>(&smem[row][0]);
}

__global__ void ldmatrix_x2_kernel(std::uint32_t* out) {
    __shared__ __align__(16) std::uint16_t smem[kRows][kCols];
    fill_matrix(smem);
    __syncthreads();
    const std::uintptr_t addr = row_addr(smem, threadIdx.x);
    unsigned r0 = 0, r1 = 0;
    ninfer::ops::ldmatrix_x2(r0, r1, addr);
    out[threadIdx.x * 4 + 0] = r0;
    out[threadIdx.x * 4 + 1] = r1;
}

__global__ void ldmatrix_x4_kernel(std::uint32_t* out) {
    __shared__ __align__(16) std::uint16_t smem[kRows][kCols];
    fill_matrix(smem);
    __syncthreads();
    const std::uintptr_t addr = row_addr(smem, threadIdx.x);
    unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    ninfer::ops::ldmatrix_x4(r0, r1, r2, r3, addr);
    out[threadIdx.x * 4 + 0] = r0;
    out[threadIdx.x * 4 + 1] = r1;
    out[threadIdx.x * 4 + 2] = r2;
    out[threadIdx.x * 4 + 3] = r3;
}

__global__ void ldmatrix_x2_t_kernel(std::uint32_t* out) {
    __shared__ __align__(16) std::uint16_t smem[kRows][kCols];
    fill_matrix(smem);
    __syncthreads();
    const std::uintptr_t addr = row_addr(smem, threadIdx.x);
    unsigned r0 = 0, r1 = 0;
    ninfer::ops::ldmatrix_x2_t(r0, r1, addr);
    out[threadIdx.x * 4 + 0] = r0;
    out[threadIdx.x * 4 + 1] = r1;
}

__global__ void ldmatrix_x4_t_kernel(std::uint32_t* out) {
    __shared__ __align__(16) std::uint16_t smem[kRows][kCols];
    fill_matrix(smem);
    __syncthreads();
    const std::uintptr_t addr = row_addr(smem, threadIdx.x);
    unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
    ninfer::ops::ldmatrix_x4_t(r0, r1, r2, r3, addr);
    out[threadIdx.x * 4 + 0] = r0;
    out[threadIdx.x * 4 + 1] = r1;
    out[threadIdx.x * 4 + 2] = r2;
    out[threadIdx.x * 4 + 3] = r3;
}

// Expected fragment words, host side, for the same layout.
std::uint32_t expect_plain(int tile, int l) {
    const int row = 8 * tile + (l >> 2), col = (l & 3) * 2;
    return static_cast<std::uint32_t>(element(row, col)) |
           (static_cast<std::uint32_t>(element(row, col + 1)) << 16);
}

std::uint32_t expect_trans(int tile, int l) {
    const int g = l >> 2, base = 8 * tile + 2 * (l & 3);
    return static_cast<std::uint32_t>(element(base, g)) |
           (static_cast<std::uint32_t>(element(base + 1, g)) << 16);
}

int failures = 0;

void check(const char* name, const std::vector<std::uint32_t>& got, int tiles, bool trans) {
    for (int l = 0; l < kRows; ++l) {
        for (int tile = 0; tile < tiles; ++tile) {
            const std::uint32_t want = trans ? expect_trans(tile, l) : expect_plain(tile, l);
            const std::uint32_t have = got[static_cast<std::size_t>(l) * 4 + tile];
            if (have != want) {
                std::printf("FAIL %s: lane %d tile %d: got 0x%08x want 0x%08x\n", name, l, tile,
                            have, want);
                ++failures;
            }
        }
    }
}

void run(cudaError_t (*launch)(std::uint32_t*), const char* name, std::uint32_t* scratch,
          int tiles, bool trans) {
    const std::size_t words = static_cast<std::size_t>(kRows) * 4;
    (void)cudaMemsetAsync(scratch, 0, words * sizeof(std::uint32_t));
    if (const cudaError_t err = launch(scratch); err != cudaSuccess) {
        std::printf("FAIL %s: launch: %s\n", name, cudaGetErrorString(err));
        ++failures;
        return;
    }
    std::vector<std::uint32_t> got(words);
    if (cudaMemcpy(got.data(), scratch, words * sizeof(std::uint32_t), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::printf("FAIL %s: copy back\n", name);
        ++failures;
        return;
    }
    check(name, got, tiles, trans);
}

cudaError_t launch_x2(std::uint32_t* out) {
    ldmatrix_x2_kernel<<<1, 32>>>(out);
    return cudaGetLastError();
}
cudaError_t launch_x4(std::uint32_t* out) {
    ldmatrix_x4_kernel<<<1, 32>>>(out);
    return cudaGetLastError();
}
cudaError_t launch_x2_t(std::uint32_t* out) {
    ldmatrix_x2_t_kernel<<<1, 32>>>(out);
    return cudaGetLastError();
}
cudaError_t launch_x4_t(std::uint32_t* out) {
    ldmatrix_x4_t_kernel<<<1, 32>>>(out);
    return cudaGetLastError();
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 77;
    }

    std::uint32_t* scratch = nullptr;
    if (cudaMalloc(&scratch, static_cast<std::size_t>(kRows) * 4 * sizeof(std::uint32_t)) !=
        cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        return 1;
    }

    run(launch_x2, "ldmatrix_x2", scratch, 2, false);
    run(launch_x4, "ldmatrix_x4", scratch, 4, false);
    run(launch_x2_t, "ldmatrix_x2_t", scratch, 2, true);
    run(launch_x4_t, "ldmatrix_x4_t", scratch, 4, true);

    (void)cudaFree(scratch);
    if (failures != 0) {
        std::printf("FAIL: %d fragment words wrong\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}

#endif
