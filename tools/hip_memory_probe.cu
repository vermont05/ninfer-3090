// First device contact for the HIP backend's memory system (Phase 3).
//
// It touches the three runtime-only behaviours the CUDA build never had to prove on this
// hardware, in the order the eviction pools and decode graph depend on them:
//
//   1. DeviceContext construction -- the process-wide device binding, streams and
//      property query every Op and pool sits on. This is also the ROCR_VISIBLE_DEVICES
//      iGPU landmine check: a gfx1102-only binary faults here if a non-matching GPU is
//      enumerated ahead of the eGPU.
//   2. cuMemGetAllocationGranularity + a VmmRegion create/back/release/restore round-trip --
//      the exact driver path the VMM arenas and evictable pools build on, plus the
//      granularity the pool code assumes when it rounds piece sizes.
//   3. Stream capture + graph instantiate/launch -- whether hipGraph capture is legal on
//      gfx1102 at all, which decode_graph.cpp requires to hold decode.
//
// It runs no model and asserts nothing about numerical results: it only proves the memory
// system is live on the real device before anything is asked to compute there. Exits 0 when
// every check passes, 1 on a failure, and 77 (CTest skip) when there is no usable device.

#include "core/arena.h"
#include "core/device.h"
#include "core/vmm_region.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

// cudaError_t and CUresult are the same integer typedef under HIP, so the two checks cannot
// be overloads of one name -- the compiler folds them into a redefinition. The runtime check is
// the macro (it covers the cuda* calls); the one driver call names check_cu directly.
void check_cuda(cudaError_t err, const char* expr) {
    if (err == cudaSuccess) { return; }
    throw std::runtime_error(std::string(expr) + " failed: " + cudaGetErrorString(err));
}

void check_cu(CUresult result, const char* expr) {
    if (result == CUDA_SUCCESS) { return; }
    const char* name = nullptr;
    (void)cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(expr) + " failed: " +
                              (name != nullptr ? name : "unknown CUresult"));
}

#define NINFER_CHECK(expr) ::check_cuda((expr), #expr)

// A graph legal-check capture: a single memset into a device buffer, captured on the
// context's compute stream. Nothing about the work matters; only that the driver will
// accept the capture, hand back a graph, instantiate it, and launch the executable.
bool capture_probe(const ninfer::DeviceContext& ctx, std::string& note) {
    ninfer::DeviceBuffer buf(4096);
    cudaStream_t stream = ctx.stream;

    NINFER_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    NINFER_CHECK(cudaMemsetAsync(buf.p, 0x5a, 1024, stream));
    cudaGraph_t graph = nullptr;
    NINFER_CHECK(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t exec = nullptr;
    NINFER_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    NINFER_CHECK(cudaGraphLaunch(exec, stream));
    NINFER_CHECK(cudaStreamSynchronize(stream));

    std::uint32_t word = 0;
    NINFER_CHECK(cudaMemcpy(&word, buf.p, sizeof(word), cudaMemcpyDeviceToHost));

    (void)cudaGraphExecDestroy(exec);
    (void)cudaGraphDestroy(graph);

    if (buf.bytes != 4096) { throw std::runtime_error("DeviceBuffer reported wrong size"); }
    if (word != 0x5a5a5a5au) {
        note = "graph launch memset wrote unexpected value " + std::to_string(word);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    try {
        int count = 0;
        const cudaError_t count_err = cudaGetDeviceCount(&count);
        if (count_err == cudaErrorNoDevice || count_err == cudaErrorInsufficientDriver) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        NINFER_CHECK(count_err);
        if (count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        // 1. Device context -- device init path.
        ninfer::DeviceContext ctx(0);
        std::cout << "[context] device=" << ctx.device
                  << " capability=" << ctx.compute_capability()
                  << " total_vram=" << (ctx.total_vram() >> 20) << " MiB"
                  << " stream=" << (ctx.stream != nullptr) << "\n";

        // 2. VMM granularity + a region lifecycle on the real device.
        {
            CUmemAllocationProp prop{};
            prop.type      = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id   = ctx.device;
            std::size_t granularity = 0;
            check_cu(cuMemGetAllocationGranularity(
                         &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                     "cuMemGetAllocationGranularity");
            std::cout << "[vmm] granularity=" << granularity << " bytes\n";
        }
        std::cout << "[vmm] VmmRegion::supported(0)="
                  << (ninfer::VmmRegion::supported(0) ? "yes" : "no") << "\n";
        if (ninfer::VmmRegion::supported(0)) {
            ninfer::VmmRegion region(64u << 20, ninfer::VmmRegion::Options{.device = ctx.device});
            std::cout << "[vmm] region reserved=" << (region.reserved_bytes() >> 20)
                      << " MiB backed=" << region.backed() << "\n";
            region.release_backing();
            std::cout << "[vmm] after release backed=" << region.backed() << "\n";
            region.restore_backing();
            std::cout << "[vmm] after restore backed=" << region.backed() << "\n";
        }

        // 3. Arena suballocation.
        {
            ninfer::DeviceBuffer backing(16u << 20);
            ninfer::DeviceArena arena(ninfer::DeviceSpan{backing.p, backing.bytes});
            auto a = arena.alloc_bytes(4096, 256);
            auto b = arena.alloc_bytes(8192, 256);
            std::cout << "[arena] capacity=" << (arena.capacity() >> 20)
                      << " MiB used=" << arena.used()
                      << " a=" << a.bytes << " b=" << b.bytes << "\n";
            if (arena.used() == 0 || !a.data || !b.data) {
                throw std::runtime_error("DeviceArena returned no storage");
            }
        }

        // 4. Graph capture legality + launch.
        std::string note;
        const bool graph_ok = capture_probe(ctx, note);
        std::cout << "[graph] capture+instantiate+launch="
                  << (graph_ok ? "ok" : note) << "\n";
        if (!graph_ok) { return 1; }

        std::cout << "PROBE-PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PROBE-FAIL: " << e.what() << "\n";
        return 1;
    }
}
