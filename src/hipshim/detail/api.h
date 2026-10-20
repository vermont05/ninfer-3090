#pragma once

// CUDA runtime/driver API names on top of the HIP runtime. Included by the shadow headers in
// this directory; see README.md for the rules this table follows.
//
// Every entry below was taken from a call site in the tree and checked against the HIP
// headers of the toolchain it is built with. A name that has no HIP counterpart is not here:
// the compile that needs it fails, and it gets a wrapper or a per-file guard at that point.

// hip_common.h picks a platform only when it sees a compiler that predefines one. clang++
// in a HIP pass gets it from the driver; every other compiler reaching this header —
// including clang++ compiling plain C++ in a HIP build, whose host pass includes the HIP
// headers directly — reaches it without one. This table is the AMD table, so define AMD
// unless the NVIDIA macro is already in play.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif

#include <hip/hip_runtime.h>

// Drawn in here, not only by the shadow headers that wrap them, so that the warp
// intrinsics further down are defined after every declaration of their names.
// hip/hip_bf16.h, hip/hip_fp16.h and hip/hip_cooperative_groups.h each declare some of
// __shfl_*_sync, __ballot_sync and __ldg outright: a function-like macro of one of those
// names, live while one of these is parsed, expands the declaration's parameter list
// and breaks it. Read in now, the guard in each header makes every later include a
// no-op, and no source ordering can put a macro in front of the declarations.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>

#include <cstddef>

// --- types -------------------------------------------------------------------------------
// These are the names the sources use in declarations and struct fields. An alias, not a
// wrapper: `cudaStream_t s` and `hipStream_t s` must remain interchangeable, including in the
// pointer-to-pointer parameters of the memory and stream calls.
using cudaError_t              = hipError_t;
using cudaStream_t             = hipStream_t;
using cudaEvent_t              = hipEvent_t;
using cudaDeviceProp           = hipDeviceProp_t;
using cudaMemcpyKind           = hipMemcpyKind;
using cudaMemcpyAttributes     = hipMemcpyAttributes;
using cudaStreamCaptureStatus  = hipStreamCaptureStatus;
using cudaStreamCaptureMode    = hipStreamCaptureMode;
using cudaFuncAttribute        = hipFuncAttribute;
using cudaGraph_t              = hipGraph_t;
using cudaGraphExec_t          = hipGraphExec_t;
using cudaGraphNode_t          = hipGraphNode_t;
using cudaUUID_t               = hipUUID_t;
// Extensible-launch plumbing: the HIP structs carry the CUDA layout (id, pad, val/value),
// and gdn_gating_proj's cooperative launch reads exactly those members.
using cudaLaunchAttribute      = hipLaunchAttribute;
using cudaLaunchConfig_t       = hipLaunchConfig_t;

// --- enumerators and constants -----------------------------------------------------------
// Object-like macros: the right-hand side is an enumerator or a macro in the HIP headers, and
// these are the values CUDA_CHECK and the error paths compare against.
#define cudaSuccess hipSuccess
#define CUDA_SUCCESS hipSuccess
#define CUDA_ERROR_NOT_READY hipErrorNotReady
#define CUDA_ERROR_OUT_OF_MEMORY hipErrorOutOfMemory

#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaErrorMemoryAllocation hipErrorMemoryAllocation
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorNoDevice hipErrorNoDevice
#define cudaErrorInsufficientDriver hipErrorInsufficientDriver
#define cudaErrorAlreadyMapped hipErrorAlreadyMapped
#define cudaErrorCooperativeLaunchTooLarge hipErrorCooperativeLaunchTooLarge
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorStreamCaptureIsolation hipErrorStreamCaptureIsolation

#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpySrcAccessOrderStream hipMemcpySrcAccessOrderStream

#define cudaDeviceScheduleAuto hipDeviceScheduleAuto
#define cudaDeviceScheduleSpin hipDeviceScheduleSpin
#define cudaDeviceScheduleYield hipDeviceScheduleYield
#define cudaDeviceScheduleBlockingSync hipDeviceScheduleBlockingSync
#define cudaDeviceMapHost hipDeviceMapHost
// CUDA declares this as an enum; HIP spells it as a macro constant, and device.cu uses it
// the way CUDA's header allows: as the whole-mask value behind a `&`.
#define cudaDeviceScheduleMask hipDeviceScheduleMask

#define cudaGraphExecUpdateSuccess hipGraphExecUpdateSuccess

#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCaptureStatusActive hipStreamCaptureStatusActive
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecordExternal hipEventRecordExternal
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostAllocMapped hipHostAllocMapped

#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout \
    hipFuncAttributePreferredSharedMemoryCarveout

// The carveout sentinel values from CUDA's cuda_runtime.h, kept at their CUDA values;
// they go behind the PreferredSharedMemoryCarveout attribute call sites above.
#define cudaSharedmemCarveoutDefault   (-1)
#define cudaSharedmemCarveoutMaxL1     (-2)
#define cudaSharedmemCarveoutMaxShared (-3)

#define cudaLaunchAttributeCooperative hipLaunchAttributeCooperative

// CUDA spells the device-attribute enumerators cudaDevAttr*, HIP spells them
// hipDeviceAttribute*.
#define cudaDevAttrMaxPitch hipDeviceAttributeMaxPitch
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#define cudaDevAttrMemoryClockRate hipDeviceAttributeMemoryClockRate
// HIP spells the bus-width enumerator without the "Global" prefix CUDA uses.
#define cudaDevAttrGlobalMemoryBusWidth hipDeviceAttributeMemoryBusWidth
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor

// Math constants used by the kernels; CUDA exports them from its own headers.
#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __int_as_float(0x7fffffff)
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION 12080
#endif

// --- device intrinsics ---------------------------------------------------------------------
// The warp-synchronous shuffles, votes and reductions are templates in
// amd_warp_sync_functions.h that static_assert(sizeof(mask) == 8), and every call
// site in the tree passes an `unsigned` 0xffffffff. The macro widens the mask and
// calls the HIP function behind the same name — a macro's own name in its body is
// not re-expanded, so this reaches the function, not the macro. This is why the
// vendor headers that declare these names are read in at the top of this file.
#define __shfl_sync(mask, ...)       __shfl_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __shfl_up_sync(mask, ...)    __shfl_up_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __shfl_down_sync(mask, ...)  __shfl_down_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __shfl_xor_sync(mask, ...)   __shfl_xor_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __ballot_sync(mask, ...)     __ballot_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __all_sync(mask, ...)        __all_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __any_sync(mask, ...)        __any_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __reduce_add_sync(mask, ...) __reduce_add_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __reduce_min_sync(mask, ...) __reduce_min_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __reduce_max_sync(mask, ...) __reduce_max_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __reduce_and_sync(mask, ...) __reduce_and_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __reduce_or_sync(mask, ...)  __reduce_or_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)
#define __reduce_xor_sync(mask, ...) __reduce_xor_sync((unsigned long long)(unsigned)(mask), __VA_ARGS__)

// Same mask trap in __syncwarp(mask); the no-argument call keeps working because the
// macro expands to the same call shape.
#define __syncwarp(...) __syncwarp(__VA_OPT__((unsigned long long)(unsigned)(__VA_ARGS__)))

// __ldg exists only for a fixed list of integer and vector types; the tree loads
// __nv_bfloat16, packed codes and other pointers through it. HIP's own overloads are
// plain loads, so one template covers the rest the same way.
template <class T>
__attribute__((always_inline)) inline __device__ T __hipshim_ldg(const T* p) {
    return *p;
}
#define __ldg(p) __hipshim_ldg(p)

// The other CUDA cache-hint loads, same story: HIP spells a couple of them for a
// fixed list of types only (half/half2 in hip_fp16.h), and they are plain loads
// wherever they exist. The tree reads wider vectors and other pointers through
// them, so route them to the same template.
#define __ldcg(p) __hipshim_ldg(p)
#define __ldcs(p) __hipshim_ldg(p)
#define __ldcv(p) __hipshim_ldg(p)

// Backpressure pause inside a spin loop; GCN has no instruction HIP spells under this
// name, and a busy wait is exactly what the loop does without it.
#define __nanosleep(t) ((void)(t))

// Byte dot-product. The four signed-byte products it folds are a handful of integer
// multiplies, and it appears only on SIMT-dequant routes.
__attribute__((always_inline)) inline __device__ int __dp4a(int a, int b, int c) {
    const unsigned x = static_cast<unsigned>(a);
    const unsigned y = static_cast<unsigned>(b);
    int acc = c;
    for (int i = 0; i < 4; ++i) {
        const auto lo = static_cast<int>(static_cast<signed char>(static_cast<unsigned char>(x >> (i * 8))));
        const auto hi = static_cast<int>(static_cast<signed char>(static_cast<unsigned char>(y >> (i * 8))));
        acc += lo * hi;
    }
    return acc;
}

// CUDA's packed-byte video intrinsics (cuda_video_intrinsics.h), used by the GGML and
// Q4 dequant helpers. There is no portable AMD instruction, so each is the lane loop the
// PTX opcode models; callers pass mixed int/unsigned, which the macros fold to unsigned.
__attribute__((always_inline)) inline __device__ unsigned __hipshim_vadd4(unsigned a,
                                                                            unsigned b) {
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned sum = ((a >> (8 * i)) & 0xFFu) + ((b >> (8 * i)) & 0xFFu);
        r |= (sum & 0xFFu) << (8 * i);
    }
    return r;
}
__attribute__((always_inline)) inline __device__ unsigned __hipshim_vsub4(unsigned a,
                                                                            unsigned b) {
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const unsigned diff = ((a >> (8 * i)) & 0xFFu) - ((b >> (8 * i)) & 0xFFu);
        r |= (diff & 0xFFu) << (8 * i);
    }
    return r;
}
__attribute__((always_inline)) inline __device__ unsigned __hipshim_vsubss4(unsigned a,
                                                                             unsigned b) {
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        const int d = static_cast<int>(static_cast<signed char>((a >> (8 * i)) & 0xFFu)) -
                        static_cast<int>(static_cast<signed char>((b >> (8 * i)) & 0xFFu));
        const int c = d > 127 ? 127 : (d < -128 ? -128 : d);
        r |= static_cast<unsigned>(static_cast<unsigned char>(c)) << (8 * i);
    }
    return r;
}
__attribute__((always_inline)) inline __device__ unsigned __hipshim_vcmpne4(unsigned a,
                                                                             unsigned b) {
    unsigned r = 0;
    for (int i = 0; i < 4; ++i) {
        if (((a >> (8 * i)) & 0xFFu) != ((b >> (8 * i)) & 0xFFu)) {
            r |= 0xFFu << (8 * i);
        }
    }
    return r;
}
#define __vadd4(a, b)   __hipshim_vadd4((unsigned)(a), (unsigned)(b))
#define __vsub4(a, b)   __hipshim_vsub4((unsigned)(a), (unsigned)(b))
#define __vsubss4(a, b) __hipshim_vsubss4((unsigned)(a), (unsigned)(b))
#define __vcmpne4(a, b) __hipshim_vcmpne4((unsigned)(a), (unsigned)(b))

// Address of a shared-window pointer as an integer; CUDA's cvta entry point has no ROCm
// counterpart, and on GCN the generic address of a shared object is its shared address.
__attribute__((always_inline)) inline __device__ unsigned long long
__cvta_generic_to_shared(const void* p) {
    return reinterpret_cast<unsigned long long>(p);
}

// --- functions -----------------------------------------------------------------------------
// All of these are ordinary functions or function templates in the HIP headers with the same
// parameter list the CUDA call sites use.
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
#define cudaGetErrorName hipGetErrorName
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaGetDeviceFlags hipGetDeviceFlags

#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMallocHost hipHostMalloc
#define cudaHostAlloc hipHostAlloc
#define cudaFreeHost hipHostFree
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemset2DAsync hipMemset2DAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2D hipMemcpy2D
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyBatchAsync hipMemcpyBatchAsync

#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventRecordWithFlags hipEventRecordWithFlags
#define cudaEventQuery hipEventQuery
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamGetCaptureInfo hipStreamGetCaptureInfo

#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphUpload hipGraphUpload
#define cudaGraphGetNodes hipGraphGetNodes

#define cudaLaunchKernelEx hipLaunchKernelEx
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
    hipOccupancyMaxActiveBlocksPerMultiprocessor

// --- divergent ----------------------------------------------------------------------------
// Same call, different shape in HIP. Each wrapper reproduces the CUDA prototype the call
// site uses and forwards to the HIP entry point that takes the same meaning.

// CUDA 12's cudaGraphInstantiate(exec, graph, flags); HIP splits the flags variant off into
// WithFlags and has no three-argument form.
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                         unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}

// cudaGraphInstantiate takes the flags directly at the call sites above; CUDA also has a
// two-argument form used with no flags.
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph,
                                         cudaGraphNode_t* error_node, void* log_buffer,
                                         size_t log_buffer_size) {
    (void)error_node;
    (void)log_buffer;
    (void)log_buffer_size;
    return hipGraphInstantiate(exec, graph, nullptr, nullptr, 0);
}

// CUDA 12's cudaGraphExecUpdate(exec, graph, info*) reports the failure through an info
// struct; HIP splits the two outputs it carries apart. The struct here carries exactly
// the fields the call site reads (decode_graph.cpp reads .result).
struct cudaGraphExecUpdateResultInfo {
    cudaGraphNode_t errorNode;
    hipGraphExecUpdateResult result;
};

inline cudaError_t cudaGraphExecUpdate(cudaGraphExec_t exec, cudaGraph_t graph,
                                         cudaGraphExecUpdateResultInfo* info) {
    return hipGraphExecUpdate(exec, graph, &info->errorNode, &info->result);
}

// cudaFuncSetAttribute is a template over the kernel pointer in cudart; HIP's takes a
// `const void*`, which a kernel function pointer does not convert to.
template <class Kernel>
inline cudaError_t cudaFuncSetAttribute(Kernel kernel, cudaFuncAttribute attribute,
                                         int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}

// --- kernel markers -----------------------------------------------------------------------
// CUDA's __grid_constant__ annotates by-value kernel parameters that TMA reads through
// their parameter address. HIP has no spelling for it and needs none: the parameter is
// already read-only in the kernel's parameter space. Defined empty so the TMA-path
// headers that name it parse; without TMA the annotation carries no information.
#define __grid_constant__

// CUDA's per-kernel register cap. It is an attribute-like call on the __global__
// declarator (`__maxnreg__(120) void k(...)`), and without a spelling here clang reads
// the name as the declarator's type and the whole declaration collapses. HIP has no
// equivalent — waves get a fixed share of the register file — so an empty macro is
// both the parse fix and the honest translation.
#define __maxnreg__(...)
