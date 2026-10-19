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

// CUDA spells the device-attribute enumerators cudaDevAttr*, HIP spells them
// hipDeviceAttribute*.
#define cudaDevAttrMaxPitch hipDeviceAttributeMaxPitch
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing

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
