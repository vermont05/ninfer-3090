#pragma once

// Shadow of <cuda.h>, the CUDA driver umbrella. The tree includes it only for the VMM code
// (cuda_vmm.h, vmm_arena.h, the evictable pools), and the surface those use is the virtual
// memory management entry points with their types plus the device and error-name helpers
// around them. HIP names these with the hip* spelling and holds addresses as void* instead
// of the integer CUdeviceptr, so the address-taking entries are thin wrappers over the HIP
// calls rather than renames, each with the CUDA driver prototype the call sites match.
//
// Not here: the tensor-map half (CUtensorMap, cuTensorMapEncodeTiled). Its consumers are the
// TMA and Blackwell files, and no HIP build compiles them.

#include "detail/api.h"

#include <cstddef>
#include <cstdint>

// --- types -------------------------------------------------------------------------------
// The struct types alias the HIP ones outright: the VMM code sets exactly the fields the
// HIP structs carry (type, location.type, location.id, flags) and passes the structs by
// pointer to the calls below.
using CUresult                       = hipError_t;
using CUdevice                        = hipDevice_t;
using CUdeviceptr                     = std::uintptr_t;
using CUmemAllocationProp             = hipMemAllocationProp;
using CUmemAccessDesc                 = hipMemAccessDesc;
using CUmemGenericAllocationHandle      = hipMemGenericAllocationHandle_t;
using CUmemAllocationGranularity_flags  = hipMemAllocationGranularity_flags;

// --- enumerators ---------------------------------------------------------------------------
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM hipMemAllocationGranularityMinimum
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite

// --- functions -----------------------------------------------------------------------------
// These three have the same shape as their HIP entry points.
#define cuInit hipInit
#define cuDeviceGet hipDeviceGet
#define cuDeviceGetAttribute hipDeviceGetAttribute

inline CUresult cuMemGetAllocationGranularity(std::size_t* granularity,
                                                const CUmemAllocationProp* prop,
                                                CUmemAllocationGranularity_flags option) {
    return hipMemGetAllocationGranularity(granularity, prop, option);
}

inline CUresult cuMemAddressReserve(CUdeviceptr* ptr, std::size_t size, std::size_t alignment,
                                      CUdeviceptr addr, unsigned long long flags) {
    void* address = nullptr;
    const CUresult result =
        hipMemAddressReserve(&address, size, alignment, reinterpret_cast<void*>(addr), flags);
    if (result == hipSuccess) { *ptr = reinterpret_cast<CUdeviceptr>(address); }
    return result;
}

inline CUresult cuMemAddressFree(CUdeviceptr ptr, std::size_t size) {
    return hipMemAddressFree(reinterpret_cast<void*>(ptr), size);
}

inline CUresult cuMemCreate(CUmemGenericAllocationHandle* handle, std::size_t size,
                             const CUmemAllocationProp* prop, unsigned long long flags) {
    return hipMemCreate(handle, size, prop, flags);
}

inline CUresult cuMemMap(CUdeviceptr ptr, std::size_t size, std::size_t offset,
                          CUmemGenericAllocationHandle handle, unsigned long long flags) {
    return hipMemMap(reinterpret_cast<void*>(ptr), size, offset, handle, flags);
}

inline CUresult cuMemUnmap(CUdeviceptr ptr, std::size_t size) {
    return hipMemUnmap(reinterpret_cast<void*>(ptr), size);
}

inline CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    return hipMemRelease(handle);
}

inline CUresult cuMemSetAccess(CUdeviceptr ptr, std::size_t size, const CUmemAccessDesc* desc,
                                std::size_t count) {
    return hipMemSetAccess(reinterpret_cast<void*>(ptr), size, desc, count);
}

// CUDA's two-argument error name: HIP's returns the string instead of writing it out.
inline CUresult cuGetErrorName(CUresult error, const char** str) {
    if (str != nullptr) { *str = hipGetErrorName(error); }
    return hipSuccess;
}
