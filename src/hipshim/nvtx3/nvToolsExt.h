#pragma once

// Shadow of <nvtx3/nvToolsExt.h>: an inert stub. NVTX ranges exist for Nsight to consume, and
// the HIP toolchain has a ROCm profiler to read them with at best. This declares exactly
// the types and entry points src/core/nvtx.h uses, with that header's call shapes, and
// does nothing, so the range code compiles on the HIP backend without a vendor NVTX.

#include <cstddef>
#include <cstdint>

// --- types -------------------------------------------------------------------------------
using nvtxDomainHandle_t = void*;
using nvtxStringHandle_t = void*;
using nvtxRangeId_t      = std::uint64_t;

// The event-attribute struct, shaped for the fields nvtx.h sets. Value-initialized at every
// call site, so member order does not matter.
struct nvtxEventAttributes {
    std::uint16_t version;
    std::uint32_t size;
    std::int32_t  category;
    std::uint16_t colorType;
    std::uint32_t color;
    std::uint16_t payloadType;
    union payload_t {
        std::uint64_t ullValue;
        std::int64_t  llValue;
        double        dValue;
        std::uint32_t u32Value;
        std::int32_t  i32Value;
    } payload;
    std::uint16_t messageType;
    union message_t {
        const char* ascii;
        void*       registered;
    } message;
};
using nvtxEventAttributes_t = nvtxEventAttributes;

// --- enumerators (as macros: CUDA's header spells them this way too) -----------------------
#define NVTX_VERSION 2
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE (static_cast<std::uint32_t>(sizeof(nvtxEventAttributes)))
#define NVTX_COLOR_ARGB 0
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 0
#define NVTX_MESSAGE_TYPE_REGISTERED 0

// --- functions -----------------------------------------------------------------------------
inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) { return nullptr; }
inline void nvtxDomainNameCategoryA(nvtxDomainHandle_t, std::uint32_t, const char*) {}
inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char*) {
    return nullptr;
}
inline void nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) {}
inline void nvtxDomainRangePop(nvtxDomainHandle_t) {}
inline nvtxRangeId_t nvtxDomainRangeStartEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) {
    return 0;
}
inline void nvtxDomainRangeEnd(nvtxDomainHandle_t, nvtxRangeId_t) {}
