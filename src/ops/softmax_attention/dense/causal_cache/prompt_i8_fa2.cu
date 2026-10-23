// ninfer::ops - INT8-family (INT8-G64, rk8v4, rk4v4) causal prompt attention launcher for the
// FA2-style kernel: plan the CTA shape and key splits, launch the attention kernel and, for a split
// launch, the FP32 merge.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/math.h"
#include "ops/kv_cache/d256_profile.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8_fa2_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Launches attention for a concrete (compile-time) fitted shape: the warps/stages the shape names
// and, when the plan splits, the FP32 merge. A split plan keeps its key splits at the warp count the
// shape fit to - on RDNA3 the four-warp shape the 64 KiB LDS ceiling forces. Only the stages the
// fitted shape names are instantiated: a runtime stage branch would build both stages of every
// geometry, and RDNA3 never runs the 2-stage ones.
template <typename Geometry, typename Metadata, bool PackedValues, bool PackedKeys, typename Fit>
void launch_fitted(const Tensor& q, const Tensor& positions, float scale, const Tensor& k_pages,
                   const Tensor& v_pages, const Tensor& k_scale_pages, const Tensor& v_scale_pages,
                   Metadata metadata, CausalAttentionExecutionEnvelope envelope,
                   WorkspaceArena& workspace, const CausalPromptFa2Plan& plan, Tensor& out,
                   cudaStream_t stream) {
    const auto width       = static_cast<std::int32_t>(q.ne[2]);
    const int fit_warps    = Fit::WarpCount;
    const bool fit_split   = plan.splits > 1;

    CausalPromptFa2Partials partials{};
    if (fit_split) {
        partials =
            allocate_causal_prompt_fa2_partials(workspace, Geometry::QHeads, width, plan.splits);
    }

    const auto launch = [&]<int Warps, int Stages, bool Split>() {
        using Shape       = CausalPromptFa2Shape<Warps, PackedValues, PackedKeys, Stages>;
        const auto kernel = causal_attention_prompt_i8_fa2_kernel<Geometry, Metadata, Warps, Split,
                                                                  PackedValues, PackedKeys,
                                                                  Stages>;
        configure_cuda_device_once([] {
            return cudaFuncSetAttribute(
                causal_attention_prompt_i8_fa2_kernel<Geometry, Metadata, Warps, Split,
                                                      PackedValues, PackedKeys, Stages>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, Shape::SmemBytes);
        });
        const dim3 grid(static_cast<unsigned>(div_up(width, Shape::Br)),
                        static_cast<unsigned>(Geometry::QHeads),
                        static_cast<unsigned>(Split ? plan.splits : 1));
        kernel<<<grid, Shape::Threads, Shape::SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::int8_t*>(k_pages.data),
            static_cast<const std::int8_t*>(v_pages.data),
            static_cast<const __half*>(k_scale_pages.data),
            static_cast<const __half*>(v_scale_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), width, static_cast<float*>(partials.rows.data),
            static_cast<float2*>(partials.stats.data));
        CUDA_CHECK(cudaGetLastError());
    };

    const auto run = [&]<int Warps>() {
        if constexpr (Fit::StageCount == 1) {
            if (fit_split) {
                launch.template operator()<Warps, 1, true>();
            } else {
                launch.template operator()<Warps, 1, false>();
            }
        } else {
            if (fit_split) {
                launch.template operator()<Warps, 2, true>();
            } else {
                launch.template operator()<Warps, 2, false>();
            }
        }
    };
    if (fit_warps == 4) {
        run.template operator()<4>();
    } else {
        run.template operator()<8>();
    }
    if (fit_split) {
        constexpr float Log2E = 1.4426950408889634074f;
        causal_attention_prompt_i8_fa2_merge_kernel<Geometry, Metadata>
            <<<dim3(static_cast<unsigned>(width), static_cast<unsigned>(Geometry::QHeads)),
               kCausalPromptHeadDim, 0, stream>>>(
                static_cast<const float*>(partials.rows.data),
                static_cast<const float2*>(partials.stats.data), metadata, width, plan.splits,
                scale * Log2E, static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
    }
}

template <typename Geometry, typename Metadata, bool PackedValues, bool PackedKeys>
void launch_for(const Tensor& q, const Tensor& positions, float scale, const Tensor& k_pages,
                const Tensor& v_pages, const Tensor& k_scale_pages, const Tensor& v_scale_pages,
                Metadata metadata, CausalAttentionExecutionEnvelope envelope,
                WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
                cudaStream_t stream) {
    const auto width = static_cast<std::int32_t>(q.ne[2]);
    const CausalPromptFa2Plan plan = causal_prompt_fa2_plan(
        Geometry::QHeads, width, envelope.max_visible_keys, multiprocessor_count);
    // The planned shape, descended on RDNA3 to fit the LDS ceiling (a 1-stage, and if needed
    // four-warp, variant); unchanged on sm_8x. plan.warps is a runtime value, so the requested
    // (two-stage) shape is selected by branch and the descent is a compile-time type choice.
    if (plan.warps == 4) {
        using Fit =
            typename CausalPromptFa2LdsFit<CausalPromptFa2Shape<4, PackedValues, PackedKeys, 2>>::type;
        launch_fitted<Geometry, Metadata, PackedValues, PackedKeys, Fit>(
            q, positions, scale, k_pages, v_pages, k_scale_pages, v_scale_pages, metadata, envelope,
            workspace, plan, out, stream);
    } else {
        using Fit =
            typename CausalPromptFa2LdsFit<CausalPromptFa2Shape<8, PackedValues, PackedKeys, 2>>::type;
        launch_fitted<Geometry, Metadata, PackedValues, PackedKeys, Fit>(
            q, positions, scale, k_pages, v_pages, k_scale_pages, v_scale_pages, metadata, envelope,
            workspace, plan, out, stream);
    }
}

template <typename Metadata, typename Cache>
void launch_geometry(const Tensor& q, const Tensor& positions, float scale, const Cache& cache,
                     Metadata metadata, CausalAttentionExecutionEnvelope envelope,
                     WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
                     cudaStream_t stream) {
    const auto launch = [&]<bool PackedValues, bool PackedKeys>() {
        if (q.ne[1] == CausalD256H24Kv4::QHeads) {
            launch_for<CausalD256H24Kv4, Metadata, PackedValues, PackedKeys>(
                q, positions, scale, cache.k_pages, cache.v_pages, cache.k_scale_pages,
                cache.v_scale_pages, metadata, envelope, workspace, multiprocessor_count, out,
                stream);
            return;
        }
        launch_for<CausalD256H16Kv2, Metadata, PackedValues, PackedKeys>(
            q, positions, scale, cache.k_pages, cache.v_pages, cache.k_scale_pages,
            cache.v_scale_pages, metadata, envelope, workspace, multiprocessor_count, out, stream);
    };
    switch (cache.storage) {
    case KvCacheStorage::Int8Group64:
        launch.template operator()<false, false>();
        return;
    case KvCacheStorage::RotatedInt8KeyInt4ValueGroup64:
        launch.template operator()<true, false>();
        return;
    case KvCacheStorage::RotatedLloyd4KeyInt4Value:
        launch.template operator()<true, true>();
        return;
    default:
        throw std::logic_error("FA2 prompt attention serves the INT8 cache family only");
    }
}

} // namespace

void causal_attention_prompt_i8_fa2_attention_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
    std::int32_t multiprocessor_count, Tensor& out, cudaStream_t stream) {
    const PagedKVDirectMetadata metadata{static_cast<const std::int32_t*>(cache.block_table.data)};
    launch_geometry(q, positions, scale, cache, metadata, envelope, workspace,
                    multiprocessor_count, out, stream);
}

void causal_attention_prompt_i8_fa2_batch_attention_launch(
    const Tensor& q, const Tensor& positions, const Tensor& valid_columns, const Tensor& table_rows,
    float scale, const PagedKVBatchLayerView& cache, CausalAttentionExecutionEnvelope envelope,
    WorkspaceArena& workspace, std::int32_t multiprocessor_count, Tensor& out,
    cudaStream_t stream) {
    const auto launch = [&]<bool Masked>() {
        const PagedKVBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        launch_geometry(q, positions, scale, cache, metadata, envelope, workspace,
                        multiprocessor_count, out, stream);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
