#include "defines.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_grouped_gemv_w4a16_moe_fp16.h"
#include "aclrtlaunch_grouped_gemv_w4a16_moe_bf16.h"
#include "aclrtlaunch_fused_moe_bs1_w4a16_fp16.h"
#include "aclrtlaunch_fused_moe_bs1_w4a16_bf16.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

HOST_API at::Tensor grouped_gemv_w4a16_moe(const at::Tensor &x_in, const at::Tensor &weight,
                                           const at::Tensor &scales, const at::Tensor &offsets, 
                                           const at::Tensor &expert_ids)
{
    // ----------------------------------------------------------------
    // 1. Shape & Type Checks
    // ----------------------------------------------------------------
    TORCH_CHECK(weight.dtype() == at::kInt, "weight must be int32 (packed int4)");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(x_in.dtype() == scales.dtype(), "x and scales must have the same dtype");
    TORCH_CHECK(x_in.dtype() == offsets.dtype(), "x and offsets must have the same dtype");
    
    // Validate Weight Shape: [NumExperts, InDim, OutDim_Packed]
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D: [Experts, InDim, OutDim/8]");
    
    int32_t num_experts = weight.size(0);
    int32_t in_dim = weight.size(1);
    int32_t out_dim_packed = weight.size(2);
    int32_t out_dim = out_dim_packed * 8; 

    // Validate Scales & offsets: [Experts, Groups, OutDim]
    TORCH_CHECK(scales.dim() == 3, "Scales must be 3D");
    TORCH_CHECK(offsets.dim() == 3, "offsets must be 3D");
    TORCH_CHECK(scales.size(0) == num_experts && offsets.size(0) == num_experts, "Experts dim mismatch");
    TORCH_CHECK(scales.size(2) == out_dim && offsets.size(2) == out_dim, "OutDim mismatch");
    
    int32_t num_groups = scales.size(1);
    TORCH_CHECK(offsets.size(1) == num_groups, "offsets Groups dim mismatch");
    TORCH_CHECK(num_groups > 0, "NumGroups must be > 0");
    TORCH_CHECK(in_dim % num_groups == 0, "InDim must be divisible by NumGroups");
    
    // Validate Group Size (Targeting 128)
    int32_t group_size = in_dim / num_groups;
    TORCH_CHECK(group_size == 128, "Kernel currently optimized for group_size=128");

    int32_t top_k = expert_ids.size(0);

    // ----------------------------------------------------------------
    // 2. Preprocess Input X
    // ----------------------------------------------------------------
    at::Tensor x_contiguous;
    if (x_in.dim() == 1) {
        TORCH_CHECK(x_in.size(0) == in_dim, "X dim mismatch");
        x_contiguous = x_in.view({1, in_dim}).expand({top_k, in_dim}).contiguous();
    } else if (x_in.size(0) == 1) {
        x_contiguous = x_in.expand({top_k, in_dim}).contiguous();
    } else {
        TORCH_CHECK(x_in.numel() == top_k * in_dim, "X numel mismatch");
        x_contiguous = x_in.contiguous();
    }

    bool is_fp16 = (x_in.dtype() == at::kHalf);
    at::Tensor y = at::zeros({top_k, out_dim}, x_in.options());
    
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    auto acl_stream = c10_npu::getCurrentNPUStream();

    if (is_fp16) {
        ACLRT_LAUNCH_KERNEL(grouped_gemv_w4a16_moe_fp16)
        (block_dim, acl_stream,
         const_cast<void *>(x_contiguous.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         const_cast<void *>(scales.data_ptr()),
         const_cast<void *>(offsets.data_ptr()),
         const_cast<void *>(expert_ids.data_ptr()),
         y.data_ptr(),
         top_k, in_dim, out_dim, num_experts
        );
    } else { 
        ACLRT_LAUNCH_KERNEL(grouped_gemv_w4a16_moe_bf16)
        (block_dim, acl_stream,
         const_cast<void *>(x_contiguous.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         const_cast<void *>(scales.data_ptr()),
         const_cast<void *>(offsets.data_ptr()),
         const_cast<void *>(expert_ids.data_ptr()),
         y.data_ptr(),
         top_k, in_dim, out_dim, num_experts
        );
    }

    return y;
}

// Updated API with offsets for both layers
// Updated API with offsets for both layers
HOST_API at::Tensor fused_moe_w4a16_bs1(
    const at::Tensor &x_in, 
    const at::Tensor &w13_weight, const at::Tensor &w13_scales, const at::Tensor &w13_offsets,
    const at::Tensor &w2_weight, const at::Tensor &w2_scales, const at::Tensor &w2_offsets,
    const at::Tensor &expert_ids, const at::Tensor &topk_weights)
{
    // --- 修改开始：Flatten 并在必要时使其连续 ---
    // 使用局部变量持有 flatten 后的 tensor
    at::Tensor expert_ids_flat = expert_ids.flatten().contiguous();
    at::Tensor topk_weights_flat = topk_weights.flatten().contiguous();

    // 此时 size(0) 就是总元素个数，即 top_k
    int32_t top_k = expert_ids_flat.size(0);
    // --- 修改结束 ---

    int32_t in_dim = x_in.numel();
    int32_t num_experts = w13_weight.size(0);
    
    // W13 Checks
    int32_t w13_packed_out = w13_weight.size(2);
    int32_t inter_dim_2x = w13_packed_out * 8; 
    int32_t inter_dim = inter_dim_2x / 2;
    TORCH_CHECK(w13_offsets.size(1) == (in_dim / 128), "W13 offsets Groups mismatch");

    // W2 Checks
    int32_t out_dim = w2_weight.size(2) * 8;
    TORCH_CHECK(w2_offsets.size(1) == (inter_dim / 128), "W2 offsets Groups mismatch");

    at::Tensor y = at::zeros({1, out_dim}, x_in.options());
    
    // Workspace: W13 Output + SwiGLU Output
    int64_t w13_out_elems = top_k * inter_dim * 2;
    int64_t swiglu_out_elems = top_k * inter_dim;
    at::Tensor workspace = at::empty({w13_out_elems + swiglu_out_elems}, x_in.options());

    auto acl_stream = c10_npu::getCurrentNPUStream();
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    bool is_fp16 = (x_in.dtype() == at::kHalf);

    // 注意：下面传递 data_ptr 时，必须使用 flattened 后的 tensor (expert_ids_flat, topk_weights_flat)
    if (is_fp16) {
        ACLRT_LAUNCH_KERNEL(fused_moe_bs1_w4a16_fp16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(w13_weight.data_ptr()),
            const_cast<void *>(w13_scales.data_ptr()),
            const_cast<void *>(w13_offsets.data_ptr()),
            const_cast<void *>(w2_weight.data_ptr()),
            const_cast<void *>(w2_scales.data_ptr()),
            const_cast<void *>(w2_offsets.data_ptr()),
            const_cast<void *>(expert_ids_flat.data_ptr()),   // <--- 更新这里
            const_cast<void *>(topk_weights_flat.data_ptr()), // <--- 更新这里
            const_cast<void *>(workspace.data_ptr()),
            y.data_ptr(),
            top_k, in_dim, inter_dim, out_dim, num_experts
        );
    } else {
        ACLRT_LAUNCH_KERNEL(fused_moe_bs1_w4a16_bf16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(w13_weight.data_ptr()),
            const_cast<void *>(w13_scales.data_ptr()),
            const_cast<void *>(w13_offsets.data_ptr()),
            const_cast<void *>(w2_weight.data_ptr()),
            const_cast<void *>(w2_scales.data_ptr()),
            const_cast<void *>(w2_offsets.data_ptr()),
            const_cast<void *>(expert_ids_flat.data_ptr()),   // <--- 更新这里
            const_cast<void *>(topk_weights_flat.data_ptr()), // <--- 更新这里
            const_cast<void *>(workspace.data_ptr()),
            y.data_ptr(),
            top_k, in_dim, inter_dim, out_dim, num_experts
        );
    }

    return y;
}

} // namespace npu_kernel
} // namespace sglang