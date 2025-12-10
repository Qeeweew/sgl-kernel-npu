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
                                           const at::Tensor &scales, const at::Tensor &expert_ids)
{
    // ----------------------------------------------------------------
    // 1. Shape & Type Checks
    // ----------------------------------------------------------------
    TORCH_CHECK(weight.dtype() == at::kInt, "weight must be int32 (packed int4)");
    TORCH_CHECK(expert_ids.dtype() == at::kInt, "expert_ids must be int32");
    TORCH_CHECK(x_in.dtype() == scales.dtype(), "x and scales must have the same dtype (float16 or bfloat16)");
    
    // Validate Weight Shape: [NumExperts, InDim, OutDim_Packed]
    // Python Repack logic: view(E, K, N/8) -> [E, InDim, OutDim/8]
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D: [Experts, InDim, OutDim/8]");
    
    int32_t num_experts = weight.size(0);
    int32_t in_dim = weight.size(1);
    int32_t out_dim_packed = weight.size(2);
    int32_t out_dim = out_dim_packed * 8; // 8 * 4bit per int32

    // Validate Scales Shape: [NumExperts, Groups, OutDim]
    // CompressedTensors format: [E, Groups, N]
    TORCH_CHECK(scales.dim() == 3, "Scales must be 3D: [Experts, Groups, OutDim]");
    TORCH_CHECK(scales.size(0) == num_experts, "Scales Experts dim mismatch");
    TORCH_CHECK(scales.size(2) == out_dim, "Scales OutDim mismatch (scale.shape[2] != weight.out_dim)");
    
    int32_t num_groups = scales.size(1);
    TORCH_CHECK(num_groups > 0, "NumGroups must be > 0");
    TORCH_CHECK(in_dim % num_groups == 0, "InDim must be divisible by NumGroups");
    
    // Calculate Group Size
    int32_t group_size = in_dim / num_groups;

    int32_t top_k = expert_ids.size(0);

    // ----------------------------------------------------------------
    // 2. Preprocess Input X (Broadcasting & Contiguous)
    // ----------------------------------------------------------------
    // Goal: Make x shape [TopK, InDim]
    at::Tensor x_contiguous;
    
    // Check InDim consistency first
    if (x_in.dim() == 1) {
        TORCH_CHECK(x_in.size(0) == in_dim, "X input dimension mismatch with Weight InDim");
    } else {
        TORCH_CHECK(x_in.size(-1) == in_dim, "X input last dimension mismatch");
    }

    if (x_in.dim() == 1) {
        // Case: [InDim] -> View [1, InDim] -> Expand [TopK, InDim]
        x_contiguous = x_in.view({1, in_dim}).expand({top_k, in_dim}).contiguous();
    } else if (x_in.size(0) == 1) {
        // Case: [1, InDim] -> Expand [TopK, InDim]
        x_contiguous = x_in.expand({top_k, in_dim}).contiguous();
    } else {
        // Case: [TopK, InDim] or [BS, InDim] where BS=TopK
        TORCH_CHECK(x_in.numel() == top_k * in_dim, "X numel mismatch with TopK * InDim");
        x_contiguous = x_in.contiguous();
    }

    bool is_fp16 = (x_in.dtype() == at::kHalf);
    bool is_bf16 = (x_in.dtype() == at::kBFloat16);
    TORCH_CHECK(is_fp16 || is_bf16, "x must be float16 or bfloat16");

    // ----------------------------------------------------------------
    // 3. Prepare Output & Tiling
    // ----------------------------------------------------------------
    at::Tensor y = at::zeros({top_k, out_dim}, x_in.options());
    
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    // 获取 Stream
    auto acl_stream = c10_npu::getCurrentNPUStream();

    // ----------------------------------------------------------------
    // 4. Kernel Launch (直接传参，不传 Tiling Tensor)
    // ----------------------------------------------------------------
    if (is_fp16) {
        ACLRT_LAUNCH_KERNEL(grouped_gemv_w4a16_moe_fp16)
        (block_dim, acl_stream,
         const_cast<void *>(x_contiguous.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         const_cast<void *>(scales.data_ptr()),
         const_cast<void *>(expert_ids.data_ptr()),
         y.data_ptr(),
         top_k, in_dim, out_dim, num_experts
        );
    } else { // BF16
        ACLRT_LAUNCH_KERNEL(grouped_gemv_w4a16_moe_bf16)
        (block_dim, acl_stream,
         const_cast<void *>(x_contiguous.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         const_cast<void *>(scales.data_ptr()),
         const_cast<void *>(expert_ids.data_ptr()),
         y.data_ptr(),
         top_k, in_dim, out_dim, num_experts
        );
    }

    return y;
}

HOST_API at::Tensor fused_moe_w4a16_bs1(
    const at::Tensor &x_in, 
    const at::Tensor &w13_weight, const at::Tensor &w13_scales,
    const at::Tensor &w2_weight, const at::Tensor &w2_scales,
    const at::Tensor &expert_ids, const at::Tensor &topk_weights)
{
    // 1. Checks (Similar to previous)
    // x_in: [1, InDim]
    int32_t top_k = expert_ids.size(0);
    int32_t in_dim = x_in.numel();
    int32_t num_experts = w13_weight.size(0);
    
    // W13: [E, In, 2*InterPacked]
    int32_t w13_packed_out = w13_weight.size(2);
    int32_t inter_dim_2x = w13_packed_out * 8; 
    int32_t inter_dim = inter_dim_2x / 2;

    // W2: [E, Inter, OutPacked]
    int32_t out_dim = w2_weight.size(2) * 8;

    // 2. Alloc Output Y
    // Output shape: [1, OutDim] (Since BS=1 and we sum up)
    at::Tensor y = at::zeros({1, out_dim}, x_in.options()); // MUST be zeros for AtomicAdd

    // 3. Alloc Workspace
    // Size = (TopK * 2 * Inter) + (TopK * Inter) elements
    int64_t w13_out_elems = top_k * inter_dim * 2;
    int64_t swiglu_out_elems = top_k * inter_dim;
    int64_t total_workspace_elems = w13_out_elems + swiglu_out_elems;
    
    at::Tensor workspace = at::empty({total_workspace_elems}, x_in.options());

    // 4. Launch
    auto acl_stream = c10_npu::getCurrentNPUStream();
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    bool is_fp16 = (x_in.dtype() == at::kHalf);
    bool is_bf16 = (x_in.dtype() == at::kBFloat16);
    TORCH_CHECK(is_fp16 || is_bf16, "x must be float16 or bfloat16");

    // 检查 topk_weights 必须是 float32 类型
    TORCH_CHECK(topk_weights.dtype() == at::kFloat, 
                "topk_weights must be float32, but got ", topk_weights.dtype());

    if (is_fp16) {
        ACLRT_LAUNCH_KERNEL(fused_moe_bs1_w4a16_fp16)(
            block_dim, acl_stream,
            const_cast<void *>(x_in.data_ptr()),
            const_cast<void *>(w13_weight.data_ptr()),
            const_cast<void *>(w13_scales.data_ptr()),
            const_cast<void *>(w2_weight.data_ptr()),
            const_cast<void *>(w2_scales.data_ptr()),
            const_cast<void *>(expert_ids.data_ptr()),
            const_cast<void *>(topk_weights.data_ptr()),
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
            const_cast<void *>(w2_weight.data_ptr()),
            const_cast<void *>(w2_scales.data_ptr()),
            const_cast<void *>(expert_ids.data_ptr()),
            const_cast<void *>(topk_weights.data_ptr()),
            const_cast<void *>(workspace.data_ptr()),
            y.data_ptr(),
            top_k, in_dim, inter_dim, out_dim, num_experts
        );
    }

    return y;
}


}  // namespace npu_kernel
}  // namespace sglang