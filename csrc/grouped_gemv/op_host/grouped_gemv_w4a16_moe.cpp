#include "defines.h"
#include "grouped_gemv_w4a16_moe_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_grouped_gemv_w4a16_moe.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

constexpr uint32_t PADDING_BYTE = 32U;

at::Tensor get_tiling(int32_t &block_dim, int32_t top_k, int32_t in_dim, int32_t out_dim, 
                      int32_t group_size, int32_t num_experts)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    // Use AIV cores (Vector Units)
    block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    
    // align to 32 bytes
    int32_t tiling_size = (sizeof(GroupedGemvW4A16MoeTilingData) + PADDING_BYTE - 1) / PADDING_BYTE * PADDING_BYTE;
    auto tiling_buffer = at::empty({tiling_size}, at::TensorOptions().dtype(at::kByte).device(at::kCPU));

    GroupedGemvW4A16MoeTilingData *tiling_data = reinterpret_cast<GroupedGemvW4A16MoeTilingData *>(tiling_buffer.data_ptr());
    tiling_data->top_k = top_k;
    tiling_data->in_dim = in_dim;
    tiling_data->out_dim = out_dim;
    tiling_data->group_size = group_size;
    tiling_data->num_experts = num_experts;

    auto tiling_tensor = TorchNpuHepler::CopyTensorHostToDevice(tiling_buffer);
    return tiling_tensor;
}

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

    // ----------------------------------------------------------------
    // 3. Prepare Output & Tiling
    // ----------------------------------------------------------------
    at::Tensor y = at::empty({top_k, out_dim}, x_in.options());
    
    int32_t block_dim;
    at::Tensor tiling_tensor = get_tiling(block_dim, top_k, in_dim, out_dim, group_size, num_experts);

    // ----------------------------------------------------------------
    // 4. Kernel Launch
    // ----------------------------------------------------------------
    // printf("%p %p %p %p %p", x_contiguous.data_ptr(), weight.data_ptr(), scales.data_ptr(), expert_ids.data_ptr(), y.data_ptr());
    EXEC_KERNEL_CMD(grouped_gemv_w4a16_moe, block_dim, x_contiguous, weight, scales, expert_ids, y,
                    tiling_tensor);
    
    return y;
}

}  // namespace npu_kernel
}  // namespace sglang