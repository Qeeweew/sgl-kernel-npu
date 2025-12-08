#include "defines.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_gemv_w4a16_fp16.h"
#include "aclrtlaunch_gemv_w4a16_bf16.h"
#include "torch_helper.h"

namespace sglang {
namespace npu_kernel {

HOST_API at::Tensor gemv_w4a16(const at::Tensor &x_in, const at::Tensor &weight,
                               const at::Tensor &scales)
{
    TORCH_CHECK(weight.dtype() == at::kInt, "weight must be int32");
    TORCH_CHECK(x_in.dtype() == scales.dtype(), "x and scales dtype mismatch");
    bool is_fp16 = (x_in.dtype() == at::kHalf);
    
    int32_t in_dim = weight.size(0);
    int32_t out_dim_packed = weight.size(1);
    int32_t out_dim = out_dim_packed * 8;
    
    // Scale shape check: [Groups, N]
    int32_t num_groups = scales.size(0);
    int32_t group_size = in_dim / num_groups;
    
    TORCH_CHECK(group_size == 32, "AscendC custom kernel only supports group_size=32");
    
    int32_t x_k = x_in.numel();
    TORCH_CHECK(x_k == in_dim, "X numel mismatch with Weight InDim");
    
    at::Tensor x_contig = x_in.contiguous();
    at::Tensor y = at::zeros({out_dim}, x_in.options());

    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t block_dim = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());
    auto acl_stream = c10_npu::getCurrentNPUStream();

    if (is_fp16) {
        ACLRT_LAUNCH_KERNEL(gemv_w4a16_fp16)
        (block_dim, acl_stream,
         const_cast<void *>(x_contig.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         const_cast<void *>(scales.data_ptr()),
         y.data_ptr(),
         in_dim, out_dim);
    } else {
        ACLRT_LAUNCH_KERNEL(gemv_w4a16_bf16)
        (block_dim, acl_stream,
         const_cast<void *>(x_contig.data_ptr()),
         const_cast<void *>(weight.data_ptr()),
         const_cast<void *>(scales.data_ptr()),
         y.data_ptr(),
         in_dim, out_dim);
    }
    
    return y;
}

} // namespace npu_kernel
} // namespace sglang