#ifndef GROUPED_GEMV_W4A16_MOE_TILING_H
#define GROUPED_GEMV_W4A16_MOE_TILING_H

#include <cstdint>

namespace sglang {
namespace npu_kernel {

struct GroupedGemvW4A16MoeTilingData {
    int32_t top_k;
    int32_t in_dim;
    int32_t out_dim;
    int32_t group_size;
    int32_t num_experts;
};

}  // namespace npu_kernel
}  // namespace sglang

#endif  // GROUPED_GEMV_W4A16_MOE_TILING_H