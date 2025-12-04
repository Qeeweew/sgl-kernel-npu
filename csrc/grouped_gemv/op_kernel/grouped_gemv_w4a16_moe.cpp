#include "kernel_operator.h"
#include "../op_host/grouped_gemv_w4a16_moe_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template<typename T>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, 
                                GM_ADDR y, GM_ADDR tiling)
    {

        auto tiling_data = reinterpret_cast<__gm__ sglang::npu_kernel::GroupedGemvW4A16MoeTilingData*>(tiling);
        this->top_k = tiling_data->top_k;
        this->in_dim = tiling_data->in_dim;
        this->out_dim = tiling_data->out_dim;
        this->group_size = tiling_data->group_size;
        this->num_experts = tiling_data->num_experts;

        // Init GM pointers
        // x: [TopK, InDim] -> T
        xGm.SetGlobalBuffer((__gm__ T *)x);
        // weight: [E, InDim, OutDim/8] -> int32
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        // scales: [E, Groups, OutDim] -> T
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        // expert_ids: [TopK] -> int32
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids);
        // y: [TopK, OutDim] -> T
        yGm.SetGlobalBuffer((__gm__ T *)y);

        // Pipe Init Skeleton
        // pipe.InitBuffer(inQueueX, BUFFER_NUM, ...);
    }

    __aicore__ inline void Process()
    {
        // Simple loop over TopK experts
        for (int32_t i = 0; i < this->top_k; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t index)
    {
        // ...
    }

    __aicore__ inline void Compute(int32_t index)
    {
        // ...
    }

    __aicore__ inline void CopyOut(int32_t index)
    {
        // ...
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueWeight, inQueueScales;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int32_t> weightGm;
    AscendC::GlobalTensor<T> scalesGm;
    AscendC::GlobalTensor<int32_t> expertIdsGm;
    AscendC::GlobalTensor<T> yGm;

    int32_t top_k;
    int32_t in_dim;
    int32_t out_dim;
    int32_t group_size;
    int32_t num_experts;
};

extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, 
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    
    // Defaulting to bfloat16_t for skeleton.
    // In production, dispatch based on input Dtype or Tiling flag.
    KernelGroupedGemvW4A16Moe<bfloat16_t> op;
    op.Init(x, weight, scales, expert_ids, y, tiling);
    op.Process();
}