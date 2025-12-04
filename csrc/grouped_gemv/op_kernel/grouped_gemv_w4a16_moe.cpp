#include "kernel_operator.h"
#include "../op_host/grouped_gemv_w4a16_moe_tiling.h"

using namespace AscendC;

// -----------------------------------------------------------------------------
// 常量定义
// -----------------------------------------------------------------------------
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GROUP_SIZE = 32;    // 量化分组大小
constexpr int32_t PACK_RATIO = 8;     // int32 包含 8个 int4
constexpr int32_t GROUP_TILE = 8;

// -----------------------------------------------------------------------------
// Kernel 类定义
// -----------------------------------------------------------------------------
template<typename T>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, 
                                GM_ADDR y, GM_ADDR tiling)
    {
        // 1. 解析 Tiling
        auto tiling_data = reinterpret_cast<__gm__ sglang::npu_kernel::GroupedGemvW4A16MoeTilingData*>(tiling);
        this->top_k = tiling_data->top_k;
        this->in_dim = tiling_data->in_dim;
        this->out_dim = tiling_data->out_dim;
        this->out_dim_packed = out_dim / 8;
        this->num_experts = tiling_data->num_experts;
        this->num_groups = this->in_dim / GROUP_SIZE;
        
        // 2. 初始化 GM 指针
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        // 3. 初始化 Pipe 和 Queue
        pipe.InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T) + 32); 
        pipe.InitBuffer(inQueueW, BUFFER_NUM, GROUP_SIZE * out_dim_packed * sizeof(int32_t));
        pipe.InitBuffer(inQueueScale, BUFFER_NUM, out_dim * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, out_dim * sizeof(float));

        // 4. 初始化 Workspace (calcBuf) 并预计算 Offset
        // 必须保证每个 Offset 都是 32 字节对齐。
        // out_dim=256, GROUP_SIZE=32，最小的数据块是 x_half (32*2=64B)，均满足对齐。
        uint32_t current_offset = 0;

        // Group Accumulator (float)
        this->offset_group_acc = current_offset;
        current_offset += out_dim * sizeof(float);

        // Dequantized Weights (half)
        this->offset_w_half = current_offset;
        current_offset += GROUP_TILE * out_dim * sizeof(half);

        // X temp cast (float)
        this->offset_x_float = current_offset;
        current_offset += GROUP_SIZE * sizeof(float);

        // X cast (half)
        this->offset_x_half = current_offset;
        current_offset += GROUP_SIZE * sizeof(half);

        // Scale cast (float)
        this->offset_s_float = current_offset;
        current_offset += out_dim * sizeof(float);

        // 初始化 TBuf
        pipe.InitBuffer(calcBuf, current_offset);
    }

    __aicore__ inline void Process()
    {
        // for (int32_t task_idx = GetBlockIdx(); task_idx < total_tasks; task_idx += GetBlockNum()) {
        const int32_t row_idx = GetBlockIdx() % top_k;
        const int32_t expert_id = expertIdsGm.GetValue(row_idx);
        const int32_t g_idx = GetBlockIdx() / top_k;
        const int32_t g_count = GetBlockNum() / top_k + ((row_idx < GetBlockNum() % top_k) ? 1 : 0);
        const int32_t n_start = 0;

        // 1. 获取并初始化 Global Accumulator
        // 使用 GetWithOffset 获取指定内存区域
        outQueueY.AllocTensor<T>(y_local);
        auto global_acc = y_local.template ReinterpretCast<float>();
        Duplicate(global_acc, 0.0f, out_dim);

        // 2. 循环遍历 Groups
        for (int32_t g = g_idx; g < num_groups; g += g_count) {
            CopyIn(expert_id, row_idx, g, n_start);
            Compute(global_acc); // 将 global_acc 传给 Compute 进行累加
        }

        outQueueY.EnQue(y_local);
        CopyOut(row_idx, n_start);
    }

private:
    __aicore__ inline void CopyIn(int32_t expert_id, int32_t row_idx, int32_t group_idx, int32_t n_start)
    {
        // 计算 Weight Offset
        uint64_t w_stride_k = this->out_dim / PACK_RATIO;
        uint64_t w_offset = (uint64_t)expert_id * this->in_dim * w_stride_k + 
                            (uint64_t)group_idx * GROUP_SIZE * w_stride_k + 
                            (n_start / PACK_RATIO);
        
        inQueueW.AllocTensor<int32_t>(w_local);
        
        AscendC::DataCopyExtParams w_copy_params{GROUP_SIZE, (uint32_t) (out_dim_packed * sizeof(int32_t)), (uint32_t) ((w_stride_k - out_dim_packed) * sizeof(int32_t)), 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(w_local, weightGm[w_offset], w_copy_params, padParams);

        inQueueW.EnQue(w_local);

        // Load X
        uint64_t x_offset = (uint64_t)row_idx * this->in_dim + group_idx * GROUP_SIZE;
        inQueueX.AllocTensor<T>(x_local);
        DataCopy(x_local, xGm[x_offset], GROUP_SIZE);
        inQueueX.EnQue(x_local);

        // Load Scale
        uint64_t s_offset = (uint64_t)expert_id * this->num_groups * this->out_dim +
                            (uint64_t)group_idx * this->out_dim + 
                            n_start;
        inQueueScale.AllocTensor<T>(s_local);
        DataCopy(s_local, scalesGm[s_offset], out_dim);
        inQueueScale.EnQue(s_local);
    }

    __aicore__ inline void Compute(LocalTensor<float>& global_acc)
    {

        // 使用 GetWithOffset 获取 Workspace 中的临时 Tensor
        LocalTensor<float> group_acc = calcBuf.GetWithOffset<float>(out_dim, offset_group_acc);
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(GROUP_TILE * out_dim, offset_w_half);
        LocalTensor<float> x_float_tmp = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_x_float);
        LocalTensor<half> x_half = calcBuf.GetWithOffset<half>(GROUP_SIZE, offset_x_half);
        LocalTensor<float> s_float = calcBuf.GetWithOffset<float>(out_dim, offset_s_float);

        Duplicate(group_acc, 0.0f, out_dim);

        // 1. 类型转换
        inQueueX.DeQue<T>(x_local);

        // X: bf16 -> float -> half
        Cast(x_float_tmp, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        Cast(x_half, x_float_tmp, RoundMode::CAST_RINT, GROUP_SIZE);
        inQueueScale.DeQue<T>(s_local);

        // Scale: bf16 -> float
        Cast(s_float, s_local, RoundMode::CAST_NONE, out_dim);

        inQueueW.DeQue<int32_t>(w_local);

        // Weight: int32 -> int4b -> half

        for (int i = 0; i < GROUP_SIZE; i += GROUP_TILE) {

            LocalTensor<int4b_t> w_int4 = w_local[i * out_dim_packed].ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * out_dim);

            half x_buf[GROUP_TILE];
            for (int j = 0; j < GROUP_TILE; j++) {
                x_buf[j] = x_half.GetValue(i + j);
            }

            // 2. 矩阵乘 (Group Level)
            for (int i = 0; i < GROUP_TILE; ++i) {
                // group_acc += w_row[i] * x_scalar[i]
                Axpy(group_acc, w_half[i * out_dim], x_buf[i], out_dim);
            }
        }


        // 3. 应用 Scale 并累加到 Global
        MulAddDst(global_acc, group_acc, s_float, out_dim);

        inQueueW.FreeTensor(w_local);
        inQueueX.FreeTensor(x_local);
        inQueueScale.FreeTensor(s_local);
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t n_start)
    {
        outQueueY.DeQue<T>(y_local);
        Cast(y_local, y_local.template ReinterpretCast<float>(), RoundMode::CAST_RINT, out_dim);
        PipeBarrier<PIPE_V>();
        uint64_t y_offset = (uint64_t)row_idx * this->out_dim + n_start;
        AscendC::SetAtomicAdd<T>();
        DataCopy(yGm[y_offset], y_local, out_dim);
        AscendC::SetAtomicNone();
        outQueueY.FreeTensor(y_local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueueX, inQueueW, inQueueScale;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueY;
    
    // 统一管理计算缓冲区
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    // 缓冲区内的偏移量变量
    uint32_t offset_group_acc;
    uint32_t offset_w_half;
    uint32_t offset_x_float;
    uint32_t offset_x_half;
    uint32_t offset_s_float;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int32_t> weightGm;
    AscendC::GlobalTensor<T> scalesGm;
    AscendC::GlobalTensor<int32_t> expertIdsGm;
    AscendC::GlobalTensor<T> yGm;

    LocalTensor<T> x_local;
    LocalTensor<T> s_local;
    LocalTensor<T> y_local;
    LocalTensor<int32_t> w_local;


    int32_t top_k;
    int32_t in_dim;
    int32_t out_dim;
    int32_t out_dim_packed;
    int32_t num_experts;
    int32_t num_groups;
};

extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, 
    GM_ADDR y, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelGroupedGemvW4A16Moe<bfloat16_t> op;
    op.Init(x, weight, scales, expert_ids, y, tiling);
    op.Process();
}