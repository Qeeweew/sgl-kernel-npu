#include "kernel_operator.h"
#include "../op_host/grouped_gemv_w4a16_moe_tiling.h"

using namespace AscendC;

// -----------------------------------------------------------------------------
// 常量定义
// -----------------------------------------------------------------------------
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_N = 1024;       // 每个Tile处理256个输出列
constexpr int32_t GROUP_SIZE = 32;    // 量化分组大小
constexpr int32_t PACK_RATIO = 8;     // int32 包含 8个 int4
constexpr int32_t TILE_N_PACKED = TILE_N / PACK_RATIO; // int32个数

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
        this->num_experts = tiling_data->num_experts;
        this->num_groups = this->in_dim / GROUP_SIZE;

        this->tiles_per_row = this->out_dim / TILE_N; 
        this->total_tasks = this->top_k * this->tiles_per_row;
        
        // 2. 初始化 GM 指针
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        // 3. 初始化 Pipe 和 Queue
        pipe.InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T) + 32); 
        pipe.InitBuffer(inQueueW, BUFFER_NUM, GROUP_SIZE * TILE_N_PACKED * sizeof(int32_t));
        pipe.InitBuffer(inQueueScale, BUFFER_NUM, TILE_N * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, TILE_N * sizeof(float));

        // 4. 初始化 Workspace (calcBuf) 并预计算 Offset
        // 必须保证每个 Offset 都是 32 字节对齐。
        // TILE_N=256, GROUP_SIZE=32，最小的数据块是 x_half (32*2=64B)，均满足对齐。
        uint32_t current_offset = 0;

        // Group Accumulator (float)
        this->offset_group_acc = current_offset;
        current_offset += TILE_N * sizeof(float);

        // Dequantized Weights (half)
        this->offset_w_half = current_offset;
        current_offset += GROUP_SIZE * TILE_N * sizeof(half);

        // X temp cast (float)
        this->offset_x_float = current_offset;
        current_offset += GROUP_SIZE * sizeof(float);

        // X cast (half)
        this->offset_x_half = current_offset;
        current_offset += GROUP_SIZE * sizeof(half);

        // Scale cast (float)
        this->offset_s_float = current_offset;
        current_offset += TILE_N * sizeof(float);

        // 初始化 TBuf
        pipe.InitBuffer(calcBuf, current_offset);
    }

    __aicore__ inline void Process()
    {
        for (int32_t task_idx = GetBlockIdx(); task_idx < total_tasks; task_idx += GetBlockNum()) {
            ComputeTile(task_idx);
        }
    }

private:
    __aicore__ inline void ComputeTile(int32_t task_idx)
    {
        int32_t row_idx = task_idx / tiles_per_row;
        int32_t tile_n_idx = task_idx % tiles_per_row;
        int32_t n_start = tile_n_idx * TILE_N;
        int32_t expert_id = expertIdsGm.GetValue(row_idx);

        // 1. 获取并初始化 Global Accumulator
        // 使用 GetWithOffset 获取指定内存区域
        outQueueY.AllocTensor<T>(y_local);
        auto global_acc = y_local.template ReinterpretCast<float>();
        Duplicate(global_acc, 0.0f, TILE_N);

        // 2. 循环遍历 Groups
        for (int32_t g = 0; g < num_groups; ++g) {
            CopyIn(expert_id, row_idx, g, n_start);
            Compute(global_acc); // 将 global_acc 传给 Compute 进行累加
        }

        outQueueY.EnQue(y_local);
        CopyOut(row_idx, n_start);
    }

    __aicore__ inline void CopyIn(int32_t expert_id, int32_t row_idx, int32_t group_idx, int32_t n_start)
    {
        // 计算 Weight Offset
        uint64_t w_stride_k = this->out_dim / PACK_RATIO;
        uint64_t w_offset = (uint64_t)expert_id * this->in_dim * w_stride_k + 
                            (uint64_t)group_idx * GROUP_SIZE * w_stride_k + 
                            (n_start / PACK_RATIO);
        
        inQueueW.AllocTensor<int32_t>(w_local);
        
        AscendC::DataCopyExtParams w_copy_params{GROUP_SIZE, (uint32_t) (TILE_N_PACKED * sizeof(int32_t)), (uint32_t) ((w_stride_k - TILE_N_PACKED) * sizeof(int32_t)), 0, 0};
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
        DataCopy(s_local, scalesGm[s_offset], TILE_N);
        inQueueScale.EnQue(s_local);
    }

    __aicore__ inline void Compute(LocalTensor<float>& global_acc)
    {
        inQueueW.DeQue<int32_t>(w_local);
        inQueueX.DeQue<T>(x_local);
        inQueueScale.DeQue<T>(s_local);

        // 使用 GetWithOffset 获取 Workspace 中的临时 Tensor
        LocalTensor<float> group_acc = calcBuf.GetWithOffset<float>(TILE_N, offset_group_acc);
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(GROUP_SIZE * TILE_N, offset_w_half);
        LocalTensor<float> x_float_tmp = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_x_float);
        LocalTensor<half> x_half = calcBuf.GetWithOffset<half>(GROUP_SIZE, offset_x_half);
        LocalTensor<float> s_float = calcBuf.GetWithOffset<float>(TILE_N, offset_s_float);

        // 1. 类型转换
        // Weight: int32 -> int4b -> half
        LocalTensor<int4b_t> w_int4 = w_local.ReinterpretCast<int4b_t>();
        Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_SIZE * TILE_N);

        // X: bf16 -> float -> half
        Cast(x_float_tmp, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        Cast(x_half, x_float_tmp, RoundMode::CAST_RINT, GROUP_SIZE);

        // Scale: bf16 -> float
        Cast(s_float, s_local, RoundMode::CAST_NONE, TILE_N);

        half x_buf[GROUP_SIZE];
        for (int i = 0; i < GROUP_SIZE; i++) {
            x_buf[i] = x_half.GetValue(i);
        }

        // 2. 矩阵乘 (Group Level)
        Duplicate(group_acc, 0.0f, TILE_N);
        for (int i = 0; i < GROUP_SIZE; ++i) {
            // group_acc += w_row[i] * x_scalar[i]
            Axpy(group_acc, w_half[i * TILE_N], x_buf[i], TILE_N);
        }

        // 3. 应用 Scale 并累加到 Global
        MulAddDst(global_acc, group_acc, s_float, TILE_N);

        inQueueW.FreeTensor(w_local);
        inQueueX.FreeTensor(x_local);
        inQueueScale.FreeTensor(s_local);
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t n_start)
    {
        outQueueY.DeQue<T>(y_local);
        Cast(y_local, y_local.template ReinterpretCast<float>(), RoundMode::CAST_RINT, TILE_N);
        PipeBarrier<PIPE_V>();
        uint64_t y_offset = (uint64_t)row_idx * this->out_dim + n_start;
        DataCopy(yGm[y_offset], y_local, TILE_N);
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
    int32_t num_experts;
    int32_t num_groups;
    
    int32_t tiles_per_row;
    int32_t total_tasks;
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