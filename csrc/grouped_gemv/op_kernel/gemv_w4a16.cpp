#include "kernel_operator.h"
#include "zero_out_impl.h" // 新增包含

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GROUP_SIZE = 32;     
constexpr int32_t PACK_RATIO = 8;     
constexpr int32_t GROUP_TILE = 8;
constexpr int32_t TILE_N = 1024; // 调整为 1024 确保足够的 Double Buffer 空间和 temp buffer

template<typename T>
class KernelGemvW4A16 {
public:
    __aicore__ inline KernelGemvW4A16() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR y, 
                                int32_t in_dim, int32_t out_dim)
    {
        this->pipe = pipe;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->num_groups = in_dim / GROUP_SIZE;
        this->out_dim_packed = out_dim / PACK_RATIO;
        
        // 1. Tiling by Groups (Split K)
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        int32_t groups_per_core = this->num_groups / core_num;
        int32_t remain_groups = this->num_groups % core_num;

        if (core_idx < remain_groups) {
            groups_per_core += 1;
            this->g_start = core_idx * groups_per_core;
        } else {
            this->g_start = remain_groups * (groups_per_core + 1) + 
                            (core_idx - remain_groups) * groups_per_core;
        }
        this->g_end = this->g_start + groups_per_core;
        
        if (this->g_start >= this->g_end) return;

        // 2. GM Init
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        // 3. Pipe Init
        pipe->InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T));
        pipe->InitBuffer(inQueueW, BUFFER_NUM, GROUP_SIZE * (TILE_N / PACK_RATIO) * sizeof(int32_t));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, TILE_N * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, TILE_N * sizeof(float));

        // 4. Workspace Init
        uint32_t calc_size = 0;
        
        this->offset_w_half = calc_size;
        calc_size += GROUP_TILE * TILE_N * sizeof(half);

        this->offset_x_float = calc_size;
        calc_size += GROUP_SIZE * sizeof(float);

        this->offset_x_half = calc_size;
        calc_size += GROUP_SIZE * sizeof(half);
        
        this->offset_s_float = calc_size;
        calc_size += TILE_N * sizeof(float);

        // 临时 Buffer 存放当前 Group 的 Raw Sum (W * X)，未乘 Scale 前
        this->offset_temp_acc = calc_size;
        calc_size += TILE_N * sizeof(float);

        pipe->InitBuffer(calcBuf, calc_size);
    }

    __aicore__ inline void Process()
    {
        if (this->g_start >= this->g_end) return;

        // Outer Loop: Split N into Tiles
        for (int32_t n_offset = 0; n_offset < this->out_dim; n_offset += TILE_N) {
            
            int32_t current_tile_n = TILE_N;
            if (n_offset + TILE_N > this->out_dim) {
                current_tile_n = this->out_dim - n_offset;
            }
            // Align packed width to 32 bytes roughly, handled by copy params
            int32_t current_tile_n_packed = current_tile_n / PACK_RATIO;

            // Init Accumulator for this Tile (accumulating across groups)
            outQueueY.AllocTensor<float>(y_acc_local);
            Duplicate(y_acc_local, 0.0f, current_tile_n);
            
            // Inner Loop: Groups assigned to this core
            for (int32_t g = this->g_start; g < this->g_end; ++g) {
                CopyIn(g, n_offset, current_tile_n, current_tile_n_packed);
                Compute(current_tile_n, current_tile_n_packed);
            }

            outQueueY.EnQue(y_acc_local);
            CopyOut(n_offset, current_tile_n);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t g_idx, int32_t n_offset, int32_t tile_n, int32_t tile_n_packed)
    {
        uint64_t w_stride_k = this->out_dim_packed;
        uint64_t w_gm_offset = (uint64_t)g_idx * GROUP_SIZE * w_stride_k + (n_offset / PACK_RATIO);

        inQueueW.AllocTensor<int32_t>(w_local);
        
        AscendC::DataCopyExtParams w_copy_params{
            (uint16_t)GROUP_SIZE, 
            (uint32_t)(tile_n_packed * sizeof(int32_t)), 
            (uint32_t)((w_stride_k - tile_n_packed) * sizeof(int32_t)), 
            0, 0
        };
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(w_local, weightGm[w_gm_offset], w_copy_params, padParams);
        inQueueW.EnQue(w_local);

        uint64_t x_offset = (uint64_t)g_idx * GROUP_SIZE;
        inQueueX.AllocTensor<T>(x_local);
        DataCopy(x_local, xGm[x_offset], GROUP_SIZE);
        inQueueX.EnQue(x_local);

        uint64_t s_offset = (uint64_t)g_idx * this->out_dim + n_offset;
        inQueueScale.AllocTensor<T>(s_local);
        DataCopy(s_local, scalesGm[s_offset], tile_n);
        inQueueScale.EnQue(s_local);
    }

    __aicore__ inline void Compute(int32_t tile_n, int32_t tile_n_packed)
    {
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(GROUP_TILE * tile_n, offset_w_half);
        LocalTensor<float> x_float_tmp = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_x_float);
        LocalTensor<half> x_half = calcBuf.GetWithOffset<half>(GROUP_SIZE, offset_x_half);
        LocalTensor<float> s_float = calcBuf.GetWithOffset<float>(tile_n, offset_s_float);
        LocalTensor<float> temp_acc = calcBuf.GetWithOffset<float>(tile_n, offset_temp_acc);

        inQueueX.DeQue<T>(x_local);
        inQueueW.DeQue<int32_t>(w_local);
        inQueueScale.DeQue<T>(s_local);

        // Init temp_acc for current group
        Duplicate(temp_acc, 0.0f, tile_n);

        // X Cast
        Cast(x_float_tmp, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        Cast(x_half, x_float_tmp, RoundMode::CAST_ROUND, GROUP_SIZE);

        // Scale Cast
        Cast(s_float, s_local, RoundMode::CAST_NONE, tile_n);

        // Compute W * X for current group
        for (int i = 0; i < GROUP_SIZE; i += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_local[i * tile_n_packed].ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * tile_n);

            half x_buf[GROUP_TILE];
            for (int k = 0; k < GROUP_TILE; k++) {
                x_buf[k] = x_half.GetValue(i + k);
            }

            for (int k = 0; k < GROUP_TILE; ++k) {
                Axpy(temp_acc, w_half[k * tile_n], x_buf[k], tile_n);
            }
        }

        // Apply Scale and Accumulate to Global Output Tile Accumulator
        // y_acc_local += temp_acc * s_float
        MulAddDst(y_acc_local, temp_acc, s_float, tile_n);

        inQueueX.FreeTensor(x_local);
        inQueueW.FreeTensor(w_local);
        inQueueScale.FreeTensor(s_local);
    }

    __aicore__ inline void CopyOut(int32_t n_offset, int32_t tile_n)
    {
        outQueueY.DeQue<float>(y_acc_local);
        
        LocalTensor<T> y_out = y_acc_local.ReinterpretCast<T>();
        Cast(y_out, y_acc_local, RoundMode::CAST_ROUND, tile_n);
        
        PipeBarrier<PIPE_V>();
        
        // Atomic Add to Global Memory
        // 因为我们只计算了部分 Groups 的结果，需要累加到最终的 Y
        SetAtomicAdd<T>(); 
        DataCopy(yGm[n_offset], y_out, tile_n);
        SetAtomicNone();
        
        outQueueY.FreeTensor(y_acc_local);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueueX, inQueueW, inQueueScale;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int32_t> weightGm;
    AscendC::GlobalTensor<T> scalesGm;
    AscendC::GlobalTensor<T> yGm;

    LocalTensor<T> x_local;
    LocalTensor<int32_t> w_local;
    LocalTensor<T> s_local;
    LocalTensor<float> y_acc_local;

    uint32_t offset_w_half;
    uint32_t offset_x_float;
    uint32_t offset_x_half;
    uint32_t offset_s_float;
    uint32_t offset_temp_acc;

    int32_t in_dim;
    int32_t out_dim;
    int32_t num_groups;
    int32_t out_dim_packed;
    int32_t g_start;
    int32_t g_end;
};

extern "C" __global__ __aicore__ void gemv_w4a16_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR y, 
    int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelGemvW4A16<half> op;
    op.Init(&pipe, x, weight, scales, y, in_dim, out_dim);
    op.Process();
}

extern "C" __global__ __aicore__ void gemv_w4a16_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR y, 
    int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelGemvW4A16<bfloat16_t> op;
    op.Init(&pipe, x, weight, scales, y, in_dim, out_dim);
    op.Process();
}