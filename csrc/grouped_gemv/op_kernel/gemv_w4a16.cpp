#include "kernel_operator.h"
#include "zero_out_impl.h" // 新增包含

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GROUP_SIZE = 32;     
constexpr int32_t PACK_RATIO = 8;     
constexpr int32_t GROUP_TILE = 8;
constexpr int32_t TILE_N = 2048; // 调整为 1024 确保足够的 Double Buffer 空间和 temp buffer
constexpr int32_t ALIGN_N = 1024; // N 维度切分对齐要求

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
        this->n_start = 0; this->n_end = 0;
        this->g_start = 0; this->g_end = 0;       

        // --- Tiling Strategy Update (2D Splitting) ---
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        // 1. Calculate blocks along N dimension
        // 尽量按 TILE_N 切分，但块数不能超过核心数
        int32_t n_blocks = (out_dim + TILE_N - 1) / TILE_N;
        if (n_blocks > core_num) n_blocks = core_num;

        // 2. Determine Grid (N x K)
        // 每个 N-Block 分配多少个核用于 Split-K (Groups)
        int32_t cores_per_n = core_num / n_blocks;
        
        // 计算当前核属于哪个 N-Block (n_idx) 和 哪个 K-Block (k_idx)
        int32_t n_idx = core_idx / cores_per_n;
        int32_t k_idx = core_idx % cores_per_n;

        // 处理核心数无法整除的情况，多余的核心不参与计算 (Inactive)
        if (n_idx >= n_blocks) {
            return;
        }

        // 3. Split N Dimension (Aligned to 256)
        // 将 out_dim 看作若干个 ALIGN_N 单元
        int32_t total_aligned_units = (out_dim + ALIGN_N - 1) / ALIGN_N;
        int32_t units_per_block = total_aligned_units / n_blocks;
        int32_t remain_units = total_aligned_units % n_blocks;

        int32_t current_units = 0;
        int32_t start_unit = 0;

        if (n_idx < remain_units) {
            current_units = units_per_block + 1;
            start_unit = n_idx * current_units;
        } else {
            current_units = units_per_block;
            start_unit = remain_units * (units_per_block + 1) + (n_idx - remain_units) * units_per_block;
        }

        this->n_start = start_unit * ALIGN_N;
        this->n_end = this->n_start + current_units * ALIGN_N;
        
        // 修正边界，不能超过 out_dim
        if (this->n_end > out_dim) this->n_end = out_dim;

        // 4. Split K Dimension (Groups) inside the assigned N-Block
        // 将 num_groups 分配给 cores_per_n 个核心
        int32_t groups_per_core = this->num_groups / cores_per_n;
        int32_t remain_groups_k = this->num_groups % cores_per_n;

        if (k_idx < remain_groups_k) {
            groups_per_core += 1;
            this->g_start = k_idx * groups_per_core;
        } else {
            this->g_start = remain_groups_k * (groups_per_core + 1) + 
                            (k_idx - remain_groups_k) * groups_per_core;
        }
        this->g_end = this->g_start + groups_per_core;
        
        // --- Init Buffers ---
        // GM Init
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        // Pipe Init
        pipe->InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T));
        pipe->InitBuffer(inQueueW, BUFFER_NUM, GROUP_SIZE * (TILE_N / PACK_RATIO) * sizeof(int32_t));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, TILE_N * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, TILE_N * sizeof(float));

        // Workspace Init
        uint32_t calc_size = 0;
        
        this->offset_w_half = calc_size;
        calc_size += GROUP_TILE * TILE_N * sizeof(half);

        this->offset_x_float = calc_size;
        calc_size += GROUP_SIZE * sizeof(float);

        this->offset_x_half = calc_size;
        calc_size += GROUP_SIZE * sizeof(half);
        
        this->offset_s_float = calc_size;
        calc_size += TILE_N * sizeof(float);

        this->offset_temp_acc = calc_size;
        calc_size += TILE_N * sizeof(float);

        pipe->InitBuffer(calcBuf, calc_size);
    }

    __aicore__ inline void Process()
    {
        if (this->g_start >= this->g_end || this->n_start >= this->n_end) return;

        // Loop over N tiles within the range assigned to this core [n_start, n_end)
        for (int32_t n_offset = this->n_start; n_offset < this->n_end; n_offset += TILE_N) {
            
            int32_t current_tile_n = TILE_N;
            if (n_offset + TILE_N > this->n_end) { // Check against n_end, not out_dim
                current_tile_n = this->n_end - n_offset;
            }
            int32_t current_tile_n_packed = current_tile_n / PACK_RATIO;

            // Init Accumulator
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

        Duplicate(temp_acc, 0.0f, tile_n);
        inQueueX.DeQue<T>(x_local);

        // X Cast
        Cast(x_float_tmp, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        Cast(x_half, x_float_tmp, RoundMode::CAST_ROUND, GROUP_SIZE);

        inQueueScale.DeQue<T>(s_local);
        // Scale Cast
        Cast(s_float, s_local, RoundMode::CAST_NONE, tile_n);

        inQueueW.DeQue<int32_t>(w_local);
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
        
        // Atomic Add is still needed because multiple cores might work on same N but different Groups (Split K)
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
    
    // Tiling Params
    int32_t n_start;
    int32_t n_end;
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