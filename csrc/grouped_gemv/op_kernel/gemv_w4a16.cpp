#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t COMPUTE_ROWS = 32;      // 权重计算分块大小（配合指令流水和UB限制）
constexpr int32_t PACK_RATIO = 8;         // int32 包含 8 个 int4
constexpr int32_t GROUP_TILE = 8;         // 内部向量计算循环展开粒度
constexpr int32_t TILE_N = 2048;          // N 维度切分大小
constexpr int32_t ALIGN_N = 1024;          // 对齐要求

template<typename T, int32_t QUANT_GROUP_SIZE>
class KernelGemvW4A16 {
public:
    __aicore__ inline KernelGemvW4A16() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR y, 
                                int32_t in_dim, int32_t out_dim)
    {
        // 编译期检查
        static_assert(QUANT_GROUP_SIZE % 32 == 0, "QUANT_GROUP_SIZE must be multiple of 32");

        this->pipe = pipe;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->num_quant_groups = in_dim / QUANT_GROUP_SIZE;
        this->out_dim_packed = out_dim / PACK_RATIO;
        
        // --- Tiling Strategy ---
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        // 1. N 维度切分
        int32_t n_blocks = (out_dim + TILE_N - 1) / TILE_N;
        if (n_blocks > core_num) n_blocks = core_num;
        
        int32_t cores_per_n = core_num / n_blocks;
        int32_t n_idx = core_idx / cores_per_n;
        int32_t k_idx = core_idx % cores_per_n;

        if (n_idx >= n_blocks) {
            this->is_active = false;
            return;
        }
        this->is_active = true;

        // 计算当前核负责的 N 范围
        int32_t total_aligned_units = (out_dim + ALIGN_N - 1) / ALIGN_N;
        int32_t units_per_block = total_aligned_units / n_blocks;
        int32_t remain_units = total_aligned_units % n_blocks;

        int32_t start_unit = 0;
        int32_t current_units = 0;
        if (n_idx < remain_units) {
            current_units = units_per_block + 1;
            start_unit = n_idx * current_units;
        } else {
            current_units = units_per_block;
            start_unit = remain_units * (units_per_block + 1) + (n_idx - remain_units) * units_per_block;
        }

        this->n_start = start_unit * ALIGN_N;
        this->n_end = this->n_start + current_units * ALIGN_N;
        if (this->n_end > out_dim) this->n_end = out_dim;

        // 2. K 维度 (Groups) 切分
        int32_t groups_per_core = this->num_quant_groups / cores_per_n;
        int32_t remain_groups = this->num_quant_groups % cores_per_n;

        if (k_idx < remain_groups) {
            groups_per_core += 1;
            this->g_start = k_idx * groups_per_core;
        } else {
            this->g_start = remain_groups * (groups_per_core + 1) + (k_idx - remain_groups) * groups_per_core;
        }
        this->g_end = this->g_start + groups_per_core;

        // --- Init Buffers ---
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        zerosGm.SetGlobalBuffer((__gm__ T *)offsets);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        // Pipe Init
        // X: 一次性搬运整个 Group
        pipe->InitBuffer(inQueueX, BUFFER_NUM, QUANT_GROUP_SIZE * sizeof(T));
        
        // W: 依然保持分块搬运，避免 TILE_N 较大时 UB 不够
        pipe->InitBuffer(inQueueW, BUFFER_NUM, COMPUTE_ROWS * (TILE_N / PACK_RATIO) * sizeof(int32_t));
        
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, TILE_N * sizeof(T));
        pipe->InitBuffer(inQueueOffset, BUFFER_NUM, TILE_N * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, TILE_N * sizeof(float));

        // Workspace Init
        uint32_t calc_size = 0;
        
        // 用于 W 反量化的临时 buffer
        this->offset_w_half = calc_size;
        calc_size += GROUP_TILE * TILE_N * sizeof(half); 

        // X 的整个 Group 的 Float 版本 (用于 ReduceSum)
        this->offset_x_full_float = calc_size;
        calc_size += QUANT_GROUP_SIZE * sizeof(float);

        // X 的整个 Group 的 Half 版本 (用于计算 W*X)
        this->offset_x_full_half = calc_size;
        calc_size += QUANT_GROUP_SIZE * sizeof(half);
        
        this->offset_s_float = calc_size;
        calc_size += TILE_N * sizeof(float);

        this->offset_z_float = calc_size;
        calc_size += TILE_N * sizeof(float);

        this->offset_group_acc = calc_size;
        calc_size += TILE_N * sizeof(float);

        this->offset_reduce_buf = calc_size;
        calc_size += QUANT_GROUP_SIZE * sizeof(float); // ReduceSum 可能会用到对齐的空间

        pipe->InitBuffer(calcBuf, calc_size);
    }

    __aicore__ inline void Process()
    {
        if (!this->is_active || this->g_start >= this->g_end || this->n_start >= this->n_end) return;

        for (int32_t n_offset = this->n_start; n_offset < this->n_end; n_offset += TILE_N) {
            
            int32_t current_tile_n = TILE_N;
            if (n_offset + TILE_N > this->n_end) {
                current_tile_n = this->n_end - n_offset;
            }
            int32_t current_tile_n_packed = current_tile_n / PACK_RATIO;

            // Init Output Accumulator
            outQueueY.AllocTensor<float>(y_acc_global);
            Duplicate(y_acc_global, 0.0f, current_tile_n);

            for (int32_t g = this->g_start; g < this->g_end; ++g) {
                ProcessGroup(g, n_offset, current_tile_n, current_tile_n_packed);
            }

            outQueueY.EnQue(y_acc_global);
            CopyOut(n_offset, current_tile_n);
        }
    }

private:
    __aicore__ inline void ProcessGroup(
        int32_t g_idx,
        int32_t n_offset,
        int32_t tile_n,
        int32_t tile_n_packed)
    {
        // ------------------------------------------------------------
        // 1. Load X (full quant group)
        // ------------------------------------------------------------
        inQueueX.AllocTensor<T>(x_local);
        DataCopy(x_local, xGm[g_idx * QUANT_GROUP_SIZE], QUANT_GROUP_SIZE);
        inQueueX.EnQue(x_local);

        // ------------------------------------------------------------
        // 2. Prefetch first W block
        // ------------------------------------------------------------
        int32_t k_group_start = g_idx * QUANT_GROUP_SIZE;

        CopyInW(k_group_start, n_offset, tile_n, tile_n_packed, w_local_arr[0]);

        inQueueX.DeQue(x_local);

        LocalTensor<float> x_full_float =
            calcBuf.GetWithOffset<float>(QUANT_GROUP_SIZE, offset_x_full_float);
        LocalTensor<half> x_full_half =
            calcBuf.GetWithOffset<half>(QUANT_GROUP_SIZE, offset_x_full_half);

        Cast(x_full_float, x_local, RoundMode::CAST_NONE, QUANT_GROUP_SIZE);
        Cast(x_full_half, x_full_float, RoundMode::CAST_ROUND, QUANT_GROUP_SIZE);
        inQueueX.FreeTensor(x_local);

        // ------------------------------------------------------------
        // 3. Init group accumulator
        // ------------------------------------------------------------
        LocalTensor<float> group_acc_xw =
            calcBuf.GetWithOffset<float>(tile_n, offset_group_acc);
        Duplicate(group_acc_xw, 0.0f, tile_n);


        int w_idx = 0;

        // ------------------------------------------------------------
        // 4. K-loop with pipeline
        // ------------------------------------------------------------
        for (int32_t k_inner = 0; k_inner < QUANT_GROUP_SIZE; k_inner += COMPUTE_ROWS)
        {
            // ---- Prefetch next data ----
            if (k_inner + COMPUTE_ROWS < QUANT_GROUP_SIZE) {
                CopyInW(k_group_start + k_inner + COMPUTE_ROWS, n_offset, tile_n, tile_n_packed, w_local_arr[w_idx ^ 1]);
            } else {
                // Last K block: prefetch Scale & Zero
                CopyInScaleZero(g_idx, n_offset, tile_n);
            }

            // ---- Compute current block ----
            inQueueW.DeQue(w_local_arr[w_idx]);

            ComputeChunk(
                w_local_arr[w_idx],
                k_inner,
                tile_n,
                tile_n_packed,
                x_full_half,
                group_acc_xw);

            inQueueW.FreeTensor(w_local_arr[w_idx]);
            w_idx ^= 1;
        }

        // ------------------------------------------------------------
        // 5. ReduceSum(X)
        // ------------------------------------------------------------
        LocalTensor<float> reduce_buf =
            calcBuf.GetWithOffset<float>(QUANT_GROUP_SIZE, offset_reduce_buf);
        ReduceSum(reduce_buf, x_full_float, reduce_buf, QUANT_GROUP_SIZE);
        float group_sum_x = reduce_buf.GetValue(0);

        // ------------------------------------------------------------
        // 6. Apply Scale & Zero
        // ------------------------------------------------------------
        LocalTensor<float> s_float =
            calcBuf.GetWithOffset<float>(tile_n, offset_s_float);
        LocalTensor<float> z_float =
            calcBuf.GetWithOffset<float>(tile_n, offset_z_float);

        inQueueScale.DeQue<T>(s_local);
        inQueueOffset.DeQue<T>(z_local);

        Cast(s_float, s_local, RoundMode::CAST_NONE, tile_n);
        Cast(z_float, z_local, RoundMode::CAST_NONE, tile_n);

        inQueueScale.FreeTensor(s_local);
        inQueueOffset.FreeTensor(z_local);

        // Y += Sum(X) * Z * S
        Mul(z_float, z_float, s_float, tile_n);
        Axpy(y_acc_global, z_float, group_sum_x, tile_n);

        // Y += (X * W) * S
        MulAddDst(y_acc_global, group_acc_xw, s_float, tile_n);
    }


    __aicore__ inline void CopyInScaleZero(int32_t g_idx, int32_t n_offset, int32_t tile_n)
    {
        uint64_t offset = (uint64_t)g_idx * this->out_dim + n_offset;
        inQueueScale.AllocTensor<T>(s_local);
        inQueueOffset.AllocTensor<T>(z_local);
        DataCopy(s_local, scalesGm[offset], tile_n);
        DataCopy(z_local, zerosGm[offset], tile_n);
        inQueueScale.EnQue(s_local);
        inQueueOffset.EnQue(z_local);
    }

    __aicore__ inline void CopyInW(
        int32_t k_global,
        int32_t n_offset,
        int32_t tile_n,
        int32_t tile_n_packed,
        LocalTensor<int32_t>& w_local)
    {
        uint64_t w_stride = this->out_dim_packed;
        uint64_t w_offset =
            (uint64_t)k_global * w_stride + (n_offset / PACK_RATIO);

        inQueueW.AllocTensor<int32_t>(w_local);

        AscendC::DataCopyExtParams params{
            (uint16_t)COMPUTE_ROWS,
            (uint32_t)(tile_n_packed * sizeof(int32_t)),
            (uint32_t)((w_stride - tile_n_packed) * sizeof(int32_t)),
            0, 0
        };
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};

        DataCopyPad(w_local, weightGm[w_offset], params, padParams);
        inQueueW.EnQue(w_local);
    }

    __aicore__ inline void ComputeChunk(
        LocalTensor<int32_t>& w_local,
        int32_t x_offset_idx,
        int32_t tile_n,
        int32_t tile_n_packed,
        LocalTensor<half>& x_full_half,
        LocalTensor<float>& group_acc_xw)
    {
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(
            GROUP_TILE * tile_n, offset_w_half);

        LocalTensor<uint64_t> x_u64 = x_full_half.ReinterpretCast<uint64_t>();
        constexpr int STEP = 4;
        half x_val_buf[COMPUTE_ROWS];
        uint64_t* x_val_buf_u64 = (uint64_t*)x_val_buf;
        for (int i = 0; i < COMPUTE_ROWS / STEP; ++i) {
            x_val_buf_u64[i] = x_u64.GetValue(x_offset_idx / STEP + i);
        }

        for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_local[i * tile_n_packed].ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * tile_n);
            for (int k = 0; k < GROUP_TILE; ++k) {
                Axpy(group_acc_xw, w_half[k * tile_n], x_val_buf[i + k], tile_n);
            }
        }
    }

    __aicore__ inline void CopyOut(int32_t n_offset, int32_t tile_n)
    {
        outQueueY.DeQue<float>(y_acc_global);
        LocalTensor<T> y_out = y_acc_global.ReinterpretCast<T>();
        Cast(y_out, y_acc_global, RoundMode::CAST_ROUND, tile_n);
        PipeBarrier<PIPE_V>();
        SetAtomicAdd<T>(); 
        DataCopy(yGm[n_offset], y_out, tile_n);
        SetAtomicNone();
        outQueueY.FreeTensor(y_acc_global);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueueX, inQueueW, inQueueScale, inQueueOffset;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int32_t> weightGm;
    AscendC::GlobalTensor<T> scalesGm;
    AscendC::GlobalTensor<T> zerosGm;
    AscendC::GlobalTensor<T> yGm;

    LocalTensor<T> x_local, s_local, z_local;
    LocalTensor<int32_t> w_local_arr[2];
    LocalTensor<float> y_acc_global;

    uint32_t offset_w_half;
    uint32_t offset_x_full_float;
    uint32_t offset_x_full_half;
    uint32_t offset_s_float;
    uint32_t offset_z_float;
    uint32_t offset_group_acc;
    uint32_t offset_reduce_buf;

    int32_t in_dim, out_dim;
    int32_t num_quant_groups;
    int32_t out_dim_packed;
    bool is_active;
    int32_t n_start, n_end, g_start, g_end;
};

// 实例化模板，默认 GroupSize = 128
extern "C" __global__ __aicore__ void gemv_w4a16_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR y, 
    int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    // 这里指定了 128，如果需要支持其他 Group Size，需要增加额外的 kernel entry 或 switch case
    KernelGemvW4A16<half, 128> op; 
    op.Init(&pipe, x, weight, scales, offsets, y, in_dim, out_dim);
    op.Process();
}

extern "C" __global__ __aicore__ void gemv_w4a16_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR y, 
    int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelGemvW4A16<bfloat16_t, 128> op;
    op.Init(&pipe, x, weight, scales, offsets, y, in_dim, out_dim);
    op.Process();
}