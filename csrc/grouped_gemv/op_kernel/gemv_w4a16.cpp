#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"

using namespace AscendC;

// -----------------------------------------------------------------------------
// Constants (keep same meaning as your Pipe version)
// -----------------------------------------------------------------------------
constexpr int32_t COMPUTE_ROWS = 32;   // K inner tile
constexpr int32_t PACK_RATIO   = 8;    // 1 int32 packs 8 int4
constexpr int32_t GROUP_TILE   = 8;    // unroll factor
constexpr int32_t TILE_N       = 2048; // N tile
constexpr int32_t ALIGN_N      = 1024; // N alignment in tiling

// -----------------------------------------------------------------------------
// HardEvent IDs (only 0/1/2/3; avoid 6/7)
// Same HardEvent must use the same event_id everywhere.
// -----------------------------------------------------------------------------
static constexpr int EID_MTE2_V = 0;  // MTE2 -> V
static constexpr int EID_V_MTE2 = 1;  // V -> MTE2
static constexpr int EID_V_MTE3 = 2;  // V -> MTE3
static constexpr int EID_MTE3_V = 3;  // MTE3 -> V

template<typename T, int32_t QUANT_GROUP_SIZE>
class KernelGemvW4A16 {
public:
    __aicore__ inline KernelGemvW4A16() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR y,
                               int32_t in_dim, int32_t out_dim)
    {
        static_assert(QUANT_GROUP_SIZE % 32 == 0, "QUANT_GROUP_SIZE must be multiple of 32");

        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->num_quant_groups = in_dim / QUANT_GROUP_SIZE;
        this->out_dim_packed = out_dim / PACK_RATIO;

        // --- Tiling Strategy (DO NOT change; identical to Pipe version) ---
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        // 1) N dimension split
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

        // Current core's N range
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
        this->n_end   = this->n_start + current_units * ALIGN_N;
        if (this->n_end > out_dim) this->n_end = out_dim;

        // 2) K dimension split (groups split)
        int32_t groups_per_core = this->num_quant_groups / cores_per_n;
        int32_t remain_groups = this->num_quant_groups % cores_per_n;

        if (k_idx < remain_groups) {
            groups_per_core += 1;
            this->g_start = k_idx * groups_per_core;
        } else {
            this->g_start = remain_groups * (groups_per_core + 1) +
                            (k_idx - remain_groups) * groups_per_core;
        }
        this->g_end = this->g_start + groups_per_core;

        // --- GM init ---
        xGm.SetGlobalBuffer((__gm__ T*)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t*)weight);
        scalesGm.SetGlobalBuffer((__gm__ T*)scales);
        zerosGm.SetGlobalBuffer((__gm__ T*)offsets);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        BuildLocalTensors();
    }

    __aicore__ inline void Process()
    {
        if (!this->is_active || this->g_start >= this->g_end || this->n_start >= this->n_end) return;

        for (int32_t n_offset = this->n_start; n_offset < this->n_end; n_offset += TILE_N) {
            int32_t cur_n_len = TILE_N;
            if (n_offset + TILE_N > this->n_end) cur_n_len = this->n_end - n_offset;

            // out_dim guaranteed multiple of 8 (PACK_RATIO), so no tail handling needed
            int32_t cur_n_packed = cur_n_len / PACK_RATIO;

            // Init output accumulator (fp32)
            Duplicate(y_fp32, 0.0f, cur_n_len);

            for (int32_t g = this->g_start; g < this->g_end; ++g) {
                ProcessGroup(g, n_offset, cur_n_len, cur_n_packed);
            }

            CopyOut(n_offset, cur_n_len);
        }
    }

private:
    __aicore__ inline void BuildLocalTensors()
    {
        // Keep UB layout similar to your moe low-level style (manual LocalTensor)
        uint32_t addr = 0;

        // X group
        x_local = LocalTensor<T>(TPosition::VECIN, addr, QUANT_GROUP_SIZE);
        addr += (uint32_t)(QUANT_GROUP_SIZE * sizeof(T));

        // W ping-pong buffers (int32 packed)
        constexpr int32_t TILE_N_PACKED = TILE_N / PACK_RATIO;
        w_local[0] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * TILE_N_PACKED);
        addr += (uint32_t)(COMPUTE_ROWS * TILE_N_PACKED * sizeof(int32_t));
        w_local[1] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * TILE_N_PACKED);
        addr += (uint32_t)(COMPUTE_ROWS * TILE_N_PACKED * sizeof(int32_t));

        // Scale / Zero
        s_local = LocalTensor<T>(TPosition::VECIN, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(T));
        z_local = LocalTensor<T>(TPosition::VECIN, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(T));

        // Global output accumulator (fp32)
        y_fp32 = LocalTensor<float>(TPosition::VECOUT, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));

        // Group accumulator MUST be fp32 (as requested)
        group_acc_fp32 = LocalTensor<float>(TPosition::VECCALC, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float) + 256);

        // W dequant temp
        w_half = LocalTensor<half>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += (uint32_t)(GROUP_TILE * TILE_N * sizeof(half));

        // X float for reduce
        x_float_full = LocalTensor<float>(TPosition::VECCALC, addr, QUANT_GROUP_SIZE);
        addr += (uint32_t)(QUANT_GROUP_SIZE * sizeof(float));

        reduce_buf = LocalTensor<float>(TPosition::VECCALC, addr, QUANT_GROUP_SIZE);
        addr += (uint32_t)(QUANT_GROUP_SIZE * sizeof(float));

        // s/z float for higher precision application
        s_float = LocalTensor<float>(TPosition::VECCALC, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));
        z_float = LocalTensor<float>(TPosition::VECCALC, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));
    }

    __aicore__ inline void PrefetchW(int32_t k_global, int32_t n_offset,
                                     int32_t cur_n_packed, int buf_id)
    {
        uint64_t w_stride = (uint64_t)out_dim_packed;
        uint64_t w_offset = (uint64_t)k_global * w_stride + (uint64_t)(n_offset / PACK_RATIO);

        DataCopyExtParams p{
            (uint16_t)COMPUTE_ROWS,
            (uint32_t)(cur_n_packed * sizeof(int32_t)),
            (uint32_t)((w_stride - (uint64_t)cur_n_packed) * sizeof(int32_t)),
            0, 0
        };
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(w_local[buf_id], weightGm[w_offset], p, pad);
    }

    __aicore__ inline void PrefetchSZ(int32_t g_idx, int32_t n_offset, int32_t cur_n_len)
    {
        uint64_t offset = (uint64_t)g_idx * (uint64_t)out_dim + (uint64_t)n_offset;
        DataCopy(s_local, scalesGm[offset], cur_n_len);
        DataCopy(z_local, zerosGm[offset], cur_n_len);
        // caller will issue ONE SetFlag<MTE2_V>(0) after both copies
    }

    __aicore__ inline void ComputeChunk(LocalTensor<int32_t>& w_i32,
                                        int32_t x_offset_idx,
                                        int32_t cur_n_len,
                                        int32_t cur_n_packed,
                                        LocalTensor<T>& x_full,
                                        LocalTensor<float>& acc_fp32)
    {
        LocalTensor<uint64_t> x_u64 = x_full.template ReinterpretCast<uint64_t>();

        constexpr int STEP = 4;
        half x_val_buf[COMPUTE_ROWS];
        uint64_t* x_val_buf_u64 = (uint64_t*)x_val_buf;
        for (int i = 0; i < COMPUTE_ROWS / STEP; ++i) {
            x_val_buf_u64[i] = x_u64.GetValue(x_offset_idx / STEP + i);
        }

        for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_i32[i * cur_n_packed].template ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * cur_n_len);
            for (int k = 0; k < GROUP_TILE; ++k) {
                // fp32 accumulate (required)
                Axpy(acc_fp32, w_half[k * cur_n_len], x_val_buf[i + k], cur_n_len);
            }
        }
    }

    __aicore__ inline void ProcessGroup(int32_t g_idx, int32_t n_offset,
                                        int32_t cur_n_len, int32_t cur_n_packed)
    {
        // 1) Load X (full quant group)
        DataCopy(x_local, xGm[(uint64_t)g_idx * (uint64_t)QUANT_GROUP_SIZE], QUANT_GROUP_SIZE);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 2) Prefetch first W block
        int32_t k_group_start = g_idx * QUANT_GROUP_SIZE;
        PrefetchW(k_group_start, n_offset, cur_n_packed, /*buf=*/0);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 3) Init group accumulator (fp32)
        Duplicate(group_acc_fp32, 0.0f, cur_n_len);

        // 4) ReduceSum(X): keep WholeReduceSum + BlockReduceSum (same as moe LL)
        Cast(x_float_full, x_local, RoundMode::CAST_NONE, QUANT_GROUP_SIZE);
        WholeReduceSum<float>(reduce_buf, x_float_full, 64, QUANT_GROUP_SIZE / 64, 1, 1, 8);
        BlockReduceSum<float>(reduce_buf, reduce_buf, 1, QUANT_GROUP_SIZE / 64, 1, 1, 8);

        // 5) K-loop with W ping-pong + last prefetch SZ (replicate moe LL)
        int ping = 0;
        for (int32_t k_inner = 0; k_inner < QUANT_GROUP_SIZE; k_inner += COMPUTE_ROWS) {
            // W ready
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            if (k_inner != 0) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }

            // prefetch next
            if (k_inner + COMPUTE_ROWS < QUANT_GROUP_SIZE) {
                int next = ping ^ 1;
                PrefetchW(k_group_start + k_inner + COMPUTE_ROWS, n_offset, cur_n_packed, next);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            } else {
                PrefetchSZ(g_idx, n_offset, cur_n_len);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            }

            // compute current
            ComputeChunk(w_local[ping], k_inner, cur_n_len, cur_n_packed, x_local, group_acc_fp32);

            // protect overwrite on next round
            if (k_inner + COMPUTE_ROWS < QUANT_GROUP_SIZE) {
                SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }
            ping ^= 1;
        }

        float group_sum_x = reduce_buf.GetValue(0);

        // Wait SZ ready
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 6) Apply Scale & Zero in fp32
        Cast(s_float, s_local, RoundMode::CAST_NONE, cur_n_len);
        Cast(z_float, z_local, RoundMode::CAST_NONE, cur_n_len);

        // Y += Sum(X) * (Z * S)
        Mul(z_float, z_float, s_float, cur_n_len);
        Axpy(y_fp32, z_float, group_sum_x, cur_n_len);

        // Y += (X*W) * S
        MulAddDst(y_fp32, group_acc_fp32, s_float, cur_n_len);
    }

    __aicore__ inline void CopyOut(int32_t n_offset, int32_t cur_n_len)
    {
        LocalTensor<T> y_out = y_fp32.template ReinterpretCast<T>();
        Cast(y_out, y_fp32, RoundMode::CAST_ROUND, cur_n_len);

        // V -> MTE3
        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        SetAtomicAdd<T>();
        DataCopy(yGm[(uint64_t)n_offset], y_out, cur_n_len);
        SetAtomicNone();

        // Close pair immediately (avoid outstanding MTE3_V across iterations)
        SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
    }

private:
    GlobalTensor<T>       xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<T>       scalesGm;
    GlobalTensor<T>       zerosGm;
    GlobalTensor<T>       yGm;

    // UB
    LocalTensor<T>       x_local;
    LocalTensor<int32_t> w_local[2];
    LocalTensor<T>       s_local, z_local;

    LocalTensor<float>   y_fp32;            // final accumulator per N tile (fp32)
    LocalTensor<float>   group_acc_fp32;    // group accumulator (fp32)  <-- required
    LocalTensor<half>    w_half;            // dequant temp

    LocalTensor<float>   x_float_full;      // for reduce
    LocalTensor<float>   reduce_buf;        // for reduce
    LocalTensor<float>   s_float;           // scale fp32
    LocalTensor<float>   z_float;           // zero fp32

    int32_t in_dim = 0, out_dim = 0;
    int32_t num_quant_groups = 0;
    int32_t out_dim_packed = 0;

    bool is_active = false;
    int32_t n_start = 0, n_end = 0, g_start = 0, g_end = 0;
};

// -----------------------------------------------------------------------------
// Kernel Entry
// -----------------------------------------------------------------------------
extern "C" __global__ __aicore__ void gemv_w4a16_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR y,
    int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();

    KernelGemvW4A16<half, 128> op;
    op.Init(x, weight, scales, offsets, y, in_dim, out_dim);
    op.Process();
}

extern "C" __global__ __aicore__ void gemv_w4a16_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR y,
    int32_t in_dim, int32_t out_dim)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();

    KernelGemvW4A16<bfloat16_t, 128> op;
    op.Init(x, weight, scales, offsets, y, in_dim, out_dim);
    op.Process();
}
