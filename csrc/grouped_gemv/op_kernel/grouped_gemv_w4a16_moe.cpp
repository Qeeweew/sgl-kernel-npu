#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "zero_out_impl.h"
#include <type_traits>

using namespace AscendC;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int32_t GROUP_SIZE   = 128;
constexpr int32_t COMPUTE_ROWS = 32;
constexpr int32_t PACK_RATIO   = 8;
constexpr int32_t GROUP_TILE   = 8;
constexpr int32_t TILE_N       = 2048;

// -----------------------------------------------------------------------------
// HardEvent IDs (only 0/1/2/3)
// Same HardEvent must use the same event_id everywhere.
// -----------------------------------------------------------------------------
static constexpr int EID_MTE2_V = 0;  // MTE2 -> V
static constexpr int EID_V_MTE2 = 1;  // V -> MTE2
static constexpr int EID_V_MTE3 = 2;  // V -> MTE3
static constexpr int EID_MTE3_V = 3;  // MTE3 -> V

// -----------------------------------------------------------------------------
// Simple Low-level ZeroOut (no pipe)
// -----------------------------------------------------------------------------
template<typename T, int32_t TILE = 2048>
class KernelZeroOutLL {
public:
    __aicore__ inline void Init(GM_ADDR dst, uint64_t elem_cnt) {
        dstGm.SetGlobalBuffer((__gm__ T*)dst);
        total = elem_cnt;
    }

    __aicore__ inline void Process() {
        uint32_t addr = 0;
        LocalTensor<T> tile = LocalTensor<T>(TPosition::VECOUT, addr, TILE);

        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (uint64_t base = (uint64_t)core_idx * TILE; base < total; base += (uint64_t)core_num * TILE) {
            uint32_t len = (uint32_t)((total - base < (uint64_t)TILE) ? (total - base) : TILE);

            Duplicate(tile, (T)0, len);

            // V -> MTE3
            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(dstGm[base], tile, len);

            // MTE3 -> V (close pair immediately; we don't overlap)
            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<T> dstGm;
    uint64_t total = 0;
};

// -----------------------------------------------------------------------------
// Kernel SwiGLU (Low-level, NO double-buffer; simple)
// Layout: [Gate | Value] split at inter_dim
// -----------------------------------------------------------------------------
template<typename T, uint32_t TILE_LEN = 512>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t total_rows, int32_t inter_dim) {
        inputGm.SetGlobalBuffer((__gm__ T*)input);
        outputGm.SetGlobalBuffer((__gm__ T*)output);
        this->total_rows = total_rows;
        this->inter_dim  = inter_dim;

        uint32_t addr = 0;
        t_in  = LocalTensor<T>(TPosition::VECIN,  addr, 2 * TILE_LEN);
        addr += (uint32_t)(2 * TILE_LEN * sizeof(T));
        t_out = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));

        f_in  = LocalTensor<float>(TPosition::VECCALC, addr, 2 * TILE_LEN);
        addr += (uint32_t)(2 * TILE_LEN * sizeof(float));
        f_out = LocalTensor<float>(TPosition::VECCALC, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        for (int32_t r = core_idx; r < total_rows; r += core_num) {
            ComputeRow(r);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t row_idx) {
        uint64_t in_base  = (uint64_t)row_idx * 2u * (uint64_t)inter_dim;
        uint64_t out_base = (uint64_t)row_idx * (uint64_t)inter_dim;

        bool first_tile = true;
        for (int32_t i = 0; i < inter_dim; i += (int32_t)TILE_LEN) {
            int32_t len = (inter_dim - i < (int32_t)TILE_LEN) ? (inter_dim - i) : (int32_t)TILE_LEN;

            // reverse deps to avoid overwrite
            if (!first_tile) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);     // previous compute done before overwriting t_in
                WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);     // previous store done before overwriting t_out
            }

            DataCopyParams params;
            params.blockCount = 2;
            params.blockLen   = (uint16_t)(len * (int32_t)sizeof(T) / 32);
            params.srcStride  = (uint16_t)((inter_dim - len) * (int32_t)sizeof(T) / 32);
            params.dstStride  = 0;

            DataCopy(t_in, inputGm[in_base + (uint64_t)i], params);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            Cast(f_in, t_in, RoundMode::CAST_NONE, 2 * len);
            Silu(f_out, f_in, len);
            Mul(f_out, f_out, f_in[len], len);
            Cast(t_out, f_out, RoundMode::CAST_ROUND, len);

            // mark V done for next tile's CopyIn overwrite protection
            SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);

            // store
            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[out_base + (uint64_t)i], t_out, len);

            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);

            first_tile = false;
        }

        // close last outstanding pairs (important across rows, avoid "Set twice" later)
        if (!first_tile) {
            WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<T>     t_in, t_out;
    LocalTensor<float> f_in, f_out;

    int32_t total_rows = 0;
    int32_t inter_dim  = 0;
};

// -----------------------------------------------------------------------------
// Kernel Grouped Gemv W4A16 Moe (Low-level)
// Keep pipeline intent: W double buffer; SZ prefetch overlaps last compute.
// -----------------------------------------------------------------------------
template<typename T, bool IS_BROADCAST_X = false, bool IS_WEIGHTED_SUM = false>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets,
                                GM_ADDR expert_ids, GM_ADDR y, GM_ADDR topk_weights,
                                int32_t total_tokens, int32_t in_dim, int32_t out_dim,
                                int32_t num_experts, int32_t top_k) {
        this->total_tokens = total_tokens;
        this->top_k = top_k;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->out_dim_packed = out_dim / PACK_RATIO;
        this->num_experts = num_experts;
        this->num_groups = in_dim / GROUP_SIZE;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t*)weight);
        scalesGm.SetGlobalBuffer((__gm__ T*)scales);
        zerosGm.SetGlobalBuffer((__gm__ T*)offsets);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t*)expert_ids);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        if constexpr (IS_WEIGHTED_SUM) {
            topkWeightsGm.SetGlobalBuffer((__gm__ float*)topk_weights);
        }

        BuildLocalTensors();
    }

    __aicore__ inline void Process() {
        int32_t total_tasks = total_tokens * num_groups;
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        int32_t base = total_tasks / core_num;
        int32_t rem  = total_tasks % core_num;

        int32_t start_task, task_cnt;
        if (core_idx < rem) {
            task_cnt = base + 1;
            start_task = core_idx * (base + 1);
        } else {
            task_cnt = base;
            start_task = rem * (base + 1) + (core_idx - rem) * base;
        }
        int32_t end_task = start_task + task_cnt;
        if (task_cnt <= 0) return;

        for (int32_t n_start = 0; n_start < out_dim; n_start += TILE_N) {
            int32_t cur_n_len = (out_dim - n_start < TILE_N) ? (out_dim - n_start) : TILE_N;

            Duplicate(y_fp32, 0.0f, cur_n_len);

            int32_t current_row_idx = -1;
            if (start_task < end_task) current_row_idx = start_task / num_groups;

            for (int32_t task_id = start_task; task_id < end_task; ++task_id) {
                int32_t row_idx   = task_id / num_groups;
                int32_t group_idx = task_id % num_groups;

                if (row_idx != current_row_idx) {
                    CopyOut(current_row_idx, n_start, cur_n_len);
                    Duplicate(y_fp32, 0.0f, cur_n_len);
                    current_row_idx = row_idx;
                }

                int32_t expert_id = expertIdsGm.GetValue(row_idx);
                ProcessGroup(group_idx, expert_id, row_idx, n_start, cur_n_len, y_fp32);
            }

            if (current_row_idx != -1) {
                CopyOut(current_row_idx, n_start, cur_n_len);
            }
        }
    }

private:
    __aicore__ inline void BuildLocalTensors() {
        constexpr int32_t tile_n_packed = TILE_N / PACK_RATIO;

        uint32_t addr = 0;

        x_local = LocalTensor<T>(TPosition::VECIN, addr, GROUP_SIZE);
        addr += (uint32_t)(GROUP_SIZE * sizeof(T));

        w_local[0] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tile_n_packed);
        addr += (uint32_t)(COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));
        w_local[1] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tile_n_packed);
        addr += (uint32_t)(COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));

        s_local = LocalTensor<T>(TPosition::VECIN, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(T));
        z_local = LocalTensor<T>(TPosition::VECIN, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(T));

        y_fp32 = LocalTensor<float>(TPosition::VECOUT, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));

        group_acc = LocalTensor<half>(TPosition::VECCALC, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(half) + 256);

        w_half = LocalTensor<half>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += (uint32_t)(GROUP_TILE * TILE_N * sizeof(half));

        x_float_full = LocalTensor<float>(TPosition::VECCALC, addr, GROUP_SIZE);
        addr += (uint32_t)(GROUP_SIZE * sizeof(float));

        z_float = LocalTensor<float>(TPosition::VECCALC, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));

        reduce_buf = LocalTensor<float>(TPosition::VECCALC, addr, GROUP_SIZE);
        addr += (uint32_t)(GROUP_SIZE * sizeof(float));
    }

    __aicore__ inline void PrefetchW(int32_t expert_id, int32_t group_idx, int32_t k_inner_start,
                                     int32_t n_offset, int32_t cur_n_len, int buf_id) {
        uint64_t w_stride_k = (uint64_t)out_dim_packed;
        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int32_t n_offset_packed = n_offset / PACK_RATIO;

        uint64_t w_offset = (uint64_t)expert_id * (uint64_t)in_dim * w_stride_k +
                            (uint64_t)(group_idx * GROUP_SIZE + k_inner_start) * w_stride_k +
                            (uint64_t)n_offset_packed;

        DataCopyExtParams p{
            (uint16_t)COMPUTE_ROWS,
            (uint32_t)(cur_n_packed * sizeof(int32_t)),
            (uint32_t)((w_stride_k - (uint64_t)cur_n_packed) * sizeof(int32_t)),
            0, 0
        };
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(w_local[buf_id], weightGm[w_offset], p, pad);
    }

    __aicore__ inline void PrefetchSZ(int32_t expert_id, int32_t group_idx,
                                      int32_t n_offset, int32_t cur_n_len) {
        uint64_t sz_offset = (uint64_t)expert_id * (uint64_t)num_groups * (uint64_t)out_dim +
                             (uint64_t)group_idx * (uint64_t)out_dim +
                             (uint64_t)n_offset;

        DataCopy(s_local, scalesGm[sz_offset], cur_n_len);
        DataCopy(z_local, zerosGm[sz_offset], cur_n_len);
        // caller will issue exactly ONE SetFlag<MTE2_V>(0) after these two copies
    }

    __aicore__ inline void ComputeChunk(LocalTensor<int32_t>& w_i32,
                                        int32_t x_offset_idx,
                                        int32_t tile_n,
                                        int32_t tile_n_packed,
                                        LocalTensor<T>& x_full_half,
                                        LocalTensor<half>& group_acc_xw) {
        LocalTensor<uint64_t> x_u64 = x_full_half.template ReinterpretCast<uint64_t>();

        constexpr int STEP = 4;
        half x_val_buf[COMPUTE_ROWS];
        uint64_t* x_val_buf_u64 = (uint64_t*)x_val_buf;
        for (int i = 0; i < COMPUTE_ROWS / STEP; ++i) {
            x_val_buf_u64[i] = x_u64.GetValue(x_offset_idx / STEP + i);
        }

        for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
            LocalTensor<int4b_t> w_int4 = w_i32[i * tile_n_packed].template ReinterpretCast<int4b_t>();
            Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * tile_n);
            for (int k = 0; k < GROUP_TILE; ++k) {
                Axpy(group_acc_xw, w_half[k * tile_n], x_val_buf[i + k], tile_n);
            }
        }
    }

    __aicore__ inline void ProcessGroup(int32_t g_idx, int32_t expert_id, int32_t row_idx,
                                        int32_t n_offset, int32_t cur_n_len,
                                        LocalTensor<float>& global_acc) {
        // ----- CopyIn X + W0 (MTE2) -----
        uint64_t x_offset;
        if constexpr (IS_BROADCAST_X) {
            int32_t batch_idx = row_idx / top_k;
            x_offset = (uint64_t)batch_idx * (uint64_t)in_dim + (uint64_t)g_idx * GROUP_SIZE;
        } else {
            x_offset = (uint64_t)row_idx * (uint64_t)in_dim + (uint64_t)g_idx * GROUP_SIZE;
        }

        DataCopy(x_local, xGm[x_offset], GROUP_SIZE);
        PrefetchW(expert_id, g_idx, 0, n_offset, cur_n_len, /*buf=*/0);

        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // ----- init accum -----
        Duplicate(group_acc, (half)0.0f, cur_n_len);

        Cast(x_float_full, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        ReduceSum(reduce_buf, x_float_full, reduce_buf, GROUP_SIZE);

        int32_t cur_n_packed = cur_n_len / PACK_RATIO;

        // W ping-pong
        // Important: because HardEvent::V_MTE2 uses ONE event_id globally,
        // we must ensure every SetFlag<V_MTE2>(1) is eventually waited before leaving ProcessGroup.
        bool v_mte2_set_once = false;

        int ping = 0;
        for (int32_t k_inner = 0; k_inner < GROUP_SIZE; k_inner += COMPUTE_ROWS) {
            // Wait current W ready (except first, already waited above)
            if (k_inner != 0) {
                WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }

            // Prefetch next W or SZ (overlap with current compute)
            if (k_inner + COMPUTE_ROWS < GROUP_SIZE) {
                int next_buf = ping ^ 1;

                PrefetchW(expert_id, g_idx, k_inner + COMPUTE_ROWS, n_offset, cur_n_len, next_buf);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            } else {
                // last iteration: prefetch SZ (do NOT move earlier)
                PrefetchSZ(expert_id, g_idx, n_offset, cur_n_len);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            }

            // Compute
            ComputeChunk(w_local[ping], k_inner, cur_n_len, cur_n_packed, x_local, group_acc);

            // V -> MTE2 flag (protect future overwrites)
            if (k_inner + COMPUTE_ROWS < GROUP_SIZE) {
                SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }
            ping ^= 1;
        }

        // Wait SZ ready (pairs the last SetFlag<MTE2_V>(0) from last iteration)
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);


        float group_x_sum = reduce_buf.GetValue(0);

        Mul(z_local, z_local, s_local, cur_n_len);
        Cast(z_float, z_local, RoundMode::CAST_NONE, cur_n_len);
        Axpy(global_acc, z_float, group_x_sum, cur_n_len);
        MulAddDst(global_acc, group_acc, s_local, cur_n_len);
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t n_offset, int32_t cur_n_len) {
        if (row_idx < 0) return;

        if constexpr (IS_WEIGHTED_SUM) {
            float w_val = topkWeightsGm.GetValue(row_idx);
            Muls(y_fp32, y_fp32, w_val, cur_n_len);
        }

        LocalTensor<T> y_half = y_fp32.template ReinterpretCast<T>();
        Cast(y_half, y_fp32, RoundMode::CAST_ROUND, cur_n_len);

        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        SetAtomicAdd<T>();

        if constexpr (IS_WEIGHTED_SUM) {
            int32_t batch_idx = row_idx / top_k;
            uint64_t y_offset = (uint64_t)batch_idx * (uint64_t)out_dim + (uint64_t)n_offset;
            DataCopy(yGm[y_offset], y_half, cur_n_len);
        } else {
            uint64_t y_offset = (uint64_t)row_idx * (uint64_t)out_dim + (uint64_t)n_offset;
            DataCopy(yGm[y_offset], y_half, cur_n_len);
        }

        SetAtomicNone();

        // Close pair immediately to avoid any outstanding MTE3_V when CopyOut called again.
        SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
    }

private:
    GlobalTensor<float> topkWeightsGm;

    GlobalTensor<T>       xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<T>       scalesGm;
    GlobalTensor<T>       zerosGm;
    GlobalTensor<int32_t> expertIdsGm;
    GlobalTensor<T>       yGm;

    // UB
    LocalTensor<T>        x_local;
    LocalTensor<int32_t>  w_local[2];
    LocalTensor<T>        s_local, z_local;

    LocalTensor<float>    y_fp32;

    LocalTensor<half>     group_acc;
    LocalTensor<half>     w_half;
    LocalTensor<float>    x_float_full;
    LocalTensor<float>    z_float;
    LocalTensor<float>    reduce_buf;

    int32_t top_k = 0;
    int32_t in_dim = 0;
    int32_t out_dim = 0;
    int32_t out_dim_packed = 0;
    int32_t num_experts = 0;
    int32_t num_groups = 0;
    int32_t total_tokens = 0;
};

// -----------------------------------------------------------------------------
// Fused Kernel Implementation (BS <= 4) - low level (no Pipe)
// Function name unchanged.
// -----------------------------------------------------------------------------
template<typename T>
__aicore__ inline void fused_moe_small_bs_impl(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets,
    GM_ADDR w2_weight,  GM_ADDR w2_scales,  GM_ADDR w2_offsets,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k)
{
    int32_t total_tokens = batch_size * top_k;

    GM_ADDR y = workspace;
    uint64_t y_size = (uint64_t)batch_size * (uint64_t)hidden_size * sizeof(T);

    GM_ADDR w13_out_ptr = (GM_ADDR)((__gm__ uint8_t*)y + y_size);
    uint64_t w13_out_size = (uint64_t)total_tokens * (uint64_t)(inter_size * 2) * sizeof(T);

    GM_ADDR w2_in_ptr = (GM_ADDR)((__gm__ uint8_t*)w13_out_ptr + w13_out_size);

    // Phase 0: Zero out (y + w13_out)
    {
        uint64_t elems = (y_size + w13_out_size) / sizeof(T);
        KernelZeroOutLL<T> zeroOp;
        zeroOp.Init(workspace, elems);
        zeroOp.Process();
    }

    AscendC::SyncAll();

    // Phase 1: W13 Gemv (Broadcast X)
    {
        KernelGroupedGemvW4A16Moe<T, true, false> op_w13;
        op_w13.Init(x, w13_weight, w13_scales, w13_offsets,
                    expert_ids, w13_out_ptr, nullptr,
                    total_tokens, hidden_size, inter_size * 2, num_experts, top_k);
        op_w13.Process();
    }

    AscendC::SyncAll();

    // Phase 2: SwiGLU
    {
        KernelSwiGLU<T> op_act;
        op_act.Init(w13_out_ptr, w2_in_ptr, total_tokens, inter_size);
        op_act.Process();
    }

    AscendC::SyncAll();

    // Phase 3: W2 Gemv (Weighted Sum)
    {
        KernelGroupedGemvW4A16Moe<T, false, true> op_w2;
        op_w2.Init(w2_in_ptr, w2_weight, w2_scales, w2_offsets,
                   expert_ids, y, topk_weights,
                   total_tokens, inter_size, hidden_size, num_experts, top_k);
        op_w2.Process();
    }
}

// -----------------------------------------------------------------------------
// Extern C Entry Points (names unchanged)
// -----------------------------------------------------------------------------
extern "C" __global__ __aicore__ void fused_moe_small_bs_w4a16_fp16(
    GM_ADDR x,
    GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets,
    GM_ADDR w2_weight,  GM_ADDR w2_scales,  GM_ADDR w2_offsets,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR workspace,
    int32_t batch_size, int32_t hidden_size, int32_t inter_size,
    int32_t num_experts, int32_t top_k)
{
    AscendC::InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    fused_moe_small_bs_impl<half>(x,
                                 w13_weight, w13_scales, w13_offsets,
                                 w2_weight,  w2_scales,  w2_offsets,
                                 expert_ids, topk_weights,
                                 workspace,
                                 batch_size, hidden_size, inter_size,
                                 num_experts, top_k);
}

extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets,
    GM_ADDR expert_ids, GM_ADDR y,
    int32_t total_tokens, int32_t in_dim, int32_t out_dim,
    int32_t num_experts, int32_t top_k)
{
    AscendC::InitSocState();
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    KernelGroupedGemvW4A16Moe<half, false, false> op;
    op.Init(x, weight, scales, offsets, expert_ids, y, nullptr,
            total_tokens, in_dim, out_dim, num_experts, top_k);
    op.Process();
}
