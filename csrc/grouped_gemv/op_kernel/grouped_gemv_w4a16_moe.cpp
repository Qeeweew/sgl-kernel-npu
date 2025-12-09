#include "kernel_operator.h"
#include "zero_out_impl.h"
#include <type_traits>

using namespace AscendC;

// -----------------------------------------------------------------------------
// 常量定义
// -----------------------------------------------------------------------------
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GROUP_SIZE = 32;
constexpr int32_t PACK_RATIO = 8;
constexpr int32_t GROUP_TILE = 8;

// -----------------------------------------------------------------------------
// Kernel SwiGLU (Phase 2)
// Logic: Cast(T->FP32) -> SwiGLU(FP32) -> Cast(FP32->T)
// Input Layout: [K, 2 * InterDim] (Gate, Value interleaved or concat)
// Assumption: Data is [Gate... | Value...] split at dim/2
// -----------------------------------------------------------------------------
template<typename T, uint32_t TILE_LEN=512>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR input, GM_ADDR output, 
                                int32_t total_rows, int32_t inter_dim) {
        this->pipe = pipe;
        this->inputGm.SetGlobalBuffer((__gm__ T*)input);
        this->outputGm.SetGlobalBuffer((__gm__ T*)output);
        this->total_rows = total_rows;
        this->inter_dim = inter_dim;

        // Input Queues (T type)
        pipe->InitBuffer(qGate, BUFFER_NUM, TILE_LEN * sizeof(T));
        pipe->InitBuffer(qValue, BUFFER_NUM, TILE_LEN * sizeof(T));
        
        // Output Queue (T type)
        pipe->InitBuffer(qOut, BUFFER_NUM, TILE_LEN * sizeof(T));

        // Calculation Buffer (FP32 type for SwiGLU)
        // Need Gate_FP32, Value_FP32, and Dst_FP32 is usually reused if allowed, 
        // but SwiGLU(dst, src0, src1) needs separate srcs.
        // Size: 3 * TILE_LEN * sizeof(float)
        pipe->InitBuffer(calcBuf, TILE_LEN * sizeof(float) * 3);
    }

    __aicore__ inline void Process() {
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();
        
        // Parallelize by rows. If rows < cores, some cores idle (acceptable for BS=1 small K)
        for (int32_t r = core_idx; r < total_rows; r += core_num) {
            ComputeRow(r);
        }
    }

    __aicore__ inline void ComputeRow(int32_t row_idx) {
        uint64_t gate_offset_base = (uint64_t)row_idx * 2 * inter_dim;
        uint64_t value_offset_base = gate_offset_base + inter_dim;
        uint64_t out_offset_base = (uint64_t)row_idx * inter_dim;

        for (int32_t i = 0; i < inter_dim; i += TILE_LEN) {
            int32_t len = (inter_dim - i < TILE_LEN) ? (inter_dim - i) : TILE_LEN;
            
            // 1. Alloc & CopyIn
            qGate.AllocTensor<T>(t_gate);
            qValue.AllocTensor<T>(t_value);
            
            DataCopy(t_gate, inputGm[gate_offset_base + i], len);
            DataCopy(t_value, inputGm[value_offset_base + i], len);
            
            qGate.EnQue(t_gate);
            qValue.EnQue(t_value);
            
            // 2. Compute
            qGate.DeQue<T>(t_gate); 
            qValue.DeQue<T>(t_value);
            qOut.AllocTensor<T>(t_out);

            // Get FP32 buffers
            LocalTensor<float> f_gate = calcBuf.GetWithOffset<float>(TILE_LEN, 0);
            LocalTensor<float> f_value = calcBuf.GetWithOffset<float>(TILE_LEN, TILE_LEN  * sizeof(float));
            LocalTensor<float> f_out = calcBuf.GetWithOffset<float>(TILE_LEN, 2 * TILE_LEN  * sizeof(float));

            // Cast T -> FP32
            Cast(f_gate, t_gate, RoundMode::CAST_NONE, len);
            Cast(f_value, t_value, RoundMode::CAST_NONE, len);

            Silu(f_out, f_gate, len);
            Mul(f_out, f_out, f_value, len);

            qGate.FreeTensor(t_gate);
            qValue.FreeTensor(t_value);

            // Cast FP32 -> T
            Cast(t_out, f_out, RoundMode::CAST_ROUND, len);

            // 3. CopyOut
            qOut.EnQue(t_out);
            
            qOut.DeQue(t_out);
            DataCopy(outputGm[out_offset_base + i], t_out, len);
            qOut.FreeTensor(t_out);
        }
    }

private:
    AscendC::TPipe* pipe;
    AscendC::GlobalTensor<T> inputGm;
    AscendC::GlobalTensor<T> outputGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> qGate, qValue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> qOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    LocalTensor<T> t_gate, t_value, t_out;
    int32_t total_rows;
    int32_t inter_dim;
};

// -----------------------------------------------------------------------------
// Kernel Grouped Gemv (Enhanced with Templates)
// -----------------------------------------------------------------------------
template<typename T, bool IS_BROADCAST_X=false, bool IS_WEIGHTED_SUM=false>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, 
                                GM_ADDR y, GM_ADDR topk_weights,
                                int32_t top_k, int32_t in_dim, int32_t out_dim, int32_t num_experts)
    {
        this->pipe = pipe;
        this->top_k = top_k;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->out_dim_packed = out_dim / 8;
        this->num_experts = num_experts;
        this->num_groups = this->in_dim / GROUP_SIZE;
        
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        if constexpr (IS_WEIGHTED_SUM) {
            topkWeightsGm.SetGlobalBuffer((__gm__ T *)topk_weights);
        }

        pipe->InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T) + 32); 
        pipe->InitBuffer(inQueueW, BUFFER_NUM, GROUP_SIZE * out_dim_packed * sizeof(int32_t));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, out_dim * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, out_dim * sizeof(float)); // Accumulator is Float

        uint32_t current_offset = 0;
        this->offset_group_acc = current_offset; current_offset += out_dim * sizeof(float);
        this->offset_w_half = current_offset; current_offset += GROUP_TILE * out_dim * sizeof(half);
        this->offset_x_float = current_offset; current_offset += GROUP_SIZE * sizeof(float);
        this->offset_x_half = current_offset; current_offset += GROUP_SIZE * sizeof(half);
        this->offset_s_float = current_offset; current_offset += out_dim * sizeof(float);
        pipe->InitBuffer(calcBuf, current_offset);
    }

    __aicore__ inline void Process()
    {
        const int32_t row_idx = GetBlockIdx() % top_k;
        const int32_t expert_id = expertIdsGm.GetValue(row_idx);
        const int32_t g_idx = GetBlockIdx() / top_k;
        const int32_t g_count = GetBlockNum() / top_k + ((row_idx < GetBlockNum() % top_k) ? 1 : 0);
        const int32_t n_start = 0; // Simplified for BS=1

        // 1. Alloc Global Acc
        outQueueY.AllocTensor<T>(y_local); // Effectively holding float data
        auto global_acc = y_local.template ReinterpretCast<float>();
        Duplicate(global_acc, 0.0f, out_dim);

        for (int32_t g = g_idx; g < num_groups; g += g_count) {
            CopyIn(expert_id, row_idx, g, n_start);
            Compute(global_acc); 
        }

        outQueueY.EnQue(y_local);
        CopyOut(row_idx, n_start);
    }

private:
    __aicore__ inline void CopyIn(int32_t expert_id, int32_t row_idx, int32_t group_idx, int32_t n_start)
    {
        uint64_t w_stride_k = this->out_dim / PACK_RATIO;
        uint64_t w_offset = (uint64_t)expert_id * this->in_dim * w_stride_k + 
                            (uint64_t)group_idx * GROUP_SIZE * w_stride_k + 
                            (n_start / PACK_RATIO);
        
        inQueueW.AllocTensor<int32_t>(w_local);
        AscendC::DataCopyExtParams w_copy_params{GROUP_SIZE, (uint32_t) (out_dim_packed * sizeof(int32_t)), (uint32_t) ((w_stride_k - out_dim_packed) * sizeof(int32_t)), 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(w_local, weightGm[w_offset], w_copy_params, padParams);
        inQueueW.EnQue(w_local);

        uint64_t s_offset = (uint64_t)expert_id * this->num_groups * this->out_dim +
                            (uint64_t)group_idx * this->out_dim + n_start;
        inQueueScale.AllocTensor<T>(s_local);
        DataCopy(s_local, scalesGm[s_offset], out_dim);
        inQueueScale.EnQue(s_local);

        // ... (X Loading Logic) ...
        uint64_t x_offset;
        if constexpr (IS_BROADCAST_X) {
            // Broadcast mode: always read from first row (assuming X is [1, InDim])
            x_offset = group_idx * GROUP_SIZE;
        } else {
            // Normal mode
            x_offset = (uint64_t)row_idx * this->in_dim + group_idx * GROUP_SIZE;
        }

        inQueueX.AllocTensor<T>(x_local);
        DataCopy(x_local, xGm[x_offset], GROUP_SIZE);
        inQueueX.EnQue(x_local);
    }

    __aicore__ inline void Compute(LocalTensor<float>& global_acc)
    {
        LocalTensor<float> group_acc = calcBuf.GetWithOffset<float>(out_dim, offset_group_acc);
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(GROUP_TILE * out_dim, offset_w_half);
        LocalTensor<float> x_float_tmp = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_x_float);
        LocalTensor<half> x_half = calcBuf.GetWithOffset<half>(GROUP_SIZE, offset_x_half);
        LocalTensor<float> s_float = calcBuf.GetWithOffset<float>(out_dim, offset_s_float);

        Duplicate(group_acc, 0.0f, out_dim);

        // 1. 类型转换
        inQueueX.DeQue<T>(x_local);

        // X: T -> float -> half
        Cast(x_float_tmp, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        Cast(x_half, x_float_tmp, RoundMode::CAST_ROUND, GROUP_SIZE);
        inQueueScale.DeQue<T>(s_local);

        // Scale: T -> float
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
            for (int i1 = 0; i1 < GROUP_TILE; ++i1) {
                // group_acc += w_row[i] * x_scalar[i]
                Axpy(group_acc, w_half[i1 * out_dim], x_buf[i1], out_dim);
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
        // At this point, y_local holds the data, but it is effectively float accumulator data
        outQueueY.DeQue<T>(y_local);
        
        // Use a view to manipulate the float values
        LocalTensor<float> y_fp32 = y_local.template ReinterpretCast<float>();

        if constexpr (IS_WEIGHTED_SUM && std::is_same_v<T, bfloat16_t>) {
            T w_val_t = topkWeightsGm.GetValue(row_idx);
            float w_val_f = ToFloat(w_val_t);
            Muls(y_fp32, y_fp32, w_val_f, out_dim);
        }

        // 4. Cast FP32 -> T
        // Use in-place cast (Float(4B) -> T(2B) fits in same buffer)
        Cast(y_local, y_fp32, RoundMode::CAST_ROUND, out_dim);

        if constexpr (IS_WEIGHTED_SUM && std::is_same_v<T, half>) {
            T w_val_t = topkWeightsGm.GetValue(row_idx);
            Muls(y_local, y_local, w_val_t, out_dim);
        }
        
        PipeBarrier<PIPE_V>();

        if constexpr (IS_WEIGHTED_SUM) {
            // Atomic Add to Global[0]
            AscendC::SetAtomicAdd<T>();
            // Write to n_start (effectively offset 0 relative to Y base, broadcasted reduction)
            DataCopy(yGm[n_start], y_local, out_dim);
            AscendC::SetAtomicNone();
        } else {
            // Normal Store
            AscendC::SetAtomicAdd<T>();
            uint64_t y_offset = (uint64_t)row_idx * this->out_dim + n_start;
            DataCopy(yGm[y_offset], y_local, out_dim);
            AscendC::SetAtomicNone();
        }

        outQueueY.FreeTensor(y_local);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::GlobalTensor<T> topkWeightsGm;
    
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueueX, inQueueW, inQueueScale;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    
    uint32_t offset_group_acc, offset_w_half, offset_x_float, offset_x_half, offset_s_float;

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

// -----------------------------------------------------------------------------
// Fused Kernel Implementation
// -----------------------------------------------------------------------------
template<typename T>
__aicore__ inline void fused_moe_bs1_impl(
    GM_ADDR x, 
    GM_ADDR w13_weight, GM_ADDR w13_scales, 
    GM_ADDR w2_weight, GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR workspace, GM_ADDR y,
    int32_t top_k, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts)
{ 
    AscendC::TPipe pipe;
    
    // Workspace layout calculation
    // Workspace 1: W13 Output [TopK, 2 * InterDim]
    GM_ADDR w13_out_ptr = workspace;
    uint64_t w13_out_size = (uint64_t)top_k * (inter_dim * 2) * sizeof(T);
    
    // Workspace 2: SwiGLU Output [TopK, InterDim] (Input to W2)
    GM_ADDR w2_in_ptr = (GM_ADDR)((__gm__ uint8_t*)workspace + w13_out_size);

    // Phase 0: Zero Out
    {
        KernelZeroOut<T> zeroOp;
        zeroOp.Init(&pipe, w13_out_ptr, w13_out_size);
        zeroOp.Process();
    }

    // Barrier & Reset
    AscendC::SyncAll();
    pipe.Reset();
    
    // ------------------------------------------------------------------------
    // Phase 1: W13 Gemv (Broadcast X)
    // ------------------------------------------------------------------------
    {
        // IS_BROADCAST_X = true, IS_WEIGHTED_SUM = false
        KernelGroupedGemvW4A16Moe<T, true, false> op_w13;
        op_w13.Init(&pipe, x, w13_weight, w13_scales, expert_ids, w13_out_ptr, nullptr,
                    top_k, in_dim, inter_dim * 2, num_experts);
        op_w13.Process();
    }

    // Barrier & Reset
    AscendC::SyncAll();
    pipe.Reset();

    // ------------------------------------------------------------------------
    // Phase 2: SwiGLU Activation
    // ------------------------------------------------------------------------
    {
        KernelSwiGLU<T> op_act;
        op_act.Init(&pipe, w13_out_ptr, w2_in_ptr, top_k, inter_dim);
        op_act.Process();
    }

    // Barrier & Reset
    AscendC::SyncAll();
    pipe.Reset();

    // ------------------------------------------------------------------------
    // Phase 3: W2 Gemv (Weighted Sum)
    // ------------------------------------------------------------------------
    {
        // IS_BROADCAST_X = false, IS_WEIGHTED_SUM = true
        KernelGroupedGemvW4A16Moe<T, false, true> op_w2;
        op_w2.Init(&pipe, w2_in_ptr, w2_weight, w2_scales, expert_ids, y, topk_weights,
                   top_k, inter_dim, out_dim, num_experts);
        op_w2.Process();
    }
}

// -----------------------------------------------------------------------------
// Extern C Entry Points
// -----------------------------------------------------------------------------
extern "C" __global__ __aicore__ void fused_moe_bs1_w4a16_fp16(
    GM_ADDR x, GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w2_weight, GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights, GM_ADDR workspace, GM_ADDR y,
    int32_t top_k, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    fused_moe_bs1_impl<half>(x, w13_weight, w13_scales, w2_weight, w2_scales, expert_ids, topk_weights, workspace, y, top_k, in_dim, inter_dim, out_dim, num_experts);
}

extern "C" __global__ __aicore__ void fused_moe_bs1_w4a16_bf16(
    GM_ADDR x, GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w2_weight, GM_ADDR w2_scales,
    GM_ADDR expert_ids, GM_ADDR topk_weights, GM_ADDR workspace, GM_ADDR y,
    int32_t top_k, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    fused_moe_bs1_impl<bfloat16_t>(x, w13_weight, w13_scales, w2_weight, w2_scales, expert_ids, topk_weights, workspace, y, top_k, in_dim, inter_dim, out_dim, num_experts);
}


// 导出 FP16 版本
extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, GM_ADDR y, 
    int32_t top_k, int32_t in_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelGroupedGemvW4A16Moe<half, false, false> op;
    op.Init(&pipe, x, weight, scales, expert_ids, y, nullptr, top_k, in_dim, out_dim, num_experts);
    op.Process();
}

// 导出 BF16 版本
extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR expert_ids, GM_ADDR y, 
    int32_t top_k, int32_t in_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    KernelGroupedGemvW4A16Moe<bfloat16_t, false, false> op;
    op.Init(&pipe, x, weight, scales, expert_ids, y, nullptr, top_k, in_dim, out_dim, num_experts);
    op.Process();
}
