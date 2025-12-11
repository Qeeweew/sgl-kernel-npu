#include "kernel_operator.h"
#include "zero_out_impl.h"
#include <type_traits>

using namespace AscendC;

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t GROUP_SIZE = 128; // Updated to 128
constexpr int32_t COMPUTE_ROWS = 32; // Block size for Weight loading within a group
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
        pipe->InitBuffer(qIn, BUFFER_NUM, 2 * TILE_LEN * sizeof(T));
        
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
        uint64_t in_offset_base = (uint64_t)row_idx * 2 * inter_dim;
        uint64_t out_offset_base = (uint64_t)row_idx * inter_dim;

        for (int32_t i = 0; i < inter_dim; i += TILE_LEN) {
            int32_t len = (inter_dim - i < TILE_LEN) ? (inter_dim - i) : TILE_LEN;
            DataCopyParams params;
            params.blockCount = 2;
            params.blockLen = len * sizeof(T) / 32;
            params.srcStride = (inter_dim - len) * sizeof(T) / 32;
            params.dstStride = 0;
            
            // 1. Alloc & CopyIn
            qIn.AllocTensor<T>(t_in);
            
            DataCopy(t_in, inputGm[in_offset_base + i], params);
            
            qIn.EnQue(t_in);
            
            // 2. Compute
            qIn.DeQue(t_in); 
            qOut.AllocTensor(t_out);

            // Get FP32 buffers
            LocalTensor<float> f_in = calcBuf.GetWithOffset<float>(2 * TILE_LEN, 0);
            LocalTensor<float> f_out = calcBuf.GetWithOffset<float>(TILE_LEN, 2 * TILE_LEN  * sizeof(float));

            // Cast T -> FP32
            Cast(f_in, t_in, RoundMode::CAST_NONE, 2 * len);

            // f_gate = f_in[0:len]
            Silu(f_out, f_in, len);

            // f_value = f_in[len:2*len]
            Mul(f_out, f_out, f_in[len], len);

            qIn.FreeTensor(t_in);

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
    AscendC::TQue<AscendC::TPosition::VECIN, 0> qIn;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> qOut;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    LocalTensor<T> t_in, t_out;
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

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets,
                                GM_ADDR expert_ids, GM_ADDR y, GM_ADDR topk_weights,
                                int32_t top_k, int32_t in_dim, int32_t out_dim, int32_t num_experts)
    {
        this->pipe = pipe;
        this->top_k = top_k;
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        this->out_dim_packed = out_dim / PACK_RATIO;
        this->num_experts = num_experts;
        this->num_groups = this->in_dim / GROUP_SIZE;
        
        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        zerosGm.SetGlobalBuffer((__gm__ T *)offsets);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expert_ids);
        yGm.SetGlobalBuffer((__gm__ T *)y);

        if constexpr (IS_WEIGHTED_SUM) {
            topkWeightsGm.SetGlobalBuffer((__gm__ float *)topk_weights);
        }

        // Pipe Init
        // X: One group per load
        pipe->InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T) + 32); 
        
        // W: Tiled loading (COMPUTE_ROWS) to support large GROUP_SIZE(128) without exceeding UB
        pipe->InitBuffer(inQueueW, BUFFER_NUM, COMPUTE_ROWS * out_dim_packed * sizeof(int32_t));
        
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, out_dim * sizeof(T));
        pipe->InitBuffer(inQueueOffset, BUFFER_NUM, out_dim * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, out_dim * sizeof(float)); 

        // Workspace CalcBuf calculation
        uint32_t current_offset = 0;
        this->offset_group_acc = current_offset; current_offset += out_dim * sizeof(float);
        this->offset_w_half = current_offset; current_offset += GROUP_TILE * out_dim * sizeof(half);
        this->offset_x_float = current_offset; current_offset += GROUP_SIZE * sizeof(float);
        this->offset_x_half = current_offset; current_offset += GROUP_SIZE * sizeof(half);
        this->offset_s_float = current_offset; current_offset += out_dim * sizeof(float);
        this->offset_z_float = current_offset; current_offset += out_dim * sizeof(float);
        this->offset_reduce_buf = current_offset; current_offset += GROUP_SIZE * sizeof(float);
        
        pipe->InitBuffer(calcBuf, current_offset);
    }

    __aicore__ inline void Process()
    {
        const int32_t row_idx = GetBlockIdx() % top_k;
        const int32_t expert_id = expertIdsGm.GetValue(row_idx);
        // Task allocation: Parallelize over Groups, but limited to TopK blocks
        const int32_t g_idx = GetBlockIdx() / top_k;
        const int32_t g_count = GetBlockNum() / top_k + ((row_idx < GetBlockNum() % top_k) ? 1 : 0);
        
        // Output Accumulator (Global to this core's task)
        outQueueY.AllocTensor<T>(y_local); 
        auto global_acc = y_local.template ReinterpretCast<float>();
        Duplicate(global_acc, 0.0f, out_dim);

        for (int32_t g = g_idx; g < num_groups; g += g_count) {
            CopyIn(expert_id, row_idx, g);
            Compute(g, expert_id, global_acc); 
        }

        outQueueY.EnQue(y_local);
        CopyOut(row_idx);
    }

private:
    __aicore__ inline void CopyIn(int32_t expert_id, int32_t row_idx, int32_t group_idx)
    {
        // 1. Load Scale & Offset (Zeros)
        // [Experts, Groups, OutDim]
        uint64_t sz_offset = (uint64_t)expert_id * this->num_groups * this->out_dim +
                            (uint64_t)group_idx * this->out_dim;
        
        inQueueScale.AllocTensor<T>(s_local);
        inQueueOffset.AllocTensor<T>(z_local);
        
        DataCopy(s_local, scalesGm[sz_offset], out_dim);
        DataCopy(z_local, zerosGm[sz_offset], out_dim);
        
        inQueueScale.EnQue(s_local);
        inQueueOffset.EnQue(z_local);

        // 2. Load X (One Group)
        uint64_t x_offset;
        if constexpr (IS_BROADCAST_X) {
            x_offset = group_idx * GROUP_SIZE;
        } else {
            x_offset = (uint64_t)row_idx * this->in_dim + group_idx * GROUP_SIZE;
        }

        inQueueX.AllocTensor<T>(x_local);
        DataCopy(x_local, xGm[x_offset], GROUP_SIZE);
        inQueueX.EnQue(x_local);
    }

    // Helper to load a chunk of W
    __aicore__ inline void CopyInW(int32_t expert_id, int32_t group_idx, int32_t k_inner_start)
    {
        uint64_t w_stride_k = this->out_dim_packed;
        // W layout: [Expert, InDim, OutDim/8]
        // k_idx = group_idx * 128 + k_inner_start
        uint64_t w_offset = (uint64_t)expert_id * this->in_dim * w_stride_k + 
                            (uint64_t)(group_idx * GROUP_SIZE + k_inner_start) * w_stride_k;
        
        inQueueW.AllocTensor<int32_t>(w_local);
        // Copy COMPUTE_ROWS * OutDimPacked
        AscendC::DataCopyExtParams w_copy_params{
            (uint16_t)COMPUTE_ROWS, 
            (uint32_t)(out_dim_packed * sizeof(int32_t)), 
            (uint32_t)((w_stride_k - out_dim_packed) * sizeof(int32_t)), 
            0, 0
        };
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        DataCopyPad(w_local, weightGm[w_offset], w_copy_params, padParams);
        inQueueW.EnQue(w_local);
    }

    __aicore__ inline void Compute(int32_t g_idx, int32_t expert_id, LocalTensor<float>& global_acc)
    {
        // Buffers
        LocalTensor<float> group_acc = calcBuf.GetWithOffset<float>(out_dim, offset_group_acc);
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(GROUP_TILE * out_dim, offset_w_half);
        LocalTensor<float> x_float_full = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_x_float);
        LocalTensor<half> x_half_full = calcBuf.GetWithOffset<half>(GROUP_SIZE, offset_x_half);
        
        LocalTensor<float> s_float = calcBuf.GetWithOffset<float>(out_dim, offset_s_float);
        LocalTensor<float> z_float = calcBuf.GetWithOffset<float>(out_dim, offset_z_float);
        LocalTensor<float> reduce_buf = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_reduce_buf);

        // 1. Process X (T -> Float -> Half)
        inQueueX.DeQue<T>(x_local);
        Cast(x_float_full, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        Cast(x_half_full, x_float_full, RoundMode::CAST_ROUND, GROUP_SIZE);

        // 2. Calculate Sum(X) for asymmetric quantization correction
        ReduceSum(reduce_buf, x_float_full, reduce_buf, GROUP_SIZE);
        float group_x_sum = reduce_buf.GetValue(0);

        inQueueX.FreeTensor(x_local);

        // 3. Process Scale & Offset
        inQueueScale.DeQue<T>(s_local);
        inQueueOffset.DeQue<T>(z_local);
        Cast(s_float, s_local, RoundMode::CAST_NONE, out_dim);
        Cast(z_float, z_local, RoundMode::CAST_NONE, out_dim);
        inQueueScale.FreeTensor(s_local);
        inQueueOffset.FreeTensor(z_local);

        // 4. Matrix Multiplication Loop (Tiled by COMPUTE_ROWS to save UB)
        Duplicate(group_acc, 0.0f, out_dim);
        
        for (int32_t k_inner = 0; k_inner < GROUP_SIZE; k_inner += COMPUTE_ROWS) {
            CopyInW(expert_id, g_idx, k_inner);
            inQueueW.DeQue<int32_t>(w_local);

            for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
                // W: int32 -> int4 -> half
                LocalTensor<int4b_t> w_int4 = w_local[i * out_dim_packed].ReinterpretCast<int4b_t>();
                Cast(w_half, w_int4, RoundMode::CAST_NONE, GROUP_TILE * out_dim);

                half x_val_buf[GROUP_TILE];
                for (int j = 0; j < GROUP_TILE; j++) {
                    x_val_buf[j] = x_half_full.GetValue(k_inner + i + j);
                }

                for (int j = 0; j < GROUP_TILE; ++j) {
                    Axpy(group_acc, w_half[j * out_dim], x_val_buf[j], out_dim);
                }
            }
            inQueueW.FreeTensor(w_local);
        }

        // 5. Finalize: Y = (X*W) * S + Sum(X) * Z * S
        // Calc Correction: Z * S
        Mul(z_float, z_float, s_float, out_dim);
        // Add Correction: Acc += Sum(X) * (Z*S)
        Axpy(global_acc, z_float, group_x_sum, out_dim);
        // Add Main Term: Acc += (X*W) * S
        MulAddDst(global_acc, group_acc, s_float, out_dim);
    }

    __aicore__ inline void CopyOut(int32_t row_idx)
    {
        outQueueY.DeQue<T>(y_local);
        LocalTensor<float> y_fp32 = y_local.template ReinterpretCast<float>();

        if constexpr (IS_WEIGHTED_SUM) {
            float w_val = topkWeightsGm.GetValue(row_idx);
            Muls(y_fp32, y_fp32, w_val, out_dim);
        }

        // Cast FP32 -> T
        Cast(y_local, y_fp32, RoundMode::CAST_ROUND, out_dim);

        PipeBarrier<PIPE_V>();

        // Atomic Add result to global memory
        AscendC::SetAtomicAdd<T>();
        if constexpr (IS_WEIGHTED_SUM) {
            DataCopy(yGm[0], y_local, out_dim);
        } else {
            uint64_t y_offset = (uint64_t)row_idx * this->out_dim;
            DataCopy(yGm[y_offset], y_local, out_dim);
        }
        AscendC::SetAtomicNone();

        outQueueY.FreeTensor(y_local);
    }

private:
    AscendC::TPipe* pipe;
    AscendC::GlobalTensor<float> topkWeightsGm;
    
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueueX, inQueueW, inQueueScale, inQueueOffset;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    
    uint32_t offset_group_acc, offset_w_half, offset_x_float, offset_x_half;
    uint32_t offset_s_float, offset_z_float, offset_reduce_buf;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<int32_t> weightGm;
    AscendC::GlobalTensor<T> scalesGm;
    AscendC::GlobalTensor<T> zerosGm;
    AscendC::GlobalTensor<int32_t> expertIdsGm;
    AscendC::GlobalTensor<T> yGm;

    LocalTensor<T> x_local, s_local, z_local, y_local;
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
    GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets,
    GM_ADDR w2_weight, GM_ADDR w2_scales, GM_ADDR w2_offsets,
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
        op_w13.Init(&pipe, x, w13_weight, w13_scales, w13_offsets, expert_ids, w13_out_ptr, nullptr,
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
        op_w2.Init(&pipe, w2_in_ptr, w2_weight, w2_scales, w2_offsets, expert_ids, y, topk_weights,
                   top_k, inter_dim, out_dim, num_experts);
        op_w2.Process();
    }
}

// -----------------------------------------------------------------------------
// Extern C Entry Points
// -----------------------------------------------------------------------------
extern "C" __global__ __aicore__ void fused_moe_bs1_w4a16_fp16(
    GM_ADDR x, GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets,  GM_ADDR w2_weight, GM_ADDR w2_scales, GM_ADDR w2_offsets,
    GM_ADDR expert_ids, GM_ADDR topk_weights, GM_ADDR workspace, GM_ADDR y,
    int32_t top_k, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    fused_moe_bs1_impl<half>(x, w13_weight, w13_scales, w13_offsets, w2_weight, w2_scales, w2_offsets, expert_ids, topk_weights, workspace, y, top_k, in_dim, inter_dim, out_dim, num_experts);
}

extern "C" __global__ __aicore__ void fused_moe_bs1_w4a16_bf16(
    GM_ADDR x, GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets, GM_ADDR w2_weight, GM_ADDR w2_scales, GM_ADDR w2_offsets,
    GM_ADDR expert_ids, GM_ADDR topk_weights, GM_ADDR workspace, GM_ADDR y,
    int32_t top_k, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    fused_moe_bs1_impl<bfloat16_t>(x, w13_weight, w13_scales, w13_offsets, w2_weight, w2_scales, w2_offsets, expert_ids, topk_weights, workspace, y, top_k, in_dim, inter_dim, out_dim, num_experts);
}


// 导出 FP16 版本
extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR expert_ids, GM_ADDR y, 
    int32_t top_k, int32_t in_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    KernelGroupedGemvW4A16Moe<half, false, false> op;
    op.Init(&pipe, x, weight, scales, offsets, expert_ids, y, nullptr, top_k, in_dim, out_dim, num_experts);
    op.Process();
}

// 导出 BF16 版本
extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_bf16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR expert_ids, GM_ADDR y, 
    int32_t top_k, int32_t in_dim, int32_t out_dim, int32_t num_experts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    KernelGroupedGemvW4A16Moe<bfloat16_t, false, false> op;
    op.Init(&pipe, x, weight, scales, offsets, expert_ids, y, nullptr, top_k, in_dim, out_dim, num_experts);
    op.Process();
}
