#define K_MAX_SHAPE_DIM 0
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
// Kernel Grouped Gemv (Enhanced for BS > 1 with Task Distribution)
// -----------------------------------------------------------------------------
template<typename T, bool IS_BROADCAST_X=false, bool IS_WEIGHTED_SUM=false>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets,
                                GM_ADDR expert_ids, GM_ADDR y, GM_ADDR topk_weights,
                                int32_t total_tokens, int32_t in_dim, int32_t out_dim, int32_t num_experts, int32_t top_k)
    {
        this->pipe = pipe;
        this->total_tokens = total_tokens; // BS * TopK
        this->top_k = top_k;               // 用于计算 batch index
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

        // --- N-Tiling Constants ---
        constexpr int32_t TILE_N = 2048;
        int32_t tile_n_packed = TILE_N / PACK_RATIO;

        // Pipe Init
        pipe->InitBuffer(inQueueX, BUFFER_NUM, GROUP_SIZE * sizeof(T)); 
        pipe->InitBuffer(inQueueW, BUFFER_NUM, COMPUTE_ROWS * tile_n_packed * sizeof(int32_t));
        pipe->InitBuffer(inQueueScale, BUFFER_NUM, TILE_N * sizeof(T));
        pipe->InitBuffer(inQueueOffset, BUFFER_NUM, TILE_N * sizeof(T));
        pipe->InitBuffer(outQueueY, BUFFER_NUM, TILE_N * sizeof(float)); 

        // Workspace CalcBuf calculation
        uint32_t current_offset = 0;
        this->offset_group_acc = current_offset; current_offset += TILE_N * sizeof(half) + 256;
        this->offset_w_half = current_offset; current_offset += GROUP_TILE * TILE_N * sizeof(half);
        this->offset_x_float = current_offset; current_offset += GROUP_SIZE * sizeof(float);
        this->offset_z_float = current_offset; current_offset += TILE_N * sizeof(float);
        this->offset_reduce_buf = current_offset; current_offset += GROUP_SIZE * sizeof(float);
        
        pipe->InitBuffer(calcBuf, current_offset);
    }

    __aicore__ inline void Process()
    {
        // 1. 计算总任务数 (Rows * Groups)
        int32_t total_tasks = total_tokens * num_groups;
        int32_t core_idx = GetBlockIdx();
        int32_t core_num = GetBlockNum();

        // 基本分配和余数
        int32_t base = total_tasks / core_num;
        int32_t rem  = total_tasks % core_num;

        // 计算当前核的起始任务
        int32_t start_task;
        int32_t task_cnt;

        if (core_idx < rem) {
            // 前 rem 个核多分配 1 个
            task_cnt = base + 1;
            start_task = core_idx * (base + 1);
        } else {
            task_cnt = base;
            start_task = rem * (base + 1) + (core_idx - rem) * base;
        }

        int32_t end_task = start_task + task_cnt;

        // 没有任务直接返回
        if (task_cnt <= 0) return;

        constexpr int32_t TILE_N = 2048;

        // --- Outer Loop: Tile N dimension ---
        // 外层循环 N，保证 Buffer 复用
        for (int32_t n_start = 0; n_start < out_dim; n_start += TILE_N) {
            int32_t cur_n_len = (out_dim - n_start < TILE_N) ? (out_dim - n_start) : TILE_N;
            
            // 初始化 Accumulator
            outQueueY.AllocTensor<T>(y_local); 
            auto global_acc = y_local.template ReinterpretCast<float>();
            Duplicate(global_acc, 0.0f, cur_n_len); 

            // 状态追踪：用于判断是否切换了 Row
            int32_t current_row_idx = -1;
            
            // 预先解码第一个任务的 row
            if (start_task < end_task) {
                current_row_idx = start_task / num_groups;
            }

            // --- Inner Loop: Iterate Tasks assigned to this core ---
            for (int32_t task_id = start_task; task_id < end_task; ++task_id) {
                
                int32_t row_idx = task_id / num_groups;
                int32_t group_idx = task_id % num_groups;

                // 如果切换了 Row，必须先把上一个 Row 的结果写回 (Atomic Add)，并清空 Accumulator
                if (row_idx != current_row_idx) {
                    // 1. CopyOut (Atomic Add to GM)
                    outQueueY.EnQue(y_local);
                    CopyOut(current_row_idx, n_start, cur_n_len);
                    
                    // 2. Reset Accumulator for new row
                    outQueueY.AllocTensor<T>(y_local);
                    // 重新获取 tensor 引用因为 Alloc 后地址可能变化 (虽然在 AscendC 队列机制下通常是循环buffer)
                    global_acc = y_local.template ReinterpretCast<float>();
                    Duplicate(global_acc, 0.0f, cur_n_len);
                    
                    current_row_idx = row_idx;
                }

                // 获取 Expert ID (注意：expert_ids 是 [BS, TopK] flatten 后的，直接用 row_idx 索引)
                int32_t expert_id = expertIdsGm.GetValue(row_idx);

                // 计算当前 Group 贡献
                ProcessGroup(group_idx, expert_id, row_idx, n_start, cur_n_len, global_acc); 
            }

            // Loop 结束，处理最后一个 Row 的残留数据
            if (current_row_idx != -1) {
                outQueueY.EnQue(y_local);
                CopyOut(current_row_idx, n_start, cur_n_len);
            }
        }
    }

private:
    __aicore__ inline void CopyInX(int32_t row_idx, int32_t group_idx)
    {
        uint64_t x_offset;
        if constexpr (IS_BROADCAST_X) {
            // W13 阶段：输入 X 是 [BS, InDim]
            // row_idx 范围是 [0, BS*TopK)，对应的 batch index 是 row_idx / top_k
            int32_t batch_idx = row_idx / top_k;
            x_offset = (uint64_t)batch_idx * this->in_dim + group_idx * GROUP_SIZE;
        } else {
            // W2 阶段：输入是 Workspace [BS*TopK, InterDim]
            // row_idx 直接对应行
            x_offset = (uint64_t)row_idx * this->in_dim + group_idx * GROUP_SIZE;
        }

        inQueueX.AllocTensor<T>(x_local);
        DataCopy(x_local, xGm[x_offset], GROUP_SIZE);
        inQueueX.EnQue(x_local);
    }

    __aicore__ inline void CopyInSZ(int32_t expert_id, int32_t row_idx, int32_t group_idx, int32_t n_offset, int32_t cur_n_len)
    {
        // 1. Load Scale & Offset (Zeros) for current N-Tile
        // [Experts, Groups, OutDim] -> Offset by n_offset
        uint64_t sz_offset = (uint64_t)expert_id * this->num_groups * this->out_dim +
                            (uint64_t)group_idx * this->out_dim + 
                            n_offset;
        
        inQueueScale.AllocTensor<T>(s_local);
        inQueueOffset.AllocTensor<T>(z_local);
        
        DataCopy(s_local, scalesGm[sz_offset], cur_n_len);
        DataCopy(z_local, zerosGm[sz_offset], cur_n_len);
        
        inQueueScale.EnQue(s_local);
        inQueueOffset.EnQue(z_local);
    }

    // Helper to load a chunk of W (Tiled by N)
    __aicore__ inline void CopyInW(int32_t expert_id, int32_t group_idx, int32_t k_inner_start, int32_t n_offset, int32_t cur_n_len, LocalTensor<int32_t>& w_local)
    {
        uint64_t w_stride_k = this->out_dim_packed; // Global stride is still full N/8
        int32_t cur_n_packed = cur_n_len / PACK_RATIO;
        int32_t n_offset_packed = n_offset / PACK_RATIO;

        // W layout: [Expert, InDim, OutDim/8]
        // Base Offset points to the start of the row + current N-tile offset
        uint64_t w_offset = (uint64_t)expert_id * this->in_dim * w_stride_k + 
                            (uint64_t)(group_idx * GROUP_SIZE + k_inner_start) * w_stride_k +
                            n_offset_packed;
        
        inQueueW.AllocTensor<int32_t>(w_local);
        
        // Use DataCopy with Stride to pick the sub-matrix
        // blockLen: width of current tile in bytes
        // srcStride: width of remaining part of global matrix row in bytes
        AscendC::DataCopyExtParams w_copy_params{
            (uint16_t)COMPUTE_ROWS, 
            (uint32_t)(cur_n_packed * sizeof(int32_t)), 
            (uint32_t)((w_stride_k - cur_n_packed) * sizeof(int32_t)), 
            0, 0
        };
        AscendC::DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
        
        DataCopyPad(w_local, weightGm[w_offset], w_copy_params, padParams);
        inQueueW.EnQue(w_local);
    }

    __aicore__ inline void ComputeChunk(
        LocalTensor<int32_t>& w_local,
        int32_t x_offset_idx,
        int32_t tile_n,
        int32_t tile_n_packed,
        LocalTensor<half>& x_full_half,
        LocalTensor<half>& group_acc_xw)
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


    __aicore__ inline void ProcessGroup(int32_t g_idx, int32_t expert_id, int32_t row_idx, 
                                      int32_t n_offset, int32_t cur_n_len, 
                                      LocalTensor<float>& global_acc)
    {
        // Buffers (Allocated size is TILE_N, use cur_n_len for calculation)
        LocalTensor<half> group_acc = calcBuf.GetWithOffset<half>(cur_n_len, offset_group_acc);
        LocalTensor<half> w_half = calcBuf.GetWithOffset<half>(GROUP_TILE * cur_n_len, offset_w_half);
        LocalTensor<float> x_float_full = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_x_float);
        LocalTensor<float> z_float = calcBuf.GetWithOffset<float>(cur_n_len, offset_z_float);
        LocalTensor<float> reduce_buf = calcBuf.GetWithOffset<float>(GROUP_SIZE, offset_reduce_buf);

        CopyInX(row_idx, g_idx);

        CopyInW(expert_id, g_idx, 0, n_offset, cur_n_len, w_local_arr[0]);

        // Matrix Multiplication Loop
        Duplicate(group_acc, (half) 0.0f, cur_n_len);

        // Process X (T -> Float -> Half)
        inQueueX.DeQue(x_local);


        Cast(x_float_full, x_local, RoundMode::CAST_NONE, GROUP_SIZE);
        inQueueX.FreeTensor(x_local);
        ReduceSum(reduce_buf, x_float_full, reduce_buf, GROUP_SIZE);
        
        int32_t cur_n_packed = cur_n_len / PACK_RATIO;

        int w_idx = 0;
        for (int32_t k_inner = 0; k_inner < GROUP_SIZE; k_inner += COMPUTE_ROWS) {
            // 1. 预取下一块
            if (k_inner + COMPUTE_ROWS < GROUP_SIZE) {
                CopyInW(expert_id, g_idx, k_inner + COMPUTE_ROWS, n_offset, cur_n_len, w_local_arr[w_idx ^ 1]);
            } else {
                CopyInSZ(expert_id, row_idx, g_idx, n_offset, cur_n_len);
            }

            // 2. DeQue 当前 W
            inQueueW.DeQue(w_local_arr[w_idx]);

            // 3. 计算（封装）
            ComputeChunk(
                w_local_arr[w_idx],
                k_inner,
                cur_n_len,
                cur_n_packed,
                x_local,
                group_acc);

            // 4. 释放
            inQueueW.FreeTensor(w_local_arr[w_idx]);
            w_idx ^= 1;
        }
        
        // Calculate Sum(X)
        float group_x_sum = reduce_buf.GetValue(0);

        // Process Scale & Offset
        inQueueScale.DeQue(s_local);
        inQueueOffset.DeQue(z_local);

        // Finalize Correction
        Mul(z_local, z_local, s_local, cur_n_len);
        Cast(z_float, z_local, RoundMode::CAST_NONE, cur_n_len);
        Axpy(global_acc, z_float, group_x_sum, cur_n_len);
        MulAddDst(global_acc, group_acc, s_local, cur_n_len);

        inQueueScale.FreeTensor(s_local);
        inQueueOffset.FreeTensor(z_local);
    }

    __aicore__ inline void CopyOut(int32_t row_idx, int32_t n_offset, int32_t cur_n_len)
    {
        outQueueY.DeQue<T>(y_local);
        LocalTensor<float> y_fp32 = y_local.template ReinterpretCast<float>();

        if constexpr (IS_WEIGHTED_SUM) {
            // W2 阶段：需要乘 TopK Weight
            // topkWeightsGm 是 [BS * TopK]
            float w_val = topkWeightsGm.GetValue(row_idx);
            Muls(y_fp32, y_fp32, w_val, cur_n_len);
        }

        // Cast FP32 -> T
        Cast(y_local, y_fp32, RoundMode::CAST_ROUND, cur_n_len);

        PipeBarrier<PIPE_V>();

        // Atomic Add result to global memory
        AscendC::SetAtomicAdd<T>();
        
        if constexpr (IS_WEIGHTED_SUM) {
            // W2 Output: [BS, OutDim]
            // 需要聚合回 Batch 维度
            int32_t batch_idx = row_idx / top_k;
            uint64_t y_offset = (uint64_t)batch_idx * this->out_dim + n_offset;
            DataCopy(yGm[y_offset], y_local, cur_n_len);
        } else {
            // W13 Output: [BS * TopK, OutDim] (Workspace)
            // 保持展开维度
            uint64_t y_offset = (uint64_t)row_idx * this->out_dim + n_offset;
            DataCopy(yGm[y_offset], y_local, cur_n_len);
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
    LocalTensor<int32_t> w_local_arr[2];

    int32_t top_k;
    int32_t in_dim;
    int32_t out_dim;
    int32_t out_dim_packed;
    int32_t num_experts;
    int32_t num_groups;
    int32_t total_tokens;
};

// -----------------------------------------------------------------------------
// Fused Kernel Implementation (BS <= 4)
// -----------------------------------------------------------------------------
template<typename T>
__aicore__ inline void fused_moe_small_bs_impl(
    GM_ADDR x, 
    GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets,
    GM_ADDR w2_weight, GM_ADDR w2_scales, GM_ADDR w2_offsets,
    GM_ADDR expert_ids, GM_ADDR topk_weights,
    GM_ADDR workspace, GM_ADDR y,
    int32_t total_tokens, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts, int32_t top_k)
{ 
    AscendC::TPipe pipe;
    
    // Workspace layout calculation
    // Workspace 1: W13 Output [TotalTokens, 2 * InterDim]
    GM_ADDR w13_out_ptr = workspace;
    uint64_t w13_out_size = (uint64_t)total_tokens * (inter_dim * 2) * sizeof(T);
    
    // Workspace 2: SwiGLU Output [TotalTokens, InterDim] (Input to W2)
    GM_ADDR w2_in_ptr = (GM_ADDR)((__gm__ uint8_t*)workspace + w13_out_size);

    // Phase 0: Zero Out Workspace (W13 Output area)
    {
        KernelZeroOut<T> zeroOp;
        zeroOp.Init(&pipe, w13_out_ptr, w13_out_size / sizeof(T));
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
        // W13 需要 top_k 来计算 row_idx 对应的 batch_idx，从而广播输入 X
        KernelGroupedGemvW4A16Moe<T, true, false> op_w13;
        op_w13.Init(&pipe, x, w13_weight, w13_scales, w13_offsets, expert_ids, w13_out_ptr, nullptr,
                    total_tokens, in_dim, inter_dim * 2, num_experts, top_k);
        op_w13.Process();
    }

    // Barrier & Reset
    AscendC::SyncAll();
    pipe.Reset();

    // ------------------------------------------------------------------------
    // Phase 2: SwiGLU Activation
    // ------------------------------------------------------------------------
    {
        // SwiGLU 只需要知道总 Token 数
        KernelSwiGLU<T> op_act;
        op_act.Init(&pipe, w13_out_ptr, w2_in_ptr, total_tokens, inter_dim);
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
        // W2 需要 top_k 来计算 row_idx 对应的 batch_idx，从而将结果累加到输出 Y
        KernelGroupedGemvW4A16Moe<T, false, true> op_w2;
        op_w2.Init(&pipe, w2_in_ptr, w2_weight, w2_scales, w2_offsets, expert_ids, y, topk_weights,
                   total_tokens, inter_dim, out_dim, num_experts, top_k);
        op_w2.Process();
    }
}

// -----------------------------------------------------------------------------
// Extern C Entry Points
// -----------------------------------------------------------------------------

// Fused MoE FP16
extern "C" __global__ __aicore__ void fused_moe_small_bs_w4a16_fp16(
    GM_ADDR x, GM_ADDR w13_weight, GM_ADDR w13_scales, GM_ADDR w13_offsets,  GM_ADDR w2_weight, GM_ADDR w2_scales, GM_ADDR w2_offsets,
    GM_ADDR expert_ids, GM_ADDR topk_weights, GM_ADDR workspace, GM_ADDR y,
    int32_t total_tokens, int32_t in_dim, int32_t inter_dim, int32_t out_dim, int32_t num_experts, int32_t top_k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    fused_moe_small_bs_impl<half>(x, w13_weight, w13_scales, w13_offsets, w2_weight, w2_scales, w2_offsets, expert_ids, topk_weights, workspace, y, total_tokens, in_dim, inter_dim, out_dim, num_experts, top_k);
}

// Standalone GEMV FP16 (Updated Interface)
extern "C" __global__ __aicore__ void grouped_gemv_w4a16_moe_fp16(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scales, GM_ADDR offsets, GM_ADDR expert_ids, GM_ADDR y, 
    int32_t total_tokens, int32_t in_dim, int32_t out_dim, int32_t num_experts, int32_t top_k)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    KernelGroupedGemvW4A16Moe<half, false, false> op;
    // Standalone GEMV 通常不需要 Broadcast X，top_k 主要用于兼容性或特定逻辑
    op.Init(&pipe, x, weight, scales, offsets, expert_ids, y, nullptr, total_tokens, in_dim, out_dim, num_experts, top_k);
    op.Process();
}
