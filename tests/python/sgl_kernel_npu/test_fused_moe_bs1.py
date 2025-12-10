import torch
import torch_npu
import math
import sys

try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Please ensure the kernel is compiled and installed.")
    sys.exit(1)
# -----------------------------------------------------------------------------
# 配置区域
# -----------------------------------------------------------------------------
DEVICE_STR = "npu:0"
DTYPE = torch.bfloat16  # 或者 torch.float16

# 模拟的模型参数
TOP_K = 8
NUM_EXPERTS = 128
IN_DIM = 2048
INTER_DIM = 768 # 中间维度 (SwiGLU 输入实际上是 2 * INTER_DIM)
OUT_DIM = 2048
GROUP_SIZE = 32 # 量化分组大小

# -----------------------------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------------------------

def pack_weights_for_npu(raw_weights_cpu, device):
    """
    将 CPU 上的原始 Int4 权重 (Int32存储) 打包并移动到 NPU。
    Raw Shape: [Experts, In, Out]
    Packed Shape: [Experts, In, Out/8]
    """
    print(f">> [Data] Packing weights: {raw_weights_cpu.shape} ...")
    num_experts, in_dim, out_dim = raw_weights_cpu.shape
    
    # 1. 扁平化为 [E*In, Out] 并移至 NPU
    flat_weights = raw_weights_cpu.view(-1, out_dim).to(device)
    
    # 2. 调用 NPU API 打包: [N, K] -> [N, K/8]
    # 注意：Ascend 的 int4pack 通常是按 K 维度(Row)打包的，但 Gemv 需要按 N 维度(Col)打包
    # npu_convert_weight_to_int4pack 的行为取决于具体的算子实现要求
    # 这里的 kernel 假设输入是 npu_convert_weight_to_int4pack 的结果
    packed_flat = torch_npu.npu_convert_weight_to_int4pack(flat_weights)
    
    # 3. 恢复 Experts 维度
    weight_packed = packed_flat.view(num_experts, in_dim, out_dim // 8)
    
    return weight_packed

def cpu_grouped_gemv(x, weights, scales, group_size):
    """
    CPU 模拟 Grouped GEMV: Y = (X @ W) * Scale
    X: [Rows, In]
    Weights: [In, Out] (Int32)
    Scales: [Groups, Out]
    """
    rows, in_dim = x.shape
    _, out_dim = weights.shape
    num_groups = in_dim // group_size
    
    # Reshape for grouping
    # X: [Rows, Groups, GroupSize, 1]
    x_reshaped = x.view(rows, num_groups, group_size, 1)
    
    # W: [Groups, GroupSize, Out]
    w_reshaped = weights.view(num_groups, group_size, out_dim)
    
    # Scale: [1, Groups, 1, Out]
    s_reshaped = scales.view(1, num_groups, 1, out_dim)
    
    # 1. Dot Product over GroupSize
    # [Rows, Groups, GroupSize, 1] * [Groups, GroupSize, Out] -> Sum(dim=2)
    # Result: [Rows, Groups, Out]
    dot = (x_reshaped * w_reshaped).sum(dim=2)
    
    # 2. Apply Scales
    # [Rows, Groups, Out] * [1, Groups, Out]
    scaled = dot * scales.unsqueeze(0)
    
    # 3. Sum over Groups -> [Rows, Out]
    y = scaled.sum(dim=1)
    
    return y

def compute_ground_truth(x, w13, s13, w2, s2, e_ids, topk_weights, group_size):
    """
    CPU Ground Truth 计算
    """
    print(">> [CPU] Computing Ground Truth...")
    x = x.float()
    w13 = w13.float() # [E, In, 2*Inter]
    s13 = s13.float()
    w2 = w2.float()   # [E, Inter, Out]
    s2 = s2.float()
    weights = topk_weights.float()
    
    batch_out = torch.zeros((1, OUT_DIM), dtype=torch.float32)
    
    # 因为 BS=1，x 是 [1, InDim]，但在 MoE 内部它被广播给所有 TopK Expert
    x_input = x # [1, In]
    
    # 模拟 Kernel 内部逻辑：遍历 TopK
    for k in range(TOP_K):
        eid = e_ids[0, k].item()
        w_val = weights[0, k].item()
        
        # --- Phase 1: W13 (Gate + Up) ---
        # W13 shape: [In, 2*Inter]
        w13_expert = w13[eid]
        s13_expert = s13[eid]
        
        # Gemv
        # Output: [1, 2*Inter]
        hidden = cpu_grouped_gemv(x_input, w13_expert, s13_expert, group_size)
        
        # --- Phase 2: SwiGLU ---
        # Split [1, 2*Inter] -> Gate[1, Inter], Value[1, Inter]
        gate = hidden[:, :INTER_DIM]
        value = hidden[:, INTER_DIM:]
        
        # Swish(Gate) * Value
        # Swish = x * sigmoid(x) (SiLU)
        act = torch.nn.functional.silu(gate) * value
        
        # --- Phase 3: W2 (Down) ---
        w2_expert = w2[eid]
        s2_expert = s2[eid]
        
        # Gemv
        # Output: [1, Out]
        out = cpu_grouped_gemv(act, w2_expert, s2_expert, group_size)
        
        # --- Phase 4: Weighted Sum ---
        batch_out += out * w_val
        
    return batch_out

# -----------------------------------------------------------------------------
# 主测试逻辑
# -----------------------------------------------------------------------------
def run_test():
    torch.manual_seed(42)
    device = torch.device(DEVICE_STR)
    
    print("=" * 60)
    print(f"Testing Fused MoE (BS=1) on {DEVICE_STR}")
    print(f"Shape: In={IN_DIM}, Inter={INTER_DIM}, Out={OUT_DIM}, K={TOP_K}")
    print("=" * 60)

    # 1. 生成数据 (CPU)
    # X: [1, In]
    x_cpu = torch.randn(1, IN_DIM, dtype=DTYPE)
    
    # Expert IDs: [1, K]
    expert_ids_cpu = torch.randint(0, NUM_EXPERTS, (1, TOP_K), dtype=torch.int32)
    
    # Weights: [1, K] (Normalized for realism)
    topk_weights_cpu = torch.rand(1, TOP_K, dtype=torch.float32)
    topk_weights_cpu = topk_weights_cpu / topk_weights_cpu.sum()

    # Model Weights (Int4 simulated as Int32 range -8 to 7)
    # W13: [Experts, In, 2 * Inter]
    w13_cpu = torch.randint(-7, 8, (NUM_EXPERTS, IN_DIM, INTER_DIM * 2), dtype=torch.int32)
    # W2: [Experts, Inter, Out]
    w2_cpu = torch.randint(-7, 8, (NUM_EXPERTS, INTER_DIM, OUT_DIM), dtype=torch.int32)

    # Scales
    num_groups = IN_DIM // GROUP_SIZE
    scale_factor = 0.01
    s13_cpu = torch.randn(NUM_EXPERTS, num_groups, INTER_DIM * 2, dtype=DTYPE) * scale_factor
    
    num_groups_w2 = INTER_DIM // GROUP_SIZE
    s2_cpu = torch.randn(NUM_EXPERTS, num_groups_w2, OUT_DIM, dtype=DTYPE) * scale_factor

    # 2. 准备 NPU 数据
    x_npu = x_cpu.to(device)
    e_ids_npu = expert_ids_cpu.to(device).view(-1)     # Flatten [K]
    weights_npu = topk_weights_cpu.to(device).view(-1) # Flatten [K]
    
    w13_scales_npu = s13_cpu.to(device)
    w2_scales_npu = s2_cpu.to(device)
    
    # 打包权重
    w13_packed_npu = pack_weights_for_npu(w13_cpu, device)
    w2_packed_npu = pack_weights_for_npu(w2_cpu, device)

    # 3. 计算 Ground Truth (CPU FP32)
    y_ref = compute_ground_truth(
        x_cpu, w13_cpu, s13_cpu, w2_cpu, s2_cpu, 
        expert_ids_cpu, topk_weights_cpu, GROUP_SIZE
    )

    # 4. 运行 NPU Kernel
    print(">> [NPU] Running Fused Kernel...")
    torch.npu.synchronize()
    
    # 假设你的 Binding 名称如下
    # 注意：需要确保输入是 flatten 的 (expert_ids, weights)
    y_npu = torch.ops.npu.fused_moe_w4a16_bs1(
        x_npu,
        w13_packed_npu,
        w13_scales_npu,
        w2_packed_npu,
        w2_scales_npu,
        e_ids_npu,
        weights_npu
    )
    
    torch.npu.synchronize()
    y_npu_cpu = y_npu.float().cpu()

    # 5. 验证
    print("-" * 60)
    diff = (y_npu_cpu - y_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Result Shape: {y_npu_cpu.shape}")
    print(f"Max Diff    : {max_diff:.6f}")
    print(f"Mean Diff   : {mean_diff:.6f}")
    
    print(f"Sample Ref  : {y_ref[0, :5].tolist()}")
    print(f"Sample NPU  : {y_npu_cpu[0, :5].tolist()}")

    # 这里的阈值相对宽松，因为包含了两次 GEMV 和一次 Activation 的累积误差
    # 且 bfloat16 的精度较低
    if mean_diff < 0.1 and max_diff < 1.0:
        print("\n✅ Test PASSED")
    else:
        print("\n❌ Test FAILED")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        import traceback
        traceback.print_exc()