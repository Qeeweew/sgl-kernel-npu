import torch
import torch_npu
import math
import sys

# 尝试导入自定义算子库
try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Please ensure the kernel is compiled and installed.")
    sys.exit(1)

def compute_cpu_ground_truth(x_cpu, raw_weights_cpu, scales_cpu, zeros_cpu, expert_ids_cpu, group_size):
    """
    在 CPU 上使用 Float32 计算 Ground Truth。
    包含 Offsets (zeros) 的计算逻辑。
    Kernel 逻辑为: Y = (X * W) * Scale + Sum(X) * Zero * Scale
    等价于: Y = Scale * ( (X * W) + Sum(X) * Zero )
    """
    print(">> [CPU] Computing Ground Truth...")
    
    # 类型转换确保计算精度
    x_f32 = x_cpu.float()
    w_f32 = raw_weights_cpu.float()
    s_f32 = scales_cpu.float()
    z_f32 = zeros_cpu.float() # Offsets
    e_ids = expert_ids_cpu.long()
    
    top_k, in_dim = x_f32.shape
    num_experts, _, out_dim = w_f32.shape
    num_groups = in_dim // group_size
    
    y_ref = torch.zeros((top_k, out_dim), dtype=torch.float32, device="cpu")

    for i in range(top_k):
        eid = e_ids[i].item()
        
        if eid < 0 or eid >= num_experts:
            raise ValueError(f"Expert ID out of bounds: {eid}")

        # Input: [Groups, GroupSize, 1]
        xi = x_f32[i].view(num_groups, group_size, 1)
        
        # Weight: [Groups, GroupSize, N]
        wi = w_f32[eid].view(num_groups, group_size, out_dim)
        
        # Scale & Zero: [Groups, N]
        si = s_f32[eid]
        zi = z_f32[eid]

        # 1. Dot Product (X * W): [Groups, N]
        dot_xw = (wi * xi).sum(dim=1)
        
        # 2. Sum X: [Groups, 1]
        sum_x = xi.sum(dim=1)
        
        # 3. Calculate Correction term (Sum(X) * Z): [Groups, N]
        # sum_x broadcasts to [Groups, N]
        term_z = sum_x * zi
        
        # 4. Combine and Scale: (Dot + Correction) * Scale
        # [Groups, N]
        group_res = (dot_xw + term_z) * si
        
        # 5. Reduce Groups: [N]
        y_ref[i] = group_res.sum(dim=0)

    return y_ref

def run_npu_kernel(x_npu, weight_packed_npu, scales_npu, zeros_npu, expert_ids_npu):
    """
    执行 NPU 自定义算子。
    """
    print(">> [NPU] Executing Kernel...")
    
    # torch.npu.synchronize()
    
    # 调用算子, 传入真实的 zeros (offsets)
    y_npu = torch.ops.npu.grouped_gemv_w4a16_moe(
        x_npu, 
        weight_packed_npu, 
        scales_npu, 
        zeros_npu,
        expert_ids_npu
    )
    
    torch.npu.synchronize()
    
    return y_npu

def pack_weights_for_npu(raw_weights_cpu, device):
    """
    将 CPU 上的原始 Int4 权重打包并移动到 NPU。
    """
    print(">> [Data] Packing weights for NPU...")
    num_experts, in_dim, out_dim = raw_weights_cpu.shape
    
    flat_weights = raw_weights_cpu.view(-1, out_dim).to(device)
    packed_flat = torch_npu.npu_convert_weight_to_int4pack(flat_weights)
    weight_packed = packed_flat.view(num_experts, in_dim, out_dim // 8)
    
    return weight_packed

def find_first_error(y_ref, y_npu, threshold=0.1):
    """
    查找并打印第一个超过阈值的误差位置
    """
    diff = (y_ref - y_npu).abs()
    mask = diff > threshold
    
    error_indices = torch.nonzero(mask, as_tuple=False)
    
    if error_indices.numel() > 0:
        first_idx = error_indices[0]
        row = first_idx[0].item()
        col = first_idx[1].item()
        
        ref_val = y_ref[row, col].item()
        npu_val = y_npu[row, col].item()
        delta = diff[row, col].item()
        
        print(f"\n[Debug] First Mismatch Detected (Threshold > {threshold}):")
        print(f"  Position : Row {row}, Col {col}")
        print(f"  Ref Value: {ref_val:.6f}")
        print(f"  NPU Value: {npu_val:.6f}")
        print(f"  Abs Diff : {delta:.6f}")
        
        if col + 5 < y_ref.shape[1]:
            print(f"  Context Ref[{row}, {col}:{col+5}]: {y_ref[row, col:col+5].tolist()}")
            print(f"  Context NPU[{row}, {col}:{col+5}]: {y_npu[row, col:col+5].tolist()}")
        return False
    return True

def run_stable_test():
    # --------------------------------------------------------------------------
    # 1. 参数配置
    # --------------------------------------------------------------------------
    TOP_K = 8
    NUM_EXPERTS = 128
    IN_DIM = 2048 # 必须是 128 的倍数
    OUT_DIM = 4096 
    GROUP_SIZE = 128
    
    device_str = "npu:0"
    device = torch.device(device_str)
    dtype = torch.bfloat16
    
    torch.manual_seed(42)
    
    print("-" * 60)
    print(f"MoE Grouped GEMV Test (With Offsets)")
    print(f"TopK={TOP_K}, Experts={NUM_EXPERTS}, Shape=[{IN_DIM}, {OUT_DIM}]")
    print("-" * 60)

    # --------------------------------------------------------------------------
    # 2. 数据生成 (全部在 CPU 上生成 Golden Data)
    # --------------------------------------------------------------------------
    print(">> [Data] Generating Golden Data on CPU...")
    
    # X: [TopK, InDim]
    x_cpu = torch.randn((TOP_K, IN_DIM), dtype=dtype)
    
    # Expert IDs: [TopK]
    expert_ids_cpu = torch.randint(0, NUM_EXPERTS, (TOP_K,), dtype=torch.int32)
    
    # Weights: [Experts, InDim, OutDim] (Int4 values stored as Int32 for simplicity in python)
    # Range -8 to 7 standard int4
    raw_weights_cpu = torch.randint(-8, 8, (NUM_EXPERTS, IN_DIM, OUT_DIM), dtype=torch.int32)
    
    # Scales: [Experts, Groups, OutDim]
    num_groups = IN_DIM // GROUP_SIZE
    scales_cpu = torch.randn((NUM_EXPERTS, num_groups, OUT_DIM), dtype=torch.float32) * 0.005
    scales_cpu = scales_cpu.to(dtype=dtype)

    # Zeros (Offsets): [Experts, Groups, OutDim]
    # 模拟量化零点，通常在小范围内波动
    zeros_cpu = torch.randn((NUM_EXPERTS, num_groups, OUT_DIM), dtype=torch.float32)
    zeros_cpu = zeros_cpu.to(dtype=dtype)

    # --------------------------------------------------------------------------
    # 3. 准备 NPU 数据
    # --------------------------------------------------------------------------
    print(">> [Data] Copying data to NPU...")
    x_npu = x_cpu.to(device)
    expert_ids_npu = expert_ids_cpu.to(device)
    scales_npu = scales_cpu.to(device)
    zeros_npu = zeros_cpu.to(device)
    
    # 权重打包
    weight_packed_npu = pack_weights_for_npu(raw_weights_cpu, device)

    # --------------------------------------------------------------------------
    # 4. 执行测试
    # --------------------------------------------------------------------------
    # 计算 Ground Truth (包含 Offsets)
    y_ref = compute_cpu_ground_truth(x_cpu, raw_weights_cpu, scales_cpu, zeros_cpu, expert_ids_cpu, GROUP_SIZE)
    
    # 执行 NPU Kernel
    y_npu = run_npu_kernel(x_npu, weight_packed_npu, scales_npu, zeros_npu, expert_ids_npu)

    # --------------------------------------------------------------------------
    # 5. 结果验证
    # --------------------------------------------------------------------------
    print("-" * 60)
    print(">> [Verify] Comparing results...")
    
    y_npu_cpu = y_npu.float().cpu()
    
    diff = (y_npu_cpu - y_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max Diff      : {max_diff:.6f}")
    print(f"  Mean Diff     : {mean_diff:.6f}")
    
    print(f"\n  Ref Sample: {y_ref[0, :5].tolist()}")
    print(f"  NPU Sample: {y_npu_cpu[0, :5].tolist()}")
    
    # 误差分析
    # 注意：引入 offset 后，涉及 sum(x) 的累加，BF16 精度损失可能会略微增加
    DEBUG_THRESHOLD = 0.15 
    passed = find_first_error(y_ref, y_npu_cpu, threshold=DEBUG_THRESHOLD)
    
    if passed and mean_diff < 0.05:
        print("\n✅ Test PASSED")
    else:
        print("\n❌ Test FAILED")

if __name__ == "__main__":
    run_stable_test()