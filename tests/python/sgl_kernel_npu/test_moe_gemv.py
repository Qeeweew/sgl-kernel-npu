import torch
import torch_npu
import math

# 确保你的环境已经加载了自定义算子库
try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Make sure the kernel is compiled and linked.")

def run_stable_test():
    # --------------------------------------------------------------------------
    # 1. 配置参数
    # --------------------------------------------------------------------------
    TOP_K = 8
    NUM_EXPERTS = 128
    IN_DIM = 1024  # K
    OUT_DIM = 1024 # N
    GROUP_SIZE = 32
    NUM_GROUPS = IN_DIM // GROUP_SIZE
    
    device = "npu:0"
    dtype = torch.bfloat16

    # 设置种子以复现结果
    torch.manual_seed(1024)

    print(f"Running Numerically Stable MoE Test:")
    print(f"  Shape: [K={IN_DIM}, N={OUT_DIM}]")
    print(f"  GroupSize: {GROUP_SIZE}")
    print("-" * 60)

    # --------------------------------------------------------------------------
    # 2. 构造数据 (控制方差)
    # --------------------------------------------------------------------------
    
    # X: 标准正态分布 N(0, 1)
    x = torch.randn((TOP_K, IN_DIM), dtype=dtype, device=device)
    
    # Expert IDs
    expert_ids = torch.randint(0, NUM_EXPERTS, (TOP_K,), dtype=torch.int32, device=device)

    # --------------------------------------------------------------------------
    # 3. 构造权重与 Scale (关键步骤)
    # --------------------------------------------------------------------------
    print("Generating Weights & Normalizing Scales...")
    
    # 1. 生成 Int4 权重 (-8 到 7)
    # Int4 的标准差约为 4.6
    raw_weights = torch.randint(-8, 8, (NUM_EXPERTS, IN_DIM, OUT_DIM), dtype=torch.int32, device="cpu")
    
    # 2. 计算 Scale 的缩放因子
    # 目标: 输出方差为 1
    # Scale ~ 1 / (sqrt(K) * std_dev(Weight))
    weight_std = 4.6
    scale_factor = 1.0 / (math.sqrt(IN_DIM) * weight_std)
    
    # 生成正的 Scale，均值约为计算出的 scale_factor
    # 这样最终输出 y 应该在 [-3, 3] 之间
    scales_cpu = torch.rand((NUM_EXPERTS, NUM_GROUPS, OUT_DIM), dtype=torch.float32) * (2 * scale_factor)
    scales = scales_cpu.to(dtype=dtype, device=device)

    # 3. 打包权重 (NPU 格式)
    # [Experts, In, Out] -> [Experts*In, Out]
    flat_weights = raw_weights.view(-1, OUT_DIM).to(device)
    # 调用 NPU API 打包: [Rows, Cols] -> [Rows, Cols/8]
    packed_flat_weights = torch_npu.npu_convert_weight_to_int4pack(flat_weights)
    weight_packed = packed_flat_weights.view(NUM_EXPERTS, IN_DIM, OUT_DIM // 8)

    # --------------------------------------------------------------------------
    # 4. Ground Truth 计算 (Python + Float32 Accumulation)
    # --------------------------------------------------------------------------
    print("Computing Ground Truth (Float32)...")
    y_ref = torch.zeros((TOP_K, OUT_DIM), dtype=torch.float32, device="cpu")
    
    x_cpu = x.float().cpu()
    raw_weights_cpu = raw_weights.float()
    
    for i in range(TOP_K):
        eid = expert_ids[i].item()
        
        # Reshape inputs for Grouped GEMV logic
        # Input: [Groups, GroupSize, 1]
        xi = x_cpu[i].view(NUM_GROUPS, GROUP_SIZE, 1)
        # Weight: [Groups, GroupSize, N]
        wi = raw_weights_cpu[eid].view(NUM_GROUPS, GROUP_SIZE, OUT_DIM)
        # Scale: [Groups, N]
        si = scales_cpu[eid]

        # Calculation: sum(w * x) * s
        # 1. Dot product inside group
        dot = (wi * xi).sum(dim=1) # -> [Groups, N]
        # 2. Apply scale
        scaled = dot * si          # -> [Groups, N]
        # 3. Reduce groups
        y_ref[i] = scaled.sum(dim=0) # -> [N]

    # 统计 Ground Truth 分布情况
    print(f"Ground Truth Stats -> Mean: {y_ref.mean():.4f}, Std: {y_ref.std():.4f}")
    print(f"  (Expected Std close to 1.0)")

    y_ref = y_ref.to(dtype=dtype, device=device)

    # --------------------------------------------------------------------------
    # 5. NPU Kernel 执行
    # --------------------------------------------------------------------------
    print("Executing NPU Kernel...")
    y_npu = torch.ops.npu.grouped_gemv_w4a16_moe(
        x, 
        weight_packed, 
        scales, 
        expert_ids
    )

    # --------------------------------------------------------------------------
    # 6. 验证
    # --------------------------------------------------------------------------
    print("-" * 60)
    print("Verification:")
    
    # 转换为 float32 进行比较
    diff = (y_npu.float() - y_ref.float()).abs()
    
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Max Diff : {max_diff:.6f}")
    print(f"  Mean Diff: {mean_diff:.6f}")
    
    # 打印前几个数值对比
    print("\n  Sample Data (First 5 elements of first row):")
    print(f"    Ref: {y_ref[0, :5].float().cpu().tolist()}")
    print(f"    NPU: {y_npu[0, :5].float().cpu().tolist()}")

    # 阈值判定
    # 由于 bfloat16 的有效位数较少，且数值现在归一化到了 1.0 左右，
    # 这里的误差通常在 1e-2 到 1e-3 级别是正常的。
    if mean_diff < 0.05 and max_diff < 0.2:
        print("\n✅ Test PASSED (Stable Distribution)!")
    else:
        print("\n❌ Test FAILED (Check Precision/Logic)!")

if __name__ == "__main__":
    run_stable_test()