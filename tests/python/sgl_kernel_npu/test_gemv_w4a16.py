import torch
import torch_npu
import sys
import time
import math

# 尝试导入自定义算子库
try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Assuming it is loaded via torch.ops")

def run_test():
    # --------------------------------------------------------------------------
    # 1. 配置参数
    # --------------------------------------------------------------------------
    # 模拟 LLaMA-3-8B/70B 常见的形状
    # K=4096 (Hidden Size), N=4096 (Intermediate), GroupSize=128
    IN_DIM = 4096   
    OUT_DIM = 4096 
    GROUP_SIZE = 128
    
    # 你的自定义算子目前只支持 batch_size = 1
    BATCH_SIZE = 1 
    
    DEVICE = "npu:0"
    DTYPE = torch.float16
    
    print("=" * 70)
    print(f"Test Configuration:")
    print(f"  Shape      : X=[{BATCH_SIZE}, {IN_DIM}], W=[{IN_DIM}, {OUT_DIM}]")
    print(f"  Group Size : {GROUP_SIZE}")
    print(f"  Dtype      : {DTYPE}")
    print(f"  Compare To : torch_npu.npu_weight_quant_batchmatmul")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # 2. 数据生成
    # --------------------------------------------------------------------------
    torch.manual_seed(42)
    
    # X: [1, K]
    x = torch.randn((BATCH_SIZE, IN_DIM), dtype=DTYPE, device=DEVICE)
    
    # Scales: [Groups, N]
    num_groups = IN_DIM // GROUP_SIZE
    scales = torch.randn((num_groups, OUT_DIM), dtype=DTYPE, device=DEVICE) * 1.0 / math.sqrt(IN_DIM)
    
    # offsets (Offset): [Groups, N]
    # 注意：这里的 offsets 是浮点类型的 Offset，对应公式 Y = X * (W + Z) * S 中的 Z
    offsets = torch.randint(-8, 8, (num_groups, OUT_DIM), device=DEVICE).to(dtype=DTYPE)

    # Weights: [K, N] 原始 Int8 权重 (-8 到 7)
    # 我们生成 int32 但限制范围在 int4 内
    weight_unpacked = torch.randint(-8, 8, (IN_DIM, OUT_DIM), dtype=torch.int32, device=DEVICE)

    # --------------------------------------------------------------------------
    # 3. 权重打包 (Packing)
    # --------------------------------------------------------------------------
    print(">> Packing weights using torch_npu.npu_convert_weight_to_int4pack ...")
    # 使用华为官方 API 进行打包，确保内存布局符合 NPU 硬件要求
    # 输入: [K, N] int32, 输出: [K, N/8] int32 (内部是特殊的 NPU 格式)
    weight_packed = torch_npu.npu_convert_weight_to_int4pack(weight_unpacked)

    # --------------------------------------------------------------------------
    # 4. 运行 NPU 原生算子 (Ground Truth)
    # --------------------------------------------------------------------------
    print(">> Running Native Op (npu_weight_quant_batchmatmul)...")
    
    # Warmup
    for _ in range(10):
        y_ref = torch_npu.npu_weight_quant_batchmatmul(
            x, 
            weight_packed, 
            antiquant_scale=scales, 
            antiquant_offset=offsets, 
            antiquant_group_size=GROUP_SIZE
        )
    
    torch.npu.synchronize()
    start_native = time.time()
    
    # Actual Run
    y_ref = torch_npu.npu_weight_quant_batchmatmul(
        x, 
        weight_packed, 
        antiquant_scale=scales, 
        antiquant_offset=offsets, 
        antiquant_group_size=GROUP_SIZE
    )
    
    torch.npu.synchronize()
    time_native = (time.time() - start_native) * 1000

    # --------------------------------------------------------------------------
    # 5. 运行自定义 AscendC 算子
    # --------------------------------------------------------------------------
    print(">> Running Custom Kernel (gemv_w4a16)...")
    
    # 自定义算子输入需要 flatten 的 x: [K]
    x_flat = x.view(-1)
    
    # Warmup
    for _ in range(10):
        y_custom = torch.ops.npu.gemv_w4a16(x_flat, weight_packed, scales, offsets)
        
    torch.npu.synchronize()
    start_custom = time.time()
    
    # Actual Run
    y_custom = torch.ops.npu.gemv_w4a16(x_flat, weight_packed, scales, offsets)
    
    torch.npu.synchronize()
    time_custom = (time.time() - start_custom) * 1000

    # --------------------------------------------------------------------------
    # 6. 结果对比
    # --------------------------------------------------------------------------
    # 将自定义输出 reshape 回 [1, N] 以便对比
    y_custom = y_custom.view(1, OUT_DIM)
    
    # 转换为 float32 进行高精度对比
    diff = (y_ref.float() - y_custom.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # 打印性能对比
    print("-" * 70)
    print(f"Performance Comparison:")
    print(f"  Native Op Time : {time_native:.3f} ms")
    print(f"  Custom Op Time : {time_custom:.3f} ms")
    if time_custom < time_native:
        print(f"  >> Speedup     : {time_native / time_custom:.2f}x 🚀")
    else:
        print(f"  >> Slowdown    : {time_native / time_custom:.2f}x")

    # 打印精度对比
    print("-" * 70)
    print(f"Accuracy Verification:")
    print(f"  Max Diff       : {max_diff:.6f}")
    print(f"  Mean Diff      : {mean_diff:.6f}")
    
    # 阈值判定
    # BF16 精度下，积累误差可能会达到 1e-2 级别，对于大矩阵乘法是正常的
    threshold = 0.05 
    
    if max_diff < threshold or mean_diff < 0.005:
        print("\n✅ Result Matches! Test PASSED.")
    else:
        print("\n❌ Result Mismatch! Test FAILED.")
        
        # Debug info
        print("\nDebug First Error:")
        mask = diff > threshold
        indices = torch.nonzero(mask, as_tuple=False)
        if indices.numel() > 0:
            idx = indices[0]
            r, c = idx[0].item(), idx[1].item()
            print(f"  At index [{r}, {c}]:")
            print(f"    Native : {y_ref[r, c].item():.6f}")
            print(f"    Custom : {y_custom[r, c].item():.6f}")
            print(f"    Diff   : {diff[r, c].item():.6f}")

if __name__ == "__main__":
    run_test()