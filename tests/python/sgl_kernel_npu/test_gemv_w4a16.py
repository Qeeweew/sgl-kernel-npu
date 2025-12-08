import torch
import torch_npu
import math
import sys

# 尝试导入自定义算子库 (通常在 torch.ops.npu 下)
try:
    # 这一步通常不需要显式 import，只要库被 torch 加载
    # 这里的 pass 只是占位，确保环境正常
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Please ensure the kernel is compiled and installed.")
    sys.exit(1)

def compute_cpu_ground_truth(x_cpu, w_unpacked_cpu, scales_cpu, group_size):
    """
    在 CPU 上计算 Ground Truth。
    """
    print(">> [CPU] Computing Ground Truth...")
    
    # 转换为 Float32 保证精度
    x_f32 = x_cpu.float()            # [1, K]
    w_f32 = w_unpacked_cpu.float()   # [K, N]
    s_f32 = scales_cpu.float()       # [Groups, N]
    
    K = x_f32.shape[1]
    N = w_f32.shape[1]
    num_groups = K // group_size
    
    # Reshape for Grouped computation
    # X: [1, K] -> [1, Groups, GroupSize]
    x_g = x_f32.view(1, num_groups, group_size)
    
    # W: [K, N] -> [Groups, GroupSize, N]
    w_g = w_f32.view(num_groups, group_size, N)
    
    # 1. Group-wise Dot Product: Sum(X * W) inside group
    # [1, Groups, GroupSize] * [Groups, GroupSize, N] -> [1, Groups, N]
    # 使用 einsum 进行乘加: b=batch(1), g=group, s=sub_group(32), n=out_dim
    dot = torch.einsum('bgs,gsn->bgn', x_g, w_g)
    
    # 2. Apply Scales: [1, Groups, N] * [Groups, N] (Broadcasting)
    # Scales are per-group, per-channel
    scaled_dot = dot * s_f32.unsqueeze(0)
    
    # 3. Sum over Groups to get final output
    # [1, Groups, N] -> [1, N]
    y_ref = scaled_dot.sum(dim=1)
    
    return y_ref

def pack_weights_for_npu(w_unpacked_cpu, device):
    """
    模拟 Python 侧的打包逻辑。
    Input: [K, N] int32 (values -8..7)
    Output: [K, N // 8] int32 (packed)
    """
    print(">> [Data] Packing weights for NPU...")
    
    # 1. Move to NPU
    w_npu = w_unpacked_cpu.to(device)
    
    # 2. Call NPU API to pack
    # npu_convert_weight_to_int4pack expects [K, N] and returns [K, N/8]
    # ensuring the memory layout is compatible with the kernel's expectation
    w_packed = torch_npu.npu_convert_weight_to_int4pack(w_npu)
    
    return w_packed

def find_first_error(y_ref, y_npu, threshold=0.1):
    diff = (y_ref - y_npu).abs()
    mask = diff > threshold
    error_indices = torch.nonzero(mask, as_tuple=False)
    
    if error_indices.numel() > 0:
        first_idx = error_indices[0]
        col = first_idx[1].item() # Batch is 0
        
        ref_val = y_ref[0, col].item()
        npu_val = y_npu[0, col].item()
        delta = diff[0, col].item()
        
        print(f"\n[Debug] First Mismatch Detected (Threshold > {threshold}):")
        print(f"  Position : Col {col}")
        print(f"  Ref Value: {ref_val:.6f}")
        print(f"  NPU Value: {npu_val:.6f}")
        print(f"  Abs Diff : {delta:.6f}")
        return False
    return True

def run_test():
    # --------------------------------------------------------------------------
    # 1. 配置
    # --------------------------------------------------------------------------
    IN_DIM = 1024 # K
    OUT_DIM = 2048 # N
    GROUP_SIZE = 32    # 固定
    
    device_str = "npu:0"
    device = torch.device(device_str)
    
    # 测试 BF16 和 FP16
    dtypes = [torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        print("=" * 60)
        print(f"Testing GEMV W4A16 Custom Kernel with dtype={dtype}")
        print(f"Shape: [{IN_DIM}, {OUT_DIM}], GroupSize={GROUP_SIZE}")
        print("=" * 60)

        torch.manual_seed(42)

        # --------------------------------------------------------------------------
        # 2. 数据生成 (CPU)
        # --------------------------------------------------------------------------
        # X: [1, K] (Batch Size = 1)
        x_cpu = torch.randn((1, IN_DIM), dtype=dtype)
        
        # Weights: [K, N] (Int4 values stored as Int32)
        w_unpacked_cpu = torch.randint(-8, 8, (IN_DIM, OUT_DIM), dtype=torch.int32)
        
        # Scales: [Groups, N]
        num_groups = IN_DIM // GROUP_SIZE
        weight_std = 4.6
        scale_factor = 1.0 / (math.sqrt(IN_DIM) * weight_std)
        # 随机生成 Scale，范围控制在合理区间防止溢出
        scales_cpu = torch.randn((num_groups, OUT_DIM), dtype=dtype) * (2 * scale_factor)

        # --------------------------------------------------------------------------
        # 3. 准备 NPU 数据
        # --------------------------------------------------------------------------
        x_npu = x_cpu.to(device)
        scales_npu = scales_cpu.to(device)
        
        # 打包权重
        w_packed_npu = pack_weights_for_npu(w_unpacked_cpu, device)

        # --------------------------------------------------------------------------
        # 4. 执行 Ground Truth
        # --------------------------------------------------------------------------
        y_ref = compute_cpu_ground_truth(x_cpu, w_unpacked_cpu, scales_cpu, GROUP_SIZE)

        # --------------------------------------------------------------------------
        # 5. 执行 Custom Kernel
        # --------------------------------------------------------------------------
        print(">> [NPU] Executing Custom Kernel (gemv_w4a16)...")
        # 确保输入是 contiguous 1D
        x_flat = x_npu.view(-1)
        
        # 调用 C++ 绑定的算子
        # 签名: gemv_w4a16(Tensor x, Tensor weight, Tensor scales) -> Tensor
        import time
        torch.npu.synchronize()
        start_time = time.time()
        
        y_npu = torch.ops.npu.gemv_w4a16(x_flat, w_packed_npu, scales_npu)
        
        torch.npu.synchronize()
        print(f">> Kernel Execution Time: {(time.time() - start_time)*1000:.3f} ms")

        # --------------------------------------------------------------------------
        # 6. 验证
        # --------------------------------------------------------------------------
        y_npu_cpu = y_npu.float().cpu().view(1, OUT_DIM)
        
        diff = (y_npu_cpu - y_ref).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print("-" * 40)
        print(f"Result Verification ({dtype}):")
        print(f"  Max Diff  : {max_diff:.6f}")
        print(f"  Mean Diff : {mean_diff:.6f}")
        
        # 精度阈值: BF16 精度较低，容忍度高一点
        threshold = 0.05 if dtype == torch.float16 else 0.15
        
        passed = find_first_error(y_ref, y_npu_cpu, threshold=threshold)
        
        if passed and mean_diff < (threshold / 2):
            print("\n✅ Test PASSED")
        else:
            print("\n❌ Test FAILED")
        print("\n")

if __name__ == "__main__":
    run_test()