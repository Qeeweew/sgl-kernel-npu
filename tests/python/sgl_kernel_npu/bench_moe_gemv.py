import torch
import torch_npu
import sys
import time

# 尝试导入自定义算子库
try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Please ensure the kernel is compiled and installed.")
    sys.exit(1)

def pack_weights_for_npu(raw_weights_cpu, device):
    """
    将 Int32 存储的权重打包为 Int4 格式
    """
    num_experts, in_dim, out_dim = raw_weights_cpu.shape
    # 展平以便打包
    flat_weights = raw_weights_cpu.view(-1, out_dim).to(device)
    # 调用 NPU API 打包 (Int32 -> Int4 Tensor)
    packed_flat = torch_npu.npu_convert_weight_to_int4pack(flat_weights)
    # 恢复形状: [E, N, K // 8]
    return packed_flat.view(num_experts, in_dim, out_dim // 8)

def benchmark_one_config(in_dim, out_dim, num_warmup=1, num_steps=200):
    # --------------------------------------------------------------------------
    # 1. 固定配置
    # --------------------------------------------------------------------------
    TOP_K = 8
    NUM_EXPERTS = 128
    GROUP_SIZE = 32
    
    device = torch.device("npu:0")
    dtype = torch.bfloat16
    
    print(f">> Preparing: K={TOP_K}, E={NUM_EXPERTS}, Shape=[{in_dim}, {out_dim}], G={GROUP_SIZE}")

    # --------------------------------------------------------------------------
    # 2. 数据准备
    # --------------------------------------------------------------------------
    torch.manual_seed(42)
    
    # Input X: [TopK, InDim]
    x = torch.randn((TOP_K, in_dim), dtype=dtype, device=device)
    
    # Expert IDs: [TopK]
    # 要求不重复: 使用 randperm 生成不重复的随机 ID
    expert_ids_long = torch.randperm(NUM_EXPERTS, device=device)[:TOP_K]
    expert_ids = expert_ids_long.int() # 转为 int32
    
    # Scales: [Experts, Groups, OutDim]
    num_groups = in_dim // GROUP_SIZE
    scales = torch.randn((NUM_EXPERTS, num_groups, out_dim), dtype=dtype, device=device)
    
    # Weights: CPU 生成 -> 打包 -> NPU
    # 注意：raw_weights 只是为了打包流程，实际数值不影响性能
    raw_w_cpu = torch.randint(-8, 8, (NUM_EXPERTS, in_dim, out_dim), dtype=torch.int32)
    weights_packed = pack_weights_for_npu(raw_w_cpu, device)

    # --------------------------------------------------------------------------
    # 3. 定义执行闭包 (Closure)
    # --------------------------------------------------------------------------
    # 在 Graph 模式下，输入 Tensor 的内存地址必须固定
    def run_op():
        return torch.ops.npu.grouped_gemv_w4a16_moe(x, weights_packed, scales, expert_ids)

    # --------------------------------------------------------------------------
    # 4. 预热 (Warmup) & Graph 录制
    # --------------------------------------------------------------------------
    # 充分预热以激活 NPU
    for _ in range(num_warmup):
        run_op()
    torch.npu.synchronize()

    # 初始化 Graph
    # 在 torch_npu 环境下，torch.npu.CUDAGraph 会被映射为 NPU Graph
    g = torch.npu.NPUGraph()
    
    # 录制 Graph
    try:
        with torch.npu.graph(g):
            static_output = run_op()
    except Exception as e:
        print(f"❌ Graph capture failed: {e}")
        print("Fallback to normal execution loop not implemented for strict benchmark.")
        return

    # --------------------------------------------------------------------------
    # 5. 性能测试 (使用 Event 计时)
    # --------------------------------------------------------------------------
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)
    
    torch.npu.synchronize()
    
    # 开始计时
    start_event.record()
    for _ in range(num_steps):
        g.replay()
    end_event.record()
    
    # 等待完成
    end_event.synchronize()
    
    # --------------------------------------------------------------------------
    # 6. 计算耗时
    # --------------------------------------------------------------------------
    total_ms = start_event.elapsed_time(end_event)
    avg_us = (total_ms / num_steps) * 1000
    
    print(f"   [Result] Avg Latency: {avg_us:.3f} us")
    print("-" * 60)

def main():
    torch.npu.set_device("npu:0")
    print("=" * 60)
    print(f"{'MoE GEMV NPU Graph Benchmark':^60}")
    print("=" * 60)

    # Config 1
    benchmark_one_config(in_dim=768, out_dim=2048)
    
    # Config 2
    benchmark_one_config(in_dim=2048, out_dim=1536)

if __name__ == "__main__":
    main()