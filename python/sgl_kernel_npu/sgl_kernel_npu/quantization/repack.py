import torch
import triton
import triton.language as tl
import time
from sgl_kernel_npu.utils.triton_utils import get_device_properties

import torch
import triton
import triton.language as tl
import time
from sgl_kernel_npu.utils.triton_utils import get_device_properties

@triton.jit
def int4_repack_kernel(
    in_ptr, out_ptr,
    N, K,
    stride_in_n, stride_in_k8,
    stride_out_k, stride_out_n8,
    NUM_CORES: tl.constexpr,
    BLOCK_N8: tl.constexpr
):
    """
    1. 保留 NUM_CORES 绑定，维持 Persistent Kernel 模式。
    2. 直接在 K 维度并行：将 K//8 行均分给 NUM_CORES。
    3. 移除多余的任务计算参数，代码极致清爽。
    4. N 维度在内部通过 1D 向量化处理。
    """
    pid = tl.program_id(0)
    
    # 直接计算 K 维度的总行数
    total_k8_rows = K // 8
    
    # 计算当前 Core 负责的 K 行范围
    rows_per_core = (total_k8_rows + NUM_CORES - 1) // NUM_CORES
    start_row = pid * rows_per_core
    
    # 如果当前 Core 分配不到任务，直接退出
    if start_row >= total_k8_rows:
        return
        
    end_row = tl.minimum(start_row + rows_per_core, total_k8_rows)

    # 外层循环：当前 Core 负责的 K 维度行
    for k8_idx in range(start_row, end_row):
        
        # 内层循环：沿 N 维度步进处理
        num_n8 = N // 8
        for n8_base in range(0, num_n8, BLOCK_N8):
            n8_idx = n8_base + tl.arange(0, BLOCK_N8)
            mask_out_n = n8_idx < num_n8

            # 纯 1D 向量寄存器
            out_0 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_1 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_2 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_3 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_4 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_5 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_6 = tl.zeros([BLOCK_N8], dtype=tl.uint32)
            out_7 = tl.zeros([BLOCK_N8], dtype=tl.uint32)

            # 静态展开 8 次子提取
            for i in tl.static_range(8):
                n_idx = n8_idx * 8 + i
                mask_n = n_idx < N
                
                # k8_idx 是标量，实现 1D 连续内存加载
                in_ptrs = in_ptr + n_idx * stride_in_n + k8_idx * stride_in_k8
                packed_in = tl.load(in_ptrs, mask=mask_n, other=0).to(tl.uint32)
                
                # 纯标量位移，NPU 完美向量化
                shift_n = i * 4
                out_0 |= ((((packed_in >> 0)  & 0xF) - 8) & 0xF) << shift_n
                out_1 |= ((((packed_in >> 4)  & 0xF) - 8) & 0xF) << shift_n
                out_2 |= ((((packed_in >> 8)  & 0xF) - 8) & 0xF) << shift_n
                out_3 |= ((((packed_in >> 12) & 0xF) - 8) & 0xF) << shift_n
                out_4 |= ((((packed_in >> 16) & 0xF) - 8) & 0xF) << shift_n
                out_5 |= ((((packed_in >> 20) & 0xF) - 8) & 0xF) << shift_n
                out_6 |= ((((packed_in >> 24) & 0xF) - 8) & 0xF) << shift_n
                out_7 |= ((((packed_in >> 28) & 0xF) - 8) & 0xF) << shift_n

            # 写回显存，基准 K_idx
            k_idx_base = k8_idx * 8
            tl.store(out_ptr + (k_idx_base + 0) * stride_out_k + n8_idx * stride_out_n8, out_0.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 1) * stride_out_k + n8_idx * stride_out_n8, out_1.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 2) * stride_out_k + n8_idx * stride_out_n8, out_2.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 3) * stride_out_k + n8_idx * stride_out_n8, out_3.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 4) * stride_out_k + n8_idx * stride_out_n8, out_4.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 5) * stride_out_k + n8_idx * stride_out_n8, out_5.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 6) * stride_out_k + n8_idx * stride_out_n8, out_6.to(tl.int32), mask=mask_out_n)
            tl.store(out_ptr + (k_idx_base + 7) * stride_out_k + n8_idx * stride_out_n8, out_7.to(tl.int32), mask=mask_out_n)


def repack_int4_npu(weight_packed_t: torch.Tensor) -> torch.Tensor:
    K_8, N = weight_packed_t.shape
    K = K_8 * 8
    
    out = torch.empty((K, N // 8), device=weight_packed_t.device, dtype=torch.int32)
    
    # 纯 1D 模式下，可以拉高 BLOCK_N8 榨干 Vector Core 吞吐
    BLOCK_N8 = 256
    
    # 获取 Vector Core 数量
    try:
        _, num_vectorcore = get_device_properties()
    except Exception:
        num_vectorcore = 32

    # Grid 锁定为 num_vectorcore
    int4_repack_kernel[(num_vectorcore,)](
        weight_packed_t, out,
        N, K,
        weight_packed_t.stride(1), weight_packed_t.stride(0),
        out.stride(0), out.stride(1),
        NUM_CORES=num_vectorcore,
        BLOCK_N8=BLOCK_N8
    )
    
    return out

def torch_reference_implementation(weight_t: torch.Tensor, K: int, N: int) -> torch.Tensor:
    """
    原生 PyTorch 参考实现：
    由于输入已经是 [K//8, N]，逻辑大幅简化，彻底消除了全局 Transpose。
    """
    # 1. Unpack & Offset: 直接沿第 0 维(K_8)解包，结果直接就是 [K, N]
    unpacked_weight = torch.zeros((K, N), device=weight_t.device, dtype=torch.int32)
    for i in range(8):
        # 提取当前 int32 中的第 i 个 nibble，赋值给 K 维度的第 i 行
        unpacked_weight[i::8, :] = (weight_t >> (4 * i)) & 0xF
    unpacked_weight = (unpacked_weight - 8).to(torch.int8)
    
    # 2. Repack: 沿第 1 维(N)打包，直接变成 [K, N // 8]
    repacked_weight = torch.zeros((K, N // 8), device=weight_t.device, dtype=torch.int32)
    for i in range(8):
        val = (unpacked_weight[:, i::8].to(torch.int32)) & 0xF
        repacked_weight |= (val << (4 * i))
        
    return repacked_weight

# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    device = "npu"
        
    N = 4096
    K = 4096
    
    print(f"Testing fused uint4 repack on {device.upper()}")
    print(f"Shape: Input [K//8, N]=[{K//8}, {N}], Output [K, N//8]=[{K}, {N//8}]")
    
    # 模拟提前转置好的输入 [K // 8, N]
    weight_packed_t = torch.randint(-2147483648, 2147483647, (K // 8, N), dtype=torch.int32, device=device)
    
    # 1. 运行 PyTorch 基准
    _ = torch_reference_implementation(weight_packed_t, K, N)

    torch.npu.synchronize()
    t0 = time.time()
    out_ref = torch_reference_implementation(weight_packed_t, K, N)
    torch.npu.synchronize()
    t1 = time.time()
    
    # 2. 运行 Triton 融合算子
    _= repack_int4_npu(weight_packed_t)
    
    torch.npu.synchronize()
    t2 = time.time()
    out_triton = repack_int4_npu(weight_packed_t)
    torch.npu.synchronize()
    t3 = time.time()
    
    # 3. 结果校验
    is_allclose = torch.allclose(out_ref, out_triton)
    print("\n--- 验证结果 ---")
    print(f"Match: {is_allclose}")
    if not is_allclose:
        diff = (out_ref != out_triton).float().mean().item() * 100
        print(f"Difference percentage: {diff:.2f}%")
        
    print("\n--- 耗时对比 ---")
    print(f"PyTorch Time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Triton Time : {(t3 - t2) * 1000:.2f} ms")