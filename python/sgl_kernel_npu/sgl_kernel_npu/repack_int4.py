import torch
import triton
import triton.language as tl

from sgl_kernel_npu.utils.triton_utils import get_device_properties

@triton.jit
def _repack_int4_npu_kernel(
    src_ptr,             # Input Pointer
    dst_ptr,             # Output Pointer
    stride_src_row,      # N (Input Last Dim)
    stride_dst_row,      # N // 8 (Output Last Dim)
    total_rows,          # Total Output Rows (Input_Rows * 8)
    num_cols_packed,     # N // 8
    num_cores,           # NPU Cores
    BLOCK_N: tl.constexpr, 
):
    # --- 1. 任务划分 (Persistent Kernel) ---
    # 计算每个 Core 负责的输出行数
    block_size_rows = (total_rows + num_cores - 1) // num_cores
    pid = tl.program_id(0)
    
    row_begin = pid * block_size_rows
    row_end = tl.minimum((pid + 1) * block_size_rows, total_rows)

    if row_begin >= total_rows:
        return

    # --- 2. 行循环 ---
    for row_idx in range(row_begin, row_end):
        
        # 映射逻辑:
        # Output Row 'r' 数据来源于 Input Row 'r // 8'
        # 内核把所有维度扁平化视为 [Total_Rows, N]
        src_row_idx = row_idx // 8
        
        # 计算移位量: Input int32 包含 8 个解包前的值
        # 我们需要第 (row_idx % 8) 个 4-bit 数据
        k_shift = (row_idx % 8) * 4

        # --- 3. 列块循环 ---
        for col_idx in range(0, num_cols_packed, BLOCK_N):
            
            # 生成列偏移 [BLOCK_N]
            cols_offs = col_idx + tl.arange(0, BLOCK_N)
            mask_n = cols_offs < num_cols_packed
            
            # 生成 2D 读取偏移 [BLOCK_N, 8]
            # row dim (BLOCK_N): 对应输出的连续列
            # col dim (8): 对应打包所需的连续输入值 (N维度)
            offs_2d_n = cols_offs[:, None] * 8 + tl.arange(0, 8)[None, :]
            
            # 扁平化寻址
            src_ptr_base = src_ptr + src_row_idx * stride_src_row
            
            # Load Data [BLOCK_N, 8]
            val_block = tl.load(src_ptr_base + offs_2d_n, mask=mask_n[:, None], other=0)

            # --- 4. 计算与打包 (No Slicing Approach) ---
            
            # Step A: 解包 (Shift)
            nibbles = (val_block >> k_shift) & 0xF
            
            # Step B: 变换 (x ^ 0x8) 等价于 ((x & 0xF) - 8)
            nibbles = nibbles ^ 0x8
            
            # Step C: 打包 (Broadcast Shift + Reduce Sum)
            # 1. 构造移位向量 [0, 4, ..., 28]
            shift_vals = tl.arange(0, 8) * 4
            
            # 2. 广播移位: [BLOCK_N, 8]
            nibbles_shifted = nibbles << shift_vals[None, :]
            
            # 3. 规约求和 (axis=1) -> [BLOCK_N]
            packed_acc = tl.sum(nibbles_shifted, axis=1)

            # --- 5. 存储 ---
            dst_offset = row_idx * stride_dst_row + cols_offs
            tl.store(dst_ptr + dst_offset, packed_acc, mask=mask_n)


def repack_int4_tensor_npu(packed_weight: torch.Tensor):
    """
    Triton NPU 实现：重排权重以适配 Ascend int4pack 格式。
    支持 2D [K, N] 或 3D [E, K, N] 输入。
    
    逻辑:
    1. 输入 K 维度 (倒数第2维) 实际上包含了 8 个压缩的 int4 值。
    2. 需要将其解包，行数 x8。
    3. 输入 N 维度 (倒数第1维) 需要按 int4pack 格式每 8 个压缩为一个 int32。
    4. 列数 /8。
    """
    # 1. 基础校验
    assert packed_weight.ndim in [2, 3], f"Expected 2D or 3D tensor, got {packed_weight.ndim}"
    if not packed_weight.is_contiguous():
        packed_weight = packed_weight.contiguous()
        
    shape = packed_weight.shape
    N = shape[-1] # 最后一维是需要打包的维度
    
    assert N % 8 == 0, f"Last dimension {N} must be divisible by 8 for packing"

    # 2. 维度计算
    # 无论 2D 还是 3D，除最后一维 N 外的所有维度乘积即为 "Input Logical Rows"
    input_logical_rows = 1
    for d in shape[:-1]:
        input_logical_rows *= d
    
    # 3. 构建输出形状
    # Output Rows = Input Rows * 8 (因为解包了 K 维度)
    total_output_rows = input_logical_rows * 8
    N_packed = N // 8
    
    # 保持输出 Tensor 的秩与输入一致
    out_shape = list(shape)
    out_shape[-2] *= 8     # K 维度扩大 8 倍
    out_shape[-1] = N_packed # N 维度缩小 8 倍
    
    new_weight = torch.empty(out_shape, dtype=torch.int32, device=packed_weight.device)
    
    # 4. 获取硬件参数
    num_vectorcore = 30 # Default for 910B
    if get_device_properties is not None:
        try:
            _, num_vectorcore = get_device_properties()
        except:
            pass

    # Grid size = 核心数 (Persistent Kernel)
    grid = (num_vectorcore, )
    
    # 5. 启动 Kernel
    # stride_src_row 始终是 N (最后一维的大小，因为是 row-major)
    # stride_dst_row 始终是 N_packed
    _repack_int4_npu_kernel[grid](
        packed_weight, 
        new_weight,
        stride_src_row=N,
        stride_dst_row=N_packed,
        total_rows=total_output_rows,
        num_cols_packed=N_packed,
        num_cores=num_vectorcore,
        BLOCK_N=128
    )
    
    return new_weight