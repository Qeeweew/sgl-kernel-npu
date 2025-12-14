import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

@triton.jit
def _awq_shuffle_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    AWQ Weight Shuffle Kernel for NPU
    Performs the bit manipulation logic:
    shifts = [0, 4, 1, 5, 2, 6, 3, 7]
    result = ((val >> shift) & 0xF) << (i * 4)
    result = result ^ 0x88888888
    """
    pid = tl.program_id(0)

    # 1. NPU Grid Strategy: 每个核心计算一部分数据
    # 计算当前核心需要处理的起始位置和结束位置
    # 类似 block_size = (total_rows - 1) // NUM_CORES + 1
    elements_per_core = (n_elements + NUM_CORES - 1) // NUM_CORES
    start_idx = pid * elements_per_core
    
    # 越界检查（针对最后一个核心可能处理不满的情况）
    if start_idx >= n_elements:
        return
        
    end_idx = tl.minimum(start_idx + elements_per_core, n_elements)

    # 2. 循环处理当前核心分配到的数据块
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx # 注意这里是 end_idx，因为是分片处理
        
        # 加载数据
        val = tl.load(x_ptr + offsets, mask=mask, other=0)
        
        # 3. Bitwise Shuffle Logic (Unrolled for performance)
        # 原始逻辑: shifts = [0, 4, 1, 5, 2, 6, 3, 7]
        # output_nibble[i] = input_nibble[shifts[i]]
        
        # 临时变量累加结果
        res = tl.zeros_like(val)

        # i=0: src shift 0 (0*4) -> dst shift 0 (0*4)
        res |= (val >> 0) & 0x0000000F
        
        # i=1: src shift 16 (4*4) -> dst shift 4 (1*4)
        # 注意: 右移后必须 & 0xF 清除高位（特别是处理负数符号位扩展时）
        res |= ((val >> 16) & 0x0000000F) << 4
        
        # i=2: src shift 4 (1*4) -> dst shift 8 (2*4)
        res |= ((val >> 4) & 0x0000000F) << 8
        
        # i=3: src shift 20 (5*4) -> dst shift 12 (3*4)
        res |= ((val >> 20) & 0x0000000F) << 12
        
        # i=4: src shift 8 (2*4) -> dst shift 16 (4*4)
        res |= ((val >> 8) & 0x0000000F) << 16
        
        # i=5: src shift 24 (6*4) -> dst shift 20 (5*4)
        res |= ((val >> 24) & 0x0000000F) << 20
        
        # i=6: src shift 12 (3*4) -> dst shift 24 (6*4)
        res |= ((val >> 12) & 0x0000000F) << 24
        
        # i=7: src shift 28 (7*4) -> dst shift 28 (7*4)
        # PyTorch int32是有符号的，右移28位如果是负数会带符号位，
        # 所以必须 & 0xF 确保只取最低4位
        res |= ((val >> 28) & 0x0000000F) << 28

        # 4. XOR 操作 (用于处理Zero Point翻转等)
        res = res ^ 0x88888888
        
        # 5. 原地写回
        tl.store(x_ptr + offsets, res, mask=mask)

def repack_qweight_inplace_npu(weight_tensor: torch.Tensor):
    """
    对 AWQ int32 权重进行原地 Shuffle。
    适用于 Huawei NPU。
    """
    # 确保张量在 NPU 上且连续
    if not weight_tensor.is_npu:
        raise ValueError("Tensor must be on NPU")
    
    # 展平处理，视作一维 int32 数组
    flat_tensor = weight_tensor.view(-1)
    if not flat_tensor.is_contiguous():
        flat_tensor = flat_tensor.contiguous()
        
    n_elements = flat_tensor.numel()
    
    # 获取 NPU 硬件属性 (Vector Cores 数量)
    # 通常 Ascend 910B 为 30 左右，具体取决于型号
    _, num_vectorcore = get_device_properties()
    
    # Grid 固定为 Vector Core 数量
    grid = (num_vectorcore, )
    
    # BLOCK_SIZE 设为 1024 或 512 以充分利用向量单元
    # 256/512/1024 都是 NPU 友好的对齐大小
    BLOCK_SIZE = 1024 

    _awq_shuffle_kernel[grid](
        flat_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_CORES=num_vectorcore,
    )
    
    return weight_tensor