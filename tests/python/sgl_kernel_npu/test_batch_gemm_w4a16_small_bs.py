"""
Test for batch_gemm_w4a16_small_bs vs npu_weight_quant_batchmatmul

Tests accuracy against FP32 CPU reference implementation.
Reference: sgl-kernel-npu/tests/python/sgl_kernel_npu/test_fused_moe.py
"""

import torch
import torch_npu
import sys
import math

try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found.")
    sys.exit(1)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
device = torch.device("npu:0")
dtype = torch.bfloat16

# Test dimensions
BATCH_SIZES = [1]
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 4096
GROUP_SIZE = 32

# Tolerance for bf16
TOLERANCE_QUANT = 0.1  # W4A16 quantization error tolerance


# ------------------------------------------------------------
# CPU Reference Implementation (FP32 Ground Truth)
# ------------------------------------------------------------
def cpu_batch_gemm_reference_fp32(
    x: torch.Tensor,
    raw_weight: torch.Tensor,  # Int32 representing Int4 [-8, 7], shape [Out, In]
    weight_scale: torch.Tensor,  # [NumGroups, Out]
    group_size: int,
):
    """
    CPU Reference: Dequantize weights to FP32 and compute batch GEMM.
    """
    # Move all to CPU and FP32
    x_cpu = x.to("cpu").float()  # [BS, In]
    w_int32 = raw_weight.to("cpu")  # [Out, In]
    scale = weight_scale.to("cpu").float()  # [NumGroups, Out]

    out_dim, in_dim = w_int32.shape
    num_groups = in_dim // group_size

    # Dequantize weight
    # scale: [NumGroups, Out] -> transpose -> [Out, NumGroups] -> expand -> [Out, In]
    w_fp32 = w_int32.float()  # [Out, In]
    scale_T = scale.transpose(0, 1)  # [Out, NumGroups]
    scale_expanded = scale_T.repeat_interleave(group_size, dim=1)  # [Out, In]
    w_dequant = w_fp32 * scale_expanded  # [Out, In]

    # Matmul: [BS, In] @ [In, Out] = [BS, Out]
    output = torch.matmul(x_cpu, w_dequant.transpose(0, 1))
    return output  # [BS, Out]


# ------------------------------------------------------------
# Utils: pack int4 weights
# ------------------------------------------------------------
def pack_int4_weights(raw_weights, device):
    """
    将 Int32 形式存储的原始 Int4 权重 (-8 ~ 7) 打包为 NPU 所需格式。
    raw_weights: [M, K]
    return: [M, K // 8] packed int32
    """
    m, k = raw_weights.shape
    flat_w = raw_weights.view(-1, k).to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(flat_w)
    return packed.view(m, k // 8)


# ------------------------------------------------------------
# Create quantized weights for testing
# ------------------------------------------------------------
def create_quantized_weights(
    output_size: int,
    input_size: int,
    group_size: int = 32,
):
    """
    Create quantized int4 weights for testing.

    Returns:
        Tuple of (raw_weight_int32, weight_scale_bf16)
        - raw_weight_int32: [output_size, input_size] Int4 stored as Int32
        - weight_scale_bf16: [num_groups, output_size] BF16
    """
    num_groups = input_size // group_size

    torch.manual_seed(42)
    # Generate random int4 weights [output_size, input_size] as int32
    raw_weight = torch.randint(-8, 8, (output_size, input_size), dtype=torch.int32, device="cpu")

    # Compute per-group scales [num_groups, output_size]
    weight_scale = torch.zeros((num_groups, output_size), dtype=torch.float32, device="cpu")
    for g in range(num_groups):
        start_idx = g * group_size
        end_idx = (g + 1) * group_size
        group_weight = raw_weight[:, start_idx:end_idx].to(torch.float32)
        max_val = group_weight.abs().max(dim=1)[0]
        scale = max_val / 7.0
        scale = torch.clamp(scale, min=1e-6)
        weight_scale[g, :] = scale

    weight_scale_bf16 = weight_scale.to(device).to(dtype)

    return raw_weight, weight_scale_bf16


# ------------------------------------------------------------
# Test function
# ------------------------------------------------------------
def test_both_kernels(batch_size, input_size, output_size, group_size=32):
    """Test both batch_gemm_w4a16_small_bs and npu_weight_quant_batchmatmul with same weights."""

    # Create weights once
    raw_weight, weight_scale = create_quantized_weights(output_size, input_size, group_size)

    # Input tensor [batch_size, input_size]
    torch.manual_seed(batch_size)
    x = torch.randn((batch_size, input_size), device=device, dtype=dtype) * (1.0 / math.sqrt(input_size))

    # Reference: FP32 CPU matmul with dequantization
    y_ref = cpu_batch_gemm_reference_fp32(x, raw_weight, weight_scale, group_size)

    # Prepare weights for both kernels
    # For batch_gemm_w4a16_small_bs: [Out, In] -> transpose -> [In, Out] -> pack -> [In, Out//8]
    weight_packed_custom = pack_int4_weights(raw_weight.transpose(0, 1).contiguous(), device)

    # For npu_weight_quant_batchmatmul: [Out, In] -> transpose -> [In, Out] -> pack -> [In, Out//8]
    weight_packed_npu = pack_int4_weights(raw_weight.transpose(0, 1).contiguous(), device)

    weight_offset = torch.zeros_like(weight_scale)

    # Run custom kernel
    y_custom = torch.ops.npu.batch_gemm_w4a16_small_bs(x, weight_packed_custom, weight_scale)

    # Run NPU native kernel
    y_npu = torch.ops.npu.npu_weight_quant_batchmatmul(
        x=x,
        weight=weight_packed_npu,
        antiquant_scale=weight_scale,
        antiquant_offset=weight_offset,
        antiquant_group_size=group_size,
        bias=None,
    )

    # Move to CPU for comparison
    y_ref_fp32 = y_ref  # Already FP32 on CPU
    y_custom_fp32 = y_custom.to("cpu").float()
    y_npu_fp32 = y_npu.to("cpu").float()

    # Compute diffs
    max_diff_custom = (y_ref_fp32 - y_custom_fp32).abs().max().item()
    mean_diff_custom = (y_ref_fp32 - y_custom_fp32).abs().mean().item()

    max_diff_npu = (y_ref_fp32 - y_npu_fp32).abs().max().item()
    mean_diff_npu = (y_ref_fp32 - y_npu_fp32).abs().mean().item()

    is_close_custom = max_diff_custom < TOLERANCE_QUANT
    is_close_npu = max_diff_npu < TOLERANCE_QUANT

    return {
        "batch_size": batch_size,
        "y_ref": y_ref_fp32,
        "y_custom": y_custom_fp32,
        "y_npu": y_npu_fp32,
        "max_diff_custom": max_diff_custom,
        "mean_diff_custom": mean_diff_custom,
        "max_diff_npu": max_diff_npu,
        "mean_diff_npu": mean_diff_npu,
        "is_close_custom": is_close_custom,
        "is_close_npu": is_close_npu,
    }


# ------------------------------------------------------------
# Main test runner
# ------------------------------------------------------------
def main():
    print("=" * 80)
    print("Test: batch_gemm_w4a16_small_bs vs npu_weight_quant_batchmatmul")
    print("Reference: FP32 CPU implementation with dequantization")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Hidden Size: {HIDDEN_SIZE}")
    print(f"  Intermediate Size: {INTERMEDIATE_SIZE}")
    print(f"  Group Size: {GROUP_SIZE}")
    print(f"  Dtype: {dtype}")
    print(f"  Tolerance: {TOLERANCE_QUANT}")
    print("=" * 80)

    all_passed = True

    input_size = HIDDEN_SIZE
    output_size = INTERMEDIATE_SIZE

    print(f"\nTest: [{input_size}] -> [{output_size}]")
    print("-" * 80)

    for batch_size in BATCH_SIZES:
        print(f"\n  Batch Size: {batch_size}")

        result = test_both_kernels(batch_size, input_size, output_size, GROUP_SIZE)

        status_custom = "PASS" if result["is_close_custom"] else "FAIL"
        status_npu = "PASS" if result["is_close_npu"] else "FAIL"

        if not result["is_close_custom"] or not result["is_close_npu"]:
            all_passed = False

        print(f"    batch_gemm_w4a16_small_bs:    {status_custom} | "
              f"max_diff={result['max_diff_custom']:.4e}, mean_diff={result['mean_diff_custom']:.4e}")
        print(f"    npu_weight_quant_batchmatmul: {status_npu} | "
              f"max_diff={result['max_diff_npu']:.4e}, mean_diff={result['mean_diff_npu']:.4e}")

        # Print first 10 elements for first sample in batch
        print(f"\n    First 10 elements (sample batch {batch_size - 1}):")
        print(f"      CPU Ref:    {result['y_ref'][-1, :10].tolist()}")
        print(f"      Custom:     {result['y_custom'][-1, :10].tolist()}")
        print(f"      NPU Native: {result['y_npu'][-1, :10].tolist()}")

    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
