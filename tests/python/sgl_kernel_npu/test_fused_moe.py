import torch
import torch_npu
import sys
import math

# 尝试导入自定义算子库
try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Please ensure the kernel is compiled and installed.")
    pass

# -----------------------------------------------------------------------------
# 1. CPU Reference Implementation (FP32 Ground Truth)
# -----------------------------------------------------------------------------
def cpu_moe_reference_fp32(
    hidden_states: torch.Tensor,
    raw_w13: torch.Tensor,   # Int32 representing Int4 [-8, 7]
    w13_scale: torch.Tensor,
    raw_w2: torch.Tensor,    # Int32 representing Int4 [-8, 7]
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    group_size: int,
):
    """
    CPU Reference: Dequantize weights to FP32 and compute MoE logic.
    This serves as the Ground Truth for logic verification.
    """
    # Move all to CPU and FP32 for high precision reference
    device = torch.device("cpu")
    x = hidden_states.to(device).float()
    t_ids = topk_ids.to(device).int()
    t_weights = topk_weights.to(device).float()
    
    # Weights are Int32 representing Int4. Convert to FP32.
    # raw_w13: [E, K, 2*Inter]
    # w13_scale: [E, K//GS, 2*Inter]
    w13_fp32 = raw_w13.to(device).float()
    w2_fp32 = raw_w2.to(device).float()
    s13 = w13_scale.to(device).float()
    s2 = w2_scale.to(device).float()

    num_experts, hidden_dim, inter_dim_2x = w13_fp32.shape
    inter_dim = inter_dim_2x // 2
    out_dim = w2_fp32.shape[-1] # Should be hidden_dim

    # --- Dequantization (Symmetric, Zero Offset) ---
    # Expand scales to match weight shape along the input dimension (dim=1 for w13, dim=1 for w2)
    # w13: Input dim is K (dim 1). Groups are along K.
    # Scale shape: [E, K//GS, Out]
    # Target shape: [E, K, Out]
    s13_expanded = s13.repeat_interleave(group_size, dim=1)
    w13_dequant = w13_fp32 * s13_expanded

    # w2: Input dim is Inter (dim 1). Groups are along Inter.
    # Scale shape: [E, Inter//GS, Out]
    s2_expanded = s2.repeat_interleave(group_size, dim=1)
    w2_dequant = w2_fp32 * s2_expanded

    # --- MoE Computation ---
    batch_size = x.shape[0]
    output = torch.zeros((batch_size, out_dim), dtype=torch.float32, device=device)

    for b in range(batch_size):
        for k in range(top_k):
            expert_id = t_ids[b, k].item()
            weight_coef = t_weights[b, k].item()

            # 1. Gate & Up Proj (W13)
            # x: [Hidden], w13: [Hidden, 2*Inter]
            # Note: raw_w13 shape in test is [E, Hidden, 2*Inter]
            w13_e = w13_dequant[expert_id] 
            h13 = torch.matmul(x[b], w13_e) # [2*Inter]

            # 2. Activation (SwiGLU)
            gate = h13[:inter_dim]
            up = h13[inter_dim:]
            h_act = torch.nn.functional.silu(gate) * up # [Inter]

            # 3. Down Proj (W2)
            # w2: [Inter, Hidden]
            w2_e = w2_dequant[expert_id]
            h_out = torch.matmul(h_act, w2_e) # [Hidden]

            # 4. Accumulate
            output[b] += h_out * weight_coef

    return output


# -----------------------------------------------------------------------------
# 2. Reference Implementation (npu_fused_experts - Ascend Official Composition)
# -----------------------------------------------------------------------------
def npu_fused_experts(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    **kwargs,
):
    # Symmetric quantization: offsets are all zeros
    use_wna16 = kwargs.get("use_wna16", False)

    original_shape = hidden_states.shape
    original_dtype = hidden_states.dtype
    scale_dtype = original_dtype if original_dtype == torch.bfloat16 else torch.float32

    # Flatten Batch
    if len(original_shape) == 3:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

    num_tokens = hidden_states.shape[0]
    num_experts = w13.shape[0]

    # Create Row Indices for Routing
    row_idx_len = num_tokens * top_k
    row_idx = (
        torch.arange(0, row_idx_len, dtype=torch.int32, device=topk_weights.device)
        .view(top_k, -1)
        .permute(1, 0)
        .contiguous()
    )

    # Init Routing
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
        )
    )

    # Compute Expert Tokens
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, num_experts
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # Symmetric quantization: offsets are zeros (same shape as scales)
    w13_offset = torch.zeros_like(w13_scale)
    w2_offset = torch.zeros_like(w2_scale)

    # --- gmm1: gate_up_proj (W13) ---
    if not use_wna16:
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        scale_args13 = {
            "scale": [w13_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args13 = {
            "antiquant_scale": [w13_scale],
            "antiquant_offset": [w13_offset],
        }

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        **scale_args13,
        split_item=2, # INT4
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # --- act_fn: swiglu ---
    hidden_states = torch_npu.npu_swiglu(hidden_states)

    # --- gmm2: down_proj (W2) ---
    if not use_wna16:
        hidden_states, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        scale_args2 = {
            "scale": [w2_scale.to(scale_dtype)],
            "per_token_scale": [pertoken_scale],
        }
    else:
        scale_args2 = {"antiquant_scale": [w2_scale], "antiquant_offset": [w2_offset]}

    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        **scale_args2,
        split_item=2, # INT4
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # Finalize Routing (Weighted Sum)
    final_hidden_states = torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights.to(dtype=hidden_states.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )

    if len(original_shape) == 3:
        final_hidden_states = final_hidden_states.view(original_shape)
    return final_hidden_states


# -----------------------------------------------------------------------------
# 3. Debug Implementation: Using Grouped GEMV Step-by-Step
# -----------------------------------------------------------------------------
def run_moe_via_grouped_gemv(x, w13, w13_scale, w2, w2_scale, topk_ids, topk_weights):
    """
    使用 grouped_gemv_w4a16_moe 算子手动拼装 MoE 计算图。
    """
    batch_size = x.shape[0]
    hidden_size = x.shape[1]
    top_k = topk_ids.shape[1]

    # 1. Expand X
    x_expanded = x.unsqueeze(1).expand(batch_size, top_k, hidden_size).reshape(-1, hidden_size).contiguous()
    expert_ids_flat = topk_ids.flatten().int()

    # 2. Phase 1: W13
    h13 = torch.ops.npu.grouped_gemv_w4a16_moe(
        x_expanded, w13, w13_scale, expert_ids_flat
    )

    # 3. Activation: SwiGLU
    inter_dim_2x = h13.shape[-1]
    inter_dim = inter_dim_2x // 2
    gate, val = torch.split(h13, inter_dim, dim=-1)
    h_act = torch.nn.functional.silu(gate) * val

    # 4. Phase 2: W2
    h_out = torch.ops.npu.grouped_gemv_w4a16_moe(
        h_act, w2, w2_scale, expert_ids_flat
    )

    # 5. Weighted Sum (Reduce)
    weights = topk_weights.view(-1, 1).to(h_out.dtype)
    h_weighted = h_out * weights
    h_weighted = h_weighted.view(batch_size, top_k, hidden_size)
    y = h_weighted.sum(dim=1)

    return y


# -----------------------------------------------------------------------------
# 4. Helper Functions
# -----------------------------------------------------------------------------
def pack_int4_weights(raw_weights, device):
    """
    将 Int32 形式存储的原始 Int4 权重 (-8 ~ 7) 打包为 NPU 所需格式。
    """
    num_experts, n, k = raw_weights.shape
    flat_w = raw_weights.view(-1, k).to(device)
    # Ensure input to pack function is on NPU if torch_npu expects it
    packed = torch_npu.npu_convert_weight_to_int4pack(flat_w)
    return packed.view(num_experts, n, k // 8)

def run_test_bs1():
    print("=" * 60)
    print("Testing Fused MoE BS=1: NPU Implementations vs CPU Reference (FP32)")
    print("Symmetric Quantization, GROUP_SIZE=32")
    print("=" * 60)

    # --- 配置参数 ---
    BATCH_SIZE = 1
    TOP_K = 8
    NUM_EXPERTS = 128
    HIDDEN_SIZE = 2048
    INTER_SIZE = 768
    GROUP_SIZE = 32

    device = torch.device("npu:0")
    dtype = torch.bfloat16

    print(f"Config: BS={BATCH_SIZE}, TopK={TOP_K}, Experts={NUM_EXPERTS}")
    print(f"        Hidden={HIDDEN_SIZE}, Inter={INTER_SIZE}, GroupSize={GROUP_SIZE}")

    # --- 数据生成 ---
    torch.manual_seed(42)

    # 1. 输入 X [BS, Hidden]
    x = torch.randn((BATCH_SIZE, HIDDEN_SIZE), dtype=dtype, device=device) * (1.0 / math.sqrt(HIDDEN_SIZE))

    # 2. 路由信息 [BS, TopK]
    topk_ids = torch.randint(0, NUM_EXPERTS, (BATCH_SIZE, TOP_K), dtype=torch.int32, device=device)
    topk_weights = torch.randn((BATCH_SIZE, TOP_K), dtype=torch.float32, device=device)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    # 3. 权重 (W13) [E, In, 2*Inter] - Keep Raw for CPU Ref
    raw_w13 = torch.randint(-8, 8, (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTER_SIZE), dtype=torch.int32)
    # Pack for NPU
    w13_packed = pack_int4_weights(raw_w13, device)

    num_groups_w13 = HIDDEN_SIZE // GROUP_SIZE
    w13_scale = torch.randn((NUM_EXPERTS, num_groups_w13, 2 * INTER_SIZE), dtype=dtype, device=device) * (1.0 / 8.0)

    # 4. 权重 (W2) [E, Inter, In] - Keep Raw for CPU Ref
    raw_w2 = torch.randint(-8, 8, (NUM_EXPERTS, INTER_SIZE, HIDDEN_SIZE), dtype=torch.int32)
    # Pack for NPU
    w2_packed = pack_int4_weights(raw_w2, device)

    num_groups_w2 = INTER_SIZE // GROUP_SIZE
    w2_scale = torch.randn((NUM_EXPERTS, num_groups_w2, HIDDEN_SIZE), dtype=dtype, device=device) * (1.0 / 8.0)

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    # 0. CPU Reference (FP32 Ground Truth)
    print("\n>> [0] Running CPU Reference (FP32 Dequant)...")
    # Ensure inputs for CPU ref are on CPU or moved inside function
    out_cpu = cpu_moe_reference_fp32(
        hidden_states=x, 
        raw_w13=raw_w13, w13_scale=w13_scale,
        raw_w2=raw_w2, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids, top_k=TOP_K,
        group_size=GROUP_SIZE
    )

    # 1. Official NPU Fused
    print(">> [1] Running Reference (npu_fused_experts - NPU)...")
    out_ref_npu = npu_fused_experts(
        hidden_states=x, w13=w13_packed, w13_scale=w13_scale,
        w2=w2_packed, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids, top_k=TOP_K,
        use_wna16=True
    )

    # 2. Step-by-Step Grouped GEMV
    print(">> [2] Running GroupedGEMV Implementation (NPU)...")
    out_gemv = run_moe_via_grouped_gemv(
        x, w13_packed, w13_scale,
        w2_packed, w2_scale,
        topk_ids, topk_weights
    )

    # 3. Custom Fused Kernel
    print(">> [3] Running Custom Fused Kernel (NPU)...")
    out_custom = torch.ops.npu.fused_moe_w4a16_small_bs(
        x, w13_packed, w13_scale,
        w2_packed, w2_scale,
        topk_ids, topk_weights
    )

    torch.npu.synchronize()

    # ---------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------
    print("\n>> Verifying Results against CPU Reference (FP32)...")

    # Move all to CPU Float for comparison
    res_cpu = out_cpu.float().cpu()
    res_ref_npu = out_ref_npu.float().cpu()
    res_gemv = out_gemv.float().cpu()
    res_custom = out_custom.float().cpu()

    def check_diff(name1, val1, name2, val2, tolerance):
        diff = (val1 - val2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # Relative error check for better scaling
        rel_diff = diff / (val2.abs() + 1e-6)
        max_rel = rel_diff.max().item()
        
        print(f"\nComparing {name1} vs {name2}:")
        print(f"  Max Abs Diff: {max_diff:.6f}")
        print(f"  Mean Abs Diff:{mean_diff:.6f}")
        print(f"  Max Rel Diff: {max_rel:.6f}")
        
        if max_diff < tolerance:
            print(f"  ✅ PASS (Tolerance < {tolerance})")
            return True
        else:
            print(f"  ❌ FAIL (Tolerance < {tolerance})")
            return False

    # Tolerance Settings
    # NPU (W4A16) vs CPU (FP32 Dequant): Expect quantization noise. 
    # Since input is normalized, 0.1 is a reasonable absolute threshold for W4 quantization error.
    TOLERANCE_QUANT = 0.1 
    # NPU vs NPU: Should be very close (same precision)
    print("\n" + "-"*30)
    print("Analysis:")

    # A. Official NPU vs CPU
    check_diff("Official NPU", res_ref_npu, "CPU Ref (FP32)", res_cpu, TOLERANCE_QUANT)

    # B. GroupedGEMV vs CPU
    check_diff("GroupedGEMV", res_gemv, "CPU Ref (FP32)", res_cpu, TOLERANCE_QUANT)

    # C. Custom Fused vs CPU
    check_diff("Custom Fused", res_custom, "CPU Ref (FP32)", res_cpu, TOLERANCE_QUANT)

    print("\nSample Values (First 5 elements):")
    print(f"  CPU Ref    : {res_cpu[0, :5].tolist()}")
    print(f"  Official   : {res_ref_npu[0, :5].tolist()}")
    print(f"  GroupedGEMV: {res_gemv[0, :5].tolist()}")
    print(f"  Custom     : {res_custom[0, :5].tolist()}")

if __name__ == "__main__":
    run_test_bs1()
