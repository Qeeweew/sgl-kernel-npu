import torch
import torch_npu
import sys
import math

# 尝试导入自定义算子库
# 假设你的 setup.py 将扩展注册为 sgl_kernel_npu
try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found. Please ensure the kernel is compiled and installed.")
    # 为了防止直接报错退出，这里允许继续，但如果调用算子会失败
    pass

# -------------------------------------------------------------------------
# 1. Reference Implementation (npu_fused_experts - Ascend Official Composition)
# -------------------------------------------------------------------------
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
    w13_offset = kwargs.get("w13_offset", None)
    w2_offset = kwargs.get("w2_offset", None)
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

    # 注意：Official API通常期望 W13 输出维度未打包，这里假设底层能处理或输入已适配
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


# -------------------------------------------------------------------------
# 2. Debug Implementation: Using Grouped GEMV Step-by-Step
# -------------------------------------------------------------------------
def run_moe_via_grouped_gemv(x, w13, w13_scale, w13_offset, w2, w2_scale, w2_offset, topk_ids, topk_weights):
    """
    使用 grouped_gemv_w4a16_moe 算子手动拼装 MoE 计算图。
    验证逻辑：手动 Expand X -> GEMV W13 -> SwiGLU -> GEMV W2 -> Reduce
    """
    # x: [BS, Hidden], topk_ids: [BS, TopK]
    batch_size = x.shape[0]
    hidden_size = x.shape[1]
    top_k = topk_ids.shape[1]
    
    # 1. 准备输入：Expand X to [TotalTokens, Hidden]
    # 相当于 Fused Kernel 中的 Phase 1 Broadcast
    # [BS, 1, Hidden] -> [BS, TopK, Hidden] -> [BS*TopK, Hidden]
    x_expanded = x.unsqueeze(1).expand(batch_size, top_k, hidden_size).reshape(-1, hidden_size).contiguous()
    
    # expert_ids: [TotalTokens]
    expert_ids_flat = topk_ids.flatten().int()

    # 2. Phase 1: W13 (Gate + Up)
    # Output: [TotalTokens, 2*Inter]
    # 调用 standalone gemv
    h13 = torch.ops.npu.grouped_gemv_w4a16_moe(
        x_expanded, w13, w13_scale, w13_offset, expert_ids_flat
    )

    # 3. Activation: SwiGLU
    # Split last dim
    inter_dim_2x = h13.shape[-1]
    inter_dim = inter_dim_2x // 2
    gate, val = torch.split(h13, inter_dim, dim=-1)
    h_act = torch.nn.functional.silu(gate) * val # [TotalTokens, Inter]

    # 4. Phase 2: W2 (Down)
    # Output: [TotalTokens, Hidden]
    # 输入已经是 [TotalTokens, Inter]，无需 expand
    h_out = torch.ops.npu.grouped_gemv_w4a16_moe(
        h_act, w2, w2_scale, w2_offset, expert_ids_flat
    )

    # 5. Weighted Sum (Reduce)
    # topk_weights: [BS, TopK] -> [BS*TopK, 1]
    weights = topk_weights.view(-1, 1).to(h_out.dtype)
    h_weighted = h_out * weights # [TotalTokens, Hidden]
    
    # Reshape to [BS, TopK, Hidden] and Sum over TopK
    h_weighted = h_weighted.view(batch_size, top_k, hidden_size)
    y = h_weighted.sum(dim=1) # [BS, Hidden]
    
    return y


# -------------------------------------------------------------------------
# 3. 辅助函数
# -------------------------------------------------------------------------
def pack_int4_weights(raw_weights, device):
    """
    将 Int32 形式存储的原始 Int4 权重 (-8 ~ 7) 打包为 NPU 所需格式。
    Raw Shape: [E, N, K]
    Packed Shape: [E, N, K/8]
    """
    num_experts, n, k = raw_weights.shape
    # Flatten to [NumExperts * N, K] for packing API
    flat_w = raw_weights.view(-1, k).to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(flat_w)
    return packed.view(num_experts, n, k // 8)

def run_test_bs1():
    print("=" * 60)
    print("Testing Fused MoE BS=1: Custom Fused vs GroupedGEMV vs Reference")
    print("=" * 60)

    # --- 配置参数 (保持不变) ---
    BATCH_SIZE = 4
    TOP_K = 8
    NUM_EXPERTS = 128
    HIDDEN_SIZE = 2048
    INTER_SIZE = 768
    GROUP_SIZE = 128
    
    device = torch.device("npu:0")
    dtype = torch.float16

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

    # 3. 权重 (W13) [E, In, 2*Inter]
    # 注意：raw_w13 最后一维是 2*INTER_SIZE
    raw_w13 = torch.randint(-8, 8, (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTER_SIZE), dtype=torch.int32)
    w13_packed = pack_int4_weights(raw_w13, device)
    
    num_groups_w13 = HIDDEN_SIZE // GROUP_SIZE
    w13_scale = torch.randn((NUM_EXPERTS, num_groups_w13, 2 * INTER_SIZE), dtype=dtype, device=device) * (1.0 / 8.0)
    w13_offset = torch.zeros((NUM_EXPERTS, num_groups_w13, 2 * INTER_SIZE), dtype=dtype, device=device)

    # 4. 权重 (W2) [E, Inter, In]
    raw_w2 = torch.randint(-8, 8, (NUM_EXPERTS, INTER_SIZE, HIDDEN_SIZE), dtype=torch.int32)
    w2_packed = pack_int4_weights(raw_w2, device)

    num_groups_w2 = INTER_SIZE // GROUP_SIZE
    w2_scale = torch.randn((NUM_EXPERTS, num_groups_w2, HIDDEN_SIZE), dtype=dtype, device=device) * (1.0 / 8.0)
    w2_offset = torch.zeros((NUM_EXPERTS, num_groups_w2, HIDDEN_SIZE), dtype=dtype, device=device)

    # ---------------------------------------------------------------------
    # Execution
    # ---------------------------------------------------------------------

    # 1. Reference (Ascend Official)
    print("\n>> [1] Running Reference (npu_fused_experts)...")
    out_ref = npu_fused_experts(
        hidden_states=x, w13=w13_packed, w13_scale=w13_scale, w2=w2_packed, w2_scale=w2_scale,
        topk_weights=topk_weights, topk_ids=topk_ids, top_k=TOP_K,
        w13_offset=w13_offset, w2_offset=w2_offset, use_wna16=True
    )

    # 2. Step-by-Step Grouped GEMV
    print(">> [2] Running GroupedGEMV Implementation (Python Logic)...")
    out_gemv = run_moe_via_grouped_gemv(
        x, w13_packed, w13_scale, w13_offset, 
        w2_packed, w2_scale, w2_offset, 
        topk_ids, topk_weights
    )

    # 3. Custom Fused Kernel
    print(">> [3] Running Custom Fused Kernel (fused_moe_w4a16_small_bs)...")
    # 注意：这里调用的是 Host API 暴露的名称
    out_custom = torch.ops.npu.fused_moe_w4a16_small_bs(
        x, w13_packed, w13_scale, w13_offset, 
        w2_packed, w2_scale, w2_offset, 
        topk_ids, topk_weights
    )
    
    torch.npu.synchronize()

    # ---------------------------------------------------------------------
    # Verification
    # ---------------------------------------------------------------------
    print("\n>> Verifying Results...")
    
    res_ref = out_ref.float().cpu()
    res_gemv = out_gemv.float().cpu()
    res_custom = out_custom.float().cpu()

    # Helper to diff
    def check_diff(name1, val1, name2, val2):
        diff = (val1 - val2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"\nComparing {name1} vs {name2}:")
        print(f"  Max Diff    : {max_diff:.6f}")
        print(f"  Mean Diff   : {mean_diff:.6f}")
        return max_diff

    # A. Custom vs Reference
    d1 = check_diff("Custom Fused", res_custom, "Reference", res_ref)
    
    # B. GroupedGEMV vs Reference
    d2 = check_diff("GroupedGEMV", res_gemv, "Reference", res_ref)
    
    # C. Custom vs GroupedGEMV
    d3 = check_diff("Custom Fused", res_custom, "GroupedGEMV", res_gemv)

    print("\nSample Values (First 5 elements):")
    print(f"  Ref        : {res_ref[0, :5].tolist()}")
    print(f"  GroupedGEMV: {res_gemv[0, :5].tolist()}")
    print(f"  Custom     : {res_custom[0, :5].tolist()}")

    # 结论
    print("\n" + "-"*30)
    print("Analysis:")
    
    # 阈值稍微放宽一点，因为 FP16 累加顺序不同可能导致微小差异
    TOLERANCE = 1e-2 
    
    if d2 < TOLERANCE:
        print("✅ GroupedGEMV kernel logic seems CORRECT (matches Reference).")
    else:
        print("❌ GroupedGEMV kernel logic seems INCORRECT (mismatch Reference).")

    if d3 < TOLERANCE:
        print("✅ Custom Fused kernel matches GroupedGEMV.")
    else:
        print("❌ Custom Fused kernel logic mismatch GroupedGEMV.")

if __name__ == "__main__":
    run_test_bs1()
