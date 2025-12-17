import math
import torch
import torch_npu
import torch.utils.benchmark as benchmark
import sys

try:
    import sgl_kernel_npu
except ImportError:
    print("Warning: sgl_kernel_npu not found.")
    sys.exit(1)

# ------------------------------------------------------------
# Fixed constants (DO NOT CHANGE)
# ------------------------------------------------------------
TOP_K = 8
NUM_EXPERTS = 128
HIDDEN_SIZE = 2048
INTER_SIZE = 768
GROUP_SIZE = 128
ROUTING_BLOCKS = NUM_EXPERTS // TOP_K  # 16

device = torch.device("npu:0")
dtype = torch.float16


# ------------------------------------------------------------
# Reference implementation (UNCHANGED semantics)
# ------------------------------------------------------------
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
    w13_offset = kwargs["w13_offset"]
    w2_offset = kwargs["w2_offset"]
    row_idx = kwargs["row_idx"]

    original_dtype = hidden_states.dtype
    scale_dtype = torch.float32

    # ---- init routing (row_idx MUST be 2D) ----
    hidden_states, expanded_row_idx, expanded_expert_idx = (
        torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=topk_ids,
            active_num=hidden_states.shape[0],
        )
    )

    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(
        expanded_expert_idx, w13.shape[0]
    ).to(torch.int64)

    # ---- W13 ----
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w13],
        antiquant_scale=[w13_scale],
        antiquant_offset=[w13_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    hidden_states = torch_npu.npu_swiglu(hidden_states)

    # ---- W2 ----
    hidden_states = torch_npu.npu_grouped_matmul(
        x=[hidden_states],
        weight=[w2],
        antiquant_scale=[w2_scale],
        antiquant_offset=[w2_offset],
        split_item=2,
        group_list_type=0,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=original_dtype,
    )[0]

    # ---- finalize ----
    return torch_npu.npu_moe_finalize_routing(
        hidden_states,
        skip1=None,
        skip2=None,
        bias=None,
        scales=topk_weights.to(hidden_states.dtype),
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
    )


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def pack_int4_weights(raw, device):
    flat = raw.view(-1, raw.shape[-1]).to(device)
    packed = torch_npu.npu_convert_weight_to_int4pack(flat)
    return packed.view(*raw.shape[:-1], raw.shape[-1] // 8)


def build_row_idx(B):
    row_idx_len = B * TOP_K
    return (
        torch.arange(row_idx_len, dtype=torch.int32, device=device)
        .view(TOP_K, -1)
        .permute(1, 0)
        .contiguous()
    )


def build_routing_blocks():
    experts = torch.arange(NUM_EXPERTS, device=device, dtype=torch.int32)
    return experts.view(ROUTING_BLOCKS, TOP_K)


def build_weight_blocks():
    w = torch.randn((ROUTING_BLOCKS, TOP_K), device=device)
    return torch.softmax(w, dim=-1)


# ------------------------------------------------------------
# Capture graph: 16 forwards inside ONE graph
# ------------------------------------------------------------
def capture_16_forward_graph(fn, x, row_idx, topk_ids_all, topk_w_all):
    # warmup
    for i in range(ROUTING_BLOCKS):
        fn(x, row_idx, topk_ids_all[i], topk_w_all[i])
    torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        for i in range(ROUTING_BLOCKS):
            fn(x, row_idx, topk_ids_all[i], topk_w_all[i])
    torch.npu.synchronize()
    return g


# ------------------------------------------------------------
# Bytes accounting (weight traffic)
# ------------------------------------------------------------
def bytes_per_expert(w13, w13_scale, w13_offset, w2, w2_scale, w2_offset):
    return (
        w13[0].numel() * w13.element_size()
        + w13_scale[0].numel() * w13_scale.element_size()
        + w13_offset[0].numel() * w13_offset.element_size()
        + w2[0].numel() * w2.element_size()
        + w2_scale[0].numel() * w2_scale.element_size()
        + w2_offset[0].numel() * w2_offset.element_size()
    )


# ------------------------------------------------------------
# Benchmark per batch size
# ------------------------------------------------------------
def benchmark_bs(B, w13, w13_scale, w13_offset, w2, w2_scale, w2_offset):
    x = torch.randn((B, HIDDEN_SIZE), device=device, dtype=dtype)
    row_idx = build_row_idx(B)

    blocks = build_routing_blocks()
    weights = build_weight_blocks()

    # build 16 routing sets
    topk_ids_all, topk_w_all = [], []
    for i in range(ROUTING_BLOCKS):
        ids = torch.empty((B, TOP_K), device=device, dtype=torch.int32)
        wts = torch.empty((B, TOP_K), device=device)
        for b in range(B):
            blk = (i + b) % ROUTING_BLOCKS
            ids[b].copy_(blocks[blk])
            wts[b].copy_(weights[blk])
        topk_ids_all.append(ids)
        topk_w_all.append(wts)

    # reference graph
    g_ref = capture_16_forward_graph(
        lambda x, row_idx, ids, w: npu_fused_experts(
            hidden_states=x,
            w13=w13, w13_scale=w13_scale,
            w2=w2, w2_scale=w2_scale,
            topk_weights=w,
            topk_ids=ids,
            top_k=TOP_K,
            w13_offset=w13_offset,
            w2_offset=w2_offset,
            row_idx=row_idx,
        ),
        x, row_idx, topk_ids_all, topk_w_all,
    )

    # custom fused graph
    g_custom = capture_16_forward_graph(
        lambda x, row_idx, ids, w: torch.ops.npu.fused_moe_w4a16_small_bs(
            x,
            w13, w13_scale, w13_offset,
            w2, w2_scale, w2_offset,
            ids, w,
        ),
        x, row_idx, topk_ids_all, topk_w_all,
    )

    def run_ref():
        g_ref.replay()
        torch.npu.synchronize()

    def run_custom():
        g_custom.replay()
        torch.npu.synchronize()

    t_ref = benchmark.Timer(
        stmt="run_ref()",
        globals={"run_ref": run_ref}
    ).blocked_autorange(min_run_time=2.0)

    t_custom = benchmark.Timer(
        stmt="run_custom()",
        globals={"run_custom": run_custom}
    ).blocked_autorange(min_run_time=2.0)

    bytes_total = (
        B * TOP_K * ROUTING_BLOCKS *
        bytes_per_expert(w13, w13_scale, w13_offset, w2, w2_scale, w2_offset)
    )

    return {
        "B": B,
        "ref_us": t_ref.mean * 1e6 / ROUTING_BLOCKS,
        "custom_us": t_custom.mean * 1e6 / ROUTING_BLOCKS,
        "ref_GBs": bytes_total / t_ref.mean / 1e9,
        "custom_GBs": bytes_total / t_custom.mean / 1e9,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    torch.manual_seed(0)

    raw_w13 = torch.randint(
        -8, 8, (NUM_EXPERTS, HIDDEN_SIZE, 2 * INTER_SIZE), dtype=torch.int32
    )
    raw_w2 = torch.randint(
        -8, 8, (NUM_EXPERTS, INTER_SIZE, HIDDEN_SIZE), dtype=torch.int32
    )

    w13 = pack_int4_weights(raw_w13, device)
    w2 = pack_int4_weights(raw_w2, device)

    w13_scale = torch.randn(
        (NUM_EXPERTS, HIDDEN_SIZE // GROUP_SIZE, 2 * INTER_SIZE),
        device=device, dtype=dtype
    )
    w13_offset = torch.zeros_like(w13_scale)

    w2_scale = torch.randn(
        (NUM_EXPERTS, INTER_SIZE // GROUP_SIZE, HIDDEN_SIZE),
        device=device, dtype=dtype
    )
    w2_offset = torch.zeros_like(w2_scale)

    print("BS | Ref(us) | Custom(us) | Ref GB/s | Custom GB/s")
    print("-" * 56)

    for B in [1, 2, 3, 4]:
        r = benchmark_bs(B, w13, w13_scale, w13_offset, w2, w2_scale, w2_offset)
        print(
            f"{B:>2} | {r['ref_us']:>7.2f} | {r['custom_us']:>10.2f} | "
            f"{r['ref_GBs']:>8.2f} | {r['custom_GBs']:>10.2f}"
        )


if __name__ == "__main__":
    main()
