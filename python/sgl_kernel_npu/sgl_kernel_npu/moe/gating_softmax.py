import torch
import triton
import triton.language as tl
from sgl_kernel_npu.utils.triton_utils import get_device_properties

@triton.jit
def _moe_topk_softmax_aiv_kernel(
    logits_ptr,   # [R, E]
    out_w_ptr,    # [R, K] fp32
    out_id_ptr,   # [R, K] int32
    R,
    E: tl.constexpr,
    K: tl.constexpr,          # assume K <= 16
    NUM_CORES: tl.constexpr,
    RENORM: tl.constexpr,     # 0/1
    VEC: tl.constexpr = 32 # force 16-aligned vector size for NPU
):

    block_size = (R - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    row_begin = pid * block_size
    if row_begin >= R:
        return
    row_end = tl.minimum((pid + 1) * block_size, R)

    e_offs = tl.arange(0, E)

    for r in range(row_begin, row_end):
        row_start = r * E
        x = tl.load(logits_ptr + row_start + e_offs).to(tl.float32)

        # softmax
        x = x - tl.max(x, axis=0)
        p = tl.exp(x)
        p = p / tl.sum(p, axis=0)

        # topk：K 次 argmax + mask-out
        topv = tl.full([VEC], -float("inf"), tl.float32)
        topi = tl.full([VEC], 0, tl.int32)

        work = p
        kv = tl.arange(0, VEC)

        for t in tl.static_range(0, K):
            vmax = tl.max(work, axis=0)
            imax = tl.argmax(work, axis=0)

            topv = tl.where(kv == t, vmax, topv)
            topi = tl.where(kv == t, imax.to(tl.int32), topi)

            work = tl.where(e_offs == imax, -float("inf"), work)

        # renormalize (only over first K)
        if RENORM:
            valid = kv < K
            s = tl.sum(tl.where(valid, topv, 0.0), axis=0)
            topv = tl.where(valid, topv / s, topv)

        # store only first K (masked)
        k_offs = kv
        mask = k_offs < K
        out_base = r * K + k_offs  # ok; masked prevents OOB store

        tl.store(out_w_ptr + out_base, topv.to(tl.float32), mask=mask)
        tl.store(out_id_ptr + out_base, topi.to(tl.int32), mask=mask)

def triton_moe_gating_topk_softmax(router_logits: torch.Tensor, k: int, renormalize: bool):
    assert router_logits.ndim == 2
    R, E = router_logits.shape
    assert k <= 16, "该实现适合小 K（2/4/8）；K 很大需要换算法"

    out_w = torch.empty((R, k), device=router_logits.device, dtype=torch.float32)
    out_id = torch.empty((R, k), device=router_logits.device, dtype=torch.int32)

    _, num_vectorcore = get_device_properties()
    grid = (min(R, num_vectorcore),)
    _moe_topk_softmax_aiv_kernel[grid](
        router_logits,
        out_w,
        out_id,
        R=R,
        E=E,
        K=k,
        NUM_CORES=num_vectorcore,
        RENORM=1 if renormalize else 0,
        multibuffer=True,
    )
    return out_w, out_id, None


# --------------------------
# Test code (fixed E=128, K=8)
# --------------------------
def reference_topk_softmax(logits: torch.Tensor, k: int, renormalize: bool):
    # logits: [R, E]
    p = torch.softmax(logits.float(), dim=-1)               # [R, E]
    topv, topi = torch.topk(p, k, dim=-1, largest=True, sorted=True)  # [R, K]
    if renormalize:
        topv = topv / topv.sum(dim=-1, keepdim=True)
    return topv.float(), topi.int()

def run_test(device=None, R=1024, E=128, K=8, renormalize=True, dtype=torch.float16, seed=0):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "npu"

    torch.manual_seed(seed)
    logits = torch.randn((R, E), device=device, dtype=dtype)

    # triton output
    w_triton, id_triton, _ = triton_moe_gating_topk_softmax(logits, k=K, renormalize=renormalize)

    # reference output
    w_ref, id_ref = reference_topk_softmax(logits, k=K, renormalize=renormalize)

    # check ids (topk indices)
    same_ids = (id_triton.cpu() == id_ref.cpu()).all().item()

    # check weights
    # note: allow small numeric diff due to exp/softmax differences
    atol = 5e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    rtol = 5e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-6
    max_abs = (w_triton.cpu() - w_ref.cpu()).abs().max().item()
    ok_w = torch.allclose(w_triton.cpu(), w_ref.cpu(), rtol=rtol, atol=atol)

    # optional: check renorm sum==1
    if renormalize:
        sum_triton = w_triton.sum(dim=-1)
        sum_ref = w_ref.sum(dim=-1)
        max_sum_err = (sum_triton - 1).abs().max().item()
        max_sum_err_ref = (sum_ref - 1).abs().max().item()
    else:
        max_sum_err = None
        max_sum_err_ref = None

    print(f"[Test] device={device}, R={R}, E={E}, K={K}, dtype={dtype}, renorm={renormalize}")
    print(f"  ids match: {same_ids}")
    print(f"  weights allclose: {ok_w} (max_abs={max_abs:.6g}, rtol={rtol}, atol={atol})")
    if renormalize:
        print(f"  max |sum(w)-1| triton: {max_sum_err:.6g}, ref: {max_sum_err_ref:.6g}")

    # show a few rows if mismatch
    if (not same_ids) or (not ok_w):
        idx = 0
        print("  example row 0:")
        print("    id_triton:", id_triton[idx].cpu().tolist())
        print("    id_ref   :", id_ref[idx].cpu().tolist())
        print("    w_triton :", w_triton[idx].cpu().tolist())
        print("    w_ref    :", w_ref[idx].cpu().tolist())

    return same_ids and ok_w

if __name__ == "__main__":
    # basic correctness tests
    run_test(R=256, E=256, K=8, renormalize=False, dtype=torch.float16, seed=0)
    run_test(R=256, E=256, K=8, renormalize=True,  dtype=torch.float16, seed=1)

    # a bigger R
    run_test(R=4096, E=256, K=8, renormalize=True, dtype=torch.float16, seed=2)
    run_test(R=1, E=256, K=8, renormalize=True, dtype=torch.float16, seed=2)
