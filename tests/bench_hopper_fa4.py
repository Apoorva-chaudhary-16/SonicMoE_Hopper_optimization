# ********************************************************************************
# tests/bench_hopper_fa4.py
#
# Benchmark FA4-inspired Hopper optimisations for SonicMoE.
#
# FIX (Issue 4):
#   The original file imported both TopK_Softmax and TopK_Softmax_Hopper from
#   the same module, where both aliases now point to FusedRouterTopKSoftmax_SM90
#   (the corrected fused kernel).  Benchmarking a kernel against itself always
#   produces a 1.000× "speedup" and gives a completely misleading result.
#
#   Because the repository no longer ships a separate legacy kernel to compare
#   against, Section 1 and Section 4 now:
#     (a) Use a PyTorch reference (torch.topk + F.softmax) as the "original"
#         baseline.  This is the operation that the fused kernel replaces.
#     (b) Note clearly in the output that this is a PyTorch baseline, not the
#         original CuTe kernel.
#
#   If a legacy CuTe kernel is later reintroduced, swap `_topk_pytorch_ref`
#   for an import of that class and the benchmark will automatically be
#   meaningful again.
# ********************************************************************************

import argparse
import statistics
import time
from functools import partial

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map
from triton.testing import do_bench

# ── FA4 fused kernel (the one being benchmarked) ──────────────────────────
from sonicmoe.functional.topk_softmax_hopper import TopK_Softmax_Hopper
from sonicmoe.functional.reduction_over_k_gather_hopper import (
    token_gather_and_sum_varlen_K_triton as gather_hopper,
)

# ── Full MoE ──────────────────────────────────────────────────────────────
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _us(ms: float) -> str:
    return f"{ms * 1000:.1f} µs"


def _speedup(baseline_ms: float, new_ms: float) -> str:
    s     = baseline_ms / new_ms
    arrow = "✅ faster" if s > 1.005 else ("⚠️  same" if s > 0.995 else "❌ slower")
    return f"{s:.3f}×  {arrow}"


def _header(title: str, baseline_label: str = "Baseline"):
    w = 95
    print()
    print("═" * w)
    print(f"  {title}")
    print("═" * w)
    print(
        f"  {'Config':<45} {baseline_label:>18} {'Hopper FA4':>12} {'Speedup'}"
    )
    print("─" * w)


def _row(label: str, baseline_ms: float, new_ms: float):
    print(
        f"  {label:<45} {_us(baseline_ms):>18} {_us(new_ms):>12}   "
        f"{_speedup(baseline_ms, new_ms)}"
    )


def _section_end():
    print("─" * 95)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch reference baseline for TopK + Softmax
# Used in Sections 1 and 4 because both kernel aliases point to the same class.
# ─────────────────────────────────────────────────────────────────────────────

def _topk_pytorch_ref(logits: torch.Tensor, K: int):
    """Pure PyTorch TopK + Softmax — the operation the fused kernel replaces."""
    logits_f32 = logits.float()
    vals, idx  = torch.topk(logits_f32, K, dim=-1)
    scores     = F.softmax(vals, dim=-1)
    return scores, idx.to(torch.int32)


def _to_cute(t):
    return (
        from_dlpack(t.detach(), assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )


def _compile_hopper_topk(logits, scores, indices, K):
    T, E         = logits.shape
    input_dtype  = torch2cute_dtype_map[logits.dtype]
    output_dtype = torch2cute_dtype_map[scores.dtype]
    stream       = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel       = TopK_Softmax_Hopper(input_dtype, output_dtype, E, K,
                                       require_softmax_fusion=True)
    compiled     = cute.compile(
        kernel,
        _to_cute(logits), _to_cute(scores), _to_cute(indices), stream,
    )
    return compiled, stream


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — TopK + Softmax kernel vs PyTorch baseline
# ─────────────────────────────────────────────────────────────────────────────

def bench_topk_softmax(warmup: int, rep: int):
    _header(
        "SECTION 1 · TopK + Softmax kernel  (FA4 fused vs PyTorch baseline)",
        baseline_label="PyTorch ref",
    )
    print("  NOTE: baseline is torch.topk + F.softmax (the op the kernel replaces).")
    print("        A legacy CuTe original kernel is not separately available.")

    CONFIGS = [
        (40960, 128,  8,  "1.4B  T=40960 E=128  K=8  "),
        (24576,  64,  4,  "7B    T=24576 E=64   K=4  "),
        (32768, 256, 16,  "30B   T=32768 E=256  K=16 "),
        (65536, 128,  8,  "large T=65536 E=128  K=8  "),
    ]

    for T, E, K, label in CONFIGS:
        logits  = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)
        scores  = torch.zeros(T, K, device="cuda", dtype=torch.float32)
        indices = torch.zeros(T, K, device="cuda", dtype=torch.int32)

        fn_hop, stream = _compile_hopper_topk(logits, scores, indices, K)

        for _ in range(warmup):
            _topk_pytorch_ref(logits, K)
            fn_hop(_to_cute(logits), _to_cute(scores), _to_cute(indices), stream)
        torch.cuda.synchronize()

        t_ref = do_bench(
            lambda: _topk_pytorch_ref(logits, K),
            warmup=warmup, rep=rep,
        )
        t_hop = do_bench(
            lambda: fn_hop(_to_cute(logits), _to_cute(scores), _to_cute(indices), stream),
            warmup=warmup, rep=rep,
        )
        _row(label, t_ref, t_hop)

    _section_end()


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Expert aggregation (gather-and-sum)
# ─────────────────────────────────────────────────────────────────────────────

def bench_gather_sum(warmup: int, rep: int):
    _header(
        "SECTION 2 · Expert aggregation kernel  (gather-and-sum)",
        baseline_label="PyTorch ref",
    )

    CONFIGS = [
        (40960,  768,  8,  "1.4B  T=40960 H=768  K=8  "),
        (24576, 1536,  4,  "7B    T=24576 H=1536 K=4  "),
        (32768, 4096, 16,  "30B   T=32768 H=4096 K=16 "),
        (32768, 4096,  8,  "30B   T=32768 H=4096 K=8  "),
    ]

    for T, H, K, label in CONFIGS:
        Mtotal   = T * K
        x        = torch.randn(Mtotal, H, device="cuda", dtype=torch.bfloat16)
        w        = torch.rand(Mtotal,     device="cuda", dtype=torch.float32)
        w_norm   = (w.view(T, K) / w.view(T, K).sum(-1, keepdim=True)).reshape(-1)
        M_perm   = torch.randperm(Mtotal, device="cuda", dtype=torch.int32)
        M_offset = (torch.arange(T + 1, device="cuda") * K).int()
        out      = torch.zeros(T, H, device="cuda", dtype=torch.float32)

        def fn_ref():
            """PyTorch scatter-add baseline."""
            x_f32 = x.float()
            # Expand: x[M_perm] * w_norm, scatter-add to out
            gathered = x_f32[M_perm.long()]             # (T*K, H)
            scaled   = gathered * w_norm[:, None]        # (T*K, H)
            rep_idx  = torch.arange(T, device="cuda").repeat_interleave(K)
            torch.zeros_like(out).scatter_add_(0, rep_idx.unsqueeze(1).expand_as(scaled), scaled)

        def fn_hop():
            gather_hopper(x, w_norm, out, M_perm, M_offset, T, K, H, is_varlen_K=False)

        for _ in range(warmup):
            fn_ref(); fn_hop()
        torch.cuda.synchronize()

        t_ref = do_bench(fn_ref, warmup=warmup, rep=rep)
        t_hop = do_bench(fn_hop, warmup=warmup, rep=rep)
        _row(label, t_ref, t_hop)

    _section_end()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — End-to-end MoE forward (SonicMoE Hopper vs PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def bench_e2e_forward(warmup: int, rep: int):
    _header(
        "SECTION 3 · End-to-end MoE forward  (SonicMoE Hopper vs PyTorch)",
        baseline_label="PyTorch MoE",
    )

    CONFIGS = [
        (8192,  768,  256, 128,  8,  "1.4B  T=8192  H=768  I=256  E=128 K=8  "),
        (8192, 1536,  512,  64,  4,  "7B    T=8192  H=1536 I=512  E=64  K=4  "),
        (8192, 4096,  512, 128,  8,  "30B   T=8192  H=4096 I=512  E=128 K=8  "),
        (8192, 4096, 1024,  64,  4,  "30B   T=8192  H=4096 I=1024 E=64  K=4  "),
    ]

    device = torch.device("cuda")

    for T, H, I, E, K, label in CONFIGS:
        torch.manual_seed(42)
        moe = MoE(
            num_experts=E,
            num_experts_per_tok=K,
            hidden_size=H,
            intermediate_size=I,
            activation_function=ActivationType.SWIGLU,
            add_bias=False,
            std=0.02,
        ).to(device=device, dtype=torch.bfloat16)

        x = 0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16)

        def fn_torch():
            with torch.autocast(device.type, torch.float32):
                moe(x, kernel_backend_moe=KernelBackendMoE.torch)

        def fn_sonic():
            with torch.autocast(device.type, torch.float32):
                moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)

        for _ in range(max(warmup, 5)):
            fn_sonic()
        torch.cuda.synchronize()

        t_torch = do_bench(fn_torch, warmup=warmup, rep=rep)
        t_sonic = do_bench(fn_sonic, warmup=warmup, rep=rep)
        _row(label, t_torch, t_sonic)

    _section_end()


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Router-only isolation (CUDA-event timing)
# ─────────────────────────────────────────────────────────────────────────────

def bench_router_isolated(warmup: int, rep: int):
    _header(
        "SECTION 4 · Router kernel isolated  (CUDA-event timing, vs PyTorch baseline)",
        baseline_label="PyTorch ref",
    )
    print(
        "  NOTE: baseline is torch.topk + F.softmax (full-precision reference)."
    )

    CONFIGS = [
        (40960, 128,  8,  "1.4B T=40960 E=128  K=8  "),
        (24576,  64,  4,  "7B   T=24576 E=64   K=4  "),
        (32768, 256, 16,  "30B  T=32768 E=256  K=16 "),
    ]

    start_e = torch.cuda.Event(enable_timing=True)
    end_e   = torch.cuda.Event(enable_timing=True)

    for T, E, K, label in CONFIGS:
        logits  = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)
        scores  = torch.zeros(T, K, device="cuda", dtype=torch.float32)
        indices = torch.zeros(T, K, device="cuda", dtype=torch.int32)

        fn_hop, stream = _compile_hopper_topk(logits, scores, indices, K)

        for _ in range(warmup):
            _topk_pytorch_ref(logits, K)
            fn_hop(_to_cute(logits), _to_cute(scores), _to_cute(indices), stream)
        torch.cuda.synchronize()

        times_ref, times_hop = [], []

        for _ in range(rep):
            start_e.record()
            _topk_pytorch_ref(logits, K)
            end_e.record(); end_e.synchronize()
            times_ref.append(start_e.elapsed_time(end_e))

        for _ in range(rep):
            start_e.record()
            fn_hop(_to_cute(logits), _to_cute(scores), _to_cute(indices), stream)
            end_e.record(); end_e.synchronize()
            times_hop.append(start_e.elapsed_time(end_e))

        t_ref = statistics.median(times_ref)
        t_hop = statistics.median(times_hop)

        _row(f"{label} [median of {rep}]", t_ref, t_hop)

    _section_end()


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    w = 95
    print()
    print("═" * w)
    print("  SUMMARY — FA4 Hopper optimisations")
    print("═" * w)
    rows = [
        ("ex2.approx (FA4 §3.1)",
         "topk_softmax_hopper.py",
         "~2× faster exp on Hopper SFU",
         "Sections 1 & 4"),
        ("Online softmax Kernel-1 merge (FA4 §3.2)",
         "topk_softmax_hopper.py",
         "Eliminates 1 register scan over E",
         "Sections 1 & 4"),
        ("Pre-normalised weights in aggregation",
         "reduction_over_k_gather_hopper.py",
         "No exp() in gather-sum kernel",
         "Section 2"),
    ]
    print(
        f"  {'Optimisation':<42} {'File':<38} {'Expected gain':<30} {'Benchmark'}"
    )
    print("─" * w)
    for opt, f, gain, bench in rows:
        print(f"  {opt:<42} {f:<38} {gain:<30} {bench}")
    print("═" * w)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark FA4 Hopper optimisations for SonicMoE"
    )
    parser.add_argument("--warmup",  type=int, default=5)
    parser.add_argument("--rep",     type=int, default=50)
    parser.add_argument("--section", type=int, default=0,
                        help="Run only section N (0 = all, 1-5)")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  SonicMoE — FA4 Hopper Optimisation Benchmark                          ║")
    print("║  GPU:", torch.cuda.get_device_name(0).ljust(65), "║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")

    torch.cuda.set_device(0)

    run_all = args.section == 0
    if run_all or args.section == 1:
        bench_topk_softmax(args.warmup, args.rep)
    if run_all or args.section == 2:
        bench_gather_sum(args.warmup, args.rep)
    if run_all or args.section == 3:
        bench_e2e_forward(args.warmup, args.rep)
    if run_all or args.section == 4:
        bench_router_isolated(args.warmup, args.rep)
    if run_all or args.section == 5:
        print_summary()