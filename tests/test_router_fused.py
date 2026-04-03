# tests/test_router_fused.py
# ******************************************************************************
# Correctness tests, HBM analysis, and wall-clock benchmarks
# for the Fused Router GEMM kernel
# ******************************************************************************

import argparse
import time
import torch
import torch.nn.functional as F


# ─── Stage 1: HBM Traffic Analysis (no GPU needed) ───

def hbm_analysis():
    """
    Exact byte-count comparison between original 3-kernel pipeline
    and fused router GEMM, across all model configs + sparse configs.
    """
    configs = [
        # (name, T, d, E, K)
        ("1.4B",   40960, 768,  128, 8),
        ("7B",     24576, 1536, 128, 8),
        ("30B",    32768, 4096, 256, 16),
        ("Sparse", 40960, 768,  512, 4),
        # NEW sparse configs (Proposal 4)
        ("Qwen3-like",  32768, 2048, 512, 10),
        ("DeepSeek-like", 32768, 7168, 256, 8),
    ]
    
    print("=" * 80)
    print("HBM Traffic Analysis: Original vs Fused Router GEMM")
    print("=" * 80)
    
    for name, T, d, E, K in configs:
        # Original pipeline HBM:
        #   Read X:        T * d * 2 bytes
        #   Read W_router: E * d * 2 bytes
        #   Write S:       T * E * 2 bytes  ← ELIMINATED
        #   Read S:        T * E * 2 bytes  ← ELIMINATED (softmax)
        #   Write S:       T * E * 2 bytes  ← ELIMINATED (softmax output)
        #   Read S:        T * E * 2 bytes  ← ELIMINATED (topK)
        #   Write results: T * K * 8 bytes (4 float + 4 int)
        
        original_hbm = (
            T * d * 2 +          # read X (matmul)
            E * d * 2 +          # read W_router (matmul)
            T * E * 2 +          # write S (matmul output)
            T * E * 2 +          # read S (softmax input)
            T * E * 2 +          # write S (softmax output)
            T * E * 2 +          # read S (topK input)
            T * K * 8            # write results
        )
        
        # Fused pipeline HBM:
        #   Read X:        T * d * 2 bytes (via TMA)
        #   Read W_router: E * d * 2 bytes (via TMA)
        #   Write results: T * K * 8 bytes
        #   NO score matrix I/O at all
        
        fused_hbm = (
            T * d * 2 +          # read X (TMA)
            E * d * 2 +          # read W_router (TMA)
            T * K * 8            # write results
        )
        
        score_matrix_eliminated = T * E * 2 * 4  # 4 passes eliminated
        reduction_pct = (1 - fused_hbm / original_hbm) * 100
        
        print(f"\n{name} (T={T}, d={d}, E={E}, K={K}):")
        print(f"  Original HBM:  {original_hbm / 1e6:.1f} MB")
        print(f"  Fused HBM:     {fused_hbm / 1e6:.1f} MB")
        print(f"  Score matrix eliminated: {score_matrix_eliminated / 1e6:.1f} MB")
        print(f"  Reduction: {reduction_pct:.1f}%")


# ─── Stage 2: Correctness (requires GPU) ───

def correctness_test(T=8192, d=768, E=128, K=8, dtype=torch.bfloat16):
    """
    Compare fused router output against PyTorch reference.
    """
    from sonicmoe.functional.router_forward import fused_router_forward
    
    device = "cuda"
    X = torch.randn(T, d, device=device, dtype=dtype)
    W = torch.randn(E, d, device=device, dtype=dtype)
    
    # Reference: PyTorch
    scores = X.float() @ W.float().T
    ref_topk_vals, ref_topk_idx = torch.topk(scores, K, dim=-1)
    ref_softmax = F.softmax(ref_topk_vals, dim=-1)
    
    # Fused kernel
    fused_idx, fused_vals = fused_router_forward(X, W, K, use_fused_gemm=True)
    torch.cuda.synchronize()
    
    # Compare — indices may be in different order, so sort both
    ref_sorted_idx, ref_order = ref_topk_idx.sort(dim=-1)
    fused_sorted_idx, fused_order = fused_idx.long().sort(dim=-1)
    
    idx_match = torch.all(ref_sorted_idx == fused_sorted_idx).item()
    
    if idx_match:
        ref_vals_ordered = ref_softmax.gather(1, ref_order)
        fused_vals_ordered = fused_vals.gather(1, fused_order)
        max_err = (ref_vals_ordered - fused_vals_ordered).abs().max().item()
    else:
        # Check per-row match rate
        match_rate = (ref_sorted_idx == fused_sorted_idx).float().mean().item()
        max_err = float("inf")
        print(f"  Index match rate: {match_rate*100:.1f}%")
    
    status = "PASS" if idx_match and max_err < 1e-2 else "FAIL"
    print(f"[{status}] T={T}, d={d}, E={E}, K={K}: idx_match={idx_match}, max_err={max_err:.2e}")
    return status == "PASS"


def run_correctness_tests():
    """Run correctness across standard + sparse configs."""
    configs = [
        (8192,  768,  128, 8),   # 1.4B standard
        (8192,  1536, 64,  4),   # 7B standard
        (8192,  4096, 128, 8),   # 30B standard
        (8192,  4096, 256, 16),  # 30B fine-grained
        (8192,  768,  512, 4),   # Sparse — key new config
        (4096,  2048, 512, 10),  # Qwen3-like — key new config
    ]
    
    print("\n" + "=" * 80)
    print("Correctness Tests: Fused Router GEMM vs PyTorch Reference")
    print("=" * 80)
    
    all_pass = True
    for T, d, E, K in configs:
        if not correctness_test(T, d, E, K):
            all_pass = False
    
    return all_pass


# ─── Stage 3: Benchmarks (requires GPU) ───

def benchmark_router(T, d, E, K, warmup=10, rep=100):
    """
    Wall-clock comparison: original 3-kernel vs fused GEMM.
    Uses CUDA events for accurate timing.
    """
    from sonicmoe.functional.router_forward import fused_router_forward, _fallback_router_forward
    
    device = "cuda"
    dtype = torch.bfloat16
    X = torch.randn(T, d, device=device, dtype=dtype)
    W = torch.randn(E, d, device=device, dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        _fallback_router_forward(X, W, K)
        fused_router_forward(X, W, K, use_fused_gemm=True)
    torch.cuda.synchronize()
    
    # Benchmark original (cuBLAS matmul + TopK+Softmax)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(rep):
        _fallback_router_forward(X, W, K)
    end.record()
    torch.cuda.synchronize()
    original_ms = start.elapsed_time(end) / rep
    
    # Benchmark fused GEMM
    start.record()
    for _ in range(rep):
        fused_router_forward(X, W, K, use_fused_gemm=True)
    end.record()
    torch.cuda.synchronize()
    fused_ms = start.elapsed_time(end) / rep
    
    speedup = original_ms / fused_ms
    print(f"  T={T}, d={d}, E={E}, K={K}: "
          f"Original={original_ms:.3f}ms, Fused={fused_ms:.3f}ms, "
          f"Speedup={speedup:.2f}x")
    
    return original_ms, fused_ms, speedup


def run_benchmarks():
    """Benchmark across all configs including sparse."""
    configs = [
        # Standard configs (same as previous experiments)
        ("1.4B",    40960, 768,  128, 8),
        ("7B",      24576, 1536, 64,  4),
        ("30B",     32768, 4096, 128, 8),
        ("30B-fg",  32768, 4096, 256, 16),
        # NEW: Sparse configs where routing fraction is larger (Proposal 4)
        ("Sparse",  40960, 768,  512, 4),
        ("Qwen3",   32768, 2048, 512, 10),
        ("DSv3",    32768, 7168, 256, 8),
    ]
    
    print("\n" + "=" * 80)
    print("Router Benchmark: Original 3-Kernel vs Fused WGMMA Router GEMM")
    print("=" * 80)
    
    for name, T, d, E, K in configs:
        print(f"\n[{name}]")
        benchmark_router(T, d, E, K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true",
                        help="Run HBM analysis only (no GPU)")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    
    if args.analysis_only:
        hbm_analysis()
    elif args.correctness_only:
        run_correctness_tests()
    elif args.benchmark_only:
        run_benchmarks()
    else:
        hbm_analysis()
        run_correctness_tests()
        run_benchmarks()
        