# benchmarks/bench_sparse_configs.py
# ******************************************************************************
# End-to-end MoE benchmarks targeting sparse/fine-grained configurations
# where routing accounts for 20-25% of total time (vs 10-15% at standard)
# ******************************************************************************

"""
Sparse configs to add to moe-cute.py benchmark:

  # Standard configs (already benchmarked):
  --thiek 40960,768,256,128,8     # 1.4B
  --thiek 24576,1536,512,64,4     # 7B
  --thiek 32768,4096,1024,128,8   # 30B

  # NEW sparse configs (Proposal 4):
  --thiek 40960,768,256,512,4     # 1.4B sparse (E=512, K=4) — 43% HBM reduction
  --thiek 32768,2048,512,512,10   # Qwen3-like (E=512, K=10)
  --thiek 32768,4096,256,512,4    # 30B ultra-sparse
  --thiek 32768,7168,2048,256,8   # DeepSeek-V3-like

These configs have:
  1. Large E (512) → routing is a bigger fraction of total time
  2. Small K relative to E → more routing overhead per useful expert
  3. Small n = T*K/E → fine-grained expert sizes where SonicMoE excels
"""

import subprocess
import sys


SPARSE_CONFIGS = [
    # (T, H, I, E, K, name)
    (40960, 768,  256,  512, 4,  "1.4B-sparse-E512"),
    (32768, 2048, 512,  512, 10, "Qwen3-like-E512"),
    (32768, 4096, 256,  512, 4,  "30B-ultra-sparse"),
    (32768, 4096, 1024, 256, 16, "30B-fine-grained"),
]


def run_moe_cute_benchmark(T, H, I, E, K, name, activation="swiglu"):
    """Run moe-cute.py for a specific config."""
    thiek = f"{T},{H},{I},{E},{K}"
    
    print(f"\n{'='*60}")
    print(f"Config: {name} (T={T}, H={H}, I={I}, E={E}, K={K})")
    print(f"  n = T*K/E = {T*K//E} tokens/expert")
    print(f"  Score matrix size: {T*E*2/1e6:.1f} MB (eliminated by fused GEMM)")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "benchmarks/moe-cute.py",
        "--thiek", thiek,
        "--activation", activation,
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    for T, H, I, E, K, name in SPARSE_CONFIGS:
        run_moe_cute_benchmark(T, H, I, E, K, name)