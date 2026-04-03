# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************

import argparse
import random
import time
from typing import Tuple, Type

import cutlass
import torch
import torch.nn.functional as F
from rich import print as print0
from triton.testing import do_bench

from sonicmoe import MoE
from sonicmoe.enums import ActivationType, is_glu
from sonicmoe.functional import moe_TC_softmax_topk_layer


# ─────────────────────────────────────────────
# Activation functions (unchanged)
# ─────────────────────────────────────────────
def swiglu(x): return x[..., 1::2] * F.silu(x[..., ::2])
def geglu(x): return F.gelu(x[..., ::2].float()).to(x.dtype) * x[..., 1::2]
def gelu(x): return F.gelu(x.float()).to(x.dtype)
def reglu(x): return (F.relu(x[..., ::2].float()) * x[..., 1::2]).to(x.dtype)
def relu(x): return F.relu(x)
def relu_sq(x): return F.relu(x) ** 2
def silu(x): return F.silu(x)


def parse_comma_separated_ints(s: str):
    return tuple([int(x.strip()) for x in s.split(",")])


# ─────────────────────────────────────────────
# MODIFIED: added --sparse-configs
# ─────────────────────────────────────────────
def parse_arguments():
    parser = argparse.ArgumentParser(description="SonicMoE benchmark")

    parser.add_argument("--thiek", type=parse_comma_separated_ints,
                        default=(32768, 4096, 1024, 128, 8))

    parser.add_argument("--dtype", type=cutlass.dtype,
                        default=cutlass.BFloat16)

    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--add_bias", action="store_true")

    parser.add_argument("--activation",
        choices=["swiglu","geglu","reglu","relu_sq","relu","silu","gelu"],
        default="swiglu"
    )

    # 🔥 NEW
    parser.add_argument("--sparse-configs", action="store_true",
                        help="Run sparse MoE configs (E=512 etc.)")

    args = parser.parse_args()
    return args


# ─────────────────────────────────────────────
# CORE RUN (unchanged)
# ─────────────────────────────────────────────
def run(thiek, dtype, skip_test, add_bias, activation):

    torch_dtype = {
        cutlass.BFloat16: torch.bfloat16,
        cutlass.Float16: torch.float16
    }[dtype]

    activation = ActivationType(activation)

    T, H, I, E, K = thiek
    print(f"\n[T={T}, H={H}, I={I}, E={E}, K={K}]")

    random.seed(1111)
    torch.manual_seed(1111)
    torch.cuda.manual_seed_all(1111)

    moe = MoE(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        activation_function=activation,
        add_bias=add_bias,
        std=0.02,
    ).to(dtype=torch_dtype).cuda()

    x = 0.2 * torch.randn(T, H, device="cuda", dtype=torch_dtype, requires_grad=True)
    dout = 0.2 * torch.randn_like(x, requires_grad=True)

    w1, w2, router_w = moe.c_fc.weight, moe.c_proj.weight, moe.router.weight
    b1, b2 = moe.c_fc.bias, moe.c_proj.bias

    # ─────────────────────────────
    # Benchmark
    # ─────────────────────────────
    repeats, warmup = 200, 5

    def forward():
        o, _, _ = moe_TC_softmax_topk_layer(
            x, router_w,
            w1.permute(1, 2, 0), b1,
            w2.permute(1, 2, 0), b2,
            moe.top_k, moe.stream_id,
            activation, False
        )
        return o

    def fwd_bwd():
        o = forward()
        o.backward(dout, retain_graph=True)
        x.grad = w1.grad = w2.grad = router_w.grad = None

    time.sleep(0.2)

    fwd_time = do_bench(forward, warmup=warmup, rep=repeats)
    e2e_time = do_bench(fwd_bwd, warmup=warmup, rep=repeats)

    print0(f"[cyan]Forward: {fwd_time:.3f} ms")
    print0(f"[green]Fwd+Bwd: {e2e_time:.3f} ms")


# ─────────────────────────────────────────────
# MAIN (MODIFIED)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_arguments()

    # 🔥 NEW: sparse configs (main contribution)
    if args.sparse_configs:
        configs = [
            (40960, 768, 256, 512, 4),   # 1.4B sparse
            (32768, 2048, 512, 512, 10), # Qwen3-like
            (32768, 4096, 256, 512, 4),  # ultra sparse
        ]

        print("\n===== Running Sparse Config Benchmarks =====")
        for cfg in configs:
            run(cfg, args.dtype, args.skip_test, args.add_bias, args.activation)

    else:
        run(args.thiek, args.dtype, args.skip_test, args.add_bias, args.activation)

    print("\nPASS")