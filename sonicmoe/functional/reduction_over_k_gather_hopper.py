# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
#
# reduction_over_k_gather_hopper.py
# FA4-inspired Hopper optimisation — gather-and-sum kernel.
# Weights arrive pre-normalised from TopK_Softmax_Hopper so no exp() here.
# ********************************************************************************

from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass import Float32, const_expr

from ..utils import get_powers_of_2


# ─────────────────────────────────────────────────────────────────────────────
# Triton gather-and-sum kernel
# Required by backward.py and forward.py
# ─────────────────────────────────────────────────────────────────────────────

def _get_triton_autotune_configs() -> list[triton.Config]:
    configs = []
    for BLOCK_H in get_powers_of_2(256, 4096):
        for BLOCK_K in get_powers_of_2(1, 128):
            for num_warps in [4, 8]:
                if BLOCK_K * BLOCK_H <= 32768:
                    configs.append(
                        triton.Config({"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K}, num_warps=num_warps, num_stages=4)
                    )
    return configs


def _prune_triton_autotune_config(configs, nargs, **kw):
    pruned_configs = []
    for c in configs:
        BLOCK_H = c.kwargs["BLOCK_H"]
        BLOCK_K = c.kwargs["BLOCK_K"]
        H = kw["H"]
        MAX_K = kw["MAX_K"]
        if (
            BLOCK_H <= triton.next_power_of_2(H)
            and BLOCK_K <= triton.next_power_of_2(MAX_K)
            and min(H * MAX_K, 1024) <= (BLOCK_H * BLOCK_K)
        ):
            pruned_configs.append(c)
    if len(pruned_configs) == 0:
        return configs
    return pruned_configs


@triton.autotune(
    configs=_get_triton_autotune_configs(),
    key=["H", "MAX_K", "w_is_None", "is_varlen_K"],
    prune_configs_by={"early_config_prune": _prune_triton_autotune_config},
)
@triton.jit
def token_gather_sum_kernel(
    x_ptr,
    w_ptr,
    M_perm_ptr,
    M_offset_ptr,
    out_ptr,
    T,
    H: tl.constexpr,
    MAX_K: tl.constexpr,
    stride_xM: tl.constexpr,
    stride_xH: tl.constexpr,
    stride_outT: tl.constexpr,
    stride_outH: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    w_is_None: tl.constexpr,
    is_varlen_K: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    t_idx = pid_t.to(tl.uint32)

    if is_varlen_K:
        Ms = tl.load(M_offset_ptr + t_idx).to(tl.uint32)
        Me = tl.load(M_offset_ptr + t_idx + 1).to(tl.uint32)
        K_this_token = Me - Ms
    else:
        Ms = MAX_K * t_idx
        K_this_token: tl.constexpr = MAX_K

    for h_tile in tl.static_range(triton.cdiv(H, BLOCK_H)):
        h_idx = (h_tile * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.uint32)
        m_h = h_idx < H
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        for k_tile in tl.range(tl.cdiv(K_this_token, BLOCK_K)):
            k_offset = k_tile * BLOCK_K
            k_idx = (k_offset + tl.arange(0, BLOCK_K)).to(tl.uint32)
            m_k = k_idx < K_this_token
            m_abs = Ms + k_idx
            perm_idx = tl.load(M_perm_ptr + m_abs, mask=m_k, other=0).to(tl.uint32)
            x_ptrs = x_ptr + perm_idx[:, None] * stride_xM + h_idx[None, :] * stride_xH
            x_mask = m_k[:, None] & m_h[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            if w_is_None:
                acc += tl.sum(x_vals, axis=0)
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=m_k, other=0.0).to(tl.float32)
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t_idx * stride_outT + h_idx * stride_outH
        tl.store(out_ptrs, acc, mask=m_h)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,
    w: Optional[torch.Tensor],
    out: torch.Tensor,
    M_perm: torch.Tensor,
    M_offset: torch.Tensor,
    T: int,
    MAX_K: int,
    H: int,
    is_varlen_K: bool,
):
    """
    Gather and weighted sum over K experts per token.
    Weights are pre-normalised from FA4 TopK_Softmax_Hopper — no exp() needed.

    out[i, :] = sum_{j=0..K[i]-1}  x[M_perm[M_offset[i] + j], :] * w[M_offset[i] + j]
    """
    token_gather_sum_kernel[(T,)](
        x,
        w,
        M_perm,
        M_offset,
        out,
        T=T,
        H=H,
        MAX_K=MAX_K,
        stride_xM=x.stride(0),
        stride_xH=x.stride(1),
        stride_outT=out.stride(0),
        stride_outH=out.stride(1),
        w_is_None=(w is None),
        is_varlen_K=is_varlen_K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FA4 exp2 helpers (used by topk_softmax_hopper.py)
# ─────────────────────────────────────────────────────────────────────────────

LOG2E = 1.4426950408889634


@cute.jit
def fast_exp(x: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.exp2(x * cutlass.Float32(LOG2E))


@cute.jit
def online_softmax_update(
    new_val: cutlass.Float32,
    running_max: cutlass.Float32,
    running_sum: cutlass.Float32,
    running_exp_val: cutlass.Float32,
) -> tuple:
    new_max = cute.arch.fmax(running_max, new_val)
    scale = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
    rescaled_sum = running_sum * scale
    rescaled_exp_val = running_exp_val * scale
    new_exp = cute.arch.exp2((new_val - new_max) * cutlass.Float32(LOG2E))
    new_sum = rescaled_sum + new_exp
    return new_max, new_sum, rescaled_exp_val + new_exp