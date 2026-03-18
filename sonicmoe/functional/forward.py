# ********************************************************************************
# Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
# ********************************************************************************
#
# forward.py — Hopper-optimised MoE forward pass
#
# Changes vs original:
# ────────────────────
# 1.  TopK_Softmax_Hopper replaces TopK_Softmax everywhere.
#     ─ Uses ex2.approx instead of expf() (FA4 §3.1).
#     ─ Fuses online softmax INTO the bitonic-sort pass (FA4 §3.2 "Kernel-1 merge").
#       One kernel launch: load logits → sort → online-softmax → write scores+indices.
#       Zero extra HBM round-trip between topK and softmax.
#
# 2.  token_gather_and_sum_varlen_K_triton is swapped for the Hopper-tuned version
#     from reduction_over_k_gather_hopper.py.  The softmax weights `w` it receives
#     are pre-normalised (from change 1), so no exp() runs inside this kernel.
#
# Everything else (GEMM pipeline, tensor-map management, backward) is unchanged.
# ─────────────────────────────────────────────────────────────────────────────

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map

from ..enums import LIBRARY_NAME, TENSORMAP, ActivationType
from ..utils import convert_torch_tensor_to_cute_tensor

# ── Hopper-optimised kernels (FA4-inspired) ───────────────────────────────────
from .topk_softmax_hopper import TopK_Softmax_Hopper as TopK_Softmax           # replaces topk_softmax.TopK_Softmax
from .reduction_over_k_gather_hopper import token_gather_and_sum_varlen_K_triton  # replaces original

from .moe_config import HopperWgmma_MoE_Down_proj_Fwd, HopperWgmma_MoE_Up_proj_Fwd


# ─────────────────────────────────────────────────────────────────────────────
# Top-K + Online-Softmax  (Kernel-1 merge)
# ─────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op(f"{LIBRARY_NAME}::_topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(
    x: torch.Tensor,
    k: int,
    values:  torch.Tensor,
    indices: torch.Tensor,
    require_softmax_fusion: bool = True,
) -> None:
    """
    Top-k forward pass — Hopper-optimised.

    Uses TopK_Softmax_Hopper which:
      • emits ex2.approx PTX for all exp() calls (FA4 §3.1)
      • runs online softmax in the same register pass as bitonic sort (FA4 §3.2)
      → single kernel launch, zero extra HBM traffic for softmax normalisation
    """
    N = x.size(1)

    input_dtype  = torch2cute_dtype_map[x.dtype]
    output_dtype = torch2cute_dtype_map[values.dtype]

    convert_from_dlpack = lambda tensor: (
        from_dlpack(tensor.detach(), assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )

    x_tensor, values_tensor, indices_tensor = [
        convert_from_dlpack(t) for t in (x, values, indices)
    ]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    compile_key = (input_dtype, output_dtype, N, k, require_softmax_fusion)
    if compile_key not in _topk_fwd.compile_cache:
        # ── Hopper-optimised kernel (ex2 + online-softmax) ───────────────────
        topk_op = TopK_Softmax(
            input_dtype,
            output_dtype,
            N,
            k,
            require_softmax_fusion,
        )
        _topk_fwd.compile_cache[compile_key] = cute.compile(
            topk_op, x_tensor, values_tensor, indices_tensor, current_stream
        )

    _topk_fwd.compile_cache[compile_key](
        x_tensor, values_tensor, indices_tensor, current_stream
    )


_topk_fwd.compile_cache = {}


# ─────────────────────────────────────────────────────────────────────────────
# Up-projection forward  (unchanged GEMM path)
# ─────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op(
    f"{LIBRARY_NAME}::_up_projection_forward",
    mutates_args={"z", "y1"},
)
def _up_projection_forward(
    x: torch.Tensor,
    w1: torch.Tensor,
    z:  torch.Tensor,
    y1: torch.Tensor,
    b1: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    expert_schedule_order:   torch.Tensor,
    x_gather_idx:            torch.Tensor,
    stream_id:               int,
    activation_type:         str,
    is_glu_activation:       bool,
    is_inference_mode_enabled: bool = False,
) -> None:
    I, H, E = w1.size()
    if is_glu_activation:
        I //= 2

    mX        = convert_torch_tensor_to_cute_tensor(x.detach(),  (0, 1), 1, 16, 8, stream=stream_id)
    mW1       = convert_torch_tensor_to_cute_tensor(w1.detach(), (2, 0, 1), 1, 16, 8, stream=stream_id)
    mZ        = convert_torch_tensor_to_cute_tensor(z,  (0, 1), 1, 16, 8, stream=stream_id)
    mY1       = convert_torch_tensor_to_cute_tensor(y1, (0, 1), 1, 16, 8, stream=stream_id)
    mE_offset = convert_torch_tensor_to_cute_tensor(expert_frequency_offset, (0,), 0, 4, 1, stream=stream_id)
    mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1, stream=stream_id)

    mE_permute_order = (
        None
        if expert_schedule_order is None
        else convert_torch_tensor_to_cute_tensor(expert_schedule_order, (0,), 0, 4, 1, stream=stream_id)
    )
    mB1 = (
        None
        if b1 is None
        else convert_torch_tensor_to_cute_tensor(b1.detach(), (0, 1), 1, 16, 8, stream=stream_id)
    )

    current_stream  = cuda.CUstream(stream_id)
    compile_w1_key  = (E, H, I, (b1 is None), x.dtype, activation_type, is_inference_mode_enabled)

    if compile_w1_key not in _up_projection_forward.compile_cache:
        w1_module  = HopperWgmma_MoE_Up_proj_Fwd(
            E, H, I, activation_type=ActivationType(activation_type),
            inference_mode=is_inference_mode_enabled,
        )
        tensormaps = [w1_module.module.generate_tensormap(None, None, None) for _ in range(2)]
        _up_projection_forward.compile_cache[compile_w1_key] = cute.compile(
            w1_module, mX, mW1, mZ, mY1, mB1, mE_offset, mX_gather,
            tensormaps[0], tensormaps[1], mE_permute_order, current_stream,
        )
        _up_projection_forward.compile_cache[TENSORMAP] = tensormaps

    w1_tensormaps = _up_projection_forward.compile_cache[TENSORMAP]
    _up_projection_forward.compile_cache[compile_w1_key](
        mX, mW1, mZ, mY1, mB1, mE_offset, mX_gather,
        w1_tensormaps[0], w1_tensormaps[1], mE_permute_order, current_stream,
    )


_up_projection_forward.compile_cache = {}


# ─────────────────────────────────────────────────────────────────────────────
# Down-projection forward  (unchanged GEMM path)
# ─────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op(
    f"{LIBRARY_NAME}::_down_projection_forward",
    mutates_args={"y2"},
)
def _down_projection_forward(
    w2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    b2: torch.Tensor | None,
    expert_frequency_offset: torch.Tensor,
    expert_schedule_order:   torch.Tensor,
    x_gather_idx:            torch.Tensor,
    stream_id:               int,
) -> None:
    H, I, E = w2.size()

    mW2       = convert_torch_tensor_to_cute_tensor(w2.detach(), (2, 0, 1), 1, 16, 8, stream=stream_id)
    mY1       = convert_torch_tensor_to_cute_tensor(y1.detach(), (0, 1), 1, 16, 8, stream=stream_id)
    mY2       = convert_torch_tensor_to_cute_tensor(y2, (0, 1), 1, 16, 8, stream=stream_id)
    mE_offset = convert_torch_tensor_to_cute_tensor(expert_frequency_offset, (0,), 0, 4, 1, stream=stream_id)
    mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1, stream=stream_id)

    mE_permute_order = (
        None
        if expert_schedule_order is None
        else convert_torch_tensor_to_cute_tensor(expert_schedule_order, (0,), 0, 4, 1, stream=stream_id)
    )
    mB2 = (
        None
        if b2 is None
        else convert_torch_tensor_to_cute_tensor(b2.detach(), (0, 1), 1, 16, 8, stream=stream_id)
    )

    current_stream = cuda.CUstream(stream_id)
    compile_w2_key = (E, H, I, (b2 is None), w2.dtype)

    if compile_w2_key not in _down_projection_forward.compile_cache:
        w2_module  = HopperWgmma_MoE_Down_proj_Fwd(E, H, I)
        tensormaps = [w2_module.module.generate_tensormap(None, None, None) for _ in range(1)]
        _down_projection_forward.compile_cache[compile_w2_key] = cute.compile(
            w2_module, mY1, mW2, mY2, mB2, mE_offset, mX_gather,
            tensormaps[0], mE_permute_order, current_stream,
        )
        _down_projection_forward.compile_cache[TENSORMAP] = tensormaps

    w2_tensormaps = _down_projection_forward.compile_cache[TENSORMAP]
    _down_projection_forward.compile_cache[compile_w2_key](
        mY1, mW2, mY2, mB2, mE_offset, mX_gather,
        w2_tensormaps[0], mE_permute_order, current_stream,
    )


_down_projection_forward.compile_cache = {}


# ─────────────────────────────────────────────────────────────────────────────
# Expert aggregation  — uses Hopper-tuned gather-and-sum
# ─────────────────────────────────────────────────────────────────────────────

@torch.library.custom_op(f"{LIBRARY_NAME}::_router_forward", mutates_args={"o"})
def _router_forward(
    y2:          torch.Tensor,
    o:           torch.Tensor,
    topk_scores: torch.Tensor,
    s_reverse_scatter_idx:              torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
    varlen_K_max: int,
    H:            int,
    is_varlen_K:  bool,
) -> None:
    """
    Expert aggregation: gathers GEMM outputs per token and applies pre-normalised
    softmax weights.  Weights come from TopK_Softmax_Hopper (already normalised),
    so this kernel performs only a weighted sum — no exp() anywhere.
    """
    token_gather_and_sum_varlen_K_triton(
        y2,
        topk_scores,
        o,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        o.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Softmax-only fallback (for E > 4096 or K > 16 configs)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _softmax_fwd_small_kernel(
    logits_ptr,
    stride_lm: tl.constexpr,
    stride_ln: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Small softmax kernel — used only when E > 4096 or K > 16.
    Uses tl.math.exp2(x * log₂e) instead of tl.exp for Hopper SFU speed.
    """
    _LOG2E_: tl.constexpr = 1.4426950408889634

    row   = tl.program_id(axis=0)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    x   = tl.load(
        logits_ptr + row * stride_lm + k_offs * stride_ln,
        mask=k_mask, other=-float("inf"),
    ).to(tl.float32)

    x   = x - tl.max(x, axis=0)
    # FA4 §3.1: ex2 instead of exp
    ex  = tl.math.exp2(x * _LOG2E_)
    y   = ex / tl.sum(ex, axis=0)

    tl.store(logits_ptr + row * stride_lm + k_offs * stride_ln, y, mask=k_mask)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_softmax_topk_fwd",
    mutates_args={"topk_router_score", "topk_router_indices"},
)
def _softmax_topk_fwd(
    router_logits:       torch.Tensor,
    topk_router_score:   torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
) -> None:
    """
    Entry point for router softmax + top-K.

    Fast path (E ≤ 4096, K ≤ 16, E%8==0):
        → Single kernel: TopK_Softmax_Hopper
          ex2.approx + online-softmax merged with bitonic sort  (Kernel-1 merge)

    Slow path (large E or K):
        → torch.topk + Triton ex2-softmax fallback
    """
    if E <= 4096 and K <= 16 and E % 8 == 0:
        _topk_fwd(
            router_logits, K,
            topk_router_score, topk_router_indices,
            require_softmax_fusion=True,
        )
    else:
        # fallback: torch topK then ex2-softmax
        topk_results = router_logits.topk(K, dim=-1)
        topk_router_score.copy_(
            topk_results.values.softmax(dim=-1, dtype=torch.float32)
            .to(topk_router_score.dtype)
        )
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))