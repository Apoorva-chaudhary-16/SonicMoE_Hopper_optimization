# ********************************************************************************
# forward.py — Hopper-optimised MoE forward pass (FA4 exp2 path)
# ********************************************************************************

from .router_forward import fused_router_forward
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map
from ..enums import LIBRARY_NAME, TENSORMAP, ActivationType
from ..utils import convert_torch_tensor_to_cute_tensor
from .topk_softmax_hopper import topk_softmax_triton
from .topk_softmax_hopper import TopK_Softmax_Hopper as TopK_Softmax
from .reduction_over_k_gather_hopper import token_gather_and_sum_varlen_K_triton
from .moe_config import HopperWgmma_MoE_Down_proj_Fwd, HopperWgmma_MoE_Up_proj_Fwd


# ─────────────────────────────────────────────────────────────
# TopK kernel (unchanged)
# ─────────────────────────────────────────────────────────────
@torch.library.custom_op(f"{LIBRARY_NAME}::_topk_fwd", mutates_args={"values", "indices"})
def _topk_fwd(x: torch.Tensor, k: int, values: torch.Tensor, indices: torch.Tensor, require_softmax_fusion: bool = True) -> None:
    topk_softmax_triton(x, k, values, indices, require_softmax_fusion)

_topk_fwd.compile_cache = {}


# ─────────────────────────────────────────────────────────────
# NEW: Fused Router GEMM (MAIN CHANGE)
# ─────────────────────────────────────────────────────────────
@torch.library.custom_op(f"{LIBRARY_NAME}::_fused_router_gemm_fwd",
                        mutates_args={"topk_router_score", "topk_router_indices"})
def _fused_router_gemm_fwd(
    x: torch.Tensor,                    # (T, d)
    w_router: torch.Tensor,             # (E, d)
    topk_router_score: torch.Tensor,    # (T, K)
    topk_router_indices: torch.Tensor,  # (T, K)
    K: int,
) -> None:
    """
    Fused router:
    X @ W_router.T + Softmax + TopK in ONE kernel.
    """
    indices, weights = fused_router_forward(x, w_router, K, use_fused_gemm=True)
    topk_router_score.copy_(weights)
    topk_router_indices.copy_(indices)


# ─────────────────────────────────────────────────────────────
# Up Projection (unchanged)
# ─────────────────────────────────────────────────────────────
@torch.library.custom_op(f"{LIBRARY_NAME}::_up_projection_forward", mutates_args={"z", "y1"})
def _up_projection_forward(x: torch.Tensor, w1: torch.Tensor, z: torch.Tensor, y1: torch.Tensor, b1: torch.Tensor | None, expert_frequency_offset: torch.Tensor, expert_schedule_order: torch.Tensor, x_gather_idx: torch.Tensor, stream_id: int, activation_type: str, is_glu_activation: bool, is_inference_mode_enabled: bool = False) -> None:
    I, H, E = w1.size()
    if is_glu_activation:
        I //= 2

    mX = convert_torch_tensor_to_cute_tensor(x.detach(), (0, 1), 1, 16, 8, stream=stream_id)
    mW1 = convert_torch_tensor_to_cute_tensor(w1.detach(), (2, 0, 1), 1, 16, 8, stream=stream_id)
    mZ = convert_torch_tensor_to_cute_tensor(z, (0, 1), 1, 16, 8, stream=stream_id)
    mY1 = convert_torch_tensor_to_cute_tensor(y1, (0, 1), 1, 16, 8, stream=stream_id)

    mE_offset = convert_torch_tensor_to_cute_tensor(expert_frequency_offset, (0,), 0, 4, 1, stream=stream_id)
    mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1, stream=stream_id)

    mE_permute_order = None if expert_schedule_order is None else convert_torch_tensor_to_cute_tensor(expert_schedule_order, (0,), 0, 4, 1, stream=stream_id)
    mB1 = None if b1 is None else convert_torch_tensor_to_cute_tensor(b1.detach(), (0, 1), 1, 16, 8, stream=stream_id)

    current_stream = cuda.CUstream(stream_id)

    compile_w1_key = (E, H, I, (b1 is None), x.dtype, activation_type, is_inference_mode_enabled)

    if compile_w1_key not in _up_projection_forward.compile_cache:
        w1_module = HopperWgmma_MoE_Up_proj_Fwd(E, H, I, activation_type=ActivationType(activation_type), inference_mode=is_inference_mode_enabled)
        tensormaps = [w1_module.module.generate_tensormap(None, None, None) for _ in range(2)]

        _up_projection_forward.compile_cache[compile_w1_key] = cute.compile(
            w1_module, mX, mW1, mZ, mY1, mB1,
            mE_offset, mX_gather,
            tensormaps[0], tensormaps[1],
            mE_permute_order, current_stream
        )
        _up_projection_forward.compile_cache[TENSORMAP] = tensormaps

    w1_tensormaps = _up_projection_forward.compile_cache[TENSORMAP]

    _up_projection_forward.compile_cache[compile_w1_key](
        mX, mW1, mZ, mY1, mB1,
        mE_offset, mX_gather,
        w1_tensormaps[0], w1_tensormaps[1],
        mE_permute_order, current_stream
    )

_up_projection_forward.compile_cache = {}


# ─────────────────────────────────────────────────────────────
# Down Projection (unchanged)
# ─────────────────────────────────────────────────────────────
@torch.library.custom_op(f"{LIBRARY_NAME}::_down_projection_forward", mutates_args={"y2"})
def _down_projection_forward(w2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, b2: torch.Tensor | None, expert_frequency_offset: torch.Tensor, expert_schedule_order: torch.Tensor, x_gather_idx: torch.Tensor, stream_id: int) -> None:
    H, I, E = w2.size()

    mW2 = convert_torch_tensor_to_cute_tensor(w2.detach(), (2, 0, 1), 1, 16, 8, stream=stream_id)
    mY1 = convert_torch_tensor_to_cute_tensor(y1.detach(), (0, 1), 1, 16, 8, stream=stream_id)
    mY2 = convert_torch_tensor_to_cute_tensor(y2, (0, 1), 1, 16, 8, stream=stream_id)

    mE_offset = convert_torch_tensor_to_cute_tensor(expert_frequency_offset, (0,), 0, 4, 1, stream=stream_id)
    mX_gather = convert_torch_tensor_to_cute_tensor(x_gather_idx, (0,), 0, 4, 1, stream=stream_id)

    mE_permute_order = None if expert_schedule_order is None else convert_torch_tensor_to_cute_tensor(expert_schedule_order, (0,), 0, 4, 1, stream=stream_id)
    mB2 = None if b2 is None else convert_torch_tensor_to_cute_tensor(b2.detach(), (0, 1), 1, 16, 8, stream=stream_id)

    current_stream = cuda.CUstream(stream_id)

    compile_w2_key = (E, H, I, (b2 is None), w2.dtype)

    if compile_w2_key not in _down_projection_forward.compile_cache:
        w2_module = HopperWgmma_MoE_Down_proj_Fwd(E, H, I)
        tensormaps = [w2_module.module.generate_tensormap(None, None, None)]

        _down_projection_forward.compile_cache[compile_w2_key] = cute.compile(
            w2_module, mY1, mW2, mY2, mB2,
            mE_offset, mX_gather,
            tensormaps[0], mE_permute_order, current_stream
        )
        _down_projection_forward.compile_cache[TENSORMAP] = tensormaps

    w2_tensormaps = _down_projection_forward.compile_cache[TENSORMAP]

    _down_projection_forward.compile_cache[compile_w2_key](
        mY1, mW2, mY2, mB2,
        mE_offset, mX_gather,
        w2_tensormaps[0], mE_permute_order, current_stream
    )

_down_projection_forward.compile_cache = {}


# ─────────────────────────────────────────────────────────────
# Router aggregation (unchanged)
# ─────────────────────────────────────────────────────────────
@torch.library.custom_op(f"{LIBRARY_NAME}::_router_forward", mutates_args={"o"})
def _router_forward(y2: torch.Tensor, o: torch.Tensor, topk_scores: torch.Tensor, s_reverse_scatter_idx: torch.Tensor, num_activated_expert_per_token_offset: torch.Tensor, varlen_K_max: int, H: int, is_varlen_K: bool) -> None:
    token_gather_and_sum_varlen_K_triton(
        y2, topk_scores, o,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        o.size(0), varlen_K_max, H, is_varlen_K
    )


# ─────────────────────────────────────────────────────────────
# OLD PATH (kept as fallback)
# ─────────────────────────────────────────────────────────────
@torch.library.custom_op(f"{LIBRARY_NAME}::_softmax_topk_fwd",
                        mutates_args={"topk_router_score", "topk_router_indices"})
def _softmax_topk_fwd(router_logits: torch.Tensor,
                     topk_router_score: torch.Tensor,
                     topk_router_indices: torch.Tensor,
                     E: int, K: int) -> None:

    if E <= 4096 and K <= 16 and E % 8 == 0:
        _topk_fwd(router_logits, K, topk_router_score, topk_router_indices, require_softmax_fusion=True)
    else:
        topk_results = router_logits.topk(K, dim=-1)
        topk_router_score.copy_(
            topk_results.values.softmax(dim=-1, dtype=torch.float32).to(topk_router_score.dtype)
        )
        topk_router_indices.copy_(
            topk_results.indices.to(topk_router_indices.dtype)
        )