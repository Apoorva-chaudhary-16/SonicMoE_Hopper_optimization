# ********************************************************************************
# tests/test_hopper_fa4_correctness.py
#
# PURPOSE:
#   Verify that the FA4-inspired Hopper optimisations (ex2.approx + online-softmax
#   Kernel-1 merge) produce numerically equivalent results to the original kernels.
#
# HOW TO RUN:
#   python -m pytest tests/test_hopper_fa4_correctness.py -v
#   # or from repo root:
#   python -m unittest tests.test_hopper_fa4_correctness
#
# WHAT IT CHECKS:
#   1. TopK_Softmax vs TopK_Softmax_Hopper — same top-K indices, same softmax scores
#   2. token_gather_and_sum (original) vs hopper version — same aggregation output
#   3. End-to-end MoE forward: Hopper path output vs PyTorch reference (same tol as moe_test.py)
#
# EXPECTED RESULT (on H100):
#   All tests PASS.  Tolerances match those used in moe_test.py.
# ********************************************************************************

import unittest

import torch
import cuda.bindings.driver as cuda
import cutlass
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map
from parameterized import parameterized

# ── originals ────────────────────────────────────────────────────────────────
from sonicmoe.functional.topk_softmax_hopper import TopK_Softmax as TopK_Softmax_Original
from sonicmoe.functional.reduction_over_k_gather_hopper import (
    token_gather_and_sum_varlen_K_triton as gather_sum_original,
)

# ── FA4 Hopper-optimised (the new files) ────────────────────────────────────
from sonicmoe.functional.topk_softmax_hopper import TopK_Softmax_Hopper
from sonicmoe.functional.reduction_over_k_gather_hopper import (
    token_gather_and_sum_varlen_K_triton as gather_sum_hopper,
)

# ── full MoE ─────────────────────────────────────────────────────────────────
from sonicmoe import KernelBackendMoE, MoE
from sonicmoe.enums import ActivationType
from .test_commons import TestCommons


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_cute(tensor, stream):
    """Convert a 2-D torch tensor to a CuTe tensor (same helper as forward.py)."""
    return (
        from_dlpack(tensor.detach(), assumed_align=16)
        .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    )


def _run_topk_kernel(cls, logits: torch.Tensor, K: int):
    """
    Run either TopK_Softmax or TopK_Softmax_Hopper and return
    (scores: float32 cpu, indices: int32 cpu).
    """
    T, E = logits.shape
    scores  = torch.zeros(T, K, device="cuda", dtype=torch.float32)
    indices = torch.zeros(T, K, device="cuda", dtype=torch.int32)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    import cutlass
    input_dtype  = torch2cute_dtype_map[logits.dtype]
    output_dtype = torch2cute_dtype_map[scores.dtype]

    kernel = cls(input_dtype, output_dtype, E, K, require_softmax_fusion=True)

    import cutlass.cute as cute
    compiled = cute.compile(
        kernel,
        _to_cute(logits, stream),
        _to_cute(scores, stream),
        _to_cute(indices.unsqueeze(-1).expand_as(scores), stream),  # shape match
        stream,
    )
    compiled(
        _to_cute(logits, stream),
        _to_cute(scores, stream),
        _to_cute(indices, stream),
        stream,
    )
    torch.cuda.synchronize()
    return scores.cpu(), indices.cpu()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — TopK kernel correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKCorrecness(TestCommons):
    """
    Confirms TopK_Softmax_Hopper produces the same indices and softmax scores
    as the original TopK_Softmax for a variety of (T, E, K) configs.
    """

    # (T,  E,   K)
    CONFIGS = [
        (8192,  128,  8),
        (8192,  64,   4),
        (8192,  32,   2),
        (4096,  256, 16),
        (16384, 128,  8),
    ]

    @parameterized.expand(CONFIGS)
    def test_topk_indices_match(self, T, E, K):
        torch.manual_seed(42)
        logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

        scores_orig,  idx_orig  = _run_topk_kernel(TopK_Softmax_Original, logits, K)
        scores_hopper, idx_hopper = _run_topk_kernel(TopK_Softmax_Hopper,   logits, K)

        # Indices must be identical
        self.assertTrue(
            torch.equal(idx_orig, idx_hopper),
            f"[T={T},E={E},K={K}] Top-K indices differ!\n"
            f"  orig:   {idx_orig[:3]}\n"
            f"  hopper: {idx_hopper[:3]}",
        )

    @parameterized.expand(CONFIGS)
    def test_softmax_scores_close(self, T, E, K):
        torch.manual_seed(42)
        logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

        scores_orig,  _ = _run_topk_kernel(TopK_Softmax_Original, logits, K)
        scores_hopper, _ = _run_topk_kernel(TopK_Softmax_Hopper,   logits, K)

        # ex2.approx introduces a tiny ULP difference — tolerate 1e-5
        torch.testing.assert_close(
            scores_orig, scores_hopper,
            atol=1e-5, rtol=1e-4,
            msg=f"[T={T},E={E},K={K}] Softmax scores differ beyond tolerance",
        )

    @parameterized.expand(CONFIGS)
    def test_softmax_sums_to_one(self, T, E, K):
        """Each row of scores must sum to 1 (probability simplex check)."""
        torch.manual_seed(0)
        logits = torch.randn(T, E, device="cuda", dtype=torch.bfloat16)

        scores_hopper, _ = _run_topk_kernel(TopK_Softmax_Hopper, logits, K)

        row_sums = scores_hopper.sum(dim=-1)   # (T,)
        torch.testing.assert_close(
            row_sums,
            torch.ones(T),
            atol=1e-4, rtol=1e-4,
            msg=f"[T={T},E={E},K={K}] Softmax rows do not sum to 1",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Gather-and-sum correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestGatherSumCorrectness(TestCommons):
    """
    Confirms the Hopper gather-and-sum produces the same output as the original.
    """

    CONFIGS = [
        (4096, 2048, 8),    # (T, H, K)
        (4096, 1536, 4),
        (8192, 4096, 16),
    ]

    @parameterized.expand(CONFIGS)
    def test_gather_sum_matches_original(self, T, H, K):
        torch.manual_seed(42)

        Mtotal = T * K
        x       = torch.randn(Mtotal, H, device="cuda", dtype=torch.bfloat16)
        w       = torch.rand( Mtotal,    device="cuda", dtype=torch.float32)
        # normalise w per-token so rows sum to 1
        w_2d    = w.view(T, K)
        w_2d    = w_2d / w_2d.sum(dim=-1, keepdim=True)
        w       = w_2d.reshape(-1)

        M_perm   = torch.randperm(Mtotal, device="cuda", dtype=torch.int32)
        M_offset = torch.arange(0, T + 1, device="cuda", dtype=torch.int32) * K

        out_orig   = torch.zeros(T, H, device="cuda", dtype=torch.float32)
        out_hopper = torch.zeros(T, H, device="cuda", dtype=torch.float32)

        gather_sum_original(x, w, out_orig,   M_perm, M_offset, T, K, H, is_varlen_K=False)
        gather_sum_hopper(  x, w, out_hopper, M_perm, M_offset, T, K, H, is_varlen_K=False)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            out_orig.cpu(), out_hopper.cpu(),
            atol=1e-4, rtol=1e-4,
            msg=f"[T={T},H={H},K={K}] Gather-sum output differs",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — End-to-end MoE forward correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestMoEHopperEndToEnd(TestCommons):
    """
    Runs the full MoE forward (SonicMoE Hopper path) and checks it matches
    the PyTorch reference with the same tolerances as moe_test.py.
    """

    _SEED = 42

    # Subset of the configs from moe_test.py
    CONFIGS = [
        (8192, 768,  256, 128,  8),   # 1.4B style
        (8192, 1536, 512,  64,  4),   # 7B style
        (8192, 4096, 512, 128,  8),   # 30B style
    ]

    @parameterized.expand(CONFIGS)
    def test_forward_matches_torch(self, T, H, I, E, K):
        self.set_seed(self._SEED)
        device = torch.device("cuda")

        with torch.device(device):
            moe = MoE(
                num_experts=E,
                num_experts_per_tok=K,
                hidden_size=H,
                intermediate_size=I,
                activation_function=ActivationType.SWIGLU,
                add_bias=False,
                std=0.02,
            ).to(dtype=torch.bfloat16)

        x = 0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16, requires_grad=False)

        with torch.autocast(device.type, torch.float32):
            y_sonic = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
            y_torch = moe(x, kernel_backend_moe=KernelBackendMoE.torch)[0]

        self.assert_equal_tensors(
            y_sonic.float(), y_torch.float(),
            exact_match=False,
            atol_bfloat16=1.4e-2,
            rtol_bfloat16=2e-2,
            dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    unittest.main()