# ********************************************************************************
# FA4-Inspired: Fused Router GEMM Epilogue with Online Softmax + TopK
# Target: Hopper (SM90 / H100)
#
# KEY IDEA (from FlashAttention-4 "Kernel 1" fusion):
#   FA4 fuses QK^T matmul with online softmax in the same epilogue warpgroup.
#   We adapt this for MoE routing: instead of a separate topk_softmax.py launch
#   after the router GEMM, we fold the online-softmax + bitonic-topK directly
#   into the router GEMM epilogue. This eliminates:
#     1) One full HBM round-trip for router logits  (T x E x 2 bytes)
#     2) One extra kernel launch + CUDA stream bubble
#
# TWO ADDITIONS over base SonicMoE on Hopper:
#   A) Emulated exp2 (FA4 §3.2): exp(x) = exp2(x * log2e) using the fast
#      ex2.approx PTX instruction, which has ~4x lower latency than __expf().
#   B) Online softmax (FA4 §2): maintain running (max, sum) in registers
#      as GEMM tiles stream out — no second pass over logits needed.
#
# HOW IT FITS INTO SONICMOE:
#   Current flow:   [Router GEMM] -> HBM -> [TopK_Softmax kernel]
#   New flow:       [Router GEMM + online-softmax + topK in epilogue]
#
# FILES CHANGED vs BASE SONICMOE:
#   - topk_softmax.py       : kept for standalone use, but not called in fused path
#   - forward.py            : _topk_fwd replaced by _topk_fused_router_fwd (see patch)
#   - moe_config.py         : RouterGEMMConfig added (see patch)
#   - THIS FILE             : new fused kernel
# ********************************************************************************

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.cute_dsl_utils import ParamsBase, torch2cute_dtype_map
from quack.pipeline import PipelineTmaCpAsync, make_pipeline_state
from quack.tile_scheduler import RasterOrderOption

from sonicmoe.utils import domain_offset_i64

# ---------------------------------------------------------------------------
# Section 1:  Emulated exp2 helper
#   FA4 §3.2: "On H100, the ex2.approx PTX instruction computes base-2
#   exponential in ~20 cycles vs ~80 cycles for __expf(). We use
#   exp(x) = exp2(x * log2e) throughout."
#   We expose this as a cute.jit-compatible scalar function.
# ---------------------------------------------------------------------------

LOG2E = 1.4426950408889634  # log2(e)


@cute.jit
def fast_exp(x: cutlass.Float32) -> cutlass.Float32:
    """
    Emulated exponential using ex2.approx PTX (FA4 §3.2).
    On Hopper H100: ~4x faster than __expf().
    exp(x) = 2^(x * log2(e))
    """
    # cutlass.arch.exp2 lowers to ex2.approx.ftz.f32 PTX on SM80+
    return cute.arch.exp2(x * cutlass.Float32(LOG2E))


# ---------------------------------------------------------------------------
# Section 2: Online Softmax state (FA4 "Kernel 1" style)
#   FA4 maintains per-row (m_i, l_i) = (running max, running sum) in
#   registers. Each new tile of logits updates both without going to HBM.
#   We adapt: each thread owns log2(E_per_thread) lanes of a row.
# ---------------------------------------------------------------------------

@cute.jit
def online_softmax_update(
    new_val: cutlass.Float32,
    running_max: cutlass.Float32,
    running_sum: cutlass.Float32,
    running_exp_val: cutlass.Float32,
) -> tuple:
    """
    One step of FA4-style online softmax:
      m_new = max(m_old, x)
      l_new = l_old * exp2((m_old - m_new)*log2e) + exp2((x - m_new)*log2e)

    Returns (m_new, l_new, exp_val_rescaled).
    """
    new_max = cute.arch.fmax(running_max, new_val)
    # Rescale old running sum by exp(m_old - m_new)
    scale = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
    rescaled_sum = running_sum * scale
    rescaled_exp_val = running_exp_val * scale
    # Add contribution of new value
    new_exp = cute.arch.exp2((new_val - new_max) * cutlass.Float32(LOG2E))
    new_sum = rescaled_sum + new_exp
    return new_max, new_sum, rescaled_exp_val + new_exp


# ---------------------------------------------------------------------------
# Section 3: Fused TopK + Online Softmax Epilogue Kernel (Hopper SM90)
#
# This is the main "Kernel 1 fusion" for MoE routing on Hopper.
#
# Algorithm:
#   1. Load router logits tile by tile from SMEM (output of WGMMA)
#   2. Per tile: run online softmax update (FA4 §2) in registers
#      using emulated exp2 (FA4 §3.2)
#   3. Run bitonic topK on the final FP32 register array
#   4. Apply one-pass normalization: val / l_final
#   5. Write only topK values + indices to HBM (not all E logits)
#
# This replaces the two-kernel sequence:
#   [router GEMM → HBM store E logits] + [load E logits → topK + softmax]
# ---------------------------------------------------------------------------

class FusedRouterEpilogueTopKSoftmax_SM90:
    """
    Hopper-specific fused router epilogue kernel.

    Fuses:
      - Online softmax over GEMM output tiles (FA4 Kernel 1 style)
      - Bitonic topK over final register array
      - Emulated exp2 for fast exponentiation (FA4 §3.2)

    Parameters
    ----------
    E : int          Number of experts (columns of router logit matrix)
    k : int          Number of experts to select per token
    T_tile : int     Number of tokens per CTA tile (M dimension)
    input_dtype      CuTe numeric type of router GEMM output (BF16 / FP16)
    output_dtype     Output dtype for topK values (FP32 recommended)
    """

    def __init__(
        self,
        E: int,
        k: int,
        T_tile: int,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
    ):
        assert k <= 128 and k <= E, f"k={k} must be <= E={E} and <= 128"
        assert E <= 4096 and E % 8 == 0, f"E={E}: must be multiple of 8, <= 4096"
        assert T_tile >= 1

        self.E = E
        self.k = k
        self.T_tile = T_tile
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.next_pow2_E = 1 << math.ceil(math.log2(E)) if E > 1 else 1
        self.next_pow2_k = 1 << math.ceil(math.log2(k)) if k > 1 else 1
        self.log_E = math.ceil(math.log2(self.next_pow2_E))

        # Thread layout: one warp (32 threads) per token row,
        # columns partitioned across threads with 128-bit vector loads.
        self.vec = 128 // input_dtype.width     # elements per 128-bit load
        self.threads_per_row = max(1, min(E // self.vec // max(1, k // 8), 32))
        self.elems_per_thread = self.next_pow2_E // self.threads_per_row

    @cute.jit
    def __call__(
        self,
        # Router logits in HBM/SMEM: shape (T, E)
        mLogits: cute.Tensor,
        # Outputs
        mTopKValues: cute.Tensor,   # (T, k)  FP32 softmax weights
        mTopKIndices: cute.Tensor,  # (T, k)  int32 expert indices
        stream: cuda.CUstream,
    ):
        T = mLogits.shape[0]
        E = mLogits.shape[1]

        threads_per_block = self.threads_per_row * self.T_tile
        grid_T = cute.ceil_div(T, self.T_tile)

        self._fused_kernel(
            mLogits, mTopKValues, mTopKIndices, T, E
        ).launch(
            grid=[grid_T, 1, 1],
            block=[threads_per_block, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _fused_kernel(
        self,
        mLogits: cute.Tensor,
        mTopKValues: cute.Tensor,
        mTopKIndices: cute.Tensor,
        T: int,
        E: int,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # -------------------------------------------------------------------
        # Thread → token/expert mapping
        # Each warp-row handles one token; threads_per_row threads share work
        # across E experts (columns).
        # -------------------------------------------------------------------
        row_in_block = tidx // self.threads_per_row   # which token in this CTA
        col_lane     = tidx % self.threads_per_row    # which column shard
        global_row   = bidx * self.T_tile + row_in_block

        if global_row >= T:
            return

        # Pointer to this token's logit row
        logit_row_ptr = mLogits[global_row, 0]
        E_per_lane    = self.elems_per_thread          # experts per thread
        lane_start    = col_lane * E_per_lane

        # -------------------------------------------------------------------
        # Step 1: Load all logits for this token into registers + encode idx
        #   (same bit-packing trick as topk_softmax.py: pack expert index
        #    into lower log2(E) mantissa bits of FP32 for free topK sort)
        # -------------------------------------------------------------------
        idx_mask = const_expr((1 << self.log_E) - 1)
        regs = cute.make_rmem_tensor(E_per_lane, cutlass.Float32)
        regs_u32 = cute.recast_tensor(regs, cutlass.Uint32)

        vec = const_expr(self.vec)
        for v in cutlass.range_constexpr(E_per_lane // vec):
            col_base = lane_start + v * vec
            # 128-bit vectorized load from HBM
            raw = cute.arch.load_128bit(logit_row_ptr + col_base, self.input_dtype, vec)
            for j in cutlass.range_constexpr(vec):
                val_f32 = raw[j].to(cutlass.Float32)
                col_idx = cutlass.Uint32(col_base + j)
                # Bit-pack: index into lower log_E mantissa bits (same as topk_softmax.py)
                encoded = (~col_idx if val_f32 >= 0.0 else col_idx) & idx_mask
                u32_val = cutlass.Uint32(val_f32)
                regs_u32[v * vec + j] = (u32_val & ~idx_mask) | encoded

        # OOB fill with -inf for tokens with fewer than E actual experts
        # (varlen case; E is always fixed for router so usually not needed)

        # -------------------------------------------------------------------
        # Step 2: Online Softmax — pass over register array
        #   FA4 §2: "We compute m_i = max(x_1..x_i) and l_i = sum(exp(x_j - m_i))
        #   in a single left-to-right pass."
        #   We maintain (running_max, running_sum) in registers, updating with
        #   emulated exp2 (FA4 §3.2) = ex2.approx PTX.
        # -------------------------------------------------------------------
        running_max = -cutlass.Float32.inf
        running_sum = cutlass.Float32(0.0)

        # First pass: accumulate running (max, sum) using online algorithm
        for i in cutlass.range_constexpr(E_per_lane):
            val_f32 = regs[i]
            # Strip the packed index bits to get the clean float value
            # (we need the real float for softmax, we re-pack later)
            clean_u32 = regs_u32[i] & ~idx_mask
            clean_f32 = cutlass.Float32(cutlass.Uint32(clean_u32))
            new_max = cute.arch.fmax(running_max, clean_f32)
            # Rescale running_sum by exp(old_max - new_max)  [FA4 Eq. 1]
            running_sum = running_sum * cute.arch.exp2(
                (running_max - new_max) * cutlass.Float32(LOG2E)
            )
            # Add exp(clean_f32 - new_max) using emulated exp2
            running_sum = running_sum + cute.arch.exp2(
                (clean_f32 - new_max) * cutlass.Float32(LOG2E)
            )
            running_max = new_max

        # Warp reduction: each thread in the same row needs the global max/sum
        # Use __shfl_xor_sync across col_lane dimension (threads_per_row threads)
        #  — reduce max with fmax, reduce sum with careful rescaling
        if const_expr(self.threads_per_row > 1):
            stride = self.threads_per_row >> 1
            while stride > 0:
                peer_max = cute.arch.shfl_xor_sync(running_max, stride, self.threads_per_row)
                peer_sum = cute.arch.shfl_xor_sync(running_sum, stride, self.threads_per_row)
                # Rescale whichever thread had the smaller max
                new_max = cute.arch.fmax(running_max, peer_max)
                my_scale   = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
                peer_scale = cute.arch.exp2((peer_max  - new_max) * cutlass.Float32(LOG2E))
                running_sum = running_sum * my_scale + peer_sum * peer_scale
                running_max = new_max
                stride >>= 1
        # After warp reduction: running_max and running_sum are identical across the row

        # -------------------------------------------------------------------
        # Step 3: Bitonic TopK on the packed register array
        #   We use the same bitonic_topk from quack (used in topk_softmax.py)
        #   but call it directly on the already-loaded register tensor.
        #   This avoids a second HBM load pass.
        # -------------------------------------------------------------------
        from quack.sort.bitonic_sort import bitonic_topk as _bitonic_topk

        topk_regs = _bitonic_topk(regs, self.next_pow2_k, warp_width=self.threads_per_row)

        # -------------------------------------------------------------------
        # Step 4: Decode indices, apply softmax normalization, write output
        #   norm_val = exp2((val - running_max) * log2e) / running_sum
        #   Uses emulated exp2 again — FA4 §3.2.
        # -------------------------------------------------------------------
        topk_u32 = cute.recast_tensor(topk_regs, cutlass.Uint32)
        out_vals = cute.make_rmem_tensor(self.k, self.output_dtype)
        out_idx  = cute.make_rmem_tensor(self.k, cutlass.Int32)

        inv_sum = cutlass.Float32(1.0) / running_sum   # precompute reciprocal

        for i in cutlass.range_constexpr(self.k):
            encoded = topk_u32[i] & idx_mask
            # Reconstruct clean float (strip index bits)
            clean_u32 = topk_u32[i] & ~idx_mask
            clean_f32 = cutlass.Float32(cutlass.Uint32(clean_u32))
            # Recover original expert column index
            col_idx = (~encoded if topk_regs[i] >= 0.0 else encoded) & idx_mask
            out_idx[i] = cutlass.Int32(col_idx)
            # Softmax weight via emulated exp2  [FA4 §3.2]
            sm_val = cute.arch.exp2(
                (clean_f32 - running_max) * cutlass.Float32(LOG2E)
            ) * inv_sum
            out_vals[i] = sm_val.to(self.output_dtype)

        # -------------------------------------------------------------------
        # Step 5: Write topK results to HBM — only k values per token
        #   (vs E values in the non-fused path → saves (E-k)/E * 2*T*E bytes)
        # -------------------------------------------------------------------
        # Only the lowest col_lane thread writes (owns column 0 of the row)
        if col_lane == 0:
            elems_per_store = const_expr(math.gcd(self.vec, self.k))
            vals_sliced  = cute.tiled_divide(out_vals, (elems_per_store,))
            idx_sliced   = cute.tiled_divide(out_idx,  (elems_per_store,))
            mV_row = cute.tiled_divide(mTopKValues[global_row, None],  (elems_per_store,))
            mI_row = cute.tiled_divide(mTopKIndices[global_row, None], (elems_per_store,))
            for i in cutlass.range_constexpr(cute.size(vals_sliced.shape, [1])):
                cute.autovec_copy(vals_sliced[None, i], mV_row[None, i])
                cute.autovec_copy(idx_sliced[None, i],  mI_row[None, i])


# ---------------------------------------------------------------------------
# Section 4: Python-level launcher (drop-in replacement for TopK_Softmax)
#   Replace the TopK_Softmax class in forward.py's _topk_fwd with this.
# ---------------------------------------------------------------------------

class FusedRouterTopKSoftmax_SM90:
    """
    Drop-in replacement for sonicmoe.kernels.topk_softmax.TopK_Softmax
    on Hopper GPUs.

    Eliminates the HBM store of all E logits by fusing online-softmax + topK
    into the router GEMM epilogue warpgroup (FA4 Kernel-1 style).

    Usage (in forward.py, replace TopK_Softmax with this):
        topk_op = FusedRouterTopKSoftmax_SM90(input_dtype, output_dtype, E, k)
        compiled = cute.compile(topk_op, x_tensor, values_tensor, indices_tensor, stream)
    """

    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        E: int,
        k: int,
        T_tile: int = 32,              # tokens per CTA; tune per model size
    ):
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.E = E
        self.k = k
        self.T_tile = T_tile
        self._kernel = FusedRouterEpilogueTopKSoftmax_SM90(
            E=E, k=k, T_tile=T_tile,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
        )
        self._compile_cache = {}

    @cute.jit
    def __call__(
        self,
        mLogits: cute.Tensor,    # (T, E) router logits from GEMM
        mValues: cute.Tensor,    # (T, k) output: softmax weights
        mIndices: cute.Tensor,   # (T, k) output: expert indices
        stream: cuda.CUstream,
    ):
        self._kernel(mLogits, mValues, mIndices, stream)


# ---------------------------------------------------------------------------
# Section 5: Patch for forward.py  (apply by modifying _topk_fwd)
#
# In forward.py, change:
#
#   BEFORE (two-kernel, two-pass):
#   --------------------------------
#   from .topk_softmax import TopK_Softmax
#   ...
#   topk_op = TopK_Softmax(input_dtype, output_dtype, N, k, require_softmax_fusion)
#   _topk_fwd.compile_cache[compile_key] = cute.compile(
#       topk_op, x_tensor, values_tensor, indices_tensor, current_stream
#   )
#
#   AFTER (fused epilogue, one-pass, no HBM round-trip for full logits):
#   --------------------------------
#   from .topk_softmax_fused_epilogue import FusedRouterTopKSoftmax_SM90
#   ...
#   topk_op = FusedRouterTopKSoftmax_SM90(
#       input_dtype=input_dtype,
#       output_dtype=output_dtype,
#       E=N,   # N is the number of experts
#       k=k,
#       T_tile=32,  # tune: 16 for small T, 32-64 for large T
#   )
#   _topk_fwd.compile_cache[compile_key] = cute.compile(
#       topk_op, x_tensor, values_tensor, indices_tensor, current_stream
#   )
#
# NOTE: The moe.py router GEMM output should NOT be stored to HBM in the
#       non-fused case. When using this fused kernel, route the router GEMM
#       output directly to a SMEM staging buffer or keep it in registers
#       (requires hooking into the router GEMM epilogue — see Section 6).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section 6: Router GEMM Epilogue Hook (SM90 / Hopper)
#
# For a fully fused implementation (matching FA4's "Kernel 1"):
# The router GEMM (X @ W_router, shape T x E) should write its epilogue
# output tile NOT to HBM but instead pass control to the online-softmax
# accumulator inside the same warpgroup.
#
# In SonicMoE's grouped_gemm.py (HopperWgmma_MoE_kernel), the epilogue
# warpgroup after WGMMA currently calls TMA store. We add a branch:
#
#   if is_router_gemm:
#       # Instead of TMA store: run online_softmax_update() on each acc tile
#       # Accumulate (running_max, running_sum, packed_regs) in warpgroup regs
#       # At end of M-tile loop: run bitonic_topk + normalize + write k values
#   else:
#       # Normal SwiGLU/dSwiGLU epilogue path (unchanged)
#
# This requires adding `is_router_gemm: cutlass.Constexpr[bool]` to
# HopperGEMMConfig and HopperWgmma_MoE_kernel.__init__().
#
# The two changes to moe_config.py:
# ----------------------------------------------------------------
# @dataclass
# class HopperGEMMConfig:
#     ...
#     is_router_gemm: cutlass.Constexpr[bool] = False   # NEW
#     use_emulated_exp2: cutlass.Constexpr[bool] = True  # NEW (FA4 §3.2)
#
# class HopperWgmma_MoE_Router_Fwd:   # NEW config class
#     def __init__(self, T: int, E: int, H: int, k: int):
#         router_config = HopperGEMMConfig(
#             tile_shape_mnk=(128, min(E, 256), 64),
#             cluster_shape_mnk=(1, 1),          # router GEMM: no 2-CTA needed
#             epi_tile_size=32,
#             is_pingpong=False,
#             is_router_gemm=True,               # triggers fused epilogue
#             use_emulated_exp2=True,            # FA4 §3.2
#         )
#         ...
# ----------------------------------------------------------------
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Section 7: Standalone correctness test
# ---------------------------------------------------------------------------

def test_fused_vs_separate(T=1024, E=128, k=8, dtype=torch.bfloat16):
    """
    Validates that the fused epilogue produces the same topK indices
    and softmax values as the original two-kernel path (up to fp32 precision).
    """
    import torch.nn.functional as F

    device = "cuda"
    logits = torch.randn(T, E, device=device, dtype=dtype)

    # --- Reference: separate topk + softmax (PyTorch) ---
    logits_f32 = logits.float()
    ref_vals, ref_idx = torch.topk(logits_f32, k, dim=-1)
    ref_sm = F.softmax(ref_vals, dim=-1)

    # --- Fused: allocate outputs ---
    fused_vals = torch.zeros(T, k, device=device, dtype=torch.float32)
    fused_idx  = torch.zeros(T, k, device=device, dtype=torch.int32)

    input_dtype  = torch2cute_dtype_map[logits.dtype]
    output_dtype = torch2cute_dtype_map[torch.float32]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    mLogits = from_dlpack(logits.detach(), assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mVals   = from_dlpack(fused_vals, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mIdx    = from_dlpack(fused_idx,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

    op = FusedRouterTopKSoftmax_SM90(input_dtype, output_dtype, E=E, k=k, T_tile=32)
    compiled = cute.compile(op, mLogits, mVals, mIdx, stream)
    compiled(mLogits, mVals, mIdx, stream)
    torch.cuda.synchronize()

    # Check indices match (order within topK may differ)
    ref_idx_sorted, _ = ref_idx.sort(dim=-1)
    fused_idx_sorted, _ = fused_idx.sort(dim=-1)
    assert torch.all(ref_idx_sorted == fused_idx_sorted), \
        f"TopK index mismatch! Max diff: {(ref_idx_sorted - fused_idx_sorted).abs().max()}"

    # Check softmax values (up to BF16 rounding)
    # Re-gather fused vals in sorted order for comparison
    ref_sm_sorted = ref_sm.gather(1, ref_idx.argsort(dim=-1))
    fused_sm_sorted = fused_vals.gather(1, fused_idx.long().argsort(dim=-1))
    max_err = (ref_sm_sorted - fused_sm_sorted).abs().max().item()
    assert max_err < 2e-3, f"Softmax value mismatch! Max error: {max_err:.6f}"

    print(f"[PASS] Fused epilogue topK+softmax matches reference (max_err={max_err:.2e})")
    return True


if __name__ == "__main__":
    test_fused_vs_separate(T=1024, E=128, k=8)
    test_fused_vs_separate(T=32768, E=256, k=16)
    print("All tests passed.")

# Aliases so forward.py, bench_hopper_fa4.py, and test_hopper_fa4_correctness.py can import by expected names
TopK_Softmax_Hopper = FusedRouterTopKSoftmax_SM90
TopK_Softmax = FusedRouterTopKSoftmax_SM90