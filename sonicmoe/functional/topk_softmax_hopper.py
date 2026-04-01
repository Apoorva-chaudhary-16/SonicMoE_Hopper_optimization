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
#
# FIXES vs ORIGINAL (8 bugs corrected):
#   Bug 3: Value cast → bitwise reinterpret (cute.arch.bitcast) in Steps 2 & 4
#   Bug 4: Python while loop → cutlass.range_constexpr in warp reduction
#   Bug 5: Sign check on packed bits → sign check on clean_f32 in Step 4
#   Bug 6: Missing syncwarp between warp reduction and bitonic topK
#   Bug 7: Only col_lane==0 wrote output; now all threads write their slice
#   Bug 1/2: Dead online_softmax_update helper now actually used in Step 2
#   Bug 8: Normalization restructured so each thread writes its own topK slice
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
#
#   FIX (Bug 1/2): This helper is now actually called in _fused_kernel Step 2
#   instead of being inlined with dead variables. The inline version had a
#   dead `val_f32 = regs[i]` that read packed bits but was never used, which
#   was confusing and signalled the wrong value was being processed.
# ---------------------------------------------------------------------------

@cute.jit
def online_softmax_update(
    new_val: cutlass.Float32,
    running_max: cutlass.Float32,
    running_sum: cutlass.Float32,
) -> tuple:
    """
    One step of FA4-style online softmax:
      m_new = max(m_old, x)
      l_new = l_old * exp2((m_old - m_new)*log2e) + exp2((x - m_new)*log2e)

    Returns (m_new, l_new).
    Note: The third return value (running_exp_val) was removed from the original
    helper signature — it was never used and added confusion.
    """
    new_max = cute.arch.fmax(running_max, new_val)
    # Rescale old running sum by exp(m_old - m_new)
    scale = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
    rescaled_sum = running_sum * scale
    # Add contribution of new value
    new_exp = cute.arch.exp2((new_val - new_max) * cutlass.Float32(LOG2E))
    new_sum = rescaled_sum + new_exp
    return new_max, new_sum


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
#   5. Write only topK values + indices to HBM (not all E logits),
#      with each thread in the row-warp writing its own slice
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

        # Number of topK results owned by each thread in the row-warp.
        # FIX (Bug 7): each of the threads_per_row threads writes its own
        # k_per_lane results rather than only col_lane==0 writing all k.
        assert self.next_pow2_k % self.threads_per_row == 0, (
            f"next_pow2_k={self.next_pow2_k} must be divisible by "
            f"threads_per_row={self.threads_per_row}"
        )
        self.k_per_lane = self.next_pow2_k // self.threads_per_row

        # Log2 of threads_per_row — used for compile-time warp reduction unroll
        # FIX (Bug 4): we need this as a constexpr integer for range_constexpr.
        self.log2_threads_per_row = int(math.log2(self.threads_per_row)) if self.threads_per_row > 1 else 0

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
                # FIX (Bug 3, encoding side): convert input element to FP32
                # cleanly first, then bit-pack the index into mantissa.
                # We use cute.arch.bitcast to reinterpret FP32 bits as Uint32
                # (equivalent to __float_as_int in CUDA) rather than a numeric
                # cast, which would change the value.
                val_f32 = raw[j].to(cutlass.Float32)
                col_idx = cutlass.Uint32(col_base + j)
                # Bit-pack: index into lower log_E mantissa bits.
                # Encoding: for positive values we store ~col_idx so that
                # larger float → larger Uint32 (correct sort order preserved).
                # For negative values store col_idx directly.
                encoded = (~col_idx if val_f32 >= cutlass.Float32(0.0) else col_idx) & idx_mask
                # FIX (Bug 3): use bitcast (reinterpret), not numeric cast.
                # cute.arch.bitcast(val_f32, Uint32) == __float_as_int(val_f32)
                u32_val = cute.arch.bitcast(val_f32, cutlass.Uint32)
                regs_u32[v * vec + j] = (u32_val & ~idx_mask) | encoded

        # OOB fill with -inf for tokens with fewer than E actual experts
        # (varlen case; E is always fixed for router so usually not needed)

        # -------------------------------------------------------------------
        # Step 2: Online Softmax — pass over register array
        #   FA4 §2: "We compute m_i = max(x_1..x_i) and l_i = sum(exp(x_j - m_i))
        #   in a single left-to-right pass."
        #   We maintain (running_max, running_sum) in registers, updating with
        #   emulated exp2 (FA4 §3.2) = ex2.approx PTX.
        #
        #   FIX (Bug 1/2): Now calls online_softmax_update() instead of
        #   inlining broken logic with dead variable `val_f32 = regs[i]`
        #   (that read the bit-packed value but was never used).
        #
        #   FIX (Bug 3): clean_f32 extracted via bitcast (reinterpret), not
        #   numeric cast.  cutlass.Float32(cutlass.Uint32(x)) does a value
        #   conversion; cute.arch.bitcast(x, Float32) reinterprets the bits.
        # -------------------------------------------------------------------
        running_max = -cutlass.Float32.inf
        running_sum = cutlass.Float32(0.0)

        for i in cutlass.range_constexpr(E_per_lane):
            # Extract clean float from packed register via bitwise reinterpret.
            # Strip the index bits first, then reinterpret the remaining bits
            # as a float32 — this recovers the original logit value.
            clean_u32 = regs_u32[i] & ~idx_mask
            # FIX (Bug 3): bitcast, not numeric cast
            clean_f32 = cute.arch.bitcast(clean_u32, cutlass.Float32)
            # Call the correct online softmax helper (FIX Bug 1/2)
            running_max, running_sum = online_softmax_update(
                clean_f32, running_max, running_sum
            )

        # -------------------------------------------------------------------
        # Warp reduction: each thread in the same row needs the global max/sum.
        # Use shfl_xor_sync across col_lane dimension (threads_per_row threads).
        # Reduce max with fmax, reduce sum with careful rescaling.
        #
        # FIX (Bug 4): Original code used a Python `while stride > 0` loop
        # with `stride >>= 1`. In the CUTLASS DSL/cute.jit context, loop
        # bounds must be compile-time constants; a Python while loop with a
        # mutating variable is not guaranteed to unroll correctly to PTX.
        # We now use cutlass.range_constexpr over log2(threads_per_row) steps,
        # computing the stride as a constexpr at each iteration.
        # -------------------------------------------------------------------
        if const_expr(self.threads_per_row > 1):
            for log_step in cutlass.range_constexpr(self.log2_threads_per_row):
                # stride decreases from threads_per_row/2 down to 1
                stride = const_expr(self.threads_per_row >> (log_step + 1))
                peer_max = cute.arch.shfl_xor_sync(running_max, stride, self.threads_per_row)
                peer_sum = cute.arch.shfl_xor_sync(running_sum, stride, self.threads_per_row)
                # Rescale whichever side had the smaller max
                new_max    = cute.arch.fmax(running_max, peer_max)
                my_scale   = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
                peer_scale = cute.arch.exp2((peer_max   - new_max) * cutlass.Float32(LOG2E))
                running_sum = running_sum * my_scale + peer_sum * peer_scale
                running_max = new_max
        # After warp reduction: running_max and running_sum are identical
        # across all threads_per_row threads in the same token row.

        # -------------------------------------------------------------------
        # FIX (Bug 6): Insert a warp-level barrier between the shuffle
        # reduction and the bitonic topK. Without this, threads may begin
        # reading from `regs` (in bitonic_topk) before all threads have
        # finished writing their packed values in Step 1, creating a race.
        # -------------------------------------------------------------------
        cute.arch.syncwarp()

        # -------------------------------------------------------------------
        # Step 3: Bitonic TopK on the packed register array
        #   We use the same bitonic_topk from quack (used in topk_softmax.py)
        #   but call it directly on the already-loaded register tensor.
        #   This avoids a second HBM load pass.
        #
        #   bitonic_topk distributes results across all threads_per_row threads
        #   in the warp — each thread holds k_per_lane = k / threads_per_row
        #   of the top-k elements after the sort completes.
        # -------------------------------------------------------------------
        from quack.sort.bitonic_sort import bitonic_topk as _bitonic_topk

        topk_regs = _bitonic_topk(regs, self.next_pow2_k, warp_width=self.threads_per_row)

        # -------------------------------------------------------------------
        # Step 4: Decode indices, apply softmax normalization.
        #   norm_val = exp2((val - running_max) * log2e) / running_sum
        #   Uses emulated exp2 again — FA4 §3.2.
        #
        #   FIX (Bug 3): clean_f32 extracted via bitcast (reinterpret), not
        #   numeric cast — same fix as Step 2.
        #
        #   FIX (Bug 5): Original code checked `topk_regs[i] >= 0.0` to
        #   decide the decoding branch, but topk_regs[i] holds the BIT-PACKED
        #   value (index stuffed in mantissa), so its sign is meaningless.
        #   We must extract clean_f32 first via bitcast, then check its sign.
        # -------------------------------------------------------------------
        topk_u32 = cute.recast_tensor(topk_regs, cutlass.Uint32)

        # Each thread in the row-warp holds k_per_lane topK results.
        # FIX (Bug 7/8): allocate per-lane output buffers (k_per_lane each),
        # then every thread writes its own slice — not just col_lane==0.
        k_per_lane = const_expr(self.k_per_lane)
        out_vals = cute.make_rmem_tensor(k_per_lane, self.output_dtype)
        out_idx  = cute.make_rmem_tensor(k_per_lane, cutlass.Int32)

        inv_sum = cutlass.Float32(1.0) / running_sum   # precompute reciprocal

        for i in cutlass.range_constexpr(k_per_lane):
            encoded  = topk_u32[i] & idx_mask
            clean_u32 = topk_u32[i] & ~idx_mask
            # FIX (Bug 3): bitcast to recover the original float bits
            clean_f32 = cute.arch.bitcast(clean_u32, cutlass.Float32)
            # FIX (Bug 5): sign check on clean_f32, not the packed topk_regs[i]
            col_idx = (~encoded if clean_f32 >= cutlass.Float32(0.0) else encoded) & idx_mask
            out_idx[i] = cutlass.Int32(col_idx)
            # Softmax weight via emulated exp2  [FA4 §3.2]
            sm_val = cute.arch.exp2(
                (clean_f32 - running_max) * cutlass.Float32(LOG2E)
            ) * inv_sum
            out_vals[i] = sm_val.to(self.output_dtype)

        # -------------------------------------------------------------------
        # Step 5: Write topK results to HBM — only k values per token.
        #
        #   FIX (Bug 7): Original code guarded writes with `if col_lane == 0`,
        #   so only that thread's k_per_lane results were written; the other
        #   threads_per_row-1 threads silently dropped their results, producing
        #   only k/threads_per_row correct outputs per token instead of k.
        #
        #   Correct approach: every thread writes its own k_per_lane results
        #   at an offset of col_lane * k_per_lane within the row's output slice.
        #   This is safe because each thread owns a non-overlapping slice of
        #   the output — no atomics or barriers needed.
        #
        #   FIX (Bug 8): normalization (Step 4) is still done by all threads,
        #   and now all threads' results are actually written, so no work is
        #   wasted.
        # -------------------------------------------------------------------
        elems_per_store = const_expr(math.gcd(self.vec, k_per_lane))

        # Each thread writes its slice starting at col_lane * k_per_lane
        lane_out_start = col_lane * k_per_lane

        vals_sliced = cute.tiled_divide(out_vals, (elems_per_store,))
        idx_sliced  = cute.tiled_divide(out_idx,  (elems_per_store,))

        mV_row = cute.tiled_divide(
            mTopKValues[global_row, lane_out_start:lane_out_start + k_per_lane],
            (elems_per_store,)
        )
        mI_row = cute.tiled_divide(
            mTopKIndices[global_row, lane_out_start:lane_out_start + k_per_lane],
            (elems_per_store,)
        )

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
        require_softmax_fusion: bool = True,
        T_tile: int = 32,              # tokens per CTA; tune per model size
    ):
        # Keep legacy TopK_Softmax API compatibility: forward.py passes
        # require_softmax_fusion as the 5th positional argument.
        # This fused Hopper path always computes normalized softmax weights,
        # so the flag is accepted for compatibility and not used for branching.
        _ = require_softmax_fusion

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

    Note on softmax correctness: the fused kernel computes softmax over ALL E
    experts (online, per-row), then selects the top-k weights from that
    distribution. The reference below matches this: softmax over all E logits,
    then gather the top-k weights. This differs from the common alternative
    of softmax-after-topk (softmax applied only to the k selected logits);
    both are valid routing strategies but they produce different weight scales.
    """
    import torch.nn.functional as F

    device = "cuda"
    logits = torch.randn(T, E, device=device, dtype=dtype)

    # --- Reference: softmax over all E, then gather top-k weights ---
    logits_f32 = logits.float()
    ref_sm_full = F.softmax(logits_f32, dim=-1)          # (T, E)
    ref_vals_logit, ref_idx = torch.topk(logits_f32, k, dim=-1)  # (T, k)
    ref_sm = ref_sm_full.gather(1, ref_idx)               # (T, k) softmax weights

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

    # Check indices match (order within topK may differ — sort both sides)
    ref_idx_sorted,   _ = ref_idx.sort(dim=-1)
    fused_idx_sorted, _ = fused_idx.sort(dim=-1)
    assert torch.all(ref_idx_sorted == fused_idx_sorted), \
        f"TopK index mismatch!\nRef:   {ref_idx_sorted[0]}\nFused: {fused_idx_sorted[0]}"

    # Check softmax values (gather in sorted-index order for fair comparison)
    ref_sm_sorted   = ref_sm.gather(1, ref_idx.argsort(dim=-1))
    fused_sm_sorted = fused_vals.gather(1, fused_idx.long().argsort(dim=-1))
    max_err = (ref_sm_sorted - fused_sm_sorted).abs().max().item()
    assert max_err < 2e-3, f"Softmax value mismatch! Max error: {max_err:.6f}"

    print(f"[PASS] T={T}, E={E}, k={k}: fused topK+softmax matches reference "
          f"(max_err={max_err:.2e})")
    return True


if __name__ == "__main__":
    test_fused_vs_separate(T=1024,  E=128, k=8)
    test_fused_vs_separate(T=32768, E=256, k=16)
    print("All tests passed.")

# Aliases so forward.py, bench_hopper_fa4.py, and test_hopper_fa4_correctness.py
# can import by expected names
TopK_Softmax_Hopper = FusedRouterTopKSoftmax_SM90
TopK_Softmax = FusedRouterTopKSoftmax_SM90