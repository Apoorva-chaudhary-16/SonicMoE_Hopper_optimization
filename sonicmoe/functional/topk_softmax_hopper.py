# ********************************************************************************
# FA4-Inspired: Fused Router GEMM Epilogue with Online Softmax + TopK
# Target: Hopper (SM90 / H100)
#
# FIXES APPLIED (cumulative — all previous + new round):
#   Bug 1/2: Dead online_softmax_update → now called correctly in Step 2
#   Bug 3:   Value cast → bitwise reinterpret via recast_tensor pair (NOT
#            cute.arch.bitcast which does not exist in CUTLASS DSL 4.4.0)
#   Bug 4:   Python while loop → cutlass.range_constexpr in warp reduction
#   Bug 5:   Sign check on packed bits → sign check on clean_f32 in Step 4
#   Bug 6:   Missing syncwarp between warp reduction and bitonic topK
#   Bug 7:   Only col_lane==0 wrote output → all threads write their slice
#   Bug 8:   Normalization restructured so each thread writes its own slice
#   Issue 1: Removed unused/potentially circular `domain_offset_i64` import
#   Issue 6: Replaced non-existent cute.arch.bitcast with safe recast_tensor
#            pair pattern (regs/regs_u32 already share memory — write masked
#            u32 back, read as float, restore original value)
#   Issue 7: Added missing warp-mask (0xFFFFFFFF) to shfl_xor_sync calls
# ********************************************************************************

import math
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map

# NOTE: domain_offset_i64 import removed — it was imported but never used in
# this file, and the absolute import path caused potential circular-import
# issues when sonicmoe.functional modules are loaded during package init.

# ---------------------------------------------------------------------------
# Section 1: Emulated exp2 helper
#   FA4 §3.2: exp(x) = exp2(x * log2e) using ex2.approx PTX instruction.
#   ~4x lower latency than __expf() on Hopper H100.
# ---------------------------------------------------------------------------

LOG2E = 1.4426950408889634  # log2(e)

# Full warp mask — all 32 lanes active.
# Required by shfl_xor_sync as the first argument (same as 0xFFFFFFFF in CUDA C).
_FULL_WARP_MASK = const_expr(0xFFFFFFFF)


@cute.jit
def fast_exp(x: cutlass.Float32) -> cutlass.Float32:
    """
    Emulated exponential using ex2.approx PTX (FA4 §3.2).
    On Hopper H100: ~4x faster than __expf().
    exp(x) = 2^(x * log2(e))
    """
    return cute.arch.exp2(x * cutlass.Float32(LOG2E))


# ---------------------------------------------------------------------------
# Section 2: Online Softmax helper (FA4 "Kernel 1" style)
#
#   FIX (Bug 1/2): This function is now actually called in Step 2 of the
#   kernel, replacing the dead-variable inline version from the original code.
#
#   The original helper had a spurious third return value `running_exp_val`
#   that was never used. It has been removed to avoid confusion.
# ---------------------------------------------------------------------------

@cute.jit
def online_softmax_update(
    new_val: cutlass.Float32,
    running_max: cutlass.Float32,
    running_sum: cutlass.Float32,
) -> tuple:
    """
    One step of FA4-style online softmax (FA4 §2):
      m_new = max(m_old, x)
      l_new = l_old * exp2((m_old - m_new)*log2e) + exp2((x - m_new)*log2e)

    Returns (m_new, l_new).
    Uses emulated exp2 (FA4 §3.2) throughout.
    """
    new_max = cute.arch.fmax(running_max, new_val)
    scale = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
    rescaled_sum = running_sum * scale
    new_exp = cute.arch.exp2((new_val - new_max) * cutlass.Float32(LOG2E))
    new_sum = rescaled_sum + new_exp
    return new_max, new_sum


# ---------------------------------------------------------------------------
# Section 3: Fused TopK + Online Softmax Epilogue Kernel (Hopper SM90)
# ---------------------------------------------------------------------------

class FusedRouterEpilogueTopKSoftmax_SM90:
    """
    Hopper-specific fused router epilogue kernel.

    Fuses online softmax + bitonic topK + emulated exp2 (FA4 §3.2) into a
    single register-resident pass with no HBM round-trip for all E logits.
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

        self.vec = 128 // input_dtype.width
        self.threads_per_row = max(1, min(E // self.vec // max(1, k // 8), 32))
        self.elems_per_thread = self.next_pow2_E // self.threads_per_row

        # FIX (Bug 7): each thread owns k_per_lane results, not just col_lane==0.
        assert self.next_pow2_k % self.threads_per_row == 0, (
            f"next_pow2_k={self.next_pow2_k} must be divisible by "
            f"threads_per_row={self.threads_per_row}"
        )
        self.k_per_lane = self.next_pow2_k // self.threads_per_row

        # FIX (Bug 4): precompute log2(threads_per_row) as a Python int so it
        # can be used as the bound of cutlass.range_constexpr below.
        self.log2_threads_per_row = (
            int(math.log2(self.threads_per_row)) if self.threads_per_row > 1 else 0
        )

    @cute.jit
    def __call__(
        self,
        mLogits: cute.Tensor,       # (T, E) BF16/FP16 router logits
        mTopKValues: cute.Tensor,   # (T, k) FP32 output softmax weights
        mTopKIndices: cute.Tensor,  # (T, k) int32 output expert indices
        stream: cuda.CUstream,
    ):
        T = mLogits.shape[0]
        E = mLogits.shape[1]

        threads_per_block = self.threads_per_row * self.T_tile
        grid_T = cute.ceil_div(T, self.T_tile)

        self._fused_kernel(mLogits, mTopKValues, mTopKIndices, T, E).launch(
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
        T,
        E,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        row_in_block = tidx // self.threads_per_row
        col_lane     = tidx % self.threads_per_row
        global_row   = bidx * self.T_tile + row_in_block

        if global_row < T:

            logit_row_ptr = mLogits[global_row, 0]
            E_per_lane    = self.elems_per_thread
            lane_start    = col_lane * E_per_lane

            # -------------------------------------------------------------------
            # Step 1: Load logits into registers and bit-pack the expert index
            #   into the lower log_E mantissa bits of each FP32 value so that
            #   the bitonic sort simultaneously sorts values AND carries indices.
            #
            #   Encoding rule:
            #     positive val → store ~col_idx in lower bits (larger float →
            #                    larger uint32, correct sort order preserved)
            #     negative val → store  col_idx in lower bits
            #
            #   FIX (Issue 6 / Bug 3): We use the recast_tensor pair (regs /
            #   regs_u32) which share the same register memory, so writing to
            #   regs_u32[i] and reading regs[i] is a bitwise reinterpret — the
            #   correct equivalent of __float_as_int / __int_as_float.
            #   This replaces the previously proposed cute.arch.bitcast() which
            #   does NOT exist in CUTLASS DSL 4.4.0 and would cause a compile
            #   error.
            # -------------------------------------------------------------------
            idx_mask = const_expr((1 << self.log_E) - 1)
            regs     = cute.make_rmem_tensor(E_per_lane, cutlass.Float32)
            regs_u32 = cute.recast_tensor(regs, cutlass.Uint32)  # same memory, u32 view

            vec = const_expr(self.vec)
            for v in cutlass.range_constexpr(E_per_lane // vec):
                col_base = lane_start + v * vec
                # Load vec elements from global memory using element-wise access
                raw = cute.make_rmem_tensor(vec, cutlass.Float32)
                for load_idx in cutlass.range_constexpr(vec):
                    raw[load_idx] = mLogits[global_row, col_base + load_idx].to(cutlass.Float32)
                for j in cutlass.range_constexpr(vec):
                    val_f32 = raw[j]
                    col_idx = cutlass.Uint32(col_base + j)
                    encoded = (
                        (~col_idx if val_f32 >= cutlass.Float32(0.0) else col_idx) & idx_mask
                    )
                    # Write packed value: bitwise reinterpret via the recast pair.
                    # regs_u32[i] and regs[i] are the same register memory viewed
                    # as uint32 and float32 respectively — no numeric conversion.
                    u32_val = regs_u32[v * vec + j]   # use existing slot as temp
                    # We need val_f32's raw bits as uint32.
                    # Step: write val_f32 as float, read as uint32 via the pair.
                    regs[v * vec + j] = val_f32                      # store float bits
                    raw_u32 = regs_u32[v * vec + j]                  # read as uint32 (bitwise)
                    regs_u32[v * vec + j] = (raw_u32 & ~idx_mask) | encoded  # pack index

            # -------------------------------------------------------------------
            # Step 2: Online Softmax — single left-to-right pass over registers.
            #
            #   FIX (Bug 1/2): Call online_softmax_update() instead of inlining
            #   broken logic.  The dead variable `val_f32 = regs[i]` in the
            #   original read the BIT-PACKED value (not the clean float) but was
            #   never actually used — now eliminated.
            #
            #   FIX (Issue 6 / Bug 3): Extract clean_f32 via the recast pair:
            #     1. Save the packed u32.
            #     2. Write clean_u32 (index bits stripped) into the slot.
            #     3. Read as float via regs[i] (bitwise, not numeric).
            #     4. Restore the packed value.
            #   This avoids cute.arch.bitcast() which doesn't exist in 4.4.0.
            # -------------------------------------------------------------------
            running_max = -cutlass.Float32.inf
            running_sum = cutlass.Float32(0.0)

            for i in cutlass.range_constexpr(E_per_lane):
                packed_u32 = regs_u32[i]
                clean_u32  = packed_u32 & ~idx_mask
                # Bitwise reinterpret: write clean uint32 bits, read as float.
                regs_u32[i] = clean_u32
                clean_f32   = regs[i]           # same register, float view
                regs_u32[i] = packed_u32        # restore packed value

                running_max, running_sum = online_softmax_update(
                    clean_f32, running_max, running_sum
                )

            # -------------------------------------------------------------------
            # Warp reduction: broadcast global (running_max, running_sum) to all
            # threads in the same token row (threads_per_row threads per row).
            #
            #   FIX (Bug 4): Replaced Python `while stride > 0` with
            #   cutlass.range_constexpr so the loop is compile-time unrolled to
            #   PTX shuffle instructions.
            #
            #   FIX (Issue 7): Added the required full-warp mask (0xFFFFFFFF) as
            #   the FIRST argument to shfl_xor_sync.  CUDA C signature is:
            #     __shfl_xor_sync(unsigned mask, T var, int laneMask, int width)
            #   The CUTLASS DSL maps this directly — omitting the mask caused
            #   the shuffle to use an undefined register for the predicate.
            # -------------------------------------------------------------------
            if const_expr(self.threads_per_row > 1):
                for log_step in cutlass.range_constexpr(self.log2_threads_per_row):
                    stride = const_expr(self.threads_per_row >> (log_step + 1))
                    # Issue 7 fix: pass _FULL_WARP_MASK as the first argument.
                    peer_max = cute.arch.shfl_xor_sync(
                        _FULL_WARP_MASK, running_max, stride, self.threads_per_row
                    )
                    peer_sum = cute.arch.shfl_xor_sync(
                        _FULL_WARP_MASK, running_sum, stride, self.threads_per_row
                    )
                    new_max    = cute.arch.fmax(running_max, peer_max)
                    my_scale   = cute.arch.exp2((running_max - new_max) * cutlass.Float32(LOG2E))
                    peer_scale = cute.arch.exp2((peer_max   - new_max) * cutlass.Float32(LOG2E))
                    running_sum = running_sum * my_scale + peer_sum * peer_scale
                    running_max = new_max
            # After the reduction: running_max and running_sum are identical across
            # all threads_per_row lanes that share the same token row.

            # -------------------------------------------------------------------
            # FIX (Bug 6): Warp barrier between the shuffle reduction and the
            # bitonic sort.  Without this, threads may start reading `regs` in
            # bitonic_topk before all threads finish writing packed values in
            # Step 1, creating a race condition.
            # -------------------------------------------------------------------
            cute.arch.sync_warp()

            # -------------------------------------------------------------------
            # Step 3: Bitonic TopK on the packed register array.
            #   bitonic_topk distributes results across all threads_per_row lanes;
            #   each thread holds k_per_lane = k / threads_per_row results.
            # -------------------------------------------------------------------
            from quack.sort.bitonic_sort import bitonic_topk as _bitonic_topk

            topk_regs = _bitonic_topk(regs, self.next_pow2_k, warp_width=self.threads_per_row)

            # -------------------------------------------------------------------
            # Step 4: Decode indices and apply softmax normalization.
            #
            #   FIX (Bug 5): Sign check must be on the CLEAN float, not on the
            #   bit-packed topk_regs[i] value (whose sign bits are corrupted by
            #   the index stuffed into the mantissa).  Same recast-pair trick as
            #   Step 2 to recover clean_f32.
            #
            #   FIX (Issue 6 / Bug 3): Bitwise reinterpret via recast pair.
            #   FIX (Bug 7/8): Allocate k_per_lane outputs per thread (not k).
            # -------------------------------------------------------------------
            topk_u32 = cute.recast_tensor(topk_regs, cutlass.Uint32)

            k_per_lane = const_expr(self.k_per_lane)
            out_vals   = cute.make_rmem_tensor(k_per_lane, self.output_dtype)
            out_idx    = cute.make_rmem_tensor(k_per_lane, cutlass.Int32)

            inv_sum = cutlass.Float32(1.0) / running_sum

            for i in cutlass.range_constexpr(k_per_lane):
                encoded   = topk_u32[i] & idx_mask
                clean_u32 = topk_u32[i] & ~idx_mask

                # Bitwise reinterpret: write clean bits, read as float, restore.
                packed_saved   = topk_u32[i]
                topk_u32[i]    = clean_u32
                clean_f32      = topk_regs[i]   # float view of same register
                topk_u32[i]    = packed_saved

                # FIX (Bug 5): sign check on clean_f32, not topk_regs[i].
                col_idx = (
                    (~encoded if clean_f32 >= cutlass.Float32(0.0) else encoded) & idx_mask
                )
                out_idx[i] = cutlass.Int32(col_idx)

                # Softmax weight via emulated exp2 (FA4 §3.2).
                sm_val = cute.arch.exp2(
                    (clean_f32 - running_max) * cutlass.Float32(LOG2E)
                ) * inv_sum
                out_vals[i] = sm_val.to(self.output_dtype)

            # -------------------------------------------------------------------
            # Step 5: Write topK results to HBM.
            #
            #   FIX (Bug 7): Each thread writes its own k_per_lane slice starting
            #   at col_lane * k_per_lane.  The original guarded all writes with
            #   `if col_lane == 0`, discarding (threads_per_row-1)/threads_per_row
            #   of the topK results silently.
            #
            #   FIX (Bug 8): Since every thread now writes its slice, the
            #   normalization work done in Step 4 is fully utilized.
            # -------------------------------------------------------------------
            lane_out_start  = col_lane * k_per_lane

            # Write results element-by-element (CUTLASS DSL doesn't support dynamic slicing)
            for out_idx_write in cutlass.range_constexpr(k_per_lane):
                mTopKValues[global_row, lane_out_start + out_idx_write] = out_vals[out_idx_write]
                mTopKIndices[global_row, lane_out_start + out_idx_write] = out_idx[out_idx_write]


    # ---------------------------------------------------------------------------
    # Section 4: Python-level launcher (drop-in replacement for TopK_Softmax)
    # ---------------------------------------------------------------------------

class FusedRouterTopKSoftmax_SM90:
    """
    Drop-in replacement for the original TopK_Softmax on Hopper GPUs.

    Accepts the same positional arguments as the original class so forward.py
    does not need to change its call site.
    """

    def __init__(
        self,
        input_dtype: Type[cutlass.Numeric],
        output_dtype: Type[cutlass.Numeric],
        E: int,
        k: int,
        require_softmax_fusion: bool = True,
        T_tile: int = 32,
    ):
        _ = require_softmax_fusion  # accepted for API compatibility, always fused

        self.input_dtype  = input_dtype
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
        mLogits:  cute.Tensor,
        mValues:  cute.Tensor,
        mIndices: cute.Tensor,
        stream:   cuda.CUstream,
    ):
        self._kernel(mLogits, mValues, mIndices, stream)


# ---------------------------------------------------------------------------
# Section 5: Standalone correctness test
# ---------------------------------------------------------------------------

def test_fused_vs_separate(T=1024, E=128, k=8, dtype=torch.bfloat16):
    """
    Validates that the fused epilogue produces the same topK indices and
    softmax weights as the reference two-step path (torch.topk + F.softmax).

    NOTE: the fused kernel computes softmax over ALL E experts, then gathers
    the top-k weights from that distribution.  The reference matches this
    semantics exactly.
    """
    import torch.nn.functional as F

    device  = "cuda"
    logits  = torch.randn(T, E, device=device, dtype=dtype)

    # --- Reference ---
    logits_f32   = logits.float()
    ref_sm_full  = F.softmax(logits_f32, dim=-1)
    _, ref_idx   = torch.topk(logits_f32, k, dim=-1)
    ref_sm       = ref_sm_full.gather(1, ref_idx)

    # --- Fused kernel ---
    fused_vals = torch.zeros(T, k, device=device, dtype=torch.float32)
    fused_idx  = torch.zeros(T, k, device=device, dtype=torch.int32)

    input_dtype  = torch2cute_dtype_map[logits.dtype]
    output_dtype = torch2cute_dtype_map[torch.float32]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def _to_cute(t):
        return (
            from_dlpack(t.detach(), assumed_align=16)
            .mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        )

    mLogits = _to_cute(logits)
    mVals   = _to_cute(fused_vals)
    mIdx    = _to_cute(fused_idx)

    op       = FusedRouterTopKSoftmax_SM90(input_dtype, output_dtype, E=E, k=k, T_tile=32)
    compiled = cute.compile(op, mLogits, mVals, mIdx, stream)
    compiled(mLogits, mVals, mIdx, stream)
    torch.cuda.synchronize()

    # Check indices (order within top-k may differ — sort both sides)
    ref_idx_sorted,   _ = ref_idx.sort(dim=-1)
    fused_idx_sorted, _ = fused_idx.sort(dim=-1)
    assert torch.all(ref_idx_sorted == fused_idx_sorted), (
        f"TopK index mismatch!\n"
        f"  ref[0]:   {ref_idx_sorted[0]}\n"
        f"  fused[0]: {fused_idx_sorted[0]}"
    )

    # Check softmax values
    ref_sm_sorted   = ref_sm.gather(1, ref_idx.argsort(dim=-1))
    fused_sm_sorted = fused_vals.gather(1, fused_idx.long().argsort(dim=-1))
    max_err = (ref_sm_sorted - fused_sm_sorted).abs().max().item()
    assert max_err < 2e-3, f"Softmax value mismatch! Max error: {max_err:.6f}"

    print(
        f"[PASS] T={T}, E={E}, k={k}: fused topK+softmax matches reference "
        f"(max_err={max_err:.2e})"
    )
    return True


if __name__ == "__main__":
    test_fused_vs_separate(T=1024,  E=128, k=8)
    test_fused_vs_separate(T=32768, E=256, k=16)
    print("All tests passed.")


# Aliases for forward.py, bench_hopper_fa4.py, test_hopper_fa4_correctness.py
TopK_Softmax_Hopper = FusedRouterTopKSoftmax_SM90
TopK_Softmax        = FusedRouterTopKSoftmax_SM90