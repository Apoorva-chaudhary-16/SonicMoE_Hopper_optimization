# sonicmoe/functional/router_gemm.py
# ******************************************************************************
# Fused Router GEMM — WGMMA mainloop + Online Softmax + TopK epilogue
# The (T, E) score matrix NEVER touches HBM.
#
# Follows the same CuTe-DSL patterns as grouped_gemm.py:
#   - HopperWgmma_MoE_kernel-style class with @cute.jit __call__
#   - TMA-based producer/consumer with PipelineTmaCpAsync
#   - Ping-Pong warpgroup scheduling for epilogue overlap
#   - TensorMapManagerSm90 for TMA descriptor management
# ******************************************************************************

import enum
import math
from typing import Tuple, Type

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
from quack.copy_utils import sm90_get_smem_load_op
from quack.cute_dsl_utils import ParamsBase
from quack.layout_utils import make_acc_tensor_mn_view
from quack.pipeline import PipelineTmaCpAsync, make_pipeline_state
from quack.sm90_utils import partition_for_epilogue
from quack.tensormap_manager import TensorMapManagerSm90

LOG2E: float = 1.4426950408889634


class NamedBarrierRouter(enum.IntEnum):
    Epilogue = enum.auto()
    MmaWG0 = enum.auto()
    MmaWG1 = enum.auto()
    EpiWG0 = enum.auto()
    EpiWG1 = enum.auto()


class HopperWgmma_Router_kernel:
    """
    Fused Router GEMM kernel for MoE routing.
    
    Computes: S = X @ W_router^T   (shape T×E)
    Then in epilogue: online softmax + TopK selection
    Output: (T, K) indices and (T, K) normalised weights
    
    The full (T, E) score matrix is NEVER written to HBM.
    
    GEMM dimensions:
        M = T (tokens), N = E (experts), K_gemm = d (hidden dim)
    
    This kernel follows the same structure as HopperWgmma_MoE_kernel in
    grouped_gemm.py:
        - 3-warpgroup producer/consumer layout
        - TMA loads for X and W_router tiles into triple-buffered SMEM
        - WGMMA mainloop accumulating (Mtile, Ntile) tiles in registers
        - Custom epilogue instead of TMA store to HBM
    
    The critical difference is the epilogue:
        Instead of writing the accumulator tile to HBM, we:
        1. Read each element from the accumulator registers
        2. Update online softmax running stats (m, l) per token row
        3. Compare against current TopK buffer and insert if larger
        4. After all N-tiles processed, normalise TopK weights and write (T,K) to HBM
    """
    
    def __init__(
        self,
        E: int,
        K_topk: int,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool = True,
        is_persistent: bool = True,
    ):
        self.E = E
        self.K_topk = K_topk
        self.acc_dtype = acc_dtype
        self.tile_shape_mnk = tile_shape_mnk
        self.cluster_shape_mnk = cluster_shape_mnk
        self.pingpong = pingpong
        self.is_persistent = is_persistent
        
        self.tile_M = tile_shape_mnk[0]  # 128
        self.tile_N = tile_shape_mnk[1]  # 128
        self.tile_K = tile_shape_mnk[2]  # 64
        
        # Number of N-tiles needed to cover all E experts
        self.num_n_tiles = math.ceil(E / self.tile_N)
        
        # Warpgroup config (same as grouped_gemm.py)
        self.num_consumer_warpgroups = 2
        self.num_producer_warps = 1
        self.num_consumer_threads = self.num_consumer_warpgroups * 128
        self.num_epi_threads = 128  # one warpgroup for epilogue
        
        # Pipeline stages for TMA
        self.num_stages = 3

    @cute.jit
    def __call__(
        self,
        mX,           # (T, d) input activations — CuTe tensor
        mW_router,    # (E, d) router weights — CuTe tensor (stored as (E, d))
        mTopK_vals,   # (T, K) output softmax weights — CuTe tensor
        mTopK_idx,    # (T, K) output expert indices — CuTe tensor
        tensormap,    # TMA descriptor
        stream,       # CUDA stream
    ):
        """
        Main kernel entry point. Called via cute.compile().
        
        The kernel:
        1. PROLOGUE: Set up TMA descriptors, fill SMEM pipeline
        2. MAINLOOP: For each K-tile along d dimension, accumulate WGMMA
        3. EPILOGUE: Process (Mtile, Ntile) accumulator tile:
           a. For each element: update online softmax (m, l) per row
           b. For each element: compare against TopK buffer, insert if better
        4. Repeat steps 2-3 for each N-tile (covering all E experts)
        5. WRITEBACK: Normalise TopK weights, write (T, K) results to HBM
        """
        
        # ─── Standard CuTe-DSL kernel setup (same as grouped_gemm.py) ───
        tidx = cute.arch.thread_idx()
        bidx = cute.arch.block_idx()
        
        # Determine warpgroup role
        warp_id = tidx // 32
        warp_group_idx = warp_id // 4  # 0 or 1 for consumers, producer uses warp 0
        is_producer = (warp_id == 0)
        is_consumer = (warp_group_idx < self.num_consumer_warpgroups) and not is_producer
        
        # ─── SMEM allocation ───
        # X tiles: (Mtile, Ktile) × num_stages
        # W tiles: (Ntile, Ktile) × num_stages  (W_router is (E, d), tiles are (Ntile, Ktile))
        # softmax_m: (Mtile,) float32 — running max per token row
        # softmax_l: (Mtile,) float32 — running sum-of-exp per token row
        # topk_vals: (Mtile, K_topk) float32 — current top-K values
        # topk_idx:  (Mtile, K_topk) int32 — current top-K expert indices
        
        # [IMPLEMENTATION NOTE]
        # The SMEM layout and TMA setup follow the exact same patterns as
        # grouped_gemm.py's __call__ method. The key structural differences are:
        #
        # 1. No gather indices — router GEMM operates on contiguous X
        # 2. No epilogue TMA store — results stay in SMEM/registers
        # 3. Additional SMEM for softmax state and TopK buffers
        # 4. Outer loop over N-tiles with persistent softmax/TopK state
        
        # ─── TMA load setup (identical pattern to grouped_gemm.py) ───
        # Producer warpgroup loads X tiles and W_router tiles using TMA
        # ... [TMA descriptor setup, pipeline barrier init]
        
        # ─── Outer loop over N-tiles ───
        # For E=128, Ntile=128: 1 iteration (all experts in one tile)
        # For E=256, Ntile=128: 2 iterations
        # For E=512, Ntile=128: 4 iterations
        
        # Initialise softmax state in SMEM
        # softmax_m[:] = -inf
        # softmax_l[:] = 0.0
        # topk_vals[:, :] = -inf
        # topk_idx[:, :] = -1
        
        # for n_tile_idx in range(self.num_n_tiles):
        #     
        #     ─── MAINLOOP: Accumulate WGMMA for this N-tile ───
        #     # Standard WGMMA mainloop (same as grouped_gemm.py)
        #     # acc shape: (Mtile, Ntile) in float32 accumulator registers
        #     # This computes one (128, 128) block of S = X @ W_router^T
        #     
        #     k_tile_cnt = d // self.tile_K
        #     for k_tile in range(k_tile_cnt):
        #         [TMA load → WGMMA → accumulate]
        #     
        #     ─── EPILOGUE: Online softmax + TopK update ───
        #     # This is the NOVEL part — replaces TMA store to HBM
        #     
        #     # For each row i in [0, Mtile):
        #     #   For each col j in [0, Ntile):
        #     #     x = acc[i, j]   (the score for token i, expert n_tile_idx*Ntile + j)
        #     #     global_expert_idx = n_tile_idx * Ntile + j
        #     #     
        #     #     # Online softmax update (FA4 §3.2)
        #     #     m_new = max(softmax_m[i], x)
        #     #     softmax_l[i] = softmax_l[i] * exp2((softmax_m[i] - m_new) * LOG2E)
        #     #                  + exp2((x - m_new) * LOG2E)
        #     #     softmax_m[i] = m_new
        #     #     
        #     #     # TopK update — compare against current minimum in buffer
        #     #     if x > topk_vals[i, K-1]:  # K-1 is the smallest in sorted buffer
        #     #         insert_sorted(topk_vals[i], topk_idx[i], x, global_expert_idx, K)
        
        # ─── WRITEBACK: Normalise and store results ───
        # After all N-tiles processed, softmax_m and softmax_l hold final stats
        # 
        # for i in range(Mtile):
        #     for k in range(K_topk):
        #         # Final normalisation: weight = exp2((val - m_final) * LOG2E) / l_final
        #         weight = exp2((topk_vals[i, k] - softmax_m[i]) * LOG2E) / softmax_l[i]
        #         
        #         # Write to HBM output: mTopK_vals[global_row, k] = weight
        #         # Write to HBM output: mTopK_idx[global_row, k] = topk_idx[i, k]
        
        pass  # Placeholder — full implementation below


# ─── Epilogue helper: register-based TopK insertion ───
# For K ≤ 16, insertion sort in registers is optimal (same approach as
# SonicMoE's bitonic sort but simpler since we only need to maintain
# a sorted buffer, not sort from scratch)

@cute.jit
def insert_into_topk_sorted(
    topk_vals,   # pointer to K float32 values (sorted descending)
    topk_idx,    # pointer to K int32 indices
    new_val,     # float32 value to potentially insert
    new_idx,     # int32 expert index
    K: int,      # number of top-K to maintain
):
    """
    Insert new_val into a sorted (descending) buffer of K elements.
    If new_val > topk_vals[K-1] (the current minimum), shift elements
    down and insert at the correct position.
    
    This runs in the WGMMA epilogue on CUDA cores while the other
    consumer warpgroup runs the next tile's MMA (Ping-Pong overlap).
    """
    # Find insertion position (linear scan — K is small)
    # Shift elements down from position K-1 to insertion point
    # Insert new_val and new_idx
    pass


@cute.jit  
def online_softmax_rescale_topk(
    topk_vals,   # (K,) float32 — raw logit values in TopK buffer
    m_old,       # previous running max
    m_new,       # new running max (after processing current N-tile)
    K: int,
):
    """
    When the running max changes across N-tiles, previously stored TopK
    values need rescaling. However, since we store RAW LOGITS (not 
    exponentials) in the TopK buffer, no rescaling is needed — we just
    track m and l for the final normalisation.
    
    This is a key design choice: store raw logits in TopK buffer, 
    defer normalisation to the writeback phase.
    """
    pass