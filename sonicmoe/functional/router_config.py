# sonicmoe/functional/router_config.py
# ******************************************************************************
# Router GEMM configurations for different model sizes
# Follows the same pattern as moe_config.py's HopperGEMMConfig
# ******************************************************************************

import math
from dataclasses import dataclass
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr

from .router_gemm import HopperWgmma_Router_kernel


@dataclass
class RouterGEMMConfig:
    tile_shape_mnk: Tuple[int, int, int] = (128, 128, 64)
    cluster_shape_mnk: Tuple[int, int] = (1, 1)
    is_pingpong: bool = True
    num_stages: int = 3


class HopperWgmma_Router_Fwd:
    """
    Fused Router GEMM forward pass.
    
    Replaces:
        scores  = X @ W_router.T           # Kernel 1: cuBLAS
        scores  = F.softmax(scores, dim=-1) # Kernel 2: elementwise
        w, idx  = scores.topk(K, dim=-1)    # Kernel 3: SonicMoE TopK
    
    With a single WGMMA kernel that never materialises the (T, E) score matrix.
    """
    
    def __init__(self, T: int, d: int, E: int, K: int):
        self.T = T
        self.d = d
        self.E = E
        self.K = K
        
        # Router GEMM config — simpler than expert GEMMs because:
        # 1. No gather indices (X is contiguous)
        # 2. Lightweight epilogue (softmax + TopK vs SwiGLU)
        # 3. N dimension is small (E ≤ 512 typically)
        
        # For E ≤ 128: single N-tile, no Ping-Pong needed
        # For E > 128: multiple N-tiles, Ping-Pong helps overlap
        if E <= 128:
            config = RouterGEMMConfig(
                tile_shape_mnk=(128, 128, 64),
                cluster_shape_mnk=(2, 1),
                is_pingpong=False,  # Single N-tile, epilogue is cheap
                num_stages=3,
            )
        elif E <= 256:
            config = RouterGEMMConfig(
                tile_shape_mnk=(128, 128, 64),
                cluster_shape_mnk=(2, 1),
                is_pingpong=True,  # 2 N-tiles, overlap epilogue with MMA
                num_stages=3,
            )
        else:  # E <= 512
            config = RouterGEMMConfig(
                tile_shape_mnk=(128, 128, 64),
                cluster_shape_mnk=(2, 1),
                is_pingpong=True,  # 4 N-tiles, Ping-Pong essential
                num_stages=3,
            )
        
        self.config = config
        self.module = HopperWgmma_Router_kernel(
            E=E,
            K_topk=K,
            acc_dtype=cutlass.Float32,
            tile_shape_mnk=config.tile_shape_mnk,
            cluster_shape_mnk=(*config.cluster_shape_mnk, 1),
            pingpong=config.is_pingpong,
            is_persistent=True,
        )
        
        self.max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            config.cluster_shape_mnk[0] * config.cluster_shape_mnk[1]
        )
    
    @cute.jit
    def __call__(self, mX, mW_router, mTopK_vals, mTopK_idx, tensormap, stream):
        return self.module(mX, mW_router, mTopK_vals, mTopK_idx, tensormap, stream)