# sonicmoe/functional/router_forward.py
# ******************************************************************************
# Public API for fused router forward pass
# ******************************************************************************

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from quack.cute_dsl_utils import torch2cute_dtype_map

from .router_config import HopperWgmma_Router_Fwd
from .topk_softmax_hopper import topk_softmax_triton  # fallback


_compile_cache = {}


def fused_router_forward(
    X: torch.Tensor,         # (T, d) input activations, bf16/fp16
    W_router: torch.Tensor,  # (E, d) router weight matrix, bf16/fp16
    K: int,                  # number of top-K experts
    use_fused_gemm: bool = True,  # set False to use fallback path
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused router forward: computes routing weights and indices without
    materialising the (T, E) score matrix in HBM.
    
    Returns:
        indices: (T, K) int32 — selected expert indices per token
        weights: (T, K) float32 — normalised softmax weights per token
    
    Falls back to separate cuBLAS + TopK+Softmax when:
        - E > 512 (too many N-tiles)
        - d < 512 (WGMMA underutilised)  
        - use_fused_gemm=False (explicit fallback)
    """
    T, d = X.shape
    E = W_router.shape[0]
    
    # Fallback conditions
    if not use_fused_gemm or E > 512 or d < 512:
        return _fallback_router_forward(X, W_router, K)
    
    # Allocate outputs
    topk_vals = torch.empty(T, K, device=X.device, dtype=torch.float32)
    topk_idx = torch.empty(T, K, device=X.device, dtype=torch.int32)
    
    # Compile cache key
    cache_key = (T, d, E, K, X.dtype)
    
    if cache_key not in _compile_cache:
        # Create kernel module
        router_module = HopperWgmma_Router_Fwd(T, d, E, K)
        
        # Convert to CuTe tensors
        mX = from_dlpack(X)
        mW = from_dlpack(W_router)
        mV = from_dlpack(topk_vals)
        mI = from_dlpack(topk_idx)
        
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        
        # Generate TMA descriptor
        tensormap = router_module.module.generate_tensormap(None, None, None)
        
        # Compile
        _compile_cache[cache_key] = cute.compile(
            router_module, mX, mW, mV, mI, tensormap, current_stream
        )
        _compile_cache[f"{cache_key}_tensormap"] = tensormap
    
    # Launch compiled kernel
    mX = from_dlpack(X)
    mW = from_dlpack(W_router)
    mV = from_dlpack(topk_vals)
    mI = from_dlpack(topk_idx)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    tensormap = _compile_cache[f"{cache_key}_tensormap"]
    
    _compile_cache[cache_key](mX, mW, mV, mI, tensormap, current_stream)
    
    return topk_idx, topk_vals


def _fallback_router_forward(
    X: torch.Tensor,
    W_router: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback: separate cuBLAS matmul + fused TopK+Softmax kernel."""
    scores = X.float() @ W_router.float().T  # (T, E) — materialised in HBM
    topk_vals = torch.empty(X.shape[0], K, device=X.device, dtype=torch.float32)
    topk_idx = torch.empty(X.shape[0], K, device=X.device, dtype=torch.int32)
    topk_softmax_triton(scores.to(X.dtype), K, topk_vals, topk_idx)
    return topk_idx, topk_vals