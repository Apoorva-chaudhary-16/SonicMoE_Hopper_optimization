#!/usr/bin/env python3
"""SonicMoE Docker Test Suite"""

import torch
import sys
import time

def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_env():
    print_header("Environment")
    print(f"  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.version.cuda}")
    print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"  Arch:    SM{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    try:
        import triton
        print(f"  Triton:  {triton.__version__}")
    except:
        print(f"  Triton:  not available")

def test_topk_softmax():
    print_header("Test 1: TopK + Softmax (FA4 exp2)")
    from sonicmoe.functional.topk_softmax_hopper import topk_softmax_triton
    
    logits = torch.randn(2048, 128, device='cuda', dtype=torch.bfloat16)
    values = torch.zeros(2048, 8, device='cuda', dtype=torch.float32)
    indices = torch.zeros(2048, 8, device='cuda', dtype=torch.int32)
    
    topk_softmax_triton(logits, 8, values, indices)
    torch.cuda.synchronize()
    
    # Verify correctness
    ref_vals, ref_idx = torch.topk(logits.float(), 8, dim=-1)
    ref_sm = torch.softmax(ref_vals, dim=-1)
    
    ref_sorted, _ = ref_idx.sort(dim=-1)
    fused_sorted, _ = indices.sort(dim=-1)
    idx_match = torch.all(ref_sorted == fused_sorted).item()
    
    row_sums = values.sum(dim=-1)
    sum_ok = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    
    print(f"  Indices match:    {'PASS' if idx_match else 'FAIL'}")
    print(f"  Softmax sums to 1: {'PASS' if sum_ok else 'FAIL'}")
    print(f"  Values[0]:        {values[0].tolist()[:4]}...")
    return idx_match and sum_ok

def test_count_cumsum():
    print_header("Test 2: count_cumsum CUDA kernel")
    from sonicmoe import count_cumsum
    
    x = torch.randint(0, 8, (4096,), device='cuda', dtype=torch.int32)
    count, cumsum = count_cumsum(x, 8, do_cumsum=True)
    
    ref_count = x.bincount(minlength=8)
    match = torch.all(count == ref_count).item()
    
    print(f"  Count match:  {'PASS' if match else 'FAIL'}")
    print(f"  Counts:       {count.tolist()}")
    return match

def test_sonicmoe_forward():
    print_header("Test 3: SonicMoE Forward Pass")
    from sonicmoe import MoE, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    
    moe = MoE(
        num_experts=8, num_experts_per_tok=2,
        hidden_size=512, intermediate_size=128,
        activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02
    ).to(device='cuda', dtype=torch.bfloat16)
    
    x = torch.randn(1024, 512, device='cuda', dtype=torch.bfloat16)
    
    print("  Compiling kernels (first run, may take 60-120s)...")
    t0 = time.time()
    output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    t1 = time.time()
    
    print(f"  First run:    {t1-t0:.1f}s (includes JIT compilation)")
    print(f"  Output shape: {output.shape}")
    print(f"  Aux loss:     {aux_loss.item():.6f}")
    
    # Second run (no compilation)
    t0 = time.time()
    output2, aux_loss2 = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    t1 = time.time()
    print(f"  Second run:   {(t1-t0)*1000:.2f}ms")
    
    ok = output.shape == (1024, 512) and not torch.isnan(output).any()
    print(f"  Status:       {'PASS' if ok else 'FAIL'}")
    return ok

def test_sonicmoe_vs_torch():
    print_header("Test 4: SonicMoE vs PyTorch Reference")
    from sonicmoe import MoE, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    
    torch.manual_seed(42)
    moe = MoE(
        num_experts=8, num_experts_per_tok=2,
        hidden_size=512, intermediate_size=128,
        activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02
    ).to(device='cuda', dtype=torch.bfloat16)
    
    x = 0.02 * torch.randn(1024, 512, device='cuda', dtype=torch.bfloat16)
    
    with torch.autocast('cuda', torch.float32):
        y_sonic = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)[0]
        y_torch = moe(x, kernel_backend_moe=KernelBackendMoE.torch)[0]
    
    diff = (y_sonic.float() - y_torch.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    ok = max_diff < 0.02
    print(f"  Max abs diff:  {max_diff:.6f}")
    print(f"  Mean abs diff: {mean_diff:.6f}")
    print(f"  Status:        {'PASS' if ok else 'FAIL'} (tol=0.02)")
    return ok

def test_benchmark():
    print_header("Test 5: Performance Benchmark")
    from sonicmoe import MoE, KernelBackendMoE
    from sonicmoe.enums import ActivationType
    import statistics
    
    moe = MoE(
        num_experts=128, num_experts_per_tok=8,
        hidden_size=4096, intermediate_size=512,
        activation_function=ActivationType.SWIGLU,
        add_bias=False, std=0.02
    ).to(device='cuda', dtype=torch.bfloat16)
    
    x = torch.randn(8192, 4096, device='cuda', dtype=torch.bfloat16)
    
    # Warmup
    print("  Warming up (compiling kernels)...")
    for _ in range(3):
        moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    print("  Running 20 iterations...")
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.time()
        moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
    
    avg = statistics.mean(times)
    std = statistics.stdev(times)
    mn = min(times)
    
    T, H, I, E, K = 8192, 4096, 512, 128, 8
    flops = 6 * T * I * H * K
    tflops = flops / (avg / 1000) / 1e12
    
    print(f"  Config:   T={T} H={H} I={I} E={E} K={K}")
    print(f"  Average:  {avg:.2f}ms +/- {std:.2f}ms")
    print(f"  Min:      {mn:.2f}ms")
    print(f"  TFLOPS:   {tflops:.1f}")
    return True

if __name__ == "__main__":
    print_env()
    
    results = {}
    results['topk'] = test_topk_softmax()
    results['cumsum'] = test_count_cumsum()
    results['forward'] = test_sonicmoe_forward()
    results['correctness'] = test_sonicmoe_vs_torch()
    
    print_header("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {name:20s}: {status}")
    
    if all_pass:
        print(f"\n  ALL TESTS PASSED!")
        
        # Only run benchmark if all tests pass
        try:
            test_benchmark()
        except Exception as e:
            print(f"\n  Benchmark skipped: {e}")
    else:
        print(f"\n  SOME TESTS FAILED")
        sys.exit(1)
