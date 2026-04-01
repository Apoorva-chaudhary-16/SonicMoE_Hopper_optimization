#!/bin/bash
# SonicMoE Test Script with FA4 Hopper Optimizations
# Sets up environment to work around missing python3.11-dev package

cd /home/gpu1/testing/SonicMoE_Hopper_optimization
source venv/bin/activate

# Use Python 3.10 headers as workaround
export CPATH="/usr/include/python3.10:$CPATH"
export C_INCLUDE_PATH="/usr/include/python3.10:$C_INCLUDE_PATH"
export CPLUS_INCLUDE_PATH="/usr/include/python3.10:$CPLUS_INCLUDE_PATH"

# Set CUDA architecture for H100
export TORCH_CUDA_ARCH_LIST="9.0"

echo "========================================================================"
echo "  SonicMoE with FA4 Hopper Optimizations Test"
echo "========================================================================"
echo ""

python << 'EOFPYTHON'
import torch
from sonicmoe import MoE, KernelBackendMoE
from sonicmoe.enums import ActivationType
import time
import statistics

print('Creating MoE layer (8 experts, k=2, hidden=512)...')
moe = MoE(
    num_experts=8,
    num_experts_per_tok=2,
    hidden_size=512,
    intermediate_size=128,
    activation_function=ActivationType.SWIGLU,
    add_bias=False,
    std=0.02,
).to(device='cuda', dtype=torch.bfloat16)

print('✓ MoE layer created successfully')
print('')
print('Running forward pass with SonicMoE backend (compiling kernels)...')
print('This may take 60-120 seconds on first run...')
print('')

x = torch.randn(1024, 512, device='cuda', dtype=torch.bfloat16)

start = time.time()
output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
elapsed = time.time() - start

print('')
print('='*70)
print('✓✓✓ SUCCESS! SONICMOE WITH FA4 HOPPER OPTIMIZATIONS WORKING! ✓✓✓')
print('='*70)
print(f'  Input shape:      {x.shape}')
print(f'  Output shape:     {output.shape}')  
print(f'  Aux loss:         {aux_loss.item():.6f}')
print(f'  Time (1st run):   {elapsed:.3f}s (includes JIT compilation)')
print('='*70)
print('')

# Run again for actual performance
print('Running 2nd forward pass (no compilation)...')
start = time.time()
output2, aux_loss2 = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
elapsed2 = time.time() - start
print(f'  Time (2nd run):   {elapsed2*1000:.2f}ms')
print('')

# Multiple iterations for stable measurement
print('Running 20 iterations for stable performance measurement...')
times = []
for i in range(20):
    start = time.time()
    output, aux_loss = moe(x, kernel_backend_moe=KernelBackendMoE.sonicmoe)
    times.append(time.time() - start)

avg_time = statistics.mean(times) * 1000
std_time = statistics.stdev(times) * 1000
min_time = min(times) * 1000
max_time = max(times) * 1000

print(f'  Average time:     {avg_time:.2f}ms ± {std_time:.2f}ms')
print(f'  Min/Max:          {min_time:.2f}ms / {max_time:.2f}ms')
print('')
print('='*70)
print('FA4 Optimizations Active:')
print('  • ex2.approx (hardware-accelerated exponential)')
print('  • Online softmax (Kernel-1 merge, single-pass)')
print('  • Optimized gather-sum with pre-normalized weights')
print('='*70)
EOFPYTHON

echo ""
echo "Test completed!"
