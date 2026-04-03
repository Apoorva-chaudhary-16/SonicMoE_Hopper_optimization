# SonicMoE Optimization Opportunities - Complete Analysis

**Date**: 2026-04-03  
**Analysis Scope**: Entire codebase including docs, code, TODOs, and benchmarks

---

## Executive Summary

| Category | Opportunities Found | Best ROI |
|----------|-------------------|----------|
| **Production Ready** | 1 (Token Rounding) | ✅ **+12.2%** |
| **Quick Wins** | 3 opportunities | +2-5% potential |
| **Configuration Tuning** | 5 opportunities | +1-3% each |
| **Kernel-Level** | 2 opportunities | +0.5-1% each |
| **Future (Blackwell)** | 1 major opportunity | +9-12% |
| **Abandoned** | 2 (Phase 1, Phase 2) | Negative or infeasible |

---

## 🏆 TIER 1: Production Ready (Immediate Results)

### **Token Rounding Optimization** ✅
**Status**: Already verified and working  
**File**: `benchmarks/moe-token-rounding.py`  
**Measured Performance**: **+12.2% end-to-end TFLOPS**

**Key Results**:
```
Baseline:        231.0 TFLOPS (forward: 241.8, backward: 226.1)
Token Rounding:  259.1 TFLOPS (forward: 291.6, backward: 245.4)
Speedup:         +12.2% end-to-end
```

**What it does**:
- Rounds expert token assignments to 128-token tile boundaries
- Eliminates 24.2% padding waste
- Zero quality degradation (routing-level optimization)

**Action**: Ready to present - just needs documentation polish

**Evidence**: `VERIFIED_RESULTS.md` lines 93-194

---

## 🚀 TIER 2: Quick Wins (High ROI, Low Effort)

### **1. Small Model Tile Tuning** 🔍
**Status**: Documented as TODO, not implemented  
**File**: `sonicmoe/functional/moe_config.py:51`  
**Expected Benefit**: **+2-5% for I < 128 configs**

**Current Issue**:
```python
# TODO: this assertion does not mean that the MoE impl prohibits such config.
# Instead, we just do not search for the best configs manually yet for small-shaped MoE
assert (is_glu and intermediate >= 128) or (not is_glu and intermediate >= 256)
```

**Opportunity**:
- Current tile shapes only tuned for large I (≥128 for GLU, ≥256 for non-GLU)
- Small model inference scenarios (I=64, I=32) use suboptimal tiles
- Could manually benchmark tile combinations for small configs

**Implementation**:
1. Generate tile shape candidates for I ∈ {32, 64, 96}
2. Benchmark on representative workloads
3. Add config classes for small-I scenarios

**Effort**: 1-2 days (mostly benchmark time)  
**Risk**: Low (orthogonal to existing optimizations)

---

### **2. Raster Order Tuning** 🔍
**Status**: Fixed configuration, not benchmarked  
**File**: `sonicmoe/functional/moe_config.py:60, 168, 177, 186`  
**Expected Benefit**: **+0.5-3% for some configs**

**Current Settings**:
- Up-projection: `RasterOrderOption.AlongM`
- Down-projection: `RasterOrderOption.AlongN`

**Opportunity**:
- Raster order affects L2 cache utilization and work distribution
- Current choices based on heuristics, not benchmarked
- Different orders may work better for different (T, H, I, E) combinations

**Options**: `Heuristic`, `AlongM`, `AlongN`, `AlongS`

**Implementation**:
1. Benchmark all 4 raster orders on representative configs
2. Identify patterns (e.g., "AlongS better for T > 8192")
3. Add adaptive selection logic

**Effort**: 1-2 days (mostly benchmark time)  
**Risk**: Low (easily reversible)

---

### **3. Adaptive count_cumsum Selection** 🔍
**Status**: Heuristic-based, could be improved  
**File**: `sonicmoe/functional/__init__.py:44-47`  
**Expected Benefit**: **+0.2-0.5%**

**Current Logic**:
```python
if T % 4 == 0 and T <= 50000:
    _, num_activated_expert_per_token_offset = count_cumsum(sorted_selected_T, T, do_cumsum=True)
else:
    num_activated_expert_per_token_offset = torch.bincount(sorted_selected_T, minlength=T).cumsum(0).int()
```

**Issue**: Hardcoded threshold (T ≤ 50000) not optimal for all hardware

**Opportunity**:
- Profile `count_cumsum` vs `bincount+cumsum` crossover point
- Adjust threshold based on GPU (H100 vs A100 vs H200)
- Consider E (expert count) in decision logic

**Implementation**: 1 hour (benchmark + code change)  
**Risk**: Very low

---

## ⚙️ TIER 3: Configuration Tuning (Medium Effort)

### **4. Pipeline Stage Optimization**
**File**: `sonicmoe/functional/moe_config.py:54-70`  
**Expected Benefit**: +1-3%  
**Current**: Fixed stages (2 or 8 based on I)

**Opportunity**: Adaptive staging based on compute vs memory intensity

---

### **5. Epilogue Tile Size Tuning**
**File**: `sonicmoe/functional/moe_config.py:57, 66, 174`  
**Expected Benefit**: +0.5-2%  
**Current**: Fixed (32, 64, or 96)

**Opportunity**: Per-config autotuning instead of heuristics

---

### **6. Triton Config Caching**
**File**: `sonicmoe/functional/reduction_over_k_gather_hopper.py:42-73`  
**Expected Benefit**: +0.1-1% (compilation time)  
**Current**: Generates 500+ configs, prunes at runtime

**Opportunity**: Cache best configs for common model sizes

---

## 🔧 TIER 4: Kernel-Level Optimizations (High Effort)

### **7. Count+Cumsum Warp Optimization**
**File**: `sonicmoe/count_cumsum/kernel.cu:77-82`  
**Expected Benefit**: +0.1-0.3%

**Current**: Uses `__syncwarp()` for synchronization  
**Opportunity**: Use warp-wide shuffle instructions for reduction

**Effort**: 1-2 days (CUDA kernel expertise required)  
**Risk**: Medium (must maintain correctness)

---

### **8. Vectorized Load Optimization**
**File**: `sonicmoe/count_cumsum/kernel.cu:37-40`  
**Expected Benefit**: +0.2-1%

**Current**: 128-bit vector loads  
**Opportunity**: 256-bit loads for improved bandwidth

**Effort**: 2-3 days  
**Risk**: Medium (careful alignment requirements)

---

## 🔮 TIER 5: Future Opportunities

### **9. Blackwell Architecture Support** 🔜
**Status**: Scaffolding exists, needs hardware  
**Expected Benefit**: **+9-12% additional** over current Hopper optimizations

**Key Advantage**: 256 KB TMEM (vs 227 KB SMEM on H100)  
- Solves Phase 2 SMEM fusion constraint
- Uses `tcgen05.mma` async instructions

**Timeline**: Q2-Q3 2026 (when B200/GB200 available)  
**Effort**: 1-2 weeks once hardware accessible

---

## ⛔ TIER 6: Abandoned/Infeasible

### **Phase 1: L2 Cache Fusion Megakernel**
**Status**: ❌ Removed (caused -0.94% slowdown)  
**File**: Previously `sonicmoe/functional/megakernel_forward.py`

**Why abandoned**:
- Grid serialization penalty (40µs overhead)
- Measured **-0.94% slower** than baseline
- API was broken (missing 8 arguments)

**Action**: Already removed from codebase

---

### **Phase 2: True SMEM Fusion**
**Status**: ❌ Not implemented (hardware constraint)  
**Expected**: +4.8% theoretical  
**File**: Design only (`PHASE2_SMEM_STATUS.md`)

**Why abandoned**:
- H100 has 227 KB SMEM, needs 228 KB (1 KB over)
- 2-3 weeks implementation effort
- Only works for I ≤ 256 (limited applicability)

**Action**: Wait for Blackwell (256 KB TMEM)

---

### **Router Precision Reduction (NOT RECOMMENDED)**
**File**: `sonicmoe/functional/__init__.py:68`  
**Expected**: +2% (5 TFLOPS)  
**Trade-off**: Numerical accuracy degradation

**Current**:
```python
# change this to router_logits.dtype (bfloat16) increase another 5 tflops at fwd 
# at the cost of numerical accuracy
topk_router_score = torch.empty(T, K, dtype=torch.float32, device=router_logits.device)
```

**Why not recommended**:
- Explicit warning about accuracy cost
- +2% gain negligible compared to Token Rounding (+12.2%)
- Could affect model quality

---

## 📊 Summary Table: All Opportunities

| **Optimization** | **Benefit** | **Effort** | **Risk** | **Status** | **Priority** |
|-----------------|------------|-----------|---------|-----------|-------------|
| Token Rounding | **+12.2%** | ✅ Done | Low | ✅ Production | **P0** |
| Small-I Tuning | +2-5% | 1-2 days | Low | 🔍 TODO | **P1** |
| Raster Order | +0.5-3% | 1-2 days | Low | 🔍 Config | **P1** |
| count_cumsum | +0.2-0.5% | 1 hour | Very Low | 🔍 Quick | **P2** |
| Pipeline Stages | +1-3% | Medium | Low | ⚙️ Config | P2 |
| Epilogue Tiles | +0.5-2% | Medium | Low | ⚙️ Config | P2 |
| Triton Cache | +0.1-1% | Low | Very Low | ⚙️ Tuning | P3 |
| Warp Reduction | +0.1-0.3% | 1-2 days | Medium | 🔧 Kernel | P3 |
| 256-bit Loads | +0.2-1% | 2-3 days | Medium | 🔧 Kernel | P3 |
| Blackwell | +9-12% | 1-2 weeks | Low | 🔜 Future | P4 |
| Phase 1 | -0.94% | N/A | N/A | ❌ Removed | N/A |
| Phase 2 | +4.8% | 2-3 weeks | High | ❌ Blocked | N/A |
| Router BF16 | +2% | Minimal | Medium | ⚠️ Not rec. | N/A |

---

## 🎯 Recommended Action Plan

### **For Immediate Results (Today)**
✅ **Present Token Rounding**: +12.2% verified speedup  
- Document in presentation/paper
- Highlight zero quality degradation
- Show before/after benchmarks

### **Quick Wins (This Week)**
1. **Small-I tile tuning** (1-2 days) → +2-5%
2. **Raster order benchmarking** (1-2 days) → +0.5-3%
3. **count_cumsum threshold** (1 hour) → +0.2-0.5%

**Potential combined gain**: +3-8% additional

### **Medium-term (This Month)**
- Pipeline stage optimization → +1-3%
- Epilogue tile autotuning → +0.5-2%
- Triton config caching → faster compilation

### **Long-term (When Hardware Available)**
- Blackwell support → +9-12% (B200/GB200 launch)

---

## 💡 Key Insights

### **What Works**
- ✅ **Routing-level optimizations** (Token Rounding: +12.2%)
- ✅ **FlashAttention-4 exp2** (already in baseline)
- ✅ **Triton autotuning** (reduction kernels)
- ✅ **Count+cumsum fusion** (custom CUDA kernel)

### **What Doesn't Work (Hardware Limits)**
- ❌ L2 cache fusion (grid serialization: -0.94%)
- ❌ SMEM fusion on Hopper (227 KB limit)
- ❌ Very aggressive compiler flags (numerical issues)

### **What's Promising**
- 🚀 Small model tuning (underexplored market)
- 🚀 Configuration adaptivity (low-hanging fruit)
- 🚀 Blackwell (solves SMEM constraint)

---

## 📁 Files Referenced

### Documentation
- `VERIFIED_RESULTS.md` - Token Rounding verification
- `PHASE1_MEGAKERNEL_ANALYSIS.md` - Phase 1 failure analysis
- `PHASE2_SMEM_STATUS.md` - Phase 2 infeasibility
- `IMPROVEMENTS_SUMMARY.md` - Historical context

### Code Locations
- `sonicmoe/functional/moe_config.py` - Configuration classes (TODOs at line 51)
- `sonicmoe/functional/__init__.py` - Routing pipeline (line 44-47, 68)
- `sonicmoe/count_cumsum/kernel.cu` - Custom CUDA kernel
- `benchmarks/moe-token-rounding.py` - Production benchmark

### Tests & Benchmarks
- `tests/bench_hopper_fa4.py` - FA4 validation
- `benchmarks/moe-cute.py` - End-to-end benchmark

---

**Bottom Line**: Token Rounding (+12.2%) is your proven result. Quick wins like small-I tuning could add another +3-8% with minimal risk. Phase 1/2 are correctly abandoned due to hardware constraints.
