# Phase 2 SMEM Fusion - Current Status

**Date**: 2026-04-03  
**Status**: ❌ **NOT IMPLEMENTED** - Only architectural design exists

---

## Quick Summary

Phase 2 SMEM fusion was **designed but never implemented** due to fundamental hardware limits:

| Aspect | Status |
|--------|--------|
| **Expected speedup** | +4.8% forward TFLOPS |
| **Implementation status** | ❌ Architectural design only, no code |
| **Primary blocker** | H100 SMEM limit: 227 KB available, 228 KB needed |
| **Config limits** | Only works for I ≤ 256 (small models) |
| **Test suite** | ❌ No tests exist |
| **Production readiness** | 0% - requires weeks of implementation |

---

## Why Phase 2 Was Abandoned

### Hardware Constraint: SMEM Budget Too Tight

**Calculation**:
```
Phase 1 buffers:
  - sA1 (X input tiles):     64 KB
  - sB  (W1 weights):        64 KB

Phase 2 buffers:
  - sA2 (Y1 reused):         64 KB
  - sW2 (W2 weights):        64 KB
  - Epilogue registers:      32 KB
  - Metadata/barriers:        4 KB

TOTAL NEEDED:  ~228 KB
H100 CAPACITY:  227 KB
GAP:            -1 KB ⚠️
```

**Result**: Cannot fit both phases in SMEM simultaneously.

### Grid Serialization Penalty

Even with Phase 1 megakernel (separate approach):
- Measured: **-0.94% slowdown** (not speedup!)
- Cause: 40µs grid serialization penalty between kernel launches
- Conclusion: L2 cache warmth doesn't overcome synchronization overhead

---

## What Would Need to Be Built

### 1. Core Fused Kernel ❌
**File**: `sonicmoe/functional/forward.py`
- Create: `_smem_fused_up_down_projection_forward()` custom op
- Single persistent kernel with 2 phases
- Named barriers for inter-phase sync
- Time-multiplexed SMEM layout

### 2. Configuration Class ❌
**File**: `sonicmoe/functional/moe_config.py`
- Create: `HopperWgmma_MoE_SMEM_Fused_Fwd` class
- Add: 2-stage TMA pipeline for W2
- Configure: SMEM reuse flags

### 3. Test Suite ❌
**Missing files**:
- `tests/test_smem_fusion.py` - correctness tests
- `tests/bench_smem_fusion.py` - performance benchmarks

### 4. Router Logic ❌
**File**: `sonicmoe/functional/__init__.py`
- Add logic to select fused path when I ≤ 256
- Fall back to separate kernels otherwise

**Effort estimate**: 2-3 weeks of full-time development

---

## Blackwell Solution (Future Hardware)

**GB200/B200 Benefits**:
- **256 KB TMEM** (vs 227 KB SMEM on H100)
- Solves the budget constraint
- Expected additional gain: **+9-12% on Blackwell**

**Problem**: Requires Blackwell hardware (SM100 architecture)
- Not yet generally available
- Different programming model (TMEM vs SMEM)

---

## Recommendation

### Option A: Abandon Phase 2 ⭐ (Recommended)
**Reasoning**:
- Hardware constraint is fundamental (227 KB limit)
- Token Rounding already delivers +12.2% (proven)
- ROI too low: weeks of work for +4.8% on limited configs
- Better to focus on other optimizations

### Option B: Implement Blackwell Phase 2 Only
**Reasoning**:
- Skip Hopper (SMEM too small)
- Wait for Blackwell availability
- Use 256 KB TMEM instead
- Timeline: Q2-Q3 2026 for hardware access

### Option C: Attempt Hopper Phase 2 Anyway
**Reasoning**:
- Academic interest / research contribution
- Prove/disprove +4.8% claim
- High risk: May still hit other blockers
- 2-3 weeks effort for uncertain payoff

---

## Current Working Optimizations

Since you need **results to show**, these are already working:

| Optimization | Status | Speedup |
|-------------|--------|---------|
| **Token Rounding** | ✅ Verified | **+12.2%** |
| **FlashAttention 4** | ✅ Working | Baseline |
| Phase 1 Megakernel | ❌ Removed | -0.94% (slower) |
| Phase 2 SMEM Fusion | ❌ Not implemented | +4.8% theoretical |

**Best ROI**: Focus on documenting/benchmarking existing optimizations rather than building Phase 2 from scratch.

---

## Files Referenced

### Documentation
- `IMPROVEMENTS_SUMMARY.md` (Lines 156-193, 252-286)
- `PHASE1_MEGAKERNEL_ANALYSIS.md` (Lines 240-251)

### Existing Implementation
- `sonicmoe/functional/grouped_gemm.py` (SMEM layout logic)
- `sonicmoe/functional/moe_config.py` (config classes)
- `sonicmoe/functional/forward.py` (separate up/down kernels)

### Missing Implementation
- ❌ Fused kernel custom op
- ❌ Phase 2-specific config class
- ❌ Test suite
- ❌ Router logic for automatic selection
