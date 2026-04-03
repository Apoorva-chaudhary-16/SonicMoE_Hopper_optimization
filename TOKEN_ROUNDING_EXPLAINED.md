# Token Rounding: Detailed Explanation & Benchmark Commands

## Question 1: Is Token Rounding a Single Kernel or Megakernel?

### **Answer: Neither! It's a ROUTING ALGORITHM, not a kernel.**

**What Token Rounding Actually Is:**
- **Type**: Routing optimization algorithm (Python-level)
- **Location**: Applied at the router stage BEFORE kernels are called
- **What it does**: Adjusts token assignments to eliminate padding waste in GEMM tiles

### How It Works:

```
┌──────────────────────────────────────────────────────────┐
│ Standard Top-K Routing (Original)                        │
├──────────────────────────────────────────────────────────┤
│ 1. Compute router scores (T tokens × E experts)          │
│ 2. Select top-K experts per token                        │
│ 3. Result: Each expert gets arbitrary token count        │
│                                                           │
│ Example at E=128, K=2, T=16384:                         │
│   Expert 0: 130 tokens → needs 256 slots (2 tiles)      │
│   Expert 1: 65 tokens  → needs 128 slots (1 tile)       │
│   Expert 2: 193 tokens → needs 256 slots (2 tiles)      │
│                        ↓                                  │
│            24.2% WASTED COMPUTE on zero-padding!         │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Token Rounding ("nr" mode in mywork)                     │
├──────────────────────────────────────────────────────────┤
│ 1. Compute router scores (same as above)                 │
│ 2. Select top-K experts per token (same)                 │
│ 3. **NEW**: Round token counts to nearest 128-multiple   │
│    - Expert with 130 tokens → round down to 128          │
│    - Expert with 65 tokens  → round up to 128            │
│    - Expert with 193 tokens → round up to 256            │
│                        ↓                                  │
│            0% WASTED COMPUTE! Perfect tile alignment     │
└──────────────────────────────────────────────────────────┘
```

**Key Point**: The GEMM kernels (megakernel or standard) **don't change**. Token Rounding just ensures they receive perfectly aligned token counts.

---

## Question 2: How to Run Benchmarks & Verify 12.6% Improvement

### **IMPORTANT CLARIFICATION**

Looking at the actual code in `sonic-moe-mywork`, I found that the **12.6% claim is documented in the docs/** folder but **I cannot find evidence of actual benchmark results proving this number**.

Here's what I discovered:

### What EXISTS in the code:

1. **Test file**: `/home/gpu1/test/sonic-moe-mywork/tests/test_tr.py`
   - Very simple correctness test (23 lines)
   - Does NOT measure performance
   - Only verifies Token Rounding runs without errors

2. **Benchmark script**: `/home/gpu1/test/sonic-moe-mywork/benchmarks/moe-token-rounding.py`
   - Full performance benchmark (450+ lines)
   - Can compare `routing="top_k"` vs `routing="nr"` (Token Rounding)
   - Measures TFLOPS for forward, backward, end-to-end

3. **Documentation**: Numbers in `docs/optimization_plan.md` and `docs/professor_report.md`
   - Claims: +20.6% forward, +12.6% end-to-end
   - Config: T=16384, H=2048, I=1024, E=128, K=2
   - **BUT**: No saved results or logs showing these numbers

### **Commands to Actually Run the Benchmarks**

Here are the exact commands to reproduce (or verify) the 12.6% claim:

---

## A. Setup Both Environments

### 1. Original SonicMoE (Baseline)

```bash
# Navigate to original code
cd /home/gpu1/test/sonic-moe

# Activate virtual environment (or use Docker)
source venv/bin/activate  # or: source .venv/bin/activate

# Install if needed
pip install -e .

# Test that it works
python -c "from sonicmoe import MoE; print('Original SonicMoE loaded!')"
```

### 2. Your Modified Version (sonic-moe-mywork)

```bash
# Navigate to your improved code
cd /home/gpu1/test/sonic-moe-mywork

# Activate virtual environment
source venv/bin/activate  # or: source .venv/bin/activate

# Install if needed
pip install -e .

# Test that it works
python -c "from sonicmoe import MoE; print('Modified SonicMoE loaded!')"
```

---

## B. Run Token Rounding Benchmark (sonic-moe-mywork)

### **Configuration from docs (claimed 12.6% improvement)**:
- T=16384 (tokens)
- H=2048 (hidden dim)
- I=1024 (intermediate dim)
- E=128 (num experts)
- K=2 (experts per token)
- Mtile=128 (tile size)

### **Command 1: Baseline Top-K Routing**

```bash
cd /home/gpu1/test/sonic-moe-mywork

python benchmarks/moe-token-rounding.py \
    --thiekq 16384,2048,1024,128,2,128 \
    --routing top_k \
    --rep 200 \
    --skip_test
```

**Expected output** (example):
```
top_k, Fwd Average time: X.XXX ms, TFLOPS: 241.2
top_k, E2E Average time: X.XXX ms, TFLOPS: 231.6
```

### **Command 2: Token Rounding (nr mode)**

```bash
cd /home/gpu1/test/sonic-moe-mywork

python benchmarks/moe-token-rounding.py \
    --thiekq 16384,2048,1024,128,2,128 \
    --routing nr \
    --rep 200 \
    --skip_test
```

**Expected output** (if claims are true):
```
nr, Fwd Average time: X.XXX ms, TFLOPS: 291.1  ← +20.6% vs 241.2
nr, E2E Average time: X.XXX ms, TFLOPS: 261.0  ← +12.6% vs 231.6
```

### **Command 3: Run Both and Compare**

```bash
cd /home/gpu1/test/sonic-moe-mywork

echo "=== BASELINE TOP-K ===" && \
python benchmarks/moe-token-rounding.py \
    --thiekq 16384,2048,1024,128,2,128 \
    --routing top_k \
    --rep 200 \
    --skip_test && \
echo "" && \
echo "=== TOKEN ROUNDING ===" && \
python benchmarks/moe-token-rounding.py \
    --thiekq 16384,2048,1024,128,2,128 \
    --routing nr \
    --rep 200 \
    --skip_test
```

---

## C. Check if Original SonicMoE Supports Token Rounding

### Does the original code have Token Rounding?

```bash
# Check if "nr" routing mode exists in original
cd /home/gpu1/test/sonic-moe
grep -r "routing.*nr\|token.*rounding" sonicmoe/ benchmarks/
```

**Expected result**: Likely **NOT FOUND** in original code (this is YOUR contribution!)

### If original doesn't have it:

The 12.6% improvement is **NOT** "original vs mywork" comparison.

Instead, it's:
- **Baseline**: Your code with standard top-k routing (`routing="top_k"`)
- **Improved**: Your code with Token Rounding (`routing="nr"`)

Both measured in the **same codebase** (sonic-moe-mywork).

---

## D. How to Actually Verify the 12.6% Claim

### **Step-by-step verification process:**

1. **Run baseline on mywork**:
   ```bash
   cd /home/gpu1/test/sonic-moe-mywork
   python benchmarks/moe-token-rounding.py \
       --thiekq 16384,2048,1024,128,2,128 \
       --routing top_k \
       --rep 200 \
       --skip_test | tee baseline_topk.log
   ```

2. **Run Token Rounding on mywork**:
   ```bash
   cd /home/gpu1/test/sonic-moe-mywork
   python benchmarks/moe-token-rounding.py \
       --thiekq 16384,2048,1024,128,2,128 \
       --routing nr \
       --rep 200 \
       --skip_test | tee improved_nr.log
   ```

3. **Compare results**:
   ```bash
   echo "=== BASELINE ===" && grep "E2E Average" baseline_topk.log
   echo "=== TOKEN ROUNDING ===" && grep "E2E Average" improved_nr.log
   ```

4. **Calculate improvement**:
   ```python
   # Example calculation
   baseline_tflops = 231.6  # From baseline log
   nr_tflops = 261.0        # From Token Rounding log
   
   improvement = (nr_tflops - baseline_tflops) / baseline_tflops * 100
   print(f"Improvement: {improvement:.1f}%")  # Should show ~12.6%
   ```

---

## E. Quick Correctness Test (Not Performance)

If you just want to verify Token Rounding **works** (not measure performance):

```bash
cd /home/gpu1/test/sonic-moe-mywork
python tests/test_tr.py
```

**Expected output**:
```
Top-K success!
Token Rounding success!
```

This only proves correctness, **NOT** the 12.6% speedup.

---

## Summary Table

| What to Run | Command | What It Tests | Where |
|-------------|---------|---------------|-------|
| **Correctness test** | `python tests/test_tr.py` | Token Rounding runs without error | sonic-moe-mywork |
| **Baseline benchmark** | `python benchmarks/moe-token-rounding.py --thiekq 16384,2048,1024,128,2,128 --routing top_k --rep 200 --skip_test` | Standard Top-K routing TFLOPS | sonic-moe-mywork |
| **Token Rounding benchmark** | `python benchmarks/moe-token-rounding.py --thiekq 16384,2048,1024,128,2,128 --routing nr --rep 200 --skip_test` | Token Rounding TFLOPS | sonic-moe-mywork |
| **Original code check** | `grep -r "nr\|token.*round" sonicmoe/` | Does original have TR? | sonic-moe |

---

## Key Findings

1. **Token Rounding is NOT a kernel** - it's a routing algorithm
2. **The 12.6% claim** is documented but needs verification by running benchmarks
3. **Comparison is within your codebase**: 
   - Baseline: `routing="top_k"` in sonic-moe-mywork
   - Improved: `routing="nr"` in sonic-moe-mywork
4. **Original SonicMoE** likely doesn't have Token Rounding at all (this is your contribution)

---

## Recommended Next Steps

1. **Run the benchmarks** using commands in Section D
2. **Save the output logs** to document actual results
3. **If results differ from 12.6%**, update documentation with actual measured numbers
4. **For presentation**: Show the benchmark command and live results (more convincing than docs)

---

Generated: April 3, 2026  
Location: /home/gpu1/testing/SonicMoE_Hopper_optimization/TOKEN_ROUNDING_EXPLAINED.md
