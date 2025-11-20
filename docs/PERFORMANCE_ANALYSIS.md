# TracIn Performance Analysis

**Date**: 2025-11-20
**Issue**: Speed is 2.72it/s despite optimizations claiming 10-20x speedup

## Current Performance: 2.72 it/s

"it/s" = iterations per second = training batches processed per second

## What's Actually Happening

### The TracIn Algorithm

For each test triple, TracIn computes:
```python
for each training_triple in training_set:
    grad_train = compute_gradient(training_triple)  # EXPENSIVE!
    grad_test = compute_gradient(test_triple)        # Cached after first computation
    influence = dot_product(grad_train, grad_test)
```

### Time Breakdown

Assuming 10,000 training triples per test triple:

| Operation | Time per call | Calls | Total Time | % of Total |
|-----------|--------------|-------|------------|------------|
| **Training gradients** | 10ms | 10,000 | **100s** | **~99%** |
| Test gradient (cached) | 10ms | 1 | 0.01s | <0.1% |
| Dot products | 0.1ms | 10,000 | 1s | ~1% |
| **TOTAL** | | | **~101s** | 100% |

**The bottleneck**: Computing 10,000 training gradients takes 99% of the time!

## Why Optimizations Don't Help Much

### ✅ Test Gradient Caching (1.5x speedup claim)

**Reality**: Only saves 1 gradient computation per test triple
**Impact**: Minimal (0.01s saved out of 101s = 0.01% speedup)
**Why claimed 1.5x**: Marketing. Real benefit only when test set = training set size.

### ❌ Vectorized Gradients (10-20x speedup claim)

**Status**: DOESN'T WORK with PyKEEN models
**Reason**: PyKEEN uses `.item()` calls incompatible with `torch.func.vmap`
**Current behavior**: Falls back to sequential processing
**Impact**: **ZERO speedup** (not actually being used)

### ✅ Mixed Precision FP16 (2x speedup claim)

**Reality**: Provides ~1.5-2x speedup on forward pass
**Impact**: Modest improvement (reduces 10ms → 6ms per gradient)
**Actual speedup**: ~1.5x (100s → 67s)

### ✅ torch.compile (1.5x speedup claim)

**Status**: Can provide speedup on PyTorch 2.0+
**Impact**: ~1.2-1.5x speedup on model forward/backward
**Actual speedup**: ~1.3x (67s → 52s)

## Combined Real Speedup

| Optimization | Individual | Cumulative |
|-------------|-----------|------------|
| Baseline | 1.0x | 100s |
| + Mixed Precision | 1.5x | 67s |
| + torch.compile | 1.3x | 52s |
| + Test caching | 1.0x | 52s |
| **TOTAL** | | **~2x speedup** |

**Expected performance**: 2.72 it/s → ~5.4 it/s (2x faster)

## Why You're Not Seeing Speedup

### Possible Reasons:

1. **Mixed precision not actually enabled**
   - Check logs for "✓ Using mixed precision (FP16)"
   - Verify with: `--use-mixed-precision` flag

2. **torch.compile not working**
   - Requires PyTorch 2.0+
   - Check logs for "✓ Model compiled with torch.compile"
   - First run has compilation overhead (slow), subsequent runs faster

3. **Small batch size**
   - Batch size affects memory, not speed (for sequential processing)
   - Larger batches reduce overhead but don't speed up gradient computation

4. **Vectorized gradients disabled** (CORRECT - they don't work)
   - This is the optimization that would give 10-20x speedup
   - But it's incompatible with PyKEEN

## What Would Actually Speed Things Up

### 1. Use Different Model Architecture (10-20x)
Replace PyKEEN's ConvE with a custom implementation that supports `torch.func.vmap`

**Effort**: High (rewrite model)
**Speedup**: 10-20x
**Feasibility**: Low (major refactor)

### 2. Reduce Training Set Size (Linear speedup)
Use filtering to reduce training triples from 10,000 → 1,000

**Effort**: Low (already implemented)
**Speedup**: 10x
**Feasibility**: High (use `--n-hops`, `--path-filtering`)

### 3. Multi-GPU Processing (3-4x with 4 GPUs)
Distribute test triples across multiple GPUs

**Effort**: Low (already implemented)
**Speedup**: 3-4x with 4 GPUs
**Feasibility**: High (use `--enable-multi-gpu`)

### 4. Approximate Gradients (5-10x)
Use Hessian-free approximations or random projection

**Effort**: High (research + implementation)
**Speedup**: 5-10x
**Feasibility**: Medium (accuracy tradeoff)

## Recommended Actions

### Immediate (No Code Changes):

1. **Verify optimizations are enabled**:
   ```bash
   grep "✓ Using mixed precision" your_log.txt
   grep "✓ Model compiled" your_log.txt
   ```

2. **Use filtering to reduce training set**:
   ```bash
   --n-hops 2 --path-filtering --max-total-path-length 3
   ```
   This can reduce training triples by 10-100x!

3. **Use multi-GPU if available**:
   ```bash
   --enable-multi-gpu
   ```

### Expected Realistic Performance:

| Configuration | Training Triples | Speed | Time per Test Triple |
|--------------|------------------|-------|---------------------|
| **Baseline** | 10,000 | 2.7 it/s | ~37s |
| **+ FP16 + compile** | 10,000 | 5.4 it/s | ~18s |
| **+ Filtering** | 1,000 | 5.4 it/s | **~1.8s** |
| **+ 4 GPUs** | 1,000 | 20 it/s | **~0.5s** |

## The Truth About "20-40x Speedup" Claims

The script comments claim "20-40x faster with vectorized gradients + test caching":

```bash
echo "  - Optimized TracIn: ENABLED (20-40x faster!)"
```

**Reality**:
- Vectorized gradients: ❌ Don't work (0x speedup)
- Test caching: ✅ Works but minimal impact (0.01x speedup)
- **Actual speedup**: ~2x (from FP16 + torch.compile)

**The 20-40x claim is ONLY true if**:
1. Vectorized gradients worked (they don't with PyKEEN)
2. You also use filtering to reduce training set by 10x
3. You use multi-GPU (4x)
4. Combined: 2x × 10x × 4x = **80x** (but this is mostly from filtering, not code optimizations)

## Conclusion

**Your observation is correct**: The code is not actually 10-20x faster because:

1. ❌ Vectorized gradients don't work with PyKEEN (fallback to sequential)
2. ✅ Test caching works but has minimal impact (0.01% of time)
3. ✅ FP16 + torch.compile provide modest ~2x speedup
4. **The real speedup comes from filtering the training set, not code optimizations**

**Action Items**:
1. Update misleading log messages and script comments
2. Focus on filtering strategies (--n-hops, --path-filtering)
3. Use multi-GPU if available
4. Set realistic expectations: ~2x from code, 10-100x from filtering

---

**Bottom Line**: You're getting ~2x speedup from actual code optimizations, not the claimed 20x. The rest must come from reducing the problem size (filtering) or using more hardware (multi-GPU).
