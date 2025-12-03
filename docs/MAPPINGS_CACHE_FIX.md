# Mappings Cache Fix: Don't Cache Test Triples

## Problem Discovered

The mappings cache was caching `test_numeric` (numeric conversion of test triples), which caused **all batch iterations to use the same test triple** - the one from the first iteration!

### Root Cause

In batch processing with `batch_tracin_with_filtering.py`:

1. **Iteration 1**: Creates temp file with test triple #1
   - Builds mappings from train.txt + temp test file
   - Converts test triple #1 to numeric
   - **Saves to cache:** `test_numeric` = [test triple #1]
   - Filters correctly for test triple #1

2. **Iteration 2**: Creates temp file with test triple #2
   - **Loads from cache:** gets cached `test_numeric` = [test triple #1]  ❌
   - Uses wrong test triple for filtering!
   - Result is identical to iteration 1

3. **Iterations 3-25**: Same problem - all use test triple #1

### User's Observation

```
2-hop paths: 3
3-hop paths: 2021
Extracted 2283 unique edges from 2024 paths

But edge kept was more than 2283
  Filtered triples: 3929
  Edges kept: 3929
```

And more importantly: **All 25 iterations produced identical filtered training sets!**

## Solution

**Test triples should NEVER be cached** because they change on every iteration in batch processing.

Only cache:
- ✅ `entity_to_idx`: Entity → ID mappings (same across all iterations)
- ✅ `idx_to_entity`: ID → Entity mappings (same across all iterations)
- ✅ `relation_to_idx`: Relation → ID mappings (same across all iterations)
- ✅ `train_numeric`: Numeric training triples (same across all iterations)
- ❌ `test_numeric`: **REMOVED from cache** (changes every iteration)

## Implementation

### Changes to [filter_training_igraph.py](../filter_training_igraph.py)

#### 1. Load Phase - Don't load `test_numeric` from cache:

```python
# Before (BROKEN):
test_numeric = cache_data['test_numeric']  # ❌ Cached test triple from iteration 1!

# After (FIXED):
# test_numeric is NOT in cache anymore
# It's always computed fresh for current test file
```

#### 2. Always Load Test Triples Fresh:

```python
# Load test triples (always loaded fresh, never cached)
logger.info(f"Loading test triples from {test_file}")
test_triples = load_triples_from_file(test_file)
```

#### 3. Always Convert Test Triples to Numeric:

```python
# Always convert test triples to numeric (never cached)
test_numeric = np.array([
    [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
    for h, r, t in test_triples
])
```

#### 4. Save Phase - Don't save `test_numeric`:

```python
# Before (BROKEN):
cache_data = {
    'entity_to_idx': entity_to_idx,
    'idx_to_entity': idx_to_entity,
    'relation_to_idx': relation_to_idx,
    'train_numeric': train_numeric,
    'test_numeric': test_numeric  # ❌ This was causing the bug!
}

# After (FIXED):
cache_data = {
    'entity_to_idx': entity_to_idx,
    'idx_to_entity': idx_to_entity,
    'relation_to_idx': relation_to_idx,
    'train_numeric': train_numeric
    # test_numeric is NOT cached
}
```

## Expected Behavior After Fix

### First Iteration
```
Loading test triples from /tmp/test_triple_0.txt  # Test triple #1
Loaded 1 test triples
Creating entity/relation mappings...
Converting training triples to numeric format...
Saving mappings to cache (test triples NOT cached)

Finding all simple paths between 1 drugs and 1 diseases...
  Filtered triples: 2283  # Correct for test triple #1
```

### Second Iteration
```
Loading mappings and train numeric from cache  # ✅ Uses cached mappings
Loading test triples from /tmp/test_triple_1.txt  # ✅ Test triple #2 (DIFFERENT!)
Loaded 1 test triples
Using cached mappings for 1 test triples

Finding all simple paths between 1 drugs and 1 diseases...
  Filtered triples: 1892  # ✅ Different! Correct for test triple #2
```

### Subsequent Iterations
Each iteration:
- ✅ Loads cached mappings (fast!)
- ✅ Loads FRESH test triple (different each time!)
- ✅ Produces DIFFERENT filtered training sets

## Performance Impact

**No performance loss!** The test conversion is trivial:

- Loading test file: ~0.001 seconds (single triple)
- Converting to numeric: ~0.0001 seconds (single triple lookup)
- **Total overhead: ~0.001 seconds per iteration**

Compare to:
- Loading/building mappings: ~0.5 seconds (SAVED by caching)
- Building graph: ~1.5 seconds (SAVED by caching)

**Net speedup still ~20x** even without caching test triples!

## Cache File Changes

### Before (Broken)
```python
cache_data = {
    'entity_to_idx': {...},      # 5 MB
    'idx_to_entity': {...},      # 5 MB
    'relation_to_idx': {...},    # 1 MB
    'train_numeric': np.array,   # 10 MB
    'test_numeric': np.array     # 0.001 MB  ❌ WRONG DATA CACHED
}
# Total: ~21 MB
```

### After (Fixed)
```python
cache_data = {
    'entity_to_idx': {...},      # 5 MB
    'idx_to_entity': {...},      # 5 MB
    'relation_to_idx': {...},    # 1 MB
    'train_numeric': np.array    # 10 MB
    # test_numeric NOT cached
}
# Total: ~21 MB (same size, but correct behavior!)
```

## How to Apply Fix

### Option 1: Delete Old Cache (Recommended)

```bash
# Delete old cache with corrupted test_numeric
rm /path/to/train_graph_cache_mappings.pkl

# Next run will create new cache WITHOUT test_numeric
bash scripts/run_batch_tracin_example.sh
```

### Option 2: Update Code (Already Done)

The fix is already in [filter_training_igraph.py](../filter_training_igraph.py) - just delete the old cache file.

## Verification

To verify the fix is working:

```bash
# Run batch processing
bash scripts/run_batch_tracin_example.sh

# Check that each iteration produces DIFFERENT output:
ls -lh /path/to/output/filtered_training/

# You should see DIFFERENT file sizes:
filtered_test_0_train.txt  # 450 KB
filtered_test_1_train.txt  # 380 KB  ← Different!
filtered_test_2_train.txt  # 520 KB  ← Different!
# etc.
```

## Related Issues

This same bug affected:
- ❌ All batch TracIn runs that used mappings cache
- ❌ Any workflow that processes multiple test triples sequentially
- ✅ Single-triple filtering (no batch processing) - unaffected

## Prevention

Added comment in code to prevent regression:

```python
# NOTE: We do NOT cache test_numeric because it changes in batch processing
```

And updated log message:
```python
logger.info(f"Mappings cache saved successfully (test triples NOT cached)")
```

## Documentation Updates

Updated:
- [MAPPINGS_CACHING_UPDATE.md](MAPPINGS_CACHING_UPDATE.md) - To reflect test triples are NOT cached
- [IGRAPH_CACHING_UPDATE.md](IGRAPH_CACHING_UPDATE.md) - Updated cache contents section

## Summary

**Before Fix:**
- All 25 batch iterations used test triple #1
- Produced identical filtered training sets
- Wrong TracIn influence scores

**After Fix:**
- Each batch iteration uses its own test triple
- Produces different filtered training sets
- Correct TracIn influence scores for each test triple

**Action Required:**
```bash
# Delete old cache
rm /path/to/*_mappings.pkl

# Run batch processing
bash scripts/run_batch_tracin_example.sh
```

The new cache will be created WITHOUT test_numeric, and each iteration will work correctly!
