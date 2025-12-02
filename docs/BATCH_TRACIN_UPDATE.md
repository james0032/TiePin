# Batch TracIn Update: NetworkX Filtering Support

## Summary

Updated [batch_tracin_with_filtering.py](batch_tracin_with_filtering.py:488) to support both **PyG** and **NetworkX** filtering methods.

## What Changed

### 1. New Function: `filter_training_data_networkx()`
- Added at line 141
- Calls `filter_training_networkx.py` instead of PyG
- Same interface as `filter_training_data()`

### 2. New Command-Line Option: `--filter-method`
- Added at line 399
- Choices: `pyg` (default) or `networkx`
- Allows switching between filtering implementations

### 3. Updated Main Logic
- Line 559: Choose filter method based on `--filter-method`
- Line 562-586: Conditional call to appropriate filter function
- Line 589: Track which method was used in results

### 4. Updated Summary Output
- Line 513: Add `filter_method` to summary JSON
- Line 678: Display filter method in final report

## Usage

### Default (PyG - Fastest)
```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/ \
    --n-hops 2
```

### NetworkX (More Transparent)
```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/ \
    --filter-method networkx \
    --n-hops 2 \
    --path-filtering
```

## Benefits

✅ **Flexibility**: Choose between speed (PyG) and transparency (NetworkX)
✅ **Validation**: Compare results from both implementations
✅ **Debugging**: Use NetworkX when PyG results are suspect
✅ **Backward Compatible**: Default behavior unchanged (uses PyG)

## Files Modified

- [batch_tracin_with_filtering.py](batch_tracin_with_filtering.py) - Main script
  - Added `filter_training_data_networkx()` function
  - Added `--filter-method` argument
  - Updated filtering logic to switch between methods
  - Updated summary output

## Files Created

- [BATCH_TRACIN_FILTERING_METHODS.md](BATCH_TRACIN_FILTERING_METHODS.md) - Detailed documentation

## Testing

Verify the update works:

```bash
# Test with NetworkX
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir test_output/ \
    --filter-method networkx \
    --n-hops 2 \
    --max-triples 1

# Check the summary
cat test_output/batch_tracin_summary.json | grep filter_method
```

Expected output:
```json
"filter_method": "networkx"
```

## Backward Compatibility

✅ All existing commands continue to work unchanged
✅ Default behavior is still PyG filtering
✅ No breaking changes to API or output format

## See Also

- [README_FILTERING.md](README_FILTERING.md) - Complete filtering guide
- [QUICK_START.md](QUICK_START.md) - Quick start for filtering
- [filter_training_networkx.py](filter_training_networkx.py) - NetworkX implementation
