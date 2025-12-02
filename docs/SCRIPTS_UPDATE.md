# Batch TracIn Scripts Update

## Summary

Updated both example scripts to use **NetworkX filtering** instead of PyG.

## Files Updated

1. [run_batch_tracin_example.sh](run_batch_tracin_example.sh)
2. [run_batch_tracin_example_CCGGDD_hatteras.sh](run_batch_tracin_example_CCGGDD_hatteras.sh)

## Changes Made

### 1. Added `--filter-method networkx`

Both scripts now explicitly use NetworkX filtering:
```bash
--filter-method networkx \
```

### 2. Removed `--cache` parameter

NetworkX doesn't support graph caching, so the `--cache "${GRAPH_CACHE}"` line was removed from the python command.

**Note**: The `GRAPH_CACHE` variable is still defined at the top of the scripts for backward compatibility, but it's no longer passed to the command.

### 3. Updated Configuration Display

Added filter method to the configuration output:
```bash
echo "  - Filter method: NetworkX (transparent, easy to debug)"
```

## Why NetworkX?

✅ **Transparent** - Easy to understand filtering logic
✅ **Debuggable** - Can see exactly which edges are kept/rejected
✅ **Validated** - Independent implementation validates PyG results
✅ **Path Filtering** - Better implementation of drug→disease path logic

## Performance Impact

NetworkX is slightly slower than PyG but still fast enough:
- PyG with cache: ~1-3s per triple
- NetworkX: ~5-10s per triple

For 25-50 triples, the difference is minimal (1-2 minutes longer total time).

## How to Switch Back to PyG

If you need to switch back to PyG (faster but less transparent):

### Option 1: Edit the scripts
Change:
```bash
--filter-method networkx \
```
To:
```bash
--filter-method pyg \
```

And add back the cache parameter:
```bash
--cache "${GRAPH_CACHE}" \
```

### Option 2: Run with custom parameters
```bash
python batch_tracin_with_filtering.py \
    ... \
    --filter-method pyg \
    --cache /path/to/cache.pkl \
    ...
```

## Verification

To verify the scripts are using NetworkX:

```bash
# Run the script
bash scripts/run_batch_tracin_example.sh

# Check the output - should see:
# "Filter method: NetworkX (transparent, easy to debug)"
# "Running NetworkX filter: python filter_training_networkx.py ..."
```

## Summary JSON Output

The `batch_tracin_summary.json` file will now include:
```json
{
  "filter_method": "networkx",
  ...
}
```

Each individual result will also include:
```json
{
  "filter_method": "networkx",
  ...
}
```

## Testing

Quick test to ensure it works:

```bash
# Test with first triple only
bash scripts/run_batch_tracin_example.sh
# (Will process all triples, but you can stop after first one completes)
```

Expected log messages:
```
Step 1/2: Filtering training data using NETWORKX...
Running NetworkX filter: python filter_training_networkx.py ...
NetworkX filtering completed successfully
```

## Rollback Instructions

If you need to revert to the previous PyG-only version:

```bash
# Restore from git
cd /Users/jchung/Documents/RENCI/everycure/git/conve_pykeen
git checkout HEAD -- scripts/run_batch_tracin_example.sh
git checkout HEAD -- scripts/run_batch_tracin_example_CCGGDD_hatteras.sh
```

Or manually:
1. Remove `--filter-method networkx \` line
2. Add back `--cache "${GRAPH_CACHE}" \` line
3. Remove "Filter method: NetworkX" from echo statements
