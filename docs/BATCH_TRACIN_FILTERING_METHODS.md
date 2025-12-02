# Batch TracIn with Multiple Filtering Methods

The `batch_tracin_with_filtering.py` script now supports **two filtering implementations**: PyG (PyTorch Geometric) and NetworkX.

## Quick Start

### Using PyG Filtering (Default - Fastest)

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/batch_tracin \
    --n-hops 2 \
    --device cuda
```

### Using NetworkX Filtering (More Transparent)

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/batch_tracin \
    --filter-method networkx \
    --n-hops 2 \
    --path-filtering \
    --device cuda
```

## Filter Method Comparison

| Feature | PyG | NetworkX |
|---------|-----|----------|
| **Speed** | ⚡⚡⚡⚡⚡ Fastest | ⚡⚡⚡ Medium |
| **Memory** | Low-Medium | Medium-High |
| **Transparency** | ⭐⭐ Less transparent | ⭐⭐⭐⭐⭐ Most transparent |
| **Debugging** | Harder | Easier |
| **Graph Caching** | Yes (--cache) | No |
| **Best For** | Production, large graphs | Debugging, validation |

## When to Use Each Method

### Use PyG (Default)
- ✅ Production environments
- ✅ Large graphs (>500K edges)
- ✅ When speed is critical
- ✅ When you want graph caching

### Use NetworkX
- ✅ Debugging filtering issues
- ✅ Validating results
- ✅ Understanding edge filtering logic
- ✅ When you suspect PyG results are incorrect
- ✅ Small to medium graphs (<500K edges)

## Command-Line Option

Add `--filter-method` to choose the filtering implementation:

```bash
--filter-method pyg       # Default: PyTorch Geometric (fastest)
--filter-method networkx  # NetworkX (most transparent)
```

## Examples

### Example 1: Standard PyG with Path Filtering

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/pyg \
    --n-hops 2 \
    --path-filtering \
    --cache graph_cache.pkl \
    --device cuda
```

### Example 2: NetworkX with Verbose Logging

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/networkx \
    --filter-method networkx \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 3 \
    --device cuda
```

### Example 3: Compare Both Methods

Run the same data with both methods and compare results:

```bash
# Run with PyG
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/comparison_pyg \
    --filter-method pyg \
    --n-hops 2 \
    --path-filtering \
    --device cuda

# Run with NetworkX
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/comparison_networkx \
    --filter-method networkx \
    --n-hops 2 \
    --path-filtering \
    --device cuda

# Compare the filtered training files
diff results/comparison_pyg/filtered_training/ results/comparison_networkx/filtered_training/
```

## Output

The summary JSON file (`batch_tracin_summary.json`) now includes the filter method used:

```json
{
  "total_triples": 100,
  "successful": 95,
  "failed_filtering": 2,
  "failed_tracin": 3,
  "skipped": 0,
  "filter_method": "networkx",
  "elapsed_time_seconds": 3600.5,
  "elapsed_time_formatted": "1h 0m 0s",
  "results": [...]
}
```

Each result also includes the filter method:

```json
{
  "index": 0,
  "triple": {"head": "DRUG1", "relation": "treats", "tail": "DISEASE1"},
  "filtering_success": true,
  "tracin_success": true,
  "filter_method": "networkx",
  "filtered_train_file": "results/filtered_training/triple_000_DRUG1_DISEASE1_filtered_train.txt",
  "output_csv": "results/triple_000_DRUG1_DISEASE1_tracin.csv"
}
```

## Troubleshooting

### Problem: NetworkX filtering is too slow

**Solution**: Use PyG instead:
```bash
--filter-method pyg --cache graph_cache.pkl
```

### Problem: Suspect PyG results are incorrect

**Solution**: Compare with NetworkX results:
```bash
# Run both and compare filtered files
--filter-method networkx  # First run
--filter-method pyg       # Second run
# Then manually compare the filtered_training/ directories
```

### Problem: Want to understand filtering decisions

**Solution**: Use NetworkX with verbose mode by running the filter directly:
```bash
# First, extract a single test triple
head -1 test.txt > single_test.txt

# Run NetworkX filter with verbose output
python filter_training_networkx.py \
    --train train.txt \
    --test single_test.txt \
    --output filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --verbose
```

## Advanced: Switching Between Methods Mid-Batch

You can use different filter methods for different ranges of triples:

```bash
# Process first 50 triples with PyG
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    ... \
    --filter-method pyg \
    --max-triples 50 \
    --output-dir results/batch

# Process next 50 triples with NetworkX (for problematic triples)
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    ... \
    --filter-method networkx \
    --start-index 50 \
    --max-triples 50 \
    --output-dir results/batch \
    --skip-existing
```

## Implementation Details

### PyG Implementation
- Uses `filter_training_data()` function
- Calls `filter_training_by_proximity_pyg.py`
- Supports graph caching via `--cache` parameter
- Optimized C++/CUDA backend

### NetworkX Implementation
- Uses `filter_training_data_networkx()` function
- Calls `filter_training_networkx.py`
- Pure Python, no caching
- More transparent filtering logic

Both implementations:
- Support all filtering modes (intersection, strict hop, path filtering)
- Accept same parameters (n-hops, min-degree, etc.)
- Produce identical output format
- Should produce identical results (if not, there's a bug!)

## Performance Expectations

For a batch of 100 test triples with ~100K training edges:

| Filter Method | Time per Triple | Total Time (100 triples) |
|--------------|----------------|--------------------------|
| **PyG (with cache)** | 1-3s | ~3-5 minutes |
| **PyG (no cache)** | 2-5s | ~5-8 minutes |
| **NetworkX** | 5-10s | ~10-17 minutes |

*Times exclude TracIn analysis, which is typically much longer*

## See Also

- [README_FILTERING.md](README_FILTERING.md) - Complete filtering documentation
- [QUICK_START.md](QUICK_START.md) - Quick start guide for filtering
- [compare_all_implementations.py](compare_all_implementations.py) - Compare PyG vs NetworkX standalone
