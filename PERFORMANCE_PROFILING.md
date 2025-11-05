# Performance Profiling Guide for filter_training_by_proximity_pyg.py

## Overview

The script now includes detailed timing measurements and profiling capabilities to help identify performance bottlenecks, especially when using cached graphs.

## Features Added

### 1. **Automatic Timing Breakdown** (Always On)

Every run now shows a detailed breakdown of where time is spent:

```
PERFORMANCE BREAKDOWN
============================================================
  3. Build entity/relation ID mappings      :    45.23s ( 52.1%)
  4. Convert triples to numeric format      :    18.67s ( 21.5%)
  1. Load training triples from disk        :    12.34s ( 14.2%)
  6. Filter triples (k-hop subgraph + degree):     5.45s (  6.3%)
  5. Initialize ProximityFilterPyG (cache)  :     2.10s (  2.4%)
  7. Convert filtered triples back to strings:    1.89s (  2.2%)
  2. Load test triples from disk            :     0.98s (  1.1%)
  8. Write filtered triples to disk         :     0.15s (  0.2%)
  9. Compute statistics                     :     0.02s (  0.0%)
------------------------------------------------------------
  TOTAL                                     :    86.83s (100.0%)
============================================================
```

**Key Insights from Timing:**
- Even with cache, steps 1-4 (file I/O and mapping) still run every time
- The cache only speeds up step 5 (graph initialization)
- Steps 3-4 (ID mapping and conversion) are often the biggest bottleneck

### 2. **Detailed Profiling with cProfile** (Optional)

Use `--profile` flag for function-level profiling:

```bash
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --cache graph_cache.pkl \
    --profile
```

This will:
- Show top 30 functions by cumulative time
- Save detailed stats to `profile.stats`
- Allow interactive analysis

## Usage Examples

### Basic Usage (Timing Only)

```bash
python filter_training_by_proximity_pyg.py \
    --train /workspace/data/train.txt \
    --test /workspace/data/test.txt \
    --output /workspace/data/train_filtered.txt \
    --cache /workspace/data/graph_cache.pkl \
    --n-hops 2 \
    --min-degree 2
```

Output will include automatic timing breakdown at the end.

### With Detailed Profiling

```bash
python filter_training_by_proximity_pyg.py \
    --train /workspace/data/train.txt \
    --test /workspace/data/test.txt \
    --output /workspace/data/train_filtered.txt \
    --cache /workspace/data/graph_cache.pkl \
    --profile
```

### Using the Test Script

```bash
./test_performance.sh train.txt test.txt graph_cache.pkl
```

This runs the script twice:
1. Once with timing breakdown
2. Once with full profiling

## Analyzing Profile Results

### View Interactively

```bash
python -m pstats profile.stats
```

### Common Commands in pstats

```
stats 20           # Show top 20 functions
sort cumulative    # Sort by cumulative time
sort time          # Sort by internal time (excludes subcalls)
callers <func>     # Show what calls this function
callees <func>     # Show what this function calls
strip             # Remove directory paths for cleaner output
```

### Generate Reports

```bash
# Top 50 functions by cumulative time
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(50)"

# Functions containing 'numpy'
python -c "import pstats; p = pstats.Stats('profile.stats'); p.print_stats('numpy')"
```

## Understanding the Bottlenecks

### Why is it slow even with cache?

The cache only stores:
- Graph structure (`edge_index`)
- Edge relations
- Node degrees
- Edge-to-triple mappings

The cache does **NOT** store:
- Entity/relation ID mappings
- Pre-converted numeric triples
- File I/O operations

### What still happens every run?

1. **Load files from disk** - Reading all triples from text files
2. **Build ID mappings** - Creating dictionaries mapping strings to integers
3. **Convert to numeric** - Transforming all triples using the mappings
4. **Convert back to strings** - Reverse mapping for output

### Expected Performance Profile

**With cache hit:**
```
Step 1-4 (I/O & mapping):    ~70-80% of time
Step 5 (graph init):         ~2-5% of time  ✓ Cached!
Step 6 (filtering):          ~10-20% of time
Step 7-9 (output):           ~5-10% of time
```

**Without cache (first run):**
```
Step 1-4 (I/O & mapping):    ~40-50% of time
Step 5 (graph init):         ~30-40% of time  ✗ Must build
Step 6 (filtering):          ~10-20% of time
Step 7-9 (output):           ~5-10% of time
```

## Optimization Recommendations

Based on profiling results, consider:

1. **Cache entity/relation mappings** - Save these to disk too
2. **Use binary formats** - Replace TSV with pickle/parquet for faster I/O
3. **Pre-compute numeric arrays** - Save converted triples
4. **Memory-map large files** - Use mmap for huge files
5. **Parallel processing** - Use multiprocessing for ID mapping

## Troubleshooting

### Profile file is huge
- This is normal for large datasets
- Use `strip` command in pstats to reduce size
- Filter results: `p.print_stats('filter_training')`

### Can't find bottleneck
- Check cumulative vs. internal time
- Look for repeated function calls
- Profile multiple runs and compare

### Memory issues
- Use `--single-triple` mode for testing
- Profile on a subset of data first
- Monitor with `memory_profiler` package

## Additional Tools

### Memory Profiling

Install: `pip install memory_profiler`

```bash
python -m memory_profiler filter_training_by_proximity_pyg.py --train ... --test ...
```

### Line-by-line Profiling

Install: `pip install line_profiler`

Add `@profile` decorator to functions and run:

```bash
kernprof -l -v filter_training_by_proximity_pyg.py --train ... --test ...
```

## Example Output

```
============================================================
PERFORMANCE BREAKDOWN
============================================================
  3. Build entity/relation ID mappings      :    45.23s ( 52.1%)  ← BOTTLENECK
  4. Convert triples to numeric format      :    18.67s ( 21.5%)  ← BOTTLENECK
  1. Load training triples from disk        :    12.34s ( 14.2%)
  6. Filter triples (k-hop subgraph + degree):     5.45s (  6.3%)
  5. Initialize ProximityFilterPyG (cache)  :     2.10s (  2.4%)  ✓ Fast with cache
  7. Convert filtered triples back to strings:    1.89s (  2.2%)
  2. Load test triples from disk            :     0.98s (  1.1%)
  8. Write filtered triples to disk         :     0.15s (  0.2%)
  9. Compute statistics                     :     0.02s (  0.0%)
------------------------------------------------------------
  TOTAL                                     :    86.83s (100.0%)
============================================================
```

**Interpretation:**
- Steps 3 & 4 take 73.6% of total time
- Cache is working (step 5 only 2.4%)
- To optimize further, need to cache ID mappings and numeric conversions
