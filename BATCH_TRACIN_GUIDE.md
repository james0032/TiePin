# Batch TracIn Analysis with Filtering - User Guide

## Overview

The `batch_tracin_with_filtering.py` script automates the entire TracIn analysis pipeline for multiple test triples:

1. **Reads test triples** from a file
2. **Filters training data** for each triple (using proximity-based filtering)
3. **Runs TracIn analysis** on filtered data
4. **Generates CSV outputs** with influence scores

This approach provides **10-50x speedup** by filtering training data before running TracIn.

---

## Quick Start

### Basic Usage

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --model-path path/to/trained_model.pt \
    --train path/to/train.txt \
    --entity-to-id path/to/entity_to_id.tsv \
    --relation-to-id path/to/relation_to_id.tsv \
    --edge-map path/to/edge_map.json \
    --node-name-dict path/to/node_name_dict.txt \
    --output-dir results/batch_tracin \
    --device cuda
```

This will:
- Process all 25 triples from the file
- Create filtered training data for each
- Run TracIn analysis on each
- Save results to `results/batch_tracin/`

---

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--test-triples` | Path to file with test triples (tab-separated) |
| `--model-path` | Path to trained ConvE model (.pt) |
| `--train` | Path to training triples file |
| `--entity-to-id` | Path to entity_to_id.tsv |
| `--relation-to-id` | Path to relation_to_id.tsv |
| `--output-dir` | Directory for output files |

### Optional Label Files

| Argument | Description |
|----------|-------------|
| `--edge-map` | Path to edge_map.json (for predicate names) |
| `--node-name-dict` | Path to node_name_dict.txt (for entity names) |

### Filtering Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-hops` | 2 | Number of hops for proximity filtering |
| `--min-degree` | 2 | Minimum degree threshold |
| `--cache` | None | Path to cache graph (speeds up filtering) |
| `--no-preserve-test-edges` | False | Don't preserve edges with test entities |

### TracIn Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--top-k` | None | Number of top influences (None = all influences) |
| `--device` | cuda | Device (cuda or cpu) |
| `--batch-size` | 512 | Batch size for processing |
| `--no-use-last-layers` | False | Track all layers (slower) |
| `--num-last-layers` | 2 | Number of last layers to track |

### Execution Control

| Argument | Default | Description |
|----------|---------|-------------|
| `--start-index` | 0 | Start from this triple index |
| `--max-triples` | None | Maximum number of triples to process |
| `--skip-filtering` | False | Skip filtering (use existing files) |
| `--skip-tracin` | False | Only do filtering, skip TracIn |

---

## Usage Examples

### Example 1: Process All 25 Top Triples

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --model-path trained_model.pt \
    --train data/train.txt \
    --entity-to-id data/entity_to_id.tsv \
    --relation-to-id data/relation_to_id.tsv \
    --edge-map data/edge_map.json \
    --node-name-dict data/node_name_dict.txt \
    --output-dir results/all_top_triples \
    --n-hops 2 \
    --min-degree 2 \
    --cache data/train_graph_cache.pkl \
    --top-k 100 \
    --device cuda \
    --batch-size 512
```

**Expected time:** ~8-10 minutes for 25 triples (with caching and last-layers optimization)

---

### Example 2: Process Top 5 Triples Only

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --model-path trained_model.pt \
    --train data/train.txt \
    --entity-to-id data/entity_to_id.tsv \
    --relation-to-id data/relation_to_id.tsv \
    --edge-map data/edge_map.json \
    --output-dir results/top5_triples \
    --max-triples 5 \
    --device cuda
```

---

### Example 3: Resume from Triple #10

If processing was interrupted, resume from triple 10:

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --model-path trained_model.pt \
    --train data/train.txt \
    --entity-to-id data/entity_to_id.tsv \
    --relation-to-id data/relation_to_id.tsv \
    --output-dir results/all_top_triples \
    --start-index 10 \
    --device cuda
```

---

### Example 4: Only Filter Training Data (Skip TracIn)

Generate filtered training files without running TracIn:

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --train data/train.txt \
    --entity-to-id data/entity_to_id.tsv \
    --relation-to-id data/relation_to_id.tsv \
    --output-dir results/filtered_only \
    --skip-tracin \
    --cache data/train_graph_cache.pkl
```

Note: `--model-path` is not required when using `--skip-tracin`

---

### Example 5: Run TracIn on Pre-Filtered Data

If you already have filtered training files:

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --model-path trained_model.pt \
    --train data/train.txt \
    --entity-to-id data/entity_to_id.tsv \
    --relation-to-id data/relation_to_id.tsv \
    --edge-map data/edge_map.json \
    --output-dir results/all_top_triples \
    --skip-filtering \
    --device cuda
```

---

## Output Structure

### Directory Layout

```
results/batch_tracin/
├── filtered_training/                           # Filtered training files
│   ├── triple_000_CHEBI_7963_MONDO_0016595_filtered_train.txt
│   ├── triple_001_CHEBI_34911_MONDO_0004525_filtered_train.txt
│   └── ... (one per test triple)
│
├── temp_triples/                                # Temporary single-triple files
│   ├── triple_000_CHEBI_7963_MONDO_0016595.txt
│   ├── triple_001_CHEBI_34911_MONDO_0004525.txt
│   └── ...
│
├── triple_000_CHEBI_7963_MONDO_0016595_tracin.json
├── triple_000_CHEBI_7963_MONDO_0016595_tracin.csv
├── triple_001_CHEBI_34911_MONDO_0004525_tracin.json
├── triple_001_CHEBI_34911_MONDO_0004525_tracin.csv
├── ... (two files per test triple)
└── batch_tracin_summary.json                    # Summary of all results
```

### Output Files

#### 1. Filtered Training Files
**Location:** `filtered_training/triple_XXX_*_filtered_train.txt`

**Format:** Tab-separated triples
```
CHEBI:123	predicate:28	MONDO:456
NCBIGene:789	predicate:29	MONDO:456
...
```

**Purpose:** Reduced training set for faster TracIn analysis

---

#### 2. TracIn CSV Files
**Location:** `triple_XXX_*_tracin.csv`

**Format:** Exact specification with labels
```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
CHEBI:34911,Permethrin,predicate:28,biolink:treats,MONDO:0004525,scabies,NCBIGene:123,GENE1,predicate:29,biolink:target_for,MONDO:0004525,scabies,0.3972
```

---

#### 3. TracIn JSON Files
**Location:** `triple_XXX_*_tracin.json`

**Format:** Detailed results with metadata
```json
[
  {
    "test_triple": [13416, 21, 70064],
    "test_triple_index": 0,
    "test_head": 13416,
    "test_head_label": "CHEBI:34911",
    "test_head_name": "Permethrin",
    "influences": [
      {
        "train_head": 12345,
        "train_relation": 29,
        "train_tail": 70064,
        "influence": 0.3972,
        "train_head_label": "NCBIGene:123",
        ...
      }
    ]
  }
]
```

---

#### 4. Summary File
**Location:** `batch_tracin_summary.json`

**Format:**
```json
{
  "total_triples": 25,
  "successful": 23,
  "failed_filtering": 1,
  "failed_tracin": 1,
  "skipped": 0,
  "elapsed_time_seconds": 487.3,
  "elapsed_time_formatted": "0h 8m 7s",
  "results": [
    {
      "index": 0,
      "triple": {"head": "CHEBI:7963", "relation": "predicate:28", "tail": "MONDO:0016595"},
      "base_name": "triple_000_CHEBI_7963_MONDO_0016595",
      "filtering_success": true,
      "tracin_success": true,
      "filtered_train_file": "...",
      "output_csv": "..."
    },
    ...
  ]
}
```

---

## Computing All Influences (No top-k Limit)

**By default, the script returns ALL TracIn scores** for every test/train triple pair, not just the top-k.

### Why This Matters

- **Complete data**: You get influence scores for every training triple
- **Larger CSV files**: Each CSV will contain ALL filtered training triples
- **Post-hoc filtering**: You can apply any threshold or ranking after the fact
- **No data loss**: Ensures no potentially important influences are missed

### Example File Sizes

**With top-k=100:**
- Each CSV: ~100 rows
- Total for 25 triples: ~2,500 rows

**With top-k=None (all influences):**
- Each CSV: ~10,000-30,000 rows (depends on filtering)
- Total for 25 triples: ~250,000-750,000 rows

### To Limit Output

If you want only top-k influences:
```bash
--top-k 100  # Get only top 100 most influential
```

---

## Performance Optimization

### 1. Use Graph Caching (Recommended)

**First run:**
```bash
--cache data/train_graph_cache.pkl
```

This creates a cached graph that is reused for all triples, providing **5-20x speedup** for filtering.

### 2. Last Layers Only (Default)

The script uses `--use-last-layers-only` by default (50x faster than full gradients).

To use all layers (slower but potentially more accurate):
```bash
--no-use-last-layers
```

### 3. Adjust Filtering Parameters

**Faster but less thorough:**
```bash
--n-hops 1 --min-degree 3
```

**More thorough but slower:**
```bash
--n-hops 3 --min-degree 1
```

### 4. Increase Batch Size (GPU Only)

For GPUs with more memory:
```bash
--batch-size 1024
```

---

## Expected Runtime

### With Recommended Settings

**Configuration:**
- 25 test triples
- n-hops=2, min-degree=2
- Use last layers only
- GPU (V100 or similar)
- Graph caching enabled

**Expected time per triple:**
- Filtering (with cache): ~5-10 seconds
- TracIn analysis: ~15-20 seconds
- **Total per triple: ~20-30 seconds**

**Total for 25 triples: ~8-12 minutes**

### Without Optimization

**Without caching or last-layers:**
- **Total for 25 triples: ~6-8 hours**

---

## Monitoring Progress

The script provides real-time logging:

```
2025-10-22 20:30:15 - INFO - Processing triple 1/25: CHEBI:7963 --[predicate:28]--> MONDO:0016595
2025-10-22 20:30:20 - INFO - Step 1/2: Filtering training data...
2025-10-22 20:30:25 - INFO - Filtering completed successfully
2025-10-22 20:30:26 - INFO - Step 2/2: Running TracIn analysis...
2025-10-22 20:30:45 - INFO - TracIn analysis completed successfully
2025-10-22 20:30:45 - INFO - ✓ Successfully completed triple 0
```

---

## Error Handling

### If Filtering Fails

The script will:
1. Log the error
2. Skip TracIn for that triple
3. Continue with next triple
4. Record failure in summary

### If TracIn Fails

The script will:
1. Log the error
2. Keep the filtered training file
3. Continue with next triple
4. Record failure in summary

### Resume After Failure

Use `--start-index` to resume:
```bash
--start-index 10  # Start from triple #10
```

---

## Analyzing Results

### Load All CSV Files

```python
import pandas as pd
from pathlib import Path
import glob

# Load all CSV files
csv_files = glob.glob('results/batch_tracin/*_tracin.csv')

# Combine into single dataframe
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    # Add source file column
    df['source_file'] = Path(f).name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

print(f"Total influences across all triples: {len(combined_df)}")
```

### Analyze Summary

```python
import json

with open('results/batch_tracin/batch_tracin_summary.json') as f:
    summary = json.load(f)

print(f"Success rate: {summary['successful']}/{summary['total_triples']}")
print(f"Average time per triple: {summary['elapsed_time_seconds']/summary['total_triples']:.1f}s")

# Check which triples failed
for result in summary['results']:
    if not result['tracin_success']:
        print(f"Failed: {result['triple']}")
```

---

## Troubleshooting

### Issue: "Module not found" error

**Cause:** Missing dependencies

**Fix:**
```bash
pip install torch torch-geometric pykeen pandas numpy
```

### Issue: Out of memory during TracIn

**Cause:** Batch size too large

**Fix:** Reduce batch size:
```bash
--batch-size 256  # or even 128
```

### Issue: Filtering takes too long

**Cause:** No graph caching

**Fix:** Use caching:
```bash
--cache data/train_graph_cache.pkl
```

### Issue: "Loaded 0 predicate names"

**Cause:** Missing or incorrect edge_map.json

**Fix:** Verify path:
```bash
--edge-map data/edge_map.json
```

---

## Best Practices

1. **Always use graph caching** for multiple triples
2. **Start with small test** (e.g., `--max-triples 2`)
3. **Use last layers only** unless accuracy is critical
4. **Check summary file** after completion
5. **Save filtered training files** for reuse

---

## Summary

✅ **Automated pipeline** for batch TracIn analysis
✅ **10-50x speedup** with filtering and optimization
✅ **Handles 25 triples in ~10 minutes** (with optimization)
✅ **Resume support** for interrupted runs
✅ **Detailed logging** and error handling
✅ **CSV output** in exact format specified

The script makes it easy to analyze influence patterns across all your top predictions!
