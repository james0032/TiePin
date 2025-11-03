# Scripts Directory

This directory contains example shell scripts for running batch TracIn analysis.

## Quick Start

### Run Batch TracIn Analysis

```bash
# From the scripts directory:
bash run_batch_tracin_example.sh

# Or from the parent directory:
bash scripts/run_batch_tracin_example.sh
```

The script will automatically navigate to the correct directory to find `batch_tracin_with_filtering.py`.

## Script: run_batch_tracin_example.sh

### What It Does

Processes multiple test triples with:
1. Automatic proximity-based filtering (n-hops + degree filtering)
2. TracIn influence score computation
3. CSV/JSON output generation

### Current Configuration

```bash
N-hops:                2
Min degree:            2
Strict hop constraint: enabled
Batch size:            16
Device:                cuda
```

### Before Running

**Update these paths in the script:**

```bash
MODEL_PATH="/workspace/data/robokop/CGGD_alltreat/checkpoints/conve_checkpoint_008.pt"
TRAIN_FILE="/workspace/data/robokop/CGGD_alltreat/train.txt"
ENTITY_TO_ID="/workspace/data/robokop/CGGD_alltreat/processed/entity_to_id.tsv"
RELATION_TO_ID="/workspace/data/robokop/CGGD_alltreat/processed/relation_to_id.tsv"
EDGE_MAP="/workspace/data/robokop/CGGD_alltreat/edge_map.json"
NODE_NAME_DICT="/workspace/data/robokop/CGGD_alltreat/node_name_dict.txt"
GRAPH_CACHE="/workspace/data/robokop/CGGD_alltreat/processed/train_graph_cache.pkl"
TEST_TRIPLES="/workspace/data/robokop/CGGD_alltreat/dmdb_results/test_scores_top50.txt"
OUTPUT_DIR="/workspace/data/robokop/CGGD_alltreat/dmdb_results/batch_tracin_top50"
```

### Output Files

After running, you'll find:

```
${OUTPUT_DIR}/
├── filtered_training/          # Filtered training data for each test triple
│   ├── triple_0_filtered.txt
│   ├── triple_1_filtered.txt
│   └── ...
├── triple_0_tracin.csv         # TracIn scores (CSV format)
├── triple_0_tracin.json        # TracIn scores (JSON format)
├── triple_1_tracin.csv
├── triple_1_tracin.json
└── batch_tracin_summary.json   # Summary statistics
```

### Customization

To change parameters, edit the script:

```bash
# In run_batch_tracin_example.sh, find this section:
python batch_tracin_with_filtering.py \
    --n-hops 2 \              # Change neighborhood size
    --min-degree 2 \          # Change degree threshold
    --strict-hop-constraint \ # Remove to disable strict mode
    --batch-size 16 \         # Adjust for your GPU memory
    --device cuda \           # Change to 'cpu' if needed
```

## Troubleshooting

### Error: "can't open file 'batch_tracin_with_filtering.py'"

**Cause**: The script couldn't find `batch_tracin_with_filtering.py`

**Solution**: Make sure you're running from either:
- The `scripts/` directory: `bash run_batch_tracin_example.sh`
- The parent directory: `bash scripts/run_batch_tracin_example.sh`

The script automatically navigates to the correct location.

### Error: CUDA out of memory

**Solution**: Reduce batch size:

```bash
--batch-size 8 \  # or even lower
```

### Error: File not found for test triples/model/etc.

**Solution**: Update the paths at the top of the script to match your environment.

## Performance Tips

### Faster Processing

1. **Increase batch size** (if GPU memory allows):
   ```bash
   --batch-size 32 \  # or higher
   ```

2. **Use GPU**:
   ```bash
   --device cuda \
   ```

3. **Cache the graph**:
   ```bash
   --cache "${GRAPH_CACHE}" \
   ```
   The cache is created on first run and reused for subsequent runs.

### Memory Optimization

If running into memory issues:

1. **Reduce n-hops**:
   ```bash
   --n-hops 1 \  # Smaller neighborhood
   ```

2. **Increase min-degree**:
   ```bash
   --min-degree 5 \  # More aggressive filtering
   ```

3. **Process fewer test triples**:
   Create a smaller test file with just the triples you want to analyze.

## Related Documentation

- [CHANGELOG_BATCH_TRACIN.md](../CHANGELOG_BATCH_TRACIN.md) - Recent changes
- [README_STRICT_HOP_CONSTRAINT.md](../README_STRICT_HOP_CONSTRAINT.md) - Strict mode documentation
- [batch_tracin_with_filtering.py](../batch_tracin_with_filtering.py) - Main script

## Support

For issues or questions:
1. Check the error message carefully
2. Review the paths in the script configuration
3. Verify all required files exist
4. Check GPU memory if using CUDA

## Example Run

```bash
$ bash run_batch_tracin_example.sh

========================================
Batch TracIn Analysis
========================================
Test triples: /workspace/data/robokop/.../test_scores_top50.txt
Output directory: /workspace/data/robokop/.../batch_tracin_top50

Configuration:
  - N-hops: 2
  - Min degree: 2
  - Strict hop constraint: enabled
  - Batch size: 16
  - Top-k influences: 100
  - Device: cuda
  - Last layers only: enabled (2 layers for speed)

Expected time: ~8-12 minutes for 25 triples
========================================

Processing test triple 1/25...
[Filtering and TracIn computation proceeds...]

========================================
Batch processing complete!
========================================

Output files:
  - Filtered training: .../batch_tracin_top50/filtered_training/
  - TracIn CSV files: .../batch_tracin_top50/*_tracin.csv
  - TracIn JSON files: .../batch_tracin_top50/*_tracin.json
  - Summary: .../batch_tracin_top50/batch_tracin_summary.json
```
