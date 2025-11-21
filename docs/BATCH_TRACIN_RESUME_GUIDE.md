# Batch TracIn Resume/Checkpoint Feature

## Overview

The `batch_tracin_with_filtering.py` script now supports **checkpoint/resume functionality** via the `--skip-existing` flag. This allows you to:

- **Resume interrupted runs** without re-processing completed triples
- **Avoid re-running** triples that already have output files
- **Save time and compute** when processing large batches

## How It Works

When `--skip-existing` is enabled, the script checks if the output CSV file already exists for each triple:

```
output_csv = f"{output_dir}/triple_{idx:03d}_{head}_{tail}_tracin.csv"
```

If the file exists → **Skip** the triple
If the file doesn't exist → **Process** the triple normally

## Usage

### Basic Resume (Process All Remaining Triples)

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/batch_tracin \
    --skip-existing \
    --device cuda
```

### Resume from Specific Index

If your run was interrupted at triple 50, resume from there:

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/batch_tracin \
    --start-index 50 \
    --skip-existing \
    --device cuda
```

### Process Specific Range

Process triples 100-200, skipping any that are already done:

```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/batch_tracin \
    --start-index 100 \
    --max-triples 100 \
    --skip-existing \
    --device cuda
```

## Output Summary

With `--skip-existing`, the summary shows:

```
========================================
BATCH PROCESSING COMPLETE
========================================
Total triples in batch: 100
Successful (new): 23
Skipped (existing): 77
Failed (filtering): 0
Failed (TracIn): 0
Elapsed time: 2h 15m 30s

✓ Resume mode enabled (--skip-existing)
  Total completed (new + existing): 100
========================================
```

## Use Cases

### 1. **Interrupted Run**
Your cluster job times out after processing 50/200 triples:

```bash
# Original run (interrupted at triple 50)
python batch_tracin_with_filtering.py ... --output-dir results/batch1

# Resume from where it left off
python batch_tracin_with_filtering.py ... --output-dir results/batch1 --skip-existing
```

### 2. **Re-run with Different Parameters**
You want to re-run only failed triples with different settings:

```bash
# First pass (some triples failed)
python batch_tracin_with_filtering.py ... --batch-size 512 --output-dir results/batch1

# Re-run with smaller batch size (only processes failed triples)
python batch_tracin_with_filtering.py ... --batch-size 256 --output-dir results/batch1 --skip-existing
```

### 3. **Parallel Processing Across Nodes**
Split work across multiple compute nodes without overlap:

```bash
# Node 1: Process triples 0-99
python batch_tracin_with_filtering.py ... --start-index 0 --max-triples 100 --skip-existing

# Node 2: Process triples 100-199
python batch_tracin_with_filtering.py ... --start-index 100 --max-triples 100 --skip-existing

# Node 3: Process triples 200-299
python batch_tracin_with_filtering.py ... --start-index 200 --max-triples 100 --skip-existing
```

### 4. **Incremental Processing**
Process a few triples at a time during development:

```bash
# Day 1: Test with first 10 triples
python batch_tracin_with_filtering.py ... --max-triples 10 --skip-existing

# Day 2: Process 10 more (skips the first 10)
python batch_tracin_with_filtering.py ... --max-triples 20 --skip-existing

# Day 3: Process all remaining
python batch_tracin_with_filtering.py ... --skip-existing
```

## Important Notes

### What Gets Skipped?

The script checks for the **output CSV file** only:
```
triple_{idx:03d}_{head}_{tail}_tracin.csv
```

If this file exists → skip

### What Doesn't Get Skipped?

- **Filtered training data**: Always regenerated unless `--skip-filtering` is used
- **Temp files**: Always recreated
- **JSON output**: Not checked (only CSV)

### Force Re-run a Specific Triple

If you need to re-run a specific triple, delete its output file:

```bash
# Remove the CSV file for triple 42
rm results/batch_tracin/triple_042_*_tracin.csv

# Re-run with --skip-existing (will process only triple 42)
python batch_tracin_with_filtering.py ... --skip-existing
```

### Combining with Other Flags

```bash
# Resume + skip filtering (use existing filtered files)
python batch_tracin_with_filtering.py ... --skip-existing --skip-filtering

# Resume + optimization flags
python batch_tracin_with_filtering.py ... \
    --skip-existing \
    --use-last-layers-only \
    --use-mixed-precision \
    --batch-size 1024
```

## Checking Progress

View which triples have been completed:

```bash
# Count completed triples
ls results/batch_tracin/triple_*_tracin.csv | wc -l

# List completed triple indices
ls results/batch_tracin/triple_*_tracin.csv | \
    sed 's/.*triple_//' | \
    sed 's/_.*//' | \
    sort -n

# Check summary
cat results/batch_tracin/batch_tracin_summary.json | jq '.successful, .skipped'
```

## Best Practices

1. **Always use `--skip-existing` for long-running jobs** to enable resumability
2. **Set reasonable `--max-triples`** for your job time limit (e.g., 50 triples per 24h job)
3. **Use `--start-index`** when manually splitting work across nodes
4. **Check the summary JSON** to verify progress between runs
5. **Keep `--output-dir` consistent** across resume attempts

## Troubleshooting

### "Skipped 0 triples but I know files exist"

Check that:
- `--output-dir` matches the directory with existing files
- File naming pattern matches: `triple_{idx:03d}_{head}_{tail}_tracin.csv`
- Files are not empty or corrupted

### "Want to force re-run all triples"

Don't use `--skip-existing`, or move existing files:

```bash
# Backup and re-run
mv results/batch_tracin results/batch_tracin_backup
python batch_tracin_with_filtering.py ... --output-dir results/batch_tracin
```

### "Some output files are corrupted"

Delete the bad files and re-run with `--skip-existing`:

```bash
# Find and remove empty CSV files
find results/batch_tracin -name "*_tracin.csv" -size 0 -delete

# Re-run to regenerate
python batch_tracin_with_filtering.py ... --skip-existing
```
