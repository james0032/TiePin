# Batch Script Update Summary

## Overview

Updated [scripts/add_ground_truth_to_all_csv.sh](scripts/add_ground_truth_to_all_csv.sh) to support the new **In_path** column functionality when batch processing TracIn CSV files.

## What Changed

### 1. **Enhanced Argument Parsing**

**Before:**
```bash
bash add_ground_truth_to_all_csv.sh <csv_folder> <ground_truth_jsonl> [output_suffix]
```

**After:**
```bash
bash add_ground_truth_to_all_csv.sh <csv_folder> <ground_truth_jsonl> [mechanistic_paths_csv] [output_suffix]
```

The script now intelligently detects if the 3rd argument is a file path (mechanistic paths) or a suffix:
- If arg 3 ends with `.txt`, `.csv`, or `.tsv` → treats it as mechanistic paths file
- Otherwise → treats it as output suffix

### 2. **Backward Compatible**

All existing usage patterns still work:

```bash
# Old usage (still works) - Only IsGroundTruth column
bash add_ground_truth_to_all_csv.sh dmdb_results ground_truth/edges.jsonl

# Old usage with custom suffix (still works)
bash add_ground_truth_to_all_csv.sh dmdb_results ground_truth/edges.jsonl "_gt"

# NEW: Add both columns
bash add_ground_truth_to_all_csv.sh dmdb_results ground_truth/edges.jsonl dedup_treats_mechanistic_paths.txt

# NEW: Both columns + custom suffix
bash add_ground_truth_to_all_csv.sh dmdb_results ground_truth/edges.jsonl dedup_treats_mechanistic_paths.txt "_annotated"
```

### 3. **Updated Features**

**Validation:**
- Checks if mechanistic paths file exists (if provided)
- Shows clear error messages

**Progress Output:**
```
==========================================
Add Ground Truth Column to TracIn CSV Files
==========================================
CSV folder:        dmdb_results/batch_tracin_top50
Ground truth file: ground_truth/drugmechdb_edges.jsonl
Mechanistic paths: dedup_treats_mechanistic_paths.txt
Will add: IsGroundTruth + In_path columns
Output suffix:     _with_gt
==========================================
```

**Command Building:**
- Uses array-based command construction for safe parameter passing
- Conditionally adds `--mechanistic-paths` argument if file is provided

## Usage Examples

### Example 1: Batch Process with Both Columns

```bash
cd /workspace/conve_pykeen

# Process all TracIn CSV files in a directory
bash scripts/add_ground_truth_to_all_csv.sh \
    dmdb_results/batch_tracin_top50 \
    ground_truth/drugmechdb_edges.jsonl \
    dedup_treats_mechanistic_paths.txt
```

**Input files:**
```
dmdb_results/batch_tracin_top50/
├── triple_000_CHEBI_17154_MONDO_0019975_tracin.csv
├── triple_001_CHEBI_15940_MONDO_0024298_tracin.csv
└── triple_002_CHEBI_17268_MONDO_0019975_tracin.csv
```

**Output files:**
```
dmdb_results/batch_tracin_top50/
├── triple_000_CHEBI_17154_MONDO_0019975_tracin_with_gt.csv  ← NEW
├── triple_001_CHEBI_15940_MONDO_0024298_tracin_with_gt.csv  ← NEW
└── triple_002_CHEBI_17268_MONDO_0019975_tracin_with_gt.csv  ← NEW
```

Each output file has **two new columns**: `IsGroundTruth` and `In_path`

### Example 2: Only Ground Truth Column (Original Behavior)

```bash
# Works exactly as before - only adds IsGroundTruth column
bash scripts/add_ground_truth_to_all_csv.sh \
    dmdb_results \
    ground_truth/drugmechdb_edges.jsonl
```

Output files have **one new column**: `IsGroundTruth`

### Example 3: Custom Output Suffix

```bash
# Add both columns with custom suffix
bash scripts/add_ground_truth_to_all_csv.sh \
    dmdb_results/batch_tracin_top50 \
    ground_truth/drugmechdb_edges.jsonl \
    dedup_treats_mechanistic_paths.txt \
    "_full_annotation"
```

Output files: `*_tracin_full_annotation.csv`

## Example Script

Created [scripts/example_batch_add_columns.sh](scripts/example_batch_add_columns.sh) demonstrating the new usage:

```bash
bash scripts/example_batch_add_columns.sh
```

This example script:
1. Shows what will be processed
2. Waits for user confirmation
3. Batch processes all CSV files
4. Adds both IsGroundTruth and In_path columns

## Technical Details

### Smart Argument Detection

```bash
# Determine if arg 3 is mechanistic paths or output suffix
if [ "$#" -ge 3 ]; then
    # If arg 3 ends with .txt, .csv, or .tsv, it's likely the mechanistic paths file
    if [[ "$3" =~ \.(txt|csv|tsv)$ ]]; then
        MECHANISTIC_PATHS_CSV="$3"
        # If arg 4 exists, it's the output suffix
        if [ "$#" -ge 4 ]; then
            OUTPUT_SUFFIX="$4"
        fi
    else
        # Otherwise, arg 3 is the output suffix
        OUTPUT_SUFFIX="$3"
    fi
fi
```

### Safe Command Building

```bash
# Build command array
CMD=(python "$PYTHON_SCRIPT" --tracin-csv "$CSV_FILE" --ground-truth "$GROUND_TRUTH_JSONL" --output "$OUTPUT_FILE")

# Conditionally add mechanistic paths
if [ -n "$MECHANISTIC_PATHS_CSV" ]; then
    CMD+=(--mechanistic-paths "$MECHANISTIC_PATHS_CSV")
fi

# Execute
"${CMD[@]}"
```

## Output Columns

When mechanistic paths are provided, each output CSV has:

| Original Columns | New Column 1 | New Column 2 |
|-----------------|--------------|--------------|
| TestHead, TestHead_label, ... | IsGroundTruth | In_path |
| TracInScore, SelfInfluence | 0 or 1 | 0 or 1 |

**IsGroundTruth**: Training edge exists in ground truth JSONL
**In_path**: Training edge connects nodes in mechanistic path

## Performance

- Processes ~100 CSV files in ~2-3 minutes (depending on file sizes)
- Each file processed independently (failures don't stop batch)
- Summary shows success/failure counts at the end

## Error Handling

The script continues processing even if individual files fail:

```
✗ Failed to process triple_042_tracin.csv

----------------------------------------
Processing: triple_043_tracin.csv
...
```

Final summary:
```
==========================================
Summary
==========================================
Total files found:  50
Successfully processed: 48
Failed: 2
==========================================
```

## See Also

- [add_ground_truth_column.py](add_ground_truth_column.py) - Main Python script
- [MECHANISTIC_PATH_ANALYSIS.md](MECHANISTIC_PATH_ANALYSIS.md) - Detailed documentation
- [scripts/example_batch_add_columns.sh](scripts/example_batch_add_columns.sh) - Example usage
