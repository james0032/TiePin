# Using run_tracin.py with CSV Export

## Overview

The `run_tracin.py` script now supports CSV export with the `--csv-output` flag. This allows you to get both JSON and CSV outputs in a single run.

---

## New Feature: CSV Export

### What's New

- **Added `--csv-output` argument**: Specify a path to save results in CSV format
- **Automatic label extraction**: Uses edge_map.json and node_name_dict.txt for labels
- **Exact header format**: CSV output matches the exact format specification
- **Works with all modes**: Single test triple or multiple test triples

---

## Usage Examples

### Example 1: Single Test Triple with CSV Output

```bash
python run_tracin.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output results/tracin_output.json \
    --csv-output results/tracin_output.csv \
    --mode single \
    --test-indices 0 \
    --top-k 100 \
    --use-last-layers-only \
    --device cuda
```

**Output:**
- `results/tracin_output.json` - JSON format with full details
- `results/tracin_output.csv` - CSV format with exact header

### Example 2: Multiple Test Triples with CSV Output

```bash
python run_tracin.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output results/tracin_output.json \
    --csv-output results/tracin_output.csv \
    --mode single \
    --test-indices 0 1 2 \
    --top-k 100 \
    --device cuda
```

**Output:**
- `results/tracin_output.json` - Combined JSON for all test triples
- `results/tracin_output_test_0.csv` - CSV for test triple 0
- `results/tracin_output_test_1.csv` - CSV for test triple 1
- `results/tracin_output_test_2.csv` - CSV for test triple 2

### Example 3: Test Mode with Output Per Triple

```bash
python run_tracin.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output results/tracin_test.json \
    --csv-output results/tracin_test.csv \
    --mode test \
    --output-per-triple \
    --max-test-triples 10 \
    --top-k 50 \
    --device cuda
```

**Output:**
- `results/tracin_test_0.json`, `results/tracin_test_1.json`, ... (JSON files)
- `results/tracin_test_0.csv`, `results/tracin_test_1.csv`, ... (CSV files)

---

## CSV Output Format

The CSV output has the **exact header format** as specified:

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
```

### Example CSV Output

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
UNII:U59UGK3IPC,Ublituximab,predicate:27,biolink:treats,MONDO:0005314,relapsing-remitting multiple sclerosis,NCBIGene:3455,IFNAR2,predicate:29,biolink:target_for,MONDO:0005314,relapsing-remitting multiple sclerosis,0.3972736001014709
UNII:U59UGK3IPC,Ublituximab,predicate:27,biolink:treats,MONDO:0005314,relapsing-remitting multiple sclerosis,NCBIGene:8698,S1PR4,predicate:29,biolink:target_for,MONDO:0005314,relapsing-remitting multiple sclerosis,0.3969393968582153
```

---

## How It Works

### 1. Edge Map JSON Loading

The script loads `edge_map.json` and extracts predicate names:

```json
{
  "{\"predicate\": \"biolink:treats\", ...}": "predicate:27"
}
```

↓ Parsed to ↓

```python
idx_to_predicate = {
    27: "biolink:treats"
}
```

### 2. CSV Label Preparation

When saving CSV, the script:
1. Uses `id_to_entity` for entity CURIEs
2. Uses `idx_to_entity_name` for entity labels
3. Creates JSON strings for relation labels: `{"predicate": "biolink:treats"}`
4. Calls `analyzer.save_influences_to_csv()` which extracts the predicate field

### 3. Predicate Extraction

The `_extract_predicate_from_json()` method in `TracInAnalyzer` extracts clean predicates:
- Input: `'{"predicate": "biolink:treats"}'`
- Output: `"biolink:treats"`

---

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to trained model (.pt) |
| `--train` | Path to training triples |
| `--entity-to-id` | Path to entity_to_id.tsv |
| `--relation-to-id` | Path to relation_to_id.tsv |
| `--output` | Output path for JSON results |

### Optional Arguments for CSV

| Argument | Description |
|----------|-------------|
| `--csv-output` | **Path to save CSV results** |
| `--edge-map` | Path to edge_map.json (for predicate names) |
| `--node-name-dict` | Path to node_name_dict.txt (for entity names) |

### Other Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `test` | Analysis mode: `test`, `self`, or `single` |
| `--test` | - | Path to test triples (required for test/single modes) |
| `--test-indices` | `[0]` | Test triple indices (for single mode) |
| `--top-k` | `10` | Number of top influences to return |
| `--device` | `cpu` | Device to run on (`cpu` or `cuda`) |
| `--use-last-layers-only` | `False` | Use last layers only (much faster!) |
| `--num-last-layers` | `2` | Number of last layers to track |
| `--batch-size` | `256` | Batch size for processing |

---

## Comparison: run_tracin.py vs tracin_to_csv.py

### run_tracin.py (General Purpose)

**Pros:**
- Flexible modes: test, self, single
- Both JSON and CSV output
- Can analyze multiple test triples
- Comprehensive logging

**Best for:**
- General TracIn analysis workflow
- When you want both JSON and CSV
- Analyzing multiple test triples

**Example:**
```bash
python run_tracin.py \
    --mode single \
    --test-indices 0 1 2 \
    --csv-output results.csv \
    --output results.json
```

### tracin_to_csv.py (CSV-Only)

**Pros:**
- Simple, focused on CSV output
- Single test triple input file
- Minimal arguments

**Best for:**
- Quick CSV export for one test triple
- When you only need CSV output
- Simpler command-line interface

**Example:**
```bash
python tracin_to_csv.py \
    --test-triple test_triple.txt \
    --output results.csv
```

---

## Complete Example Workflow

### Step 1: Prepare Your Files

You should have:
- `trained_model.pt` - Your trained ConvE model
- `train.txt` - Training triples
- `test.txt` - Test triples
- `entity_to_id.tsv` - Entity mappings
- `relation_to_id.tsv` - Relation mappings
- `edge_map.json` - Relation qualifiers with predicates
- `node_name_dict.txt` - Entity names (optional)

### Step 2: Run TracIn Analysis

```bash
python run_tracin.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output results/tracin_analysis.json \
    --csv-output results/tracin_analysis.csv \
    --mode single \
    --test-indices 0 \
    --top-k 100 \
    --use-last-layers-only \
    --num-last-layers 2 \
    --device cuda \
    --batch-size 512
```

### Step 3: Check Logs

You should see:
```
INFO - Loaded 45 predicate names  # (from edge_map.json)
INFO - Loaded 12345 entity names   # (from node_name_dict.txt)
...
INFO - Saving CSV output to results/tracin_analysis.csv
...
INFO - JSON results saved to results/tracin_analysis.json
```

### Step 4: Analyze Results

```python
import pandas as pd

# Load CSV
df = pd.read_csv('results/tracin_analysis.csv')

# View top influences
print(df.head(10))

# Group by relation
by_rel = df.groupby('TrainRel_label')['TracInScore'].agg(['mean', 'count'])
print(by_rel)
```

---

## Troubleshooting

### Issue: "Loaded 0 predicate names"

**Cause:** edge_map.json not found or incorrect format

**Fix:**
1. Check that edge_map.json exists
2. Verify the format matches:
   ```json
   {
     "{\"predicate\": \"biolink:treats\", ...}": "predicate:27"
   }
   ```
3. Ensure predicate IDs in edge_map.json match relation_to_id.tsv

### Issue: CSV shows relation IDs instead of predicate names

**Cause:** Missing `--edge-map` argument

**Fix:**
```bash
python run_tracin.py \
    --edge-map edge_map.json \  # Add this!
    --csv-output results.csv \
    ...
```

### Issue: CSV shows entity IDs instead of names

**Cause:** Missing `--node-name-dict` argument

**Fix:**
```bash
python run_tracin.py \
    --node-name-dict node_name_dict.txt \  # Add this!
    --csv-output results.csv \
    ...
```

---

## Summary

✅ **Added CSV export to run_tracin.py**
✅ **Use `--csv-output` flag to specify CSV path**
✅ **Automatic label extraction from edge_map.json**
✅ **Exact CSV header format as specified**
✅ **Works with single or multiple test triples**
✅ **Compatible with all analysis modes**

The `run_tracin.py` script now provides a complete workflow for TracIn analysis with both JSON and CSV outputs!
