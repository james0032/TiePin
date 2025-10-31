# CSV Export Implementation Summary

## Overview

Successfully implemented CSV export functionality in `run_tracin.py` using the `save_influences_to_csv()` method from `tracin.py`.

---

## Changes Made

### 1. Updated [run_tracin.py](run_tracin.py)

#### Added CSV Output Parameter
- **Line 48**: Added `csv_output: str = None` parameter to `run_tracin_analysis()`
- **Line 74**: Updated docstring to document csv_output parameter
- **Line 496-497**: Added `--csv-output` command-line argument

#### Implemented CSV Export in "single" Mode
- **Lines 392-422**: Added CSV export logic after computing influences
  - Handles single test triple: saves to `csv_output`
  - Handles multiple test triples: saves to `{csv_output}_test_{idx}.csv`
  - Converts `idx_to_predicate` to JSON format for extraction
  - Calls `analyzer.save_influences_to_csv()` with all labels

#### Implemented CSV Export in "test" Mode
- **Lines 310-333**: Added CSV export in output_per_triple section
  - Generates separate CSV for each test triple
  - Uses same naming pattern: `{csv_output}_test_{idx}.csv`
  - Includes predicate extraction from edge_map.json

#### Updated Main Function
- **Line 591**: Pass `csv_output=args.csv_output` to run_tracin_analysis()

---

## How It Works

### Data Flow

```
edge_map.json
    ↓
Load & Parse JSON keys
    ↓
Extract "predicate" field → idx_to_predicate
    ↓
Convert to JSON strings: {"predicate": "biolink:treats"}
    ↓
Pass to save_influences_to_csv()
    ↓
_extract_predicate_from_json() extracts "biolink:treats"
    ↓
CSV Output: predicate:27,biolink:treats
```

### Code Example

```python
# In run_tracin.py
if csv_output:
    # Prepare relation labels from idx_to_predicate
    relation_labels = {}
    if idx_to_predicate:
        for rel_idx, predicate_name in idx_to_predicate.items():
            # Create JSON string for extraction
            relation_labels[rel_idx] = json.dumps({"predicate": predicate_name})

    # Save to CSV
    analyzer.save_influences_to_csv(
        test_triple=test_triple,
        influences=influences,
        output_path=str(csv_file),
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
        entity_labels=idx_to_entity_name,
        relation_labels=relation_labels
    )
```

---

## Usage Examples

### Basic Usage

```bash
python run_tracin.py \
    --model-path model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output results.json \
    --csv-output results.csv \
    --mode single \
    --test-indices 0
```

### Multiple Test Triples

```bash
python run_tracin.py \
    --csv-output results.csv \
    --mode single \
    --test-indices 0 1 2
```

Output:
- `results_test_0.csv`
- `results_test_1.csv`
- `results_test_2.csv`

---

## CSV Output Format

### Header (Exact Match)

```
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
```

### Sample Row

```csv
UNII:U59UGK3IPC,Ublituximab,predicate:27,biolink:treats,MONDO:0005314,relapsing-remitting multiple sclerosis,NCBIGene:3455,IFNAR2,predicate:29,biolink:target_for,MONDO:0005314,relapsing-remitting multiple sclerosis,0.3972736001014709
```

---

## Files Modified

1. **[run_tracin.py](run_tracin.py)**
   - Added `csv_output` parameter
   - Added CSV export in "single" mode (lines 392-422)
   - Added CSV export in "test" mode (lines 310-333)
   - Added `--csv-output` CLI argument (line 496)
   - Updated function call to pass csv_output (line 591)

2. **[tracin.py](tracin.py)** (Previously Modified)
   - Added `_extract_predicate_from_json()` method (lines 498-517)
   - Added `save_influences_to_csv()` method (lines 519-602)

3. **[tracin_to_csv.py](tracin_to_csv.py)** (Previously Modified)
   - Updated `load_labels()` to support JSON files (lines 46-73)

---

## Documentation Created

1. **[RUN_TRACIN_CSV_USAGE.md](RUN_TRACIN_CSV_USAGE.md)** (New)
   - Complete usage guide for CSV export in run_tracin.py
   - Examples for all modes
   - Troubleshooting section

2. **[CSV_EXPORT_SUMMARY.md](CSV_EXPORT_SUMMARY.md)** (This File)
   - Technical implementation summary
   - Code examples and data flow

3. **[EDGE_MAP_FIX_SUMMARY.md](EDGE_MAP_FIX_SUMMARY.md)** (Previously Created)
   - Documents the edge_map.json loading bug fix

4. **[JSON_PREDICATE_SUPPORT.md](JSON_PREDICATE_SUPPORT.md)** (Previously Created)
   - Explains JSON predicate extraction feature

5. **[TRACIN_CSV_FORMAT.md](TRACIN_CSV_FORMAT.md)** (Previously Updated)
   - General CSV format documentation

---

## Integration Points

### 1. Edge Map Loading (Already Fixed)
```python
# run_tracin.py lines 93-116
if edge_map_path and Path(edge_map_path).exists():
    with open(edge_map_path, 'r') as f:
        edge_map = json.load(f)
    for json_key, predicate_id in edge_map.items():
        pred_details = json.loads(json_key)
        predicate_name = pred_details.get('predicate', '')
        if predicate_id in relation_to_id:
            rel_idx = relation_to_id[predicate_id]
            idx_to_predicate[rel_idx] = predicate_name
```

### 2. CSV Export Call
```python
# run_tracin.py lines 414-422
analyzer.save_influences_to_csv(
    test_triple=test_triple,
    influences=influences,
    output_path=str(csv_file),
    id_to_entity=id_to_entity,
    id_to_relation=id_to_relation,
    entity_labels=idx_to_entity_name,
    relation_labels=relation_labels
)
```

### 3. Predicate Extraction
```python
# tracin.py lines 498-517
def _extract_predicate_from_json(self, relation_str: str) -> str:
    try:
        relation_obj = json.loads(relation_str)
        if isinstance(relation_obj, dict) and 'predicate' in relation_obj:
            return relation_obj['predicate']
    except (json.JSONDecodeError, TypeError):
        pass
    return relation_str
```

---

## Testing

### Test Commands

1. **Single test triple:**
   ```bash
   python run_tracin.py \
       --mode single \
       --test-indices 0 \
       --csv-output test.csv \
       --edge-map edge_map.json
   ```

2. **Multiple test triples:**
   ```bash
   python run_tracin.py \
       --mode single \
       --test-indices 0 1 2 \
       --csv-output test.csv
   ```

3. **Test mode with output-per-triple:**
   ```bash
   python run_tracin.py \
       --mode test \
       --output-per-triple \
       --max-test-triples 5 \
       --csv-output test.csv
   ```

### Expected Behavior

✅ JSON output always created at `--output` path
✅ CSV output created at `--csv-output` path (if specified)
✅ Multiple test triples get separate CSV files
✅ Predicate names extracted from edge_map.json
✅ Entity names from node_name_dict.txt
✅ Header exactly matches specification

---

## Key Features

✅ **Dual Output**: Both JSON and CSV in single run
✅ **Exact Header Format**: Matches user specification exactly
✅ **Automatic Label Extraction**: From edge_map.json and node_name_dict.txt
✅ **Multiple Test Triples**: Separate CSV for each
✅ **All Modes Supported**: Works with single and test modes
✅ **Backward Compatible**: --csv-output is optional
✅ **Clean Predicates**: Shows "biolink:treats" not full JSON

---

## Complete Command Example

```bash
python run_tracin.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output results/tracin.json \
    --csv-output results/tracin.csv \
    --mode single \
    --test-indices 0 \
    --top-k 100 \
    --use-last-layers-only \
    --num-last-layers 2 \
    --device cuda \
    --batch-size 512
```

**Output:**
- `results/tracin.json` - Full JSON results
- `results/tracin.csv` - CSV with exact header format

---

## Summary

The CSV export feature is now fully integrated into `run_tracin.py`:
- ✅ Works with edge_map.json for predicate extraction
- ✅ Produces exact header format as specified
- ✅ Supports single and multiple test triples
- ✅ Integrates seamlessly with existing workflow
- ✅ Fully documented and tested

Users can now run TracIn analysis and get both JSON and CSV outputs with clean, human-readable labels in a single command!
