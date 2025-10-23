# Edge Map JSON Loading Fix

## Problem

When loading `edge_map.json` in `run_tracin.py`, the script was showing:
```
2025-10-22 23:10:19,948 - __main__ - INFO - Loading edge map from edge_map.json
2025-10-22 23:10:19,950 - __main__ - INFO -   Loaded 0 predicate names
```

This meant no predicates were being loaded from the edge_map.json file.

---

## Root Cause

The bug was in [run_tracin.py](run_tracin.py:108-113) (old code):

```python
# WRONG: Checking if predicate_id is in VALUES
if predicate_id in relation_to_id.values():
    # Then looping through to find the matching key
    for rel_curie, rel_idx in relation_to_id.items():
        if rel_curie == predicate_id:
            idx_to_predicate[rel_idx] = predicate_name
            break
```

### Why This Failed

Given the data structures:
- `edge_map`: `{"json_string": "predicate:27", ...}`
- `relation_to_id`: `{"predicate:27": 27, ...}`

The code was checking if `"predicate:27"` (a string) was in `relation_to_id.values()` (which contains integers like `27`).

This would **never match** because:
- `predicate_id = "predicate:27"` (string)
- `relation_to_id.values() = [0, 1, 2, ..., 27, ...]` (integers)
- `"predicate:27" in [0, 1, 2, ...]` → **False**

---

## Solution

Changed to check if `predicate_id` is in `relation_to_id.keys()` instead:

```python
# CORRECT: Check if predicate_id is a key in relation_to_id
if predicate_id in relation_to_id:
    # Get the index directly
    rel_idx = relation_to_id[predicate_id]
    idx_to_predicate[rel_idx] = predicate_name
```

### Why This Works

Now the check is:
- `predicate_id = "predicate:27"` (string)
- `relation_to_id = {"predicate:27": 27, ...}` (dict with string keys)
- `"predicate:27" in {"predicate:27": 27, ...}` → **True** ✓

And we can directly get the index:
- `rel_idx = relation_to_id["predicate:27"]` → `27`
- `idx_to_predicate[27] = "biolink:treats"`

---

## Files Fixed

### 1. [run_tracin.py](run_tracin.py:93-116)

**Before:**
```python
if predicate_id in relation_to_id.values():  # WRONG!
    for rel_curie, rel_idx in relation_to_id.items():
        if rel_curie == predicate_id:
            idx_to_predicate[rel_idx] = predicate_name
            break
```

**After:**
```python
if predicate_id in relation_to_id:  # CORRECT!
    rel_idx = relation_to_id[predicate_id]
    idx_to_predicate[rel_idx] = predicate_name
```

### 2. [tracin_to_csv.py](tracin_to_csv.py:46-58)

Added support for loading edge_map.json directly:

```python
if label_path.endswith('.json'):
    with open(label_path, 'r') as f:
        edge_map = json.load(f)
        for json_str, predicate_id in edge_map.items():
            if predicate_id.startswith('predicate:'):
                idx = int(predicate_id.split(':')[1])
                labels[idx] = json_str
```

### 3. [tracin.py](tracin.py:498-517)

Added `_extract_predicate_from_json()` method to extract clean predicate names:

```python
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

## Expected Output

After this fix, you should see:

```
2025-10-22 23:XX:XX,XXX - __main__ - INFO - Loading edge map from edge_map.json
2025-10-22 23:XX:XX,XXX - __main__ - INFO -   Loaded 45 predicate names
```

(The number will depend on how many relations are in your edge_map.json)

---

## How to Use

### Option 1: Using run_tracin.py

```bash
python run_tracin.py \
    --model-path model.pt \
    --train train.txt \
    --test test.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --output results/tracin_output.json \
    --mode single \
    --test-indices 0
```

Now the JSON output will include predicate names:
```json
{
  "test_relation": 27,
  "test_relation_label": "predicate:27",
  "test_relation_name": "biolink:treats",
  ...
}
```

### Option 2: Using tracin_to_csv.py

```bash
python tracin_to_csv.py \
    --model-path model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --relation-labels edge_map.json \
    --output results/tracin.csv
```

Now the CSV will show:
```csv
TestRel,TestRel_label,...
predicate:27,biolink:treats,...
```

---

## Testing

You can verify the fix works by checking the log output:

```bash
python run_tracin.py --edge-map edge_map.json ... 2>&1 | grep "Loaded.*predicate names"
```

**Before fix:** `Loaded 0 predicate names`
**After fix:** `Loaded 45 predicate names` (or however many you have)

---

## Summary

✅ **Fixed `run_tracin.py`** - Now correctly loads predicates from edge_map.json
✅ **Enhanced `tracin_to_csv.py`** - Supports JSON files as relation labels
✅ **Updated `tracin.py`** - Extracts clean predicate names from JSON
✅ **CSV output** - Shows `biolink:treats` instead of full JSON string
✅ **Backward compatible** - Works with both JSON and text label files

The edge_map.json predicates will now be properly loaded and displayed in both JSON and CSV outputs!
