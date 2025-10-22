# JSON Predicate Extraction Support

## Overview

The TracIn CSV export now supports **JSON-formatted relation strings** with automatic predicate extraction. This is particularly useful when working with qualified relations stored in `edge_map.json` format.

---

## What Changed

### 1. Automatic Predicate Extraction

The system now automatically extracts the `"predicate"` field from JSON-formatted relation strings.

**Example:**

Input relation label (JSON):
```json
{
  "object_aspect_qualifier": "activity",
  "object_direction_qualifier": "decreased",
  "predicate": "biolink:affects",
  "subject_aspect_qualifier": "",
  "subject_direction_qualifier": ""
}
```

Output in CSV:
```
biolink:affects
```

### 2. Support for edge_map.json

You can now use `edge_map.json` files directly as relation label files.

**edge_map.json format:**
```json
{
  "{\"predicate\": \"biolink:affects\", ...}": "predicate:0",
  "{\"predicate\": \"biolink:treats\", ...}": "predicate:27",
  "{\"predicate\": \"biolink:coexpressed_with\", ...}": "predicate:1"
}
```

The system will:
1. Load the JSON file
2. Create a reverse mapping: `predicate:X` → JSON string
3. Extract the predicate name for display in CSV

---

## Usage

### Using edge_map.json as Relation Labels

```bash
python tracin_to_csv.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --relation-labels edge_map.json \    # Use edge_map.json here!
    --output results/tracin_influences.csv \
    --top-k 100
```

### Output Format

The CSV will have clean, readable predicate names:

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
UNII:U59UGK3IPC,Ublituximab,predicate:27,biolink:treats,MONDO:0005314,relapsing-remitting multiple sclerosis,NCBIGene:3455,IFNAR2,predicate:29,biolink:target_for,MONDO:0005314,relapsing-remitting multiple sclerosis,0.3972736001014709
```

**Note:** The `TestRel_label` and `TrainRel_label` columns will show `biolink:treats` instead of the full JSON string.

---

## Header Format

The output CSV header is **exactly** as requested:

```
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
```

No extra spaces, no variations—it matches your specification exactly.

---

## How It Works

### 1. Loading edge_map.json

The `load_labels()` function in `tracin_to_csv.py` now detects JSON files:

```python
def load_labels(label_path: str) -> Dict[int, str]:
    # Check if it's a JSON file
    if label_path.endswith('.json'):
        with open(label_path, 'r') as f:
            edge_map = json.load(f)
            # Create reverse mapping: predicate_id -> json_string
            for json_str, predicate_id in edge_map.items():
                if predicate_id.startswith('predicate:'):
                    idx = int(predicate_id.split(':')[1])
                    labels[idx] = json_str
```

### 2. Extracting Predicates

The `_extract_predicate_from_json()` method in `TracInAnalyzer` extracts the predicate:

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

### 3. Writing CSV

The `save_influences_to_csv()` method applies extraction when writing:

```python
# For test relation
if relation_labels and test_r in relation_labels:
    test_r_label = self._extract_predicate_from_json(relation_labels[test_r])
else:
    test_r_label = self._extract_predicate_from_json(test_r_id)

# Same for train relations
```

---

## Backward Compatibility

The changes are **fully backward compatible**:

- **Simple strings** (like `predicate:27`) are passed through unchanged
- **Tab-separated files** work as before
- **Line-by-line files** work as before
- **New JSON files** are automatically detected and parsed

---

## Testing

Run the test suite to verify functionality:

```bash
python test_json_extraction_simple.py
```

**Expected output:**
```
✓✓✓ All tests passed! ✓✓✓

The CSV output will correctly show:
  TestRel: predicate:27
  TestRel_label: biolink:treats

Instead of the full JSON string.
```

---

## Files Modified

1. **[tracin.py](tracin.py)**
   - Added `_extract_predicate_from_json()` method
   - Updated `save_influences_to_csv()` to use predicate extraction
   - Ensured exact header format match

2. **[tracin_to_csv.py](tracin_to_csv.py)**
   - Updated `load_labels()` to support JSON files
   - Added automatic edge_map.json detection and parsing

3. **[TRACIN_CSV_FORMAT.md](TRACIN_CSV_FORMAT.md)**
   - Added "Option 3: JSON Edge Map" section
   - Documented edge_map.json usage

4. **[test_json_extraction_simple.py](test_json_extraction_simple.py)** (New)
   - Test suite for JSON extraction
   - Demonstrates edge_map.json loading
   - Shows expected CSV output format

---

## Example Workflow

### Step 1: Prepare Your Files

You already have:
- `train.txt` - Training triples
- `edge_map.json` - Relation mappings with qualifiers
- `entity_to_id.tsv` - Entity mappings
- `relation_to_id.tsv` - Relation mappings
- `node_name_dict.txt` - Entity labels (optional)

### Step 2: Create Test Triple File

```bash
echo -e "UNII:U59UGK3IPC\tpredicate:27\tMONDO:0005314" > test_triple.txt
```

### Step 3: Run TracIn Analysis

```bash
python tracin_to_csv.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --relation-labels edge_map.json \
    --output results/tracin_output.csv \
    --top-k 100 \
    --device cuda
```

### Step 4: Analyze Results

```python
import pandas as pd

df = pd.read_csv('results/tracin_output.csv')

# View top influences
print(df.head(10))

# Group by predicate
by_predicate = df.groupby('TrainRel_label')['TracInScore'].agg(['mean', 'count'])
print(by_predicate)
```

---

## Summary

✅ **CSV header exactly matches your specification**
✅ **Automatic JSON predicate extraction**
✅ **Support for edge_map.json format**
✅ **Backward compatible with existing files**
✅ **Fully tested and documented**

The system now recognizes edge_map.json and automatically extracts clean predicate names like `biolink:treats` for the CSV output, while keeping the full qualified information available in the original data.
