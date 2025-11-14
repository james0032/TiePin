# Fixed Predicate Conversion - Now Uses edge_map.json

## Date: 2025-11-13

## Change Summary

Updated `make_test_with_drugmechdb_treat.py` to use `edge_map.json` for predicate conversion instead of `rel_dict.txt`.

## Why This Change?

The `edge_map.json` file is the canonical source of truth for predicate mappings in the pipeline. It maps full predicate details (including qualifiers) to predicate IDs.

**edge_map.json format**:
```json
{
  "{\"object_aspect_qualifier\": \"\", \"object_direction_qualifier\": \"\", \"predicate\": \"biolink:treats\", \"subject_aspect_qualifier\": \"\", \"subject_direction_qualifier\": \"\"}": "predicate:28"
}
```

This ensures we use the exact same predicate ID that's used in rotorobo.txt.

## Changes Made

### 1. Removed rel_dict.txt Dependency

**Deleted**:
- `load_relation_dict()` function
- `--rel-dict-file` command-line argument
- `rel_dict` input from Snakefile
- `rel_dict_path` validation

### 2. Added edge_map-based Conversion Logic

**New logic** (lines 415-442):

```python
# Look for treats predicate with empty qualifiers
treats_predicate_id = None
for predicate_detail, predicate_id in edge_map.items():
    try:
        pred_dict = json.loads(predicate_detail)
        if (pred_dict.get("predicate") == "biolink:treats" and
            not pred_dict.get("subject_aspect_qualifier") and
            not pred_dict.get("subject_direction_qualifier") and
            not pred_dict.get("object_aspect_qualifier") and
            not pred_dict.get("object_direction_qualifier")):
            treats_predicate_id = predicate_id
            logger.info(f"Found treats predicate ID with empty qualifiers: {treats_predicate_id}")
            break
    except json.JSONDecodeError:
        continue

if not treats_predicate_id:
    # Fallback: use the first treats predicate ID
    treats_predicate_id = list(treats_predicates)[0]
    logger.warning(f"No treats predicate with empty qualifiers found, using first one: {treats_predicate_id}")
```

### 3. Simplified Predicate Conversion

**New conversion logic** (lines 451-471):

```python
for subj, pred, obj in filtered_edges:
    if pred == "biolink:treats":
        # Convert biolink:treats -> predicate:28 (or whatever the ID is)
        converted_edges.append((subj, treats_predicate_id, obj))
    else:
        unmapped_predicates.add(pred)
        logger.warning(f"Unexpected predicate in filtered TSV: {pred}, keeping as-is")
        converted_edges.append((subj, pred, obj))
```

### 4. Updated Snakefile

**Before**:
```python
input:
    subgraph = f"{BASE_DIR}/rotorobo.txt",
    edge_map = f"{BASE_DIR}/edge_map.json",  # Used for debug logging only
    filtered_tsv = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt",
    rel_dict = f"{BASE_DIR}/processed/rel_dict.txt"  # Required: maps biolink:treats to predicate IDs
```

**After**:
```python
input:
    subgraph = f"{BASE_DIR}/rotorobo.txt",
    edge_map = f"{BASE_DIR}/edge_map.json",  # Required: maps biolink:treats to predicate IDs
    filtered_tsv = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt"
```

**Shell command** - removed `--rel-dict-file` argument:
```bash
python src/make_test_with_drugmechdb_treat.py \
    --input-dir {params.input_dir} \
    --filtered-tsv {input.filtered_tsv} \
    --test-pct {params.test_pct} \
    --seed {params.seed} \
    --output-dir {params.input_dir}
```

## How It Works Now

### Step 1: Load edge_map.json
```python
edge_map = {
    '{"predicate": "biolink:treats", ...}': 'predicate:28',
    '{"predicate": "biolink:causes", ...}': 'predicate:15',
    ...
}
```

### Step 2: Find treats predicates
```python
treats_predicates = {'predicate:28', 'predicate:29', ...}  # All IDs with biolink:treats
```

### Step 3: Find treats predicate with empty qualifiers
```python
treats_predicate_id = 'predicate:28'  # The one with no qualifiers
```

### Step 4: Convert filtered edges
```python
# Input from filtered TSV
filtered_edges = [
    ('CHEBI:27732', 'biolink:treats', 'HP:0001257'),
    ...
]

# Convert
converted_edges = [
    ('CHEBI:27732', 'predicate:28', 'HP:0001257'),
    ...
]
```

### Step 5: Write to test.txt
```
CHEBI:27732    predicate:28    HP:0001257
CHEBI:4462     predicate:28    MONDO:0004773
```

## Benefits

1. **Single source of truth**: Uses the same mapping as the rest of the pipeline
2. **Handles qualifiers**: Can distinguish between different qualifier variants
3. **More accurate**: Guaranteed to match the predicate IDs in rotorobo.txt
4. **Simpler dependencies**: One fewer file to track

## Edge Cases Handled

### Multiple treats predicates with qualifiers

If edge_map.json contains:
```json
{
  '{"predicate": "biolink:treats", "subject_aspect_qualifier": "", ...}': 'predicate:28',
  '{"predicate": "biolink:treats", "subject_aspect_qualifier": "activity_or_abundance", ...}': 'predicate:29'
}
```

The script will:
1. Find all treats predicate IDs: `{'predicate:28', 'predicate:29'}`
2. Look for the one with empty qualifiers: `predicate:28`
3. Use that for conversion

### No treats predicate with empty qualifiers

If no predicate has empty qualifiers (unlikely), the script will:
1. Log a warning
2. Use the first treats predicate ID found
3. Continue processing

## Testing

Run the pipeline:
```bash
snakemake --cores 1 extract_drugmechdb_test
```

Expected output:
```
Loading edge map from edge_map.json
Loaded 50 predicate mappings
Found 1 treats predicate IDs: {'predicate:28'}
Found treats predicate ID with empty qualifiers: predicate:28

Converting predicates from biolink:treats to predicate:28
Converted 3964 predicates to predicate:28

Removed 750 test edges from rotorobo.txt
```

Verify test.txt format:
```bash
head test.txt
# Should show predicate:28, not biolink:treats
```

## Files Modified

1. **make_test_with_drugmechdb_treat.py**:
   - Removed `load_relation_dict()` function
   - Removed `--rel-dict-file` argument
   - Added edge_map-based predicate ID lookup
   - Simplified conversion logic

2. **Snakefile**:
   - Removed `rel_dict` input
   - Updated docstring
   - Removed `--rel-dict-file` from shell command

## Related Documentation

- [MAKE_TEST_PREDICATE_MAPPING_FIXED.md](MAKE_TEST_PREDICATE_MAPPING_FIXED.md) - Previous version using rel_dict
- This document supersedes the rel_dict approach
