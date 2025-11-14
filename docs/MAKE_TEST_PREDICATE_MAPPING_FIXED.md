# Fixed Test Edge Removal - Added Predicate Mapping

## Date: 2025-11-13

## Issue

The script was removing 0 test edges instead of 750 because of a predicate format mismatch:

```json
{
  "test_edges": 750,
  "test_edges_removed_from_rotorobo": 0,
  "train_candidate_edges": 18602343
}
```

## Root Cause

The filtered TSV file uses **biolink predicate labels** (e.g., `biolink:treats`), but rotorobo.txt uses **predicate IDs** (e.g., `predicate:28`).

**Filtered TSV format**:
```
Drug    Predicate       Disease
CHEBI:27732     biolink:treats  HP:0001257
CHEBI:4462      biolink:treats  MONDO:0004773
```

**rotorobo.txt format**:
```
CHEBI:27732     predicate:28    HP:0001257
CHEBI:4462      predicate:28    MONDO:0004773
```

When the script tried to match edges, it was comparing:
- Test edge: `(CHEBI:27732, biolink:treats, HP:0001257)`
- Rotorobo edge: `(CHEBI:27732, predicate:28, HP:0001257)`

These don't match because the predicates are different!

## Solution

Use `rel_dict.txt` to map biolink predicate labels to predicate IDs, then convert the test edges before matching.

**rel_dict.txt format**:
```
biolink:treats  predicate:28
biolink:causes  predicate:15
```

## Changes Made

### 1. Added load_relation_dict() Function (Lines 210-237)

```python
def load_relation_dict(rel_dict_path: str) -> Dict[str, str]:
    """Load relation dictionary mapping relation labels to IDs.

    Args:
        rel_dict_path: Path to rel_dict.txt file

    Returns:
        Dictionary mapping relation labels (e.g., 'biolink:treats') to IDs (e.g., 'predicate:28')
    """
    logger.info(f"Loading relation dictionary from {rel_dict_path}")
    rel_to_id = {}

    with open(rel_dict_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 2:
                logger.warning(f"Invalid line in rel_dict.txt: {line}")
                continue

            rel_label, rel_id = parts
            rel_to_id[rel_label] = rel_id

    logger.info(f"Loaded {len(rel_to_id)} relation mappings")
    return rel_to_id
```

### 2. Added --rel-dict-file Argument (Lines 308-313)

```python
parser.add_argument(
    '--rel-dict-file',
    type=str,
    default='processed/rel_dict.txt',
    help='Name of relation dictionary file (default: processed/rel_dict.txt)'
)
```

### 3. Load and Use rel_dict in main() (Lines 364, 394-396, 404)

**Add rel_dict_path**:
```python
rel_dict_path = os.path.join(input_dir, args.rel_dict_file)
```

**Validate file exists**:
```python
if not os.path.exists(rel_dict_path):
    logger.error(f"Relation dict file not found: {rel_dict_path}")
    return 1
```

**Load relation dictionary**:
```python
rel_to_id = load_relation_dict(rel_dict_path)
```

### 4. Convert Predicates from Biolink Labels to IDs (Lines 425-445)

```python
# Convert predicates from biolink labels to predicate IDs
logger.info("Converting predicates from biolink labels to predicate IDs")
converted_edges = []
unmapped_predicates = set()

for subj, pred, obj in filtered_edges:
    if pred in rel_to_id:
        # Convert biolink:treats -> predicate:28
        pred_id = rel_to_id[pred]
        converted_edges.append((subj, pred_id, obj))
    else:
        unmapped_predicates.add(pred)
        # Keep original predicate if not found (will log warning)
        converted_edges.append((subj, pred, obj))

if unmapped_predicates:
    logger.warning(f"Found {len(unmapped_predicates)} unmapped predicates in filtered TSV: {unmapped_predicates}")
    logger.warning("These edges will use original predicates, which may not match rotorobo.txt format")

logger.info(f"Converted {len(converted_edges) - len(unmapped_predicates)} predicates to IDs")
filtered_edges = converted_edges
```

### 5. Simplified Matching Logic (Lines 495-509)

**Before** (matching by (subject, object) pairs):
```python
test_edge_pairs = {(subj, obj) for subj, pred, obj in test_edges}
for edge in all_edges:
    subj, pred, obj = edge
    if (subj, obj) in test_edge_pairs and pred in treats_predicates:
        removed_count += 1
    else:
        train_candidates.append(edge)
```

**After** (exact triple matching):
```python
test_edge_set = set(test_edges)
for edge in all_edges:
    if edge in test_edge_set:
        removed_count += 1
    else:
        train_candidates.append(edge)
```

Much simpler and safer! Now we match full triples exactly.

### 6. Updated Snakefile (Line 206)

```python
shell:
    """
    python src/make_test_with_drugmechdb_treat.py \
        --input-dir {params.input_dir} \
        --filtered-tsv {input.filtered_tsv} \
        --rel-dict-file processed/rel_dict.txt \    # ← Added this line
        --test-pct {params.test_pct} \
        --seed {params.seed} \
        --output-dir {params.input_dir} \
        2>&1 | tee {log}
    """
```

## How It Works Now

### Step 1: Load Relation Dictionary
```python
rel_to_id = {
    'biolink:treats': 'predicate:28',
    'biolink:causes': 'predicate:15',
    # ...
}
```

### Step 2: Load Filtered Edges
```python
filtered_edges = [
    ('CHEBI:27732', 'biolink:treats', 'HP:0001257'),
    ('CHEBI:4462', 'biolink:treats', 'MONDO:0004773'),
    # ...
]
```

### Step 3: Convert Predicates
```python
converted_edges = [
    ('CHEBI:27732', 'predicate:28', 'HP:0001257'),    # ← Converted!
    ('CHEBI:4462', 'predicate:28', 'MONDO:0004773'),  # ← Converted!
    # ...
]
```

### Step 4: Sample Test Edges
```python
test_edges = [
    ('CHEBI:27732', 'predicate:28', 'HP:0001257'),
    # ... (750 edges total)
]
```

### Step 5: Exact Triple Matching
```python
test_edge_set = {
    ('CHEBI:27732', 'predicate:28', 'HP:0001257'),
    # ...
}

# Remove test edges from all_edges
for edge in all_edges:
    if edge in test_edge_set:  # Exact match!
        removed_count += 1
    else:
        train_candidates.append(edge)
```

## Expected Output

```
Loading relation dictionary from /workspace/data/robokop/CGGD_alltreat/processed/rel_dict.txt
Loaded 50 relation mappings

Loading filtered treats edges from drugmechdb_treats_filtered.txt
Header line: Drug	Predicate	Disease
First edge tuple: (subject='CHEBI:27732', predicate='biolink:treats', object='HP:0001257')
Loaded 3964 filtered treats edges

Converting predicates from biolink labels to predicate IDs
Converted 3964 predicates to IDs

Deduplicating filtered edges (original count: 3964)
Unique filtered edges: 3754

Sampled 750 test edges

================================================================================
DEBUG: Sample edges comparison
================================================================================
Sample test edge (from filtered TSV): ('CHEBI:27732', 'predicate:28', 'HP:0001257')
Sample rotorobo edge: ('CHEBI:27732', 'predicate:28', 'HP:0001257')
Looking for first test edge in rotorobo:
  Test edge: (CHEBI:27732, predicate:28, HP:0001257)
  FOUND: Exact match in rotorobo.txt
================================================================================

Creating train candidates by removing test edges from rotorobo.txt
Matching test edges by full triples (subject, predicate, object)
Created set of 750 unique test edges

Removed 750 test edges from rotorobo.txt
Train candidates: 18601593 edges

✓ Success!
```

## Benefits of This Fix

1. **Correct matching**: Now matches full triples exactly after predicate conversion
2. **Safer logic**: No more guessing which predicate IDs represent treats
3. **More maintainable**: Uses the canonical rel_dict.txt mapping
4. **Better debugging**: Clear logging of predicate conversion
5. **Handles edge cases**: Warns about unmapped predicates

## Files Modified

1. **make_test_with_drugmechdb_treat.py**:
   - Added `load_relation_dict()` function
   - Added `--rel-dict-file` argument
   - Added predicate conversion logic
   - Simplified matching to exact triple matching
   - Updated debug output

2. **Snakefile**:
   - Added `--rel-dict-file processed/rel_dict.txt` to shell command
   - `rel_dict` already existed as input dependency

## Testing

Run the pipeline step:

```bash
snakemake --cores 1 make_test_with_drugmechdb_treat
```

Check the output:
```bash
cat /workspace/data/robokop/CGGD_alltreat/test_statistics.json | jq '.test_edges_removed_from_rotorobo'
# Should show: 750 (not 0!)
```

Verify train_candidates.txt is smaller:
```bash
wc -l /workspace/data/robokop/CGGD_alltreat/rotorobo.txt
wc -l /workspace/data/robokop/CGGD_alltreat/train_candidates.txt
# Difference should be 750
```

## Related Issues

- Original issue: 0 test edges removed instead of 750
- Root cause: Predicate format mismatch (biolink:treats vs predicate:28)
- Solution: Use rel_dict.txt to convert predicates before matching
