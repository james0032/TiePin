# Fix: Intermediate Nodes Column Name Parsing

## Problem

The script `add_ground_truth_column.py` was showing warnings like:

```
WARNING - Row 1: Failed to parse intermediate nodes '': invalid syntax (<unknown>, line 0)
WARNING - Row 2: Failed to parse intermediate nodes '': invalid syntax (<unknown>, line 0)
WARNING - Row 3: Failed to parse intermediate nodes '': invalid syntax (<unknown>, line 0)
```

Even though the CSV file had non-empty intermediate nodes:
```csv
Drug,Disease,Intermediate_Nodes,drugmechdb_path_id
CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445, HGNC.FAMILY:862, NCBITaxon:5052]",DB00582_MESH_D055744_1
```

## Root Cause

**Column name mismatch:**

- **File had**: `Intermediate_Nodes` (with underscore, no brackets)
- **Script expected**: `[Intermediate Nodes]` (with brackets and space)

When the column didn't match, `row.get('[Intermediate Nodes]', '')` returned an empty string `''`, causing the parser to fail with:
```python
ast.literal_eval('')  # SyntaxError: invalid syntax
```

## Solution

Updated the script to try **multiple column name variants**:

```python
# Try multiple possible column names for intermediate nodes
intermediate_nodes_str = (
    row.get('[Intermediate Nodes]', '') or
    row.get('Intermediate_Nodes', '') or
    row.get('Intermediate Nodes', '') or
    row.get('[Intermediate_Nodes]', '')
).strip()
```

Now the script accepts any of these formats:
- `[Intermediate Nodes]` (original format)
- `Intermediate_Nodes` (underscore format - from your file)
- `Intermediate Nodes` (space format)
- `[Intermediate_Nodes]` (bracket + underscore)

## Testing

Created comprehensive test in `test_intermediate_nodes_parsing.py`:

```bash
$ python test_intermediate_nodes_parsing.py

Test: Underscore format (Intermediate_Nodes)
  ✓ PASS: Found 2 intermediate nodes: ['GO:0006696', 'GO:0030445']

Test: Bracket format ([Intermediate Nodes])
  ✓ PASS: Found 2 intermediate nodes: ['GO:0006696', 'GO:0030445']

Test: Space format (Intermediate Nodes)
  ✓ PASS: Found 2 intermediate nodes: ['GO:0006696', 'GO:0030445']

Test: Empty intermediate nodes
  ✓ PASS: Found 0 intermediate nodes: []

✓ All tests PASSED
```

## Changes Made

### File: `add_ground_truth_column.py`

1. **Lines 88-94**: Try multiple column name variants with fallback
2. **Lines 85-86**: Add debug logging for available columns
3. **Line 103**: Handle empty string explicitly before parsing

### Before:
```python
intermediate_nodes_str = row.get('[Intermediate Nodes]', '').strip()

if intermediate_nodes_str == '[]':
    intermediate_nodes = []
else:
    # Parse...
```

### After:
```python
# Try multiple possible column names for intermediate nodes
intermediate_nodes_str = (
    row.get('[Intermediate Nodes]', '') or
    row.get('Intermediate_Nodes', '') or
    row.get('Intermediate Nodes', '') or
    row.get('[Intermediate_Nodes]', '')
).strip()

# Handle empty string or empty list
if not intermediate_nodes_str or intermediate_nodes_str == '[]':
    intermediate_nodes = []
else:
    # Parse...
```

## Expected Behavior

### Before Fix:
```
2025-11-21 14:39:47,842 - INFO - Loading mechanistic paths from results/mechanistic_paths/drugmechdb_path_id_results.txt
2025-11-21 14:39:47,842 - WARNING - Row 1: Failed to parse intermediate nodes '': invalid syntax (<unknown>, line 0)
2025-11-21 14:39:47,842 - WARNING - Row 2: Failed to parse intermediate nodes '': invalid syntax (<unknown>, line 0)
2025-11-21 14:39:47,842 - WARNING - Row 3: Failed to parse intermediate nodes '': invalid syntax (<unknown>, line 0)
```

### After Fix:
```
2025-11-21 10:45:51,800 - INFO - Loading mechanistic paths from drugmechdb_path_id_results.txt
2025-11-21 10:45:51,801 - INFO - Loaded 1234 mechanistic paths
```

No warnings! ✅

## Impact

- **Does NOT affect `IsGroundTruth` column** - that only depends on ground truth edges
- **DOES fix `In_path` and `On_specific_path` columns** - these now correctly identify edges in mechanistic paths
- **Backward compatible** - still works with the original `[Intermediate Nodes]` format

## Usage

The fix is automatic. Just run the script as before:

```bash
python add_ground_truth_column.py \
    --tracin-csv results/triple_000_tracin.csv \
    --ground-truth ground_truth/drugmechdb_edges.jsonl \
    --mechanistic-paths results/mechanistic_paths/drugmechdb_path_id_results.txt \
    --output results/triple_000_tracin_with_gt.csv
```

Now it will work with either column name format!
