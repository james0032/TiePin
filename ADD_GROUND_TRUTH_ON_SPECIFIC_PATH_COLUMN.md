# Added "On_specific_path" Column to add_ground_truth_column.py

## Date: 2025-11-14

## Enhancement

Added a new column `On_specific_path` to distinguish between:
- **In_path**: Edge connects any nodes in the mechanistic path (loose definition)
- **On_specific_path**: Edge is on the **sequential path** from test head to test tail (strict definition)

## Motivation

The existing `In_path` column was too permissive. It marked an edge as "in path" if **both endpoints** were in the set of path nodes (test entities + intermediate nodes).

### Problem with `In_path`

**Example path**: `Drug → A → B → Disease`

```python
# Current In_path logic
path_nodes = {Drug, A, B, Disease}
if train_head in path_nodes and train_tail in path_nodes:
    In_path = 1  # TRUE
```

**Edges marked as In_path=1**:
- ✓ (Drug, A) - correct, on sequential path
- ✓ (A, B) - correct, on sequential path
- ✓ (B, Disease) - correct, on sequential path
- ✗ (Drug, B) - **wrong**, not on sequential path (shortcut)
- ✗ (A, Disease) - **wrong**, not on sequential path (shortcut)
- ✗ (Drug, Disease) - **wrong**, not on sequential path (direct edge)

The `In_path` column includes **shortcuts** and **direct connections** that aren't part of the specific mechanistic path.

## New Column: On_specific_path

### Definition

An edge is `On_specific_path=1` if it connects **consecutive nodes** in the ordered path sequence.

**Path sequence**: `[test_head, intermediate[0], intermediate[1], ..., intermediate[n], test_tail]`

### Implementation

```python
def is_on_specific_path(
    train_head: str,
    train_tail: str,
    test_head: str,
    test_tail: str,
    intermediate_nodes: List[str]
) -> bool:
    """Check if edge is on the specific sequential path."""
    # Build full path sequence
    path_sequence = [test_head] + intermediate_nodes + [test_tail]

    # Check if edge connects consecutive nodes
    for i in range(len(path_sequence) - 1):
        node_a = path_sequence[i]
        node_b = path_sequence[i + 1]

        # Check both directions (graph is undirected)
        if (train_head == node_a and train_tail == node_b) or \
           (train_head == node_b and train_tail == node_a):
            return True

    return False
```

### Example

**Given path**: `Drug → A → B → Disease`

**Edges checked**:

| Train Edge | In_path | On_specific_path | Explanation |
|------------|---------|------------------|-------------|
| (Drug, A) | 1 | 1 | ✓ Consecutive in path |
| (A, B) | 1 | 1 | ✓ Consecutive in path |
| (B, Disease) | 1 | 1 | ✓ Consecutive in path |
| (Drug, B) | 1 | 0 | ✗ Not consecutive (skips A) |
| (A, Disease) | 1 | 0 | ✗ Not consecutive (skips B) |
| (Drug, Disease) | 1 | 0 | ✗ Not consecutive (skips A, B) |
| (Drug, X) | 0 | 0 | ✗ X not in path |

## Output Format

When `--mechanistic-paths` is provided, the output CSV now has **three** new columns:

```csv
...,IsGroundTruth,In_path,On_specific_path
...,1,1,1          # Ground truth edge on specific path
...,0,1,1          # Not ground truth, but on specific path
...,0,1,0          # In path nodes, but not on specific path (shortcut)
...,0,0,0          # Not related to path
```

## Use Cases

### Analysis 1: Identify Mechanistic Edges

**Question**: Which training edges are on the **exact mechanistic path**?

**Answer**: `On_specific_path = 1`

These are the edges that form the sequential path from drug to disease.

### Analysis 2: Find Shortcuts

**Question**: Which edges connect path nodes but skip intermediate steps?

**Answer**: `In_path = 1 AND On_specific_path = 0`

These are shortcuts or alternative connections between path nodes.

### Analysis 3: TracIn Score Analysis

**Question**: Do edges on the specific path have higher TracIn scores?

**Answer**: Compare TracIn scores for:
- `On_specific_path = 1` vs `On_specific_path = 0`
- Group by: ground truth edges, non-ground truth edges

### Analysis 4: Path Validation

**Question**: Are ground truth edges concentrated on the specific path?

**Answer**: Count edges where `IsGroundTruth = 1 AND On_specific_path = 1`

## Changes Made

### Lines 162-198: New Function `is_on_specific_path`

Added function to check if edge is on the specific sequential path.

### Lines 228, 239, 259, 283, 293-295, 299, 308, 320: Updated to Handle New Column

- Added `on_specific_path_count` counter
- Added `On_specific_path` to CSV header
- Compute `On_specific_path` value for each edge
- Write `On_specific_path` to output
- Log statistics for `On_specific_path`

### Lines 8-10: Updated Docstring

Clarified the difference between `In_path` and `On_specific_path`.

### Lines 336-346: Updated Examples

Added explanation of output columns.

## Example Usage

```bash
# Add all three columns: IsGroundTruth, In_path, On_specific_path
python add_ground_truth_column.py \
    --tracin-csv results/triple_000_tracin.csv \
    --ground-truth ground_truth/drugmechdb_edges.jsonl \
    --mechanistic-paths mechanistic_paths.csv \
    --output results/triple_000_with_all_columns.csv
```

**Expected output**:
```
2025-11-14 15:30:12 - INFO - Loaded 5000 unique ground truth edges
2025-11-14 15:30:12 - INFO - Loaded 100 mechanistic paths
2025-11-14 15:30:12 - INFO - Processing TracIn CSV: results/triple_000_tracin.csv
2025-11-14 15:30:13 - INFO - Processed 10000 training edges
2025-11-14 15:30:13 - INFO - Matched 250 ground truth edges (2.50%)
2025-11-14 15:30:13 - INFO - Found 500 edges in mechanistic paths (5.00%)
2025-11-14 15:30:13 - INFO - Found 150 edges on specific sequential path (1.50%)
2025-11-14 15:30:13 - INFO - ✓ Found 250 training edges that match ground truth
2025-11-14 15:30:13 - INFO - ✓ Found 500 training edges in mechanistic paths
2025-11-14 15:30:13 - INFO - ✓ Found 150 training edges on specific sequential path
```

## Interpretation

From the example above:
- **500 edges** connect path nodes (In_path=1)
- **150 edges** are on the specific path (On_specific_path=1)
- **350 edges** are shortcuts/alternative connections (In_path=1, On_specific_path=0)

This shows that **70% of "in path" edges are actually shortcuts**, not on the sequential mechanistic path!

## Benefits

1. **More precise path analysis**: Distinguish between edges that are truly on the mechanistic path vs. shortcuts
2. **Better evaluation**: Can evaluate TracIn's ability to identify the **exact** mechanistic path
3. **Debugging**: Identify cases where the model learns shortcuts instead of the true mechanism
4. **Interpretability**: Clearer understanding of which edges form the causal chain

## Related Files

- **add_ground_truth_column.py**: Main script with the enhancement
- Input: TracIn CSV, ground truth JSONL, mechanistic paths CSV
- Output: Enhanced CSV with three indicator columns

## Testing

Test with a simple path:

**mechanistic_paths.csv**:
```csv
Drug,Disease,[Intermediate Nodes]
CHEBI:123,MONDO:456,"[PROTEIN:A, PROTEIN:B]"
```

**Expected path**: `CHEBI:123 → PROTEIN:A → PROTEIN:B → MONDO:456`

**Test edges**:

| Edge | Expected On_specific_path |
|------|---------------------------|
| (CHEBI:123, PROTEIN:A) | 1 |
| (PROTEIN:A, PROTEIN:B) | 1 |
| (PROTEIN:B, MONDO:456) | 1 |
| (CHEBI:123, PROTEIN:B) | 0 (shortcut) |
| (PROTEIN:A, MONDO:456) | 0 (shortcut) |
| (CHEBI:123, MONDO:456) | 0 (direct edge) |

All edges above would have `In_path=1` (both endpoints in path nodes), but only the first three have `On_specific_path=1`.
