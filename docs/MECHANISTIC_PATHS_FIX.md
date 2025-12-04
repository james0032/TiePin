# Mechanistic Paths CSV Parsing Fix

## Problem

The `add_ground_truth_column.py` script was not correctly reading the mechanistic-paths file because it assumed a different format.

### Expected Format

The mechanistic paths CSV file has this format:
```csv
Drug,Disease,Intermediate_Nodes,drugmechdb_path_id
CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445, HGNC.FAMILY:862, NCBITaxon:5052]",DB00582_MESH_D055744_1
```

Key characteristics:
- Standard CSV format with comma delimiters
- `Intermediate_Nodes` column contains a bracketed list: `[NODE1, NODE2, NODE3]`
- The column may be quoted (e.g., `"[...]"`) by CSV writers
- Node names can contain special characters like `:` and `.`

## What Was Wrong

The old code had overly complex logic trying to handle multiple column name variations and used `ast.literal_eval()` which didn't work well with the actual format.

## The Fix

Simplified the parsing logic in [add_ground_truth_column.py:68-139](add_ground_truth_column.py#L68-L139):

1. **Use standard CSV parsing**: Python's `csv.DictReader` handles the quoting automatically
2. **Simple bracket stripping**: Remove `[` and `]` from start/end
3. **Split by comma**: Parse the comma-separated node list
4. **Strip whitespace**: Clean up each node name

### New Parsing Logic

```python
# Get the Intermediate_Nodes column value (quotes already handled by csv.DictReader)
intermediate_nodes_str = row.get('Intermediate_Nodes', '').strip()

# Handle empty or empty list
if not intermediate_nodes_str or intermediate_nodes_str == '[]':
    intermediate_nodes = []
else:
    # Remove brackets and split by comma
    if intermediate_nodes_str.startswith('[') and intermediate_nodes_str.endswith(']'):
        nodes_str = intermediate_nodes_str[1:-1].strip()
        intermediate_nodes = [node.strip() for node in nodes_str.split(',') if node.strip()]
    else:
        # Fallback: split by comma directly
        intermediate_nodes = [node.strip() for node in intermediate_nodes_str.split(',') if node.strip()]
```

## Testing

The fix was tested with [test_mechanistic_paths_parsing.py](test_mechanistic_paths_parsing.py):

```bash
python test_mechanistic_paths_parsing.py
```

**Test Cases:**
- ✓ Path with 4 intermediate nodes: `[GO:0006696, GO:0030445, HGNC.FAMILY:862, NCBITaxon:5052]`
- ✓ Path with 1 intermediate node: `[CHEBI:12345]`
- ✓ Empty path: `[]`
- ✓ Path with special characters in node names

**All tests passed!**

## Changes Made

### File: `add_ground_truth_column.py`

1. **Removed unused import**: Deleted `import ast` (line 14)
2. **Simplified `load_mechanistic_paths()` function** (lines 68-139):
   - Removed complex column name variations handling
   - Simplified bracket parsing logic
   - Added better logging with example paths
   - Added statistics logging (average path length)

## Usage

The script now correctly handles mechanistic paths CSV files:

```bash
python add_ground_truth_column.py \
    --tracin-csv results/triple_000_tracin.csv \
    --ground-truth ground_truth/drugmechdb_edges.jsonl \
    --mechanistic-paths dedup_treats_mechanistic_paths.txt \
    --output results/triple_000_with_gt_and_path.csv
```

The output will include:
- `IsGroundTruth`: 1 if training edge matches ground truth
- `In_path`: 1 if training edge connects any nodes in the mechanistic path
- `On_specific_path`: 1 if training edge is on the sequential path from drug to disease

## Verification

When the script runs, you'll see logging like:

```
INFO - CSV columns found: ['Drug', 'Disease', 'Intermediate_Nodes', 'drugmechdb_path_id']
INFO -   Example path 1: CHEBI:10023 -> HP:0020103 via ['GO:0006696', 'GO:0030445', 'HGNC.FAMILY:862', 'NCBITaxon:5052']
INFO - Loaded 4 mechanistic paths
INFO -   3 paths have intermediate nodes (avg length: 2.7)
```

This confirms the parsing is working correctly!
