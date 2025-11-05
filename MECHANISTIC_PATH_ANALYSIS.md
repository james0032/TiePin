# Mechanistic Path Analysis for TracIn Results

## Overview

The `add_ground_truth_column.py` script now supports adding an **"In_path"** column to TracIn CSV results. This column identifies which training edges are part of the mechanistic path connecting the test drug and disease through intermediate nodes.

## What is the "In_path" Column?

The "In_path" column indicates whether a training edge connects nodes that are part of the mechanistic path:

- **In_path = 1**: The training edge connects two nodes in the path
- **In_path = 0**: The training edge is NOT part of the mechanistic path

### Which edges are considered "in path"?

A training edge `(TrainHead, TrainTail)` is marked as "in path" if **both** endpoints are in the set of:
- Test drug (head entity)
- Test disease (tail entity)
- Intermediate nodes from the mechanistic path

This covers all possible path connections:
1. `[test_drug, intermediate_node]`
2. `[intermediate_node, test_disease]`
3. `[intermediate_node_1, intermediate_node_2]`
4. All reversed versions (since the graph is undirected)

## Input Files

### 1. TracIn CSV File
The standard TracIn output with columns:
```
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore,SelfInfluence
```

Example:
```csv
CHEBI:17154,Nicotinamide,predicate:28,biolink:has_adverse_event,MONDO:0019975,pellagra,NCBIGene:10135,NAMPT,predicate:56,biolink:affects,CHEBI:13389,Nadide,0.2247,0.1017
```

### 2. Mechanistic Paths CSV File
Format: `Drug,Disease,[Intermediate Nodes]`

Example from `dedup_treats_mechanistic_paths.txt`:
```csv
Drug,Disease,[Intermediate Nodes]
CHEBI:15940,MONDO:0024298,"[CHEBI:17154, MONDO:0019975]"
CHEBI:17154,MONDO:0019975,[CHEBI:15940]
CHEBI:17268,MONDO:0019975,"[CHEBI:15940, CHEBI:17154, MESH:D009539]"
CHEBI:17688,MONDO:0019975,[]
CHEBI:45213,MONDO:0019975,"[CHEBI:15940, CHEBI:17154]"
```

### 3. Ground Truth JSONL File (existing)
The ground truth edges in JSONL format (existing functionality).

## Usage

### Basic: Only Ground Truth Column

```bash
python add_ground_truth_column.py \
    --tracin-csv dmdb_results/test_scores_tracin.csv \
    --ground-truth ground_truth/drugmechdb_edges.jsonl \
    --output dmdb_results/test_scores_tracin_with_gt.csv
```

Output columns: Original columns + **IsGroundTruth**

### Advanced: Ground Truth + Mechanistic Path Columns

```bash
python add_ground_truth_column.py \
    --tracin-csv dmdb_results/triple_000_CHEBI_17154_MONDO_0019975_tracin.csv \
    --ground-truth ground_truth/drugmechdb_edges.jsonl \
    --mechanistic-paths dedup_treats_mechanistic_paths.txt \
    --output dmdb_results/triple_000_with_gt_and_path.csv
```

Output columns: Original columns + **IsGroundTruth** + **In_path**

## Example Workflow

### For Test Triple: CHEBI:17154 (Nicotinamide) → MONDO:0019975 (pellagra)

**1. From mechanistic paths file:**
```
CHEBI:17154,MONDO:0019975,[CHEBI:15940]
```

This means the path is:
- Test Drug: `CHEBI:17154` (Nicotinamide)
- Intermediate Node: `CHEBI:15940`
- Test Disease: `MONDO:0019975` (pellagra)

**2. Run the script:**
```bash
python add_ground_truth_column.py \
    --tracin-csv triple_000_CHEBI_17154_MONDO_0019975_tracin.csv \
    --ground-truth drugmechdb_edges.jsonl \
    --mechanistic-paths dedup_treats_mechanistic_paths.txt \
    --output triple_000_with_path.csv
```

**3. Training edges marked as In_path = 1:**
- Any edge connecting `CHEBI:17154` ↔ `CHEBI:15940`
- Any edge connecting `CHEBI:15940` ↔ `MONDO:0019975`
- Any edge connecting `CHEBI:17154` ↔ `MONDO:0019975`

**4. Training edges marked as In_path = 0:**
- All other edges (e.g., `NCBIGene:10135` ↔ `CHEBI:13389`)

## Output Format

The output CSV will have all original columns plus:

| Column Name | Type | Description |
|------------|------|-------------|
| IsGroundTruth | 0/1 | 1 if edge exists in ground truth JSONL |
| In_path | 0/1 | 1 if edge connects nodes in mechanistic path |

Example output:
```csv
TestHead,TestHead_label,...,TracInScore,SelfInfluence,IsGroundTruth,In_path
CHEBI:17154,Nicotinamide,...,0.2247,0.1017,1,1
CHEBI:17154,Nicotinamide,...,0.2191,0.1017,0,0
```

## Analysis Examples

### Find all training edges in mechanistic paths:
```bash
# Get edges with In_path = 1
grep ',1$' triple_000_with_path.csv | head -10
```

### Find edges that are both ground truth AND in path:
```bash
# Get edges with IsGroundTruth = 1 AND In_path = 1
awk -F',' '$NF==1 && $(NF-1)==1' triple_000_with_path.csv
```

### Count statistics:
```python
import pandas as pd

df = pd.read_csv('triple_000_with_path.csv')

print(f"Total training edges: {len(df)}")
print(f"Edges in path: {df['In_path'].sum()}")
print(f"Ground truth edges: {df['IsGroundTruth'].sum()}")
print(f"Both in path AND ground truth: {((df['In_path']==1) & (df['IsGroundTruth']==1)).sum()}")
```

## Implementation Details

### Path Matching Logic

The function `is_in_path()` checks if a training edge is part of the mechanistic path:

```python
def is_in_path(train_head, train_tail, test_head, test_tail, intermediate_nodes):
    # Create set of all path nodes
    path_nodes = {test_head, test_tail}
    path_nodes.update(intermediate_nodes)

    # Both endpoints must be in the path
    return train_head in path_nodes and train_tail in path_nodes
```

### Handling Edge Cases

1. **Empty intermediate nodes (`[]`)**:
   - Only edges connecting test_head ↔ test_tail are marked as in path

2. **Multiple intermediate nodes**:
   - All pairwise connections are considered in path

3. **Undirected graph**:
   - Order doesn't matter: `(A, B)` and `(B, A)` are treated the same

## Troubleshooting

### No edges marked as In_path

**Possible causes:**
1. Test triple not in mechanistic paths CSV
2. Entity ID mismatch (check CURIE formats match exactly)
3. Empty intermediate nodes list

**Solution:**
```bash
# Check if test triple exists in paths file
grep "CHEBI:17154,MONDO:0019975" dedup_treats_mechanistic_paths.txt
```

### Mismatch between TracIn file and paths file

**Error:** Warning about missing test triple in paths

**Solution:** Ensure the test triple entities match exactly:
- TracIn file: Uses `TestHead` and `TestTail` columns
- Paths file: Uses `Drug` and `Disease` columns

## Performance Notes

- Loading paths: ~1-2 seconds for 1000 paths
- Processing: ~0.1-0.5 seconds per 1000 TracIn rows
- Memory: Minimal (paths loaded into dictionary)

## See Also

- [add_ground_truth_column.py](add_ground_truth_column.py) - Main script
- [example_add_path_column.sh](example_add_path_column.sh) - Example usage
- [utils/add_pair_exists_column.py](utils/add_pair_exists_column.py) - Related utility for path filtering
