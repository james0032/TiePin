# Fixed Proximity Filter to Use Intersection Instead of Union

## Date: 2025-11-14

## Issue

The proximity filter was using **UNION** of n-hop neighborhoods instead of **INTERSECTION**, which included many irrelevant edges that are NOT on paths connecting drugs to diseases.

## Root Cause

**Original code** (lines 429-450):
```python
# Extract ALL test entities (drugs + diseases together)
test_entities = set()
for h, r, t in test_triples:
    test_entities.add(int(h))  # Add drugs
    test_entities.add(int(t))  # Add diseases

# Get n-hop neighborhood around ALL entities at once (UNION)
subset_nodes, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=test_entity_list,  # ALL drugs and diseases together
    num_hops=n_hops,
    ...
)
```

This creates a **UNION** of neighborhoods:
- Nodes within n hops of ANY drug OR ANY disease
- Includes nodes far from diseases (only near drugs)
- Includes nodes far from drugs (only near diseases)
- Many edges NOT on drug→disease paths

## Solution

**New code** (lines 429-503):

### Step 1: Separate Heads and Tails

```python
# Extract test entities - separate heads (drugs) and tails (diseases)
head_entities = set()
tail_entities = set()
for h, r, t in test_triples:
    head_entities.add(int(h))
    tail_entities.add(int(t))
```

### Step 2: Compute Separate Neighborhoods

```python
# Get n-hop neighborhoods from heads (drugs) and tails (diseases) separately
head_nodes, head_edge_index, _, _ = k_hop_subgraph(
    node_idx=head_entity_list,
    num_hops=n_hops,
    ...
)

tail_nodes, tail_edge_index, _, _ = k_hop_subgraph(
    node_idx=tail_entity_list,
    num_hops=n_hops,
    ...
)
```

### Step 3: Compute Intersection

```python
# INTERSECTION: Keep only nodes reachable from BOTH heads and tails
head_nodes_set = set(head_nodes.tolist())
tail_nodes_set = set(tail_nodes.tolist())
intersect_nodes_set = head_nodes_set & tail_nodes_set

logger.info(f"Found {len(head_nodes_set)} nodes in {n_hops}-hop neighborhood of heads (drugs)")
logger.info(f"Found {len(tail_nodes_set)} nodes in {n_hops}-hop neighborhood of tails (diseases)")
logger.info(f"Intersection: {len(intersect_nodes_set)} nodes reachable from BOTH heads and tails")
```

### Step 4: Filter Edges by Intersection

```python
for i in range(subset_edge_index.shape[1]):
    src = subset_edge_index[0, i].item()
    dst = subset_edge_index[1, i].item()

    # Skip reverse edges
    if src > dst:
        continue

    # INTERSECTION CONSTRAINT: Both endpoints must be in intersection set
    # This ensures edges are on potential paths between drugs and diseases
    if src not in intersect_nodes_set or dst not in intersect_nodes_set:
        continue  # Skip edges not in intersection

    # ... rest of degree and hop filtering logic
```

## How It Works

### Before (UNION Approach)

```
Drug1 ----> X ----> Y          Z <---- W <---- Disease1
       (1 hop)  (2 hops)  (no path)  (2 hops) (1 hop)
```

**Union neighborhood**: {Drug1, X, Y, Z, W, Disease1}
- Y is 2 hops from Drug1, included even though far from diseases
- Z is 2 hops from Disease1, included even though far from drugs
- NO path from Drug1 to Disease1 through Y or Z

### After (INTERSECTION Approach)

```
Drug1 ----> X ----> M <---- W <---- Disease1
       (1 hop)  (2 hops)  (2 hops) (1 hop)
```

**Intersection neighborhood**: {Drug1, X, M, W, Disease1}
- Only nodes within n hops of BOTH drugs AND diseases
- Focuses on intermediate nodes (X, M, W) that could connect drugs to diseases
- Excludes Y and Z (far from one side)

## Expected Impact

### Filtering Selectivity

**Before (UNION)**:
- With n_hops=2: ~50-70% of edges retained
- Many irrelevant edges included

**After (INTERSECTION)**:
- With n_hops=2: ~20-40% of edges retained (more selective)
- Focuses on edges that could form drug→disease paths
- Better signal-to-noise ratio

### Example Log Output

```
Test head entities (drugs): 150
Test tail entities (diseases): 200
Computing n-hop neighborhood from head entities (drugs)...
Computing n-hop neighborhood from tail entities (diseases)...
Found 25000 nodes in 2-hop neighborhood of heads (drugs)
Found 30000 nodes in 2-hop neighborhood of tails (diseases)
Intersection: 8000 nodes reachable from BOTH heads and tails
Filtered: 18602343 → 5000000 (73.1% reduction)
```

## Biological Motivation

For drug repurposing, we care about **mechanistic paths**:

```
Drug --> Target --> Pathway --> Phenotype --> Disease
```

**UNION problems**:
- Includes targets far from diseases
- Includes phenotypes far from drugs
- Dilutes signal with irrelevant biology

**INTERSECTION benefits**:
- Focuses on intermediate nodes connecting drugs and diseases
- Keeps edges that participate in complete paths
- Better aligns with mechanistic reasoning
- More relevant for drug repurposing predictions

## Testing

### Test the Change

```bash
python filter_training_by_proximity_pyg.py \
    --train train_candidates.txt \
    --test test.txt \
    --output train_filtered.txt \
    --n-hops 2 \
    --min-degree 2
```

Expected output:
```
Test head entities (drugs): 150
Test tail entities (diseases): 200
Found 25000 nodes in 2-hop neighborhood of heads (drugs)
Found 30000 nodes in 2-hop neighborhood of tails (diseases)
Intersection: 8000 nodes reachable from BOTH heads and tails
Filtered: 18602343 → 5000000 (73.1% reduction)
```

### Verify More Selective Filtering

Compare before/after:
```bash
# Before: ~50% reduction
# After: ~70-80% reduction (more selective)
```

## Files Modified

1. **filter_training_by_proximity_pyg.py**:
   - Lines 429-438: Separate head_entities and tail_entities
   - Lines 441-458: Compute separate k_hop_subgraph for heads and tails
   - Lines 460-467: Compute intersection of neighborhoods
   - Lines 469-477: Combine edge indices and create test_entities set
   - Lines 500-503: Add intersection constraint to edge filtering loop

## Backward Compatibility

The change maintains the same API:
- Same function signature
- Same parameters
- Same output format

Only the **internal algorithm** changed from union to intersection.

## Related Documentation

- [PROXIMITY_FILTER_ALGORITHM_ANALYSIS.md](PROXIMITY_FILTER_ALGORITHM_ANALYSIS.md) - Detailed analysis of union vs intersection
- Lines 402-508 in filter_training_by_proximity_pyg.py - `filter_for_multiple_test_triples()` method
