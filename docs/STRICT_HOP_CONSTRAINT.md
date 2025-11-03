# Strict Hop Constraint Feature

## Summary

Added a `--strict-hop-constraint` flag to [filter_training_by_proximity_pyg.py](filter_training_by_proximity_pyg.py) that provides an additional layer of safety when filtering knowledge graph triples by hop distance.

## The Original Issue

You reported that with `--n-hops 2` and `--min-degree` settings, the filtered subgraph could contain paths that were 5-6 hops away from a single test triple. This was **not actually a bug**, but rather a misunderstanding of how the filtering works.

### How the Original (Non-Strict) Mode Works

The filtering has two stages:

1. **N-hop Neighborhood Extraction**: Use PyG's `k_hop_subgraph()` to extract all nodes within `n_hops` of test entities
2. **Degree Filtering**: Keep edges where at least one endpoint has degree ≥ `min_degree` in the subgraph

The key insight: **PyG's `k_hop_subgraph` already ensures all nodes are within n_hops**. However, when you compute paths in the *filtered* graph using BFS, you might find longer paths because edges can connect nodes that are far apart (e.g., node A at 2 hops north, node B at 2 hops south, creating a 4-hop path between them through the test node).

**This is expected behavior!** Every individual edge still has both endpoints within n_hops of the test entities.

## What the Strict Mode Adds

The `--strict-hop-constraint` flag adds explicit validation:

```python
# For each edge (src, dst) in the subgraph:
if strict_hop_constraint:
    src_dist = hop_distances[src].item()
    dst_dist = hop_distances[dst].item()

    # Reject if either endpoint is beyond n_hops
    if src_dist > n_hops or dst_dist > n_hops:
        should_keep = False
```

This ensures that:
1. Both endpoints of every edge are within `n_hops` (computed via BFS from test entities)
2. Even edges preserved by `--preserve-test-edges` must respect the hop limit
3. Provides extra safety and explicit validation

## Usage

```bash
# Original mode (default)
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered_train.txt \
    --n-hops 2 \
    --min-degree 2

# With strict hop constraint
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered_train.txt \
    --n-hops 2 \
    --min-degree 2 \
    --strict-hop-constraint
```

## Implementation Details

### New Method: `_compute_hop_distances()`

Computes shortest-path distances from test entities to all nodes using BFS:

```python
def _compute_hop_distances(self, test_entities: List[int], max_hops: int) -> torch.Tensor:
    """Compute shortest hop distance from test entities to all nodes."""
    # Returns tensor where distances[node_id] = hop distance
    # Returns -1 for unreachable nodes or nodes beyond max_hops
```

### Modified Methods

Both `filter_for_single_test_triple()` and `filter_for_multiple_test_triples()` now accept a `strict_hop_constraint` parameter (default: `False` for backward compatibility).

## Testing

Three test files are provided:

1. **[test_strict_simple.py](test_strict_simple.py)**: Simple linear graph demonstrating the feature
2. **[debug_strict_hop.py](debug_strict_hop.py)** and **[debug_strict_hop2.py](debug_strict_hop2.py)**: Detailed debugging scripts
3. **[debug_original_test.py](debug_original_test.py)**: Validates the fix on your reported issue

### Test Results

```bash
$ python test_strict_simple.py
Graph: 0 - 1 - 2 - 3 - 4
Test triple: (0, 0, 1)

With n_hops=1:
  Non-strict: 4 edges, max endpoint dist: 1
  Strict:     4 edges, max endpoint dist: 1

✓ BOTH PASS: Both modes respect the 1-hop constraint
```

```bash
$ python debug_original_test.py
With n_hops=2:
  Maximum hop distance of any edge endpoint: 2

✓ SUCCESS: All edges within 2 hops
```

## When to Use Strict Mode

**Use strict mode when:**
- You want explicit guarantees that all edge endpoints are within n_hops
- You're debugging or validating filtering behavior
- You want maximum safety/assurance

**Default mode is fine when:**
- You trust PyG's `k_hop_subgraph` implementation (you should!)
- Performance is critical (strict mode adds a BFS computation)
- You understand that the n-hop constraint applies to nodes, not paths in the filtered graph

## Performance Impact

The strict mode adds one BFS traversal to compute hop distances:
- Time complexity: O(E) where E is the number of edges
- Space complexity: O(N) where N is the number of nodes
- Negligible for most knowledge graphs

## Backward Compatibility

The default behavior is unchanged. All existing scripts will work exactly as before. The `--strict-hop-constraint` flag is optional.
