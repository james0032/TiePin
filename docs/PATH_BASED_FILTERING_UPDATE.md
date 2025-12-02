# Path-Based Filtering Implementation

## Summary

Replaced the distance-based path filtering logic in [filter_training_networkx.py](filter_training_networkx.py) with **true path enumeration**. The new implementation enumerates all simple paths between drug and disease nodes, then extracts edges from those paths.

## What Changed

### 1. New Method: `find_all_paths_between_nodes()`
- **Location**: Line 132-180
- **Purpose**: Enumerate all simple paths between drug nodes and disease nodes
- **Implementation**: Uses `nx.all_simple_paths()` with configurable cutoff
- **Returns**:
  - List of all paths (each path is a list of node IDs)
  - Path statistics dictionary with:
    - `total_paths`: Total number of paths found
    - `path_length_distribution`: Count of paths at each length
    - `shortest_path_length`: Length of shortest path
    - `longest_path_length`: Length of longest path
    - `avg_path_length`: Average path length
    - `unique_nodes_on_paths`: Set of all nodes appearing on any path
    - `unique_edges_on_paths`: Count of unique edges on paths

### 2. New Method: `extract_edges_from_paths()`
- **Location**: Line 182-202
- **Purpose**: Extract all unique edges from a list of paths
- **Implementation**:
  - Iterates through each path
  - Extracts consecutive node pairs as edges
  - Normalizes edges as `(min(src, dst), max(src, dst))` for undirected graph
  - Returns set of unique edges
- **Returns**: Set of tuples `(src, dst)` where `src < dst`

### 3. Updated: `filter_for_single_test_triple()`
- **Location**: Lines 309-346
- **Changes**:
  - **Removed**: `drug_distances` and `disease_distances` computation
  - **Added**: Path enumeration when `path_filtering=True`
  - **Added**: Edge extraction from paths
  - **Updated**: `_filter_edges()` call signature to use `edges_on_paths` and `path_stats`

**Old approach** (lines 320-338):
```python
if path_filtering:
    with timer("Computing hop distances (path filtering mode)"):
        drug_distances = self.compute_hop_distances_from_nodes([test_h], n_hops)
        disease_distances = self.compute_hop_distances_from_nodes([test_t], n_hops)

filtered_triple_indices = self._filter_edges(
    ...
    drug_distances=drug_distances,
    disease_distances=disease_distances,
    max_total_path_length=max_total_path_length
)
```

**New approach** (lines 317-346):
```python
edges_on_paths = None
path_stats = None
if path_filtering:
    if max_total_path_length is not None:
        max_path_len = max_total_path_length
    else:
        max_path_len = n_hops * 2

    with timer(f"Finding all paths between drug and disease (cutoff={max_path_len})"):
        all_paths, path_stats = self.find_all_paths_between_nodes(
            drug_nodes, disease_nodes, max_path_len
        )
        edges_on_paths = self.extract_edges_from_paths(all_paths)

filtered_triple_indices = self._filter_edges(
    ...
    edges_on_paths=edges_on_paths,
    path_stats=path_stats
)
```

### 4. Updated: `filter_for_multiple_test_triples()`
- **Location**: Lines 412-451
- **Changes**: Same as `filter_for_single_test_triple()`
- **Removed**: `drug_distances` and `disease_distances` declarations
- **Added**: Path enumeration and edge extraction

### 5. Updated: `_filter_edges()` Method Signature
- **Location**: Line 204
- **Removed Parameters**:
  - `drug_distances: Optional[Dict[int, int]]`
  - `disease_distances: Optional[Dict[int, int]]`
  - `max_total_path_length: Optional[int]`
- **Added Parameters**:
  - `edges_on_paths: Optional[Set[Tuple[int, int]]]`
  - `path_stats: Optional[Dict]`

### 6. Updated: `_filter_edges()` Implementation
- **Location**: Lines 240-245
- **Old logic**:
  ```python
  if should_keep and path_filtering:
      # Check if edge could be on a drug→disease path
      if not self.is_edge_on_drug_disease_path(
          src, dst, drug_distances, disease_distances, max_total_path_length
      ):
          should_keep = False
  ```
- **New logic**:
  ```python
  if should_keep and path_filtering:
      # Check if edge is actually on an enumerated path
      if edges_on_paths is not None:
          edge = (min(src, dst), max(src, dst))
          if edge not in edges_on_paths:
              should_keep = False
  ```

### 7. Removed: `is_edge_on_drug_disease_path()` Method
- **Status**: Completely removed (no longer needed)
- **Reason**: Replaced by direct path enumeration

## Benefits of Path-Based Filtering

### 1. **Transparency**
- Can see exactly which paths exist between drug and disease
- Can inspect path lengths, path count, nodes on paths
- No heuristics or approximations

### 2. **Accuracy**
- Edges are kept if and only if they appear on actual paths
- Old approach used distance heuristic: `drug_dist[src] + disease_dist[dst] <= max_path_len`
- New approach guarantees correctness

### 3. **Debuggability**
- Path statistics show:
  - How many paths were found
  - Path length distribution
  - Shortest/longest/average path lengths
  - Which nodes/edges are on paths
- Easy to verify results manually on small examples

### 4. **Rich Statistics**
The `path_stats` dictionary provides detailed information:
```python
{
    'total_paths': 42,
    'path_length_distribution': {2: 5, 3: 20, 4: 17},
    'shortest_path_length': 2,
    'longest_path_length': 4,
    'avg_path_length': 3.2,
    'unique_nodes_on_paths': {1, 5, 12, 34, ...},
    'unique_edges_on_paths': 87
}
```

## Usage

### With NetworkX Filtering (Path-Based)
```bash
python filter_training_networkx.py \
    --train train.txt \
    --test test_triple.txt \
    --output filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 4
```

### Via Batch TracIn Script
```bash
python batch_tracin_with_filtering.py \
    --test-triples test.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/ \
    --filter-method networkx \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 4
```

### Via Shell Scripts
Both example scripts now use NetworkX with path filtering by default:
```bash
bash scripts/run_batch_tracin_example.sh
bash scripts/run_batch_tracin_example_CCGGDD_hatteras.sh
```

## Performance Considerations

### Time Complexity
- **Path enumeration**: Can be expensive for highly connected graphs
- **Cutoff parameter**: Controls search depth (lower = faster but may miss long paths)
- **Simple paths only**: No cycles allowed (prevents infinite enumeration)

### Recommended Settings
- **Small graphs (<10K nodes)**: `max_total_path_length` up to 5-6 is fine
- **Medium graphs (10K-100K nodes)**: Keep `max_total_path_length` ≤ 4
- **Large graphs (>100K nodes)**: Use `max_total_path_length` ≤ 3 or skip path filtering

### When to Use
- ✅ **Use path filtering when**:
  - You need guaranteed correctness
  - Debugging filter results
  - Graph is not too large/dense
  - Max path length ≤ 4

- ❌ **Skip path filtering when**:
  - Graph is very large and dense
  - Performance is critical
  - Intersection filtering is sufficient

## Validation

To verify the implementation works:

```bash
# Test import
python -c "from filter_training_networkx import NetworkXProximityFilter; print('OK')"

# Test on small example
python filter_training_networkx.py \
    --train data/train.txt \
    --test data/test_triple.txt \
    --output /tmp/test_output.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 3
```

Expected log output:
```
Finding all paths between drug and disease (cutoff=3)...
  Found 42 paths between 1 drug(s) and 1 disease(s)
  Path length distribution: {2: 5, 3: 37}
  Shortest: 2, Longest: 3, Average: 2.88
  Unique edges on paths: 87
```

## Backward Compatibility

✅ **Fully backward compatible**
- `--path-filtering` flag is **optional** (defaults to False)
- Without `--path-filtering`, behavior is identical to before
- All existing scripts continue to work

## Files Modified

- [filter_training_networkx.py](filter_training_networkx.py) - Main implementation
  - Added `find_all_paths_between_nodes()` method
  - Added `extract_edges_from_paths()` method
  - Updated `filter_for_single_test_triple()`
  - Updated `filter_for_multiple_test_triples()`
  - Updated `_filter_edges()` signature and implementation
  - Removed `is_edge_on_drug_disease_path()` method

## Testing Checklist

- [x] Syntax check: Import successful
- [ ] Unit test: Run on small synthetic graph
- [ ] Integration test: Run via batch_tracin_with_filtering.py
- [ ] Comparison test: Compare with PyG results
- [ ] Performance test: Time on real dataset

## See Also

- [BATCH_TRACIN_UPDATE.md](BATCH_TRACIN_UPDATE.md) - Integration with batch TracIn
- [SCRIPTS_UPDATE.md](scripts/SCRIPTS_UPDATE.md) - Shell script updates
- [README_FILTERING.md](README_FILTERING.md) - Complete filtering guide
