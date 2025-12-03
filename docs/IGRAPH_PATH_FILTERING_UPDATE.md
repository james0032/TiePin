# igraph Path-Based Filtering Update

## Summary

Updated [filter_training_igraph.py](filter_training_igraph.py) to use **path enumeration** instead of distance-based heuristics, matching the NetworkX implementation.

## What Changed

### 1. New Method: `find_all_paths_between_nodes()`
- **Location**: Lines 111-190
- **Purpose**: Enumerate all simple paths between drug and disease nodes
- **Implementation**: Uses igraph's `get_all_simple_paths()` method
- **API**: `graph.get_all_simple_paths(source, to=target, cutoff=max_length)`

**Key differences from NetworkX**:
- igraph API: `graph.get_all_simple_paths(source, to=target, cutoff=...)`
- NetworkX API: `nx.all_simple_paths(graph, source, target, cutoff=...)`

**Returns**:
- List of all paths (each path is a list of node IDs)
- Path statistics dictionary:
  - `total_paths`: Total number of paths found
  - `drug_disease_pairs`: Total drug-disease pairs tested
  - `pairs_with_paths`: How many pairs have at least one path
  - `path_length_distribution`: Count of paths at each length (e.g., `{2: 5, 3: 20}`)
  - `shortest_path_length`: Minimum path length
  - `longest_path_length`: Maximum path length
  - `avg_path_length`: Average path length

### 2. New Method: `extract_edges_from_paths()`
- **Location**: Lines 192-216
- **Purpose**: Extract all unique edges from enumerated paths
- **Implementation**:
  - Iterates through paths
  - Extracts consecutive node pairs as edges
  - Normalizes edges as `(min(src, dst), max(src, dst))` for undirected graph
- **Returns**: Set of tuples `(src, dst)` where `src < dst`

### 3. Updated: `filter_for_test_triples()`
- **Location**: Lines 268-282
- **Changes**:
  - **Added**: Path enumeration when `path_filtering=True`
  - **Added**: Edge extraction from paths
  - **Updated**: Edge filtering logic to use `edges_on_paths` set

**Old approach** (lines 253-258, removed):
```python
if should_keep and path_filtering:
    if not self.is_edge_on_drug_disease_path(
        src, dst, drug_distances, disease_distances,
        n_hops, max_total_path_length
    ):
        should_keep = False
```

**New approach** (lines 268-282):
```python
edges_on_paths = None
path_stats = None
if path_filtering:
    if max_total_path_length is not None:
        max_path_len = max_total_path_length
    else:
        max_path_len = n_hops * 2

    logger.info(f"Finding all paths between drugs and diseases (cutoff={max_path_len})...")
    all_paths, path_stats = self.find_all_paths_between_nodes(
        drug_nodes, disease_nodes, max_path_len
    )
    edges_on_paths = self.extract_edges_from_paths(all_paths)
```

### 4. Updated: Edge Filtering Logic
- **Location**: Lines 318-323
- **Old logic**:
  ```python
  if should_keep and path_filtering:
      if not self.is_edge_on_drug_disease_path(...):
          should_keep = False
  ```
- **New logic**:
  ```python
  if should_keep and path_filtering:
      if edges_on_paths is not None:
          edge_tuple = (min(src, dst), max(src, dst))
          if edge_tuple not in edges_on_paths:
              should_keep = False
  ```

### 5. Removed: `is_edge_on_drug_disease_path()` Method
- **Status**: Completely removed
- **Reason**: Replaced by direct path enumeration

## Benefits of Path-Based Filtering

### 1. **Consistency with NetworkX**
- Both implementations now use the same approach
- Results should be identical (assuming no bugs)
- Easy to cross-validate

### 2. **Transparency**
- Can see exactly which paths exist
- Can inspect path counts and length distribution
- No heuristics or approximations

### 3. **Accuracy**
- Edges kept if and only if they appear on actual paths
- Old approach used distance heuristic
- New approach guarantees correctness

### 4. **Rich Statistics**
The `path_stats` dictionary provides:
```python
{
    'total_paths': 42,
    'drug_disease_pairs': 1,
    'pairs_with_paths': 1,
    'path_length_distribution': {2: 5, 3: 20, 4: 17},
    'shortest_path_length': 2,
    'longest_path_length': 4,
    'avg_path_length': 3.2
}
```

## igraph vs NetworkX API Differences

| Feature | igraph | NetworkX |
|---------|--------|----------|
| **Path enumeration** | `graph.get_all_simple_paths(src, to=dst, cutoff=n)` | `nx.all_simple_paths(graph, src, dst, cutoff=n)` |
| **Return type** | List of vertex ID lists | Generator of node lists |
| **Graph reference** | Method on graph object | Function taking graph as arg |
| **Performance** | Faster (C implementation) | Slower (pure Python) |
| **Installation** | `pip install igraph` | `pip install networkx` |

## Usage

### Basic Path Filtering
```bash
python filter_training_igraph.py \
    --train train.txt \
    --test test_triple.txt \
    --output filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 4
```

### Without Path Filtering (Intersection Only)
```bash
python filter_training_igraph.py \
    --train train.txt \
    --test test_triple.txt \
    --output filtered.txt \
    --n-hops 2
```

## Expected Output

When path filtering is enabled, you'll see:

```
Finding all simple paths between 1 drugs and 1 diseases (cutoff=4)...
Found 42 paths connecting 1/1 drug-disease pairs
Path length distribution:
  2-hop paths: 5
  3-hop paths: 20
  4-hop paths: 17
Extracted 87 unique edges from 42 paths
```

## Performance Comparison

| Implementation | Path Enumeration | Typical Speed | Language |
|---------------|------------------|---------------|----------|
| **PyG** | No (distance-based) | Fastest | Python + C++ |
| **NetworkX** | Yes (nx.all_simple_paths) | Slow | Pure Python |
| **igraph** | Yes (get_all_simple_paths) | Fast | Python + C |

**Recommendation**:
- Use **igraph** for path-based filtering when performance matters
- Use **NetworkX** for debugging and transparency
- Use **PyG** for maximum speed (but less transparent)

## Validation

To verify the implementation:

```bash
# Syntax check
python -m py_compile filter_training_igraph.py

# Run on test data (requires igraph installation)
pip install igraph

python filter_training_igraph.py \
    --train data/train.txt \
    --test data/test_triple.txt \
    --output /tmp/igraph_output.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 3
```

## Cross-Validation

Compare igraph results with NetworkX:

```bash
# Run NetworkX filter
python filter_training_networkx.py \
    --train train.txt \
    --test test.txt \
    --output networkx_filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 3

# Run igraph filter
python filter_training_igraph.py \
    --train train.txt \
    --test test.txt \
    --output igraph_filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 3

# Compare results
diff networkx_filtered.txt igraph_filtered.txt
```

**Expected**: No differences (or minimal ordering differences)

## Files Modified

- [filter_training_igraph.py](filter_training_igraph.py)
  - Added `find_all_paths_between_nodes()` method
  - Added `extract_edges_from_paths()` method
  - Updated `filter_for_test_triples()` to use path enumeration
  - Removed `is_edge_on_drug_disease_path()` method

## Backward Compatibility

✅ **Fully backward compatible**
- `--path-filtering` flag is optional (defaults to False)
- Without `--path-filtering`, behavior is identical to before
- All existing scripts continue to work

## Testing Checklist

- [x] Syntax check: Passed
- [ ] Unit test: Run on small synthetic graph
- [ ] Integration test: Compare with NetworkX results
- [ ] Performance test: Time on real dataset
- [ ] Validation: Verify path counts and edge counts match NetworkX

## See Also

- [PATH_BASED_FILTERING_UPDATE.md](PATH_BASED_FILTERING_UPDATE.md) - NetworkX implementation
- [filter_training_networkx.py](filter_training_networkx.py) - NetworkX reference implementation
- [compare_all_implementations.py](compare_all_implementations.py) - Cross-validation tool
