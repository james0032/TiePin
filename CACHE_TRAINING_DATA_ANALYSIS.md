# Analysis: Caching Training Triples in train_graph_cache.pkl

## Date: 2025-11-14

## Current State

### What's Currently Cached
```python
cache_data = {
    'edge_index': self.edge_index,              # PyTorch tensor, undirected edges
    'edge_relations': self.edge_relations,      # PyTorch tensor, relation IDs
    'node_degrees': self.node_degrees,          # PyTorch tensor, degree per node
    'edge_to_triples': self.edge_to_triples,    # Dict mapping edges to triple indices
    'data_hash': self._compute_data_hash(),     # MD5 hash for validation
    'num_triples': len(self.training_triples)   # Count only
}
```

### What's NOT Currently Cached
- `self.training_triples` (numpy array of shape (N, 3))
- String-to-ID mappings (`entity_to_idx`, `relation_to_idx`, `idx_to_entity`)
- Original string labels from train.txt

### Current Workflow
1. Load train.txt → string triples
2. Build entity/relation mappings
3. Convert to numeric format → `train_numeric`
4. Initialize `ProximityFilterPyG(train_numeric)`
5. Check cache or build graph
6. Filter triples → numeric results
7. Convert back to strings using mappings
8. Write filtered output

## Proposal: Add Training Data to Cache

### What to Add
```python
cache_data = {
    # Existing
    'edge_index': self.edge_index,
    'edge_relations': self.edge_relations,
    'node_degrees': self.node_degrees,
    'edge_to_triples': self.edge_to_triples,
    'data_hash': self._compute_data_hash(),
    'num_triples': len(self.training_triples),

    # NEW: Add training triples
    'training_triples': self.training_triples,  # Numpy array (N, 3)

    # OPTIONAL: Add mappings (needed for string conversion)
    'entity_to_idx': entity_to_idx,
    'idx_to_entity': idx_to_entity,
    'relation_to_idx': relation_to_idx,
}
```

## Benefits

### 1. **Skip train.txt Loading** ✓
- **Current**: Must load train.txt every time (Step 1)
- **With cache**: Load from pickle if cache exists
- **Time saved**: 2-30 seconds depending on file size

### 2. **Skip Entity/Relation Mapping** ✓
- **Current**: Must rebuild mappings from scratch (Step 3)
- **With cache**: Load pre-built mappings from pickle
- **Time saved**: 1-5 seconds for large graphs

### 3. **Skip Numeric Conversion** ✓
- **Current**: Must convert strings to indices (Step 4)
- **With cache**: Already in numeric format
- **Time saved**: 1-10 seconds for millions of triples

### 4. **Faster Startup for batch_tracin_with_filtering.py** ✓
- Batch processing runs filtering multiple times
- First run builds cache, subsequent runs load instantly
- **Speedup**: 5-45 seconds per test triple after first run

### 5. **Memory Efficiency** ✓
- Pickle compression is more efficient than text
- Numeric arrays are more compact than strings
- **Example**: 1M triples as strings (~100 MB) vs numpy (~24 MB)

## Caveats

### 1. **Larger Cache Files** ⚠️

**Current cache size** (graph structure only):
```
For 1M triples, ~50K entities:
- edge_index: 2 × 2M × 8 bytes = 32 MB (undirected doubles edges)
- edge_relations: 1M × 8 bytes = 8 MB
- node_degrees: 50K × 8 bytes = 0.4 MB
- edge_to_triples: ~10 MB (dict overhead)
- Total: ~50 MB
```

**With training_triples added**:
```
- training_triples: 1M × 3 × 8 bytes = 24 MB
- Total: ~74 MB (+48% increase)
```

**With mappings added**:
```
- entity_to_idx: ~50K entries × ~100 bytes = 5 MB
- idx_to_entity: ~50K entries × ~100 bytes = 5 MB
- relation_to_idx: ~100 entries × ~50 bytes = 0.005 MB
- Total: ~84 MB (+68% increase)
```

**Impact**:
- ✓ Still reasonable for most use cases
- ⚠️ For very large graphs (10M+ triples), cache could be 500+ MB
- ✓ Faster to load 84 MB pickle than parse 100 MB text file

### 2. **Cache Invalidation Complexity** ⚠️

**Current**: Cache is invalidated if `data_hash` (computed from training_triples) doesn't match

**Problem**: We can't compute the hash without loading train.txt first!

**Chicken-and-egg issue**:
```python
# To validate cache, we need to compute current hash
current_hash = self._compute_data_hash()  # Requires self.training_triples

# But self.training_triples needs to be loaded from train.txt
# So we MUST load train.txt before we can check if cache is valid!
```

**Solution Options**:

#### Option A: File-based validation (recommended)
```python
# Save file metadata in cache
cache_data['train_file_path'] = train_path
cache_data['train_file_mtime'] = os.path.getmtime(train_path)
cache_data['train_file_size'] = os.path.getsize(train_path)

# Validate without loading train.txt
if (cache['train_file_path'] == train_path and
    cache['train_file_mtime'] == os.path.getmtime(train_path) and
    cache['train_file_size'] == os.path.getsize(train_path)):
    # Cache is valid, load from cache
else:
    # Cache invalid, load train.txt
```

**Pros**: Fast validation, no need to load train.txt
**Cons**: Doesn't detect if file content changed but mtime/size didn't

#### Option B: Separate hash file
```python
# Save hash in separate .hash file
with open(f"{train_path}.hash", 'w') as f:
    f.write(compute_file_hash(train_path))

# Check hash file first
if os.path.exists(f"{train_path}.hash"):
    with open(f"{train_path}.hash", 'r') as f:
        expected_hash = f.read()
    if expected_hash == cache['data_hash']:
        # Load from cache
```

**Pros**: Content-based validation
**Cons**: Requires additional .hash file management

#### Option C: Keep current approach (load train.txt always)
```python
# Always load train.txt to validate cache
# If cache valid, ignore loaded data and use cached version
```

**Pros**: Simplest, most reliable
**Cons**: Defeats the purpose of caching training data

### 3. **Workflow Redesign Required** ⚠️

**Current workflow** (in `run_proximity_filtering` function):
```python
# Step 1: Load train.txt
train_triples = load_from_file(train_path)

# Step 2: Load test.txt
test_triples = load_from_file(test_path)

# Step 3: Build mappings from train + test
entity_to_idx, relation_to_idx = build_mappings(train_triples, test_triples)

# Step 4: Convert to numeric
train_numeric = convert_to_numeric(train_triples, entity_to_idx, relation_to_idx)

# Step 5: Create filter object
filter_obj = ProximityFilterPyG(train_numeric, cache_path=cache_path)
```

**Problem**: Mappings are built OUTSIDE the ProximityFilterPyG class, but we need them in the cache!

**Solution**: Refactor to move mapping logic into the class or cache the mappings separately.

### 4. **Test Triples Dependency** ⚠️

**Current**: Entity/relation mappings include entities from BOTH train.txt AND test.txt

```python
for triple in list(train_triples) + list(test_triples):  # Line 684
    # Build mappings
```

**Problem**: Cached mappings would be specific to a particular test.txt file!

**Implications**:
- Cache is only valid for the same train.txt + test.txt combination
- Different test.txt files would require different caches or cache invalidation
- Cache becomes less reusable

**Solutions**:

#### Option 1: Cache train-only mappings
```python
# Cache only train.txt entities
# Add test.txt entities at runtime
```
**Pros**: Cache is reusable across different test sets
**Cons**: Need to merge mappings at runtime, more complex

#### Option 2: Cache per (train, test) pair
```python
# Cache includes test.txt hash in validation
cache_key = f"{train_hash}_{test_hash}.pkl"
```
**Pros**: Correct validation
**Cons**: Less reusable, more cache files

#### Option 3: Don't cache mappings
```python
# Only cache training_triples (numeric)
# Rebuild mappings at runtime
```
**Pros**: Simpler, more reusable
**Cons**: Less speedup

### 5. **Pickle Security** ⚠️

**Warning**: Pickle files can execute arbitrary code during deserialization

**Risk**: If cache file is tampered with, could execute malicious code

**Mitigation**:
- Store cache in trusted location only
- Add integrity checks (HMAC)
- Consider using safer formats (HDF5, npz)

### 6. **Backward Compatibility** ⚠️

**Impact**: Old cache files without `training_triples` key will fail to load

**Solution**: Graceful fallback
```python
if 'training_triples' in cache_data:
    self.training_triples = cache_data['training_triples']
else:
    # Old cache format, rebuild from scratch
    return False
```

## Recommended Approach

### Option A: Full Caching (Maximum Speedup)

**Cache everything**:
```python
cache_data = {
    # Graph structure
    'edge_index': self.edge_index,
    'edge_relations': self.edge_relations,
    'node_degrees': self.node_degrees,
    'edge_to_triples': self.edge_to_triples,

    # Training data
    'training_triples': self.training_triples,

    # Validation
    'train_file_path': train_path,
    'train_file_mtime': os.path.getmtime(train_path),
    'train_file_size': os.path.getsize(train_path),
    'data_hash': self._compute_data_hash(),  # Keep for integrity
}
```

**Validation**:
```python
# Quick check: file metadata
if cache matches file metadata:
    load from cache
else:
    invalidate cache, rebuild
```

**Pros**: Maximum speedup, simple validation
**Cons**: Larger cache files, doesn't handle test.txt dependency

### Option B: Hybrid Caching (Balanced)

**Cache training data only, rebuild mappings**:
```python
cache_data = {
    # Graph + training
    'edge_index': self.edge_index,
    'training_triples': self.training_triples,
    # ... other graph data

    # Validation
    'train_file_mtime': os.path.getmtime(train_path),
}
```

**Workflow**:
1. Check cache validity (file mtime)
2. If valid: Load training_triples from cache
3. Load test.txt (always fresh)
4. Rebuild mappings from cached train + fresh test
5. Use cached graph structure

**Pros**: Reusable across different test.txt, good speedup
**Cons**: Still need to rebuild mappings

### Option C: Current Approach (Keep as-is)

**Keep current caching behavior**:
- Cache only graph structure
- Always load train.txt
- Always rebuild mappings

**Pros**: Simple, reliable, no breaking changes
**Cons**: No speedup for file loading

## Estimated Performance Impact

### Typical RoboKOP Dataset
- train.txt: 1M triples, ~100 MB
- test.txt: 100 triples, ~10 KB

### Time Breakdown (Current)
```
1. Load train.txt:          5-10 seconds
2. Load test.txt:           <1 second
3. Build mappings:          2-5 seconds
4. Convert to numeric:      3-8 seconds
5. Build graph (no cache):  10-30 seconds
   Build graph (cached):    1-2 seconds
--------------------------------------------
Total (no cache):           20-54 seconds
Total (cache):              11-26 seconds  (graph cached)
```

### Time Breakdown (With Full Caching)
```
1-4. Load from cache:       2-5 seconds  (load entire pickle)
5.   Load test.txt:         <1 second
6.   Update mappings:       <1 second
--------------------------------------------
Total (cache):              3-7 seconds  (75% faster!)
```

### Speedup Analysis
- **First run** (no cache): 20-54 seconds (same)
- **Subsequent runs** (cache hit): 3-7 seconds vs 11-26 seconds
- **Speedup**: **2-4x faster** with full caching
- **Batch processing**: Huge benefit - saves 8-19 seconds per test triple

## Recommendation

### For batch_tracin_with_filtering.py: **Option A (Full Caching)** ✓

**Rationale**:
- Batch processing runs filtering many times with same train.txt
- 8-19 second speedup per test triple is significant
- For 100 test triples: saves 13-32 minutes total!
- Cache file size (84 MB) is reasonable

**Implementation**:
1. Add `training_triples` to cache
2. Use file mtime/size for quick validation
3. Keep data_hash for integrity check
4. Add backward compatibility fallback

### For one-off filtering: **Option C (Keep Current)** ✓

**Rationale**:
- Single run doesn't benefit much from caching training data
- Simpler is better for one-time use
- Current approach is already optimized

## Next Steps

1. **Evaluate use case**: Is this for batch processing or one-off filtering?
2. **If batch processing**: Implement Option A (full caching)
3. **If one-off**: Keep current approach
4. **Testing**: Verify cache invalidation works correctly
5. **Documentation**: Document cache format and limitations
