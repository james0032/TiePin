# Troubleshooting Guide

## Common Errors and Solutions

### Error: `FileNotFoundError: [Errno 2] No such file or directory`

**Full Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'results/filtered/train_filtered.txt'
```

**Cause**: Output directory doesn't exist

**Solution**: Fixed in latest version! The script now automatically creates the output directory.

**Manual Fix** (if using older version):
```bash
# Create output directory manually
mkdir -p results/filtered

# Then run script
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output results/filtered/train_filtered.txt
```

---

### Error: `ModuleNotFoundError: No module named 'torch_geometric'`

**Cause**: PyTorch Geometric not installed

**Solution**:
```bash
# Install PyTorch Geometric
pip install torch-geometric

# Or with specific CUDA version
pip install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

### Error: Cache invalidation message

**Message**:
```
Cache invalidated: training data has changed
Built new graph with 100000 training triples
```

**Cause**: Training data differs from cached version (this is expected behavior!)

**Solution**: No action needed - the graph is automatically rebuilt with the new data.

**To force cache rebuild**:
```bash
# Delete cache file
rm train_graph.pkl

# Run script
python filter_training_by_proximity_pyg.py --cache train_graph.pkl ...
```

---

### Error: `PermissionError: [Errno 13] Permission denied`

**Cause**: No write permission for output file or cache directory

**Solution**:
```bash
# Check permissions
ls -la results/filtered/

# Fix permissions
chmod 755 results/filtered/
chmod 644 results/filtered/*.txt

# Or write to a different directory
python filter_training_by_proximity_pyg.py \
    --output /tmp/train_filtered.txt \
    ...
```

---

### Error: Out of Memory

**Symptoms**:
- Script crashes with memory error
- System becomes unresponsive

**Solutions**:

1. **Use caching** (reduces memory usage):
   ```bash
   python filter_training_by_proximity_pyg.py \
       --cache train_graph.pkl \
       ...
   ```

2. **Process smaller chunks**:
   - Split training data into smaller files
   - Filter each separately

3. **Increase system swap**:
   ```bash
   # Linux
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

---

### Warning: "Failed to save cache"

**Message**:
```
WARNING: Failed to save cache: [Errno 28] No space left on device
```

**Cause**: Not enough disk space for cache file

**Solutions**:

1. **Free up disk space**:
   ```bash
   # Check disk usage
   df -h

   # Clean up old caches
   rm -f *.pkl
   ```

2. **Use different cache location**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --cache /tmp/train_graph.pkl \  # Use temp directory
       ...
   ```

3. **Disable caching**:
   ```bash
   # Omit --cache flag
   python filter_training_by_proximity_pyg.py \
       --train train.txt \
       --test test.txt \
       --output filtered.txt
   ```

---

### Issue: Filtering is too slow

**Solutions**:

1. **Enable caching**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --cache train_graph.pkl \
       ...
   ```

2. **Reduce n-hops**:
   ```bash
   # Use 1-hop instead of 2-hop
   python filter_training_by_proximity_pyg.py \
       --n-hops 1 \
       ...
   ```

3. **Increase min-degree** (more aggressive filtering):
   ```bash
   # Use higher threshold
   python filter_training_by_proximity_pyg.py \
       --min-degree 3 \
       ...
   ```

4. **Use single-triple mode** for testing:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --single-triple \  # Process only first test triple
       ...
   ```

---

### Issue: Too few triples after filtering

**Symptoms**:
- Output file has very few triples
- Most training data removed

**Solutions**:

1. **Increase n-hops**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --n-hops 3 \  # Larger neighborhood
       ...
   ```

2. **Decrease min-degree**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --min-degree 1 \  # Keep more edges
       ...
   ```

3. **Ensure preserve-test-edges is enabled**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --preserve-test-edges \  # Keep direct influences
       ...
   ```

---

### Issue: Too many triples after filtering

**Symptoms**:
- Output file nearly as large as input
- Not enough reduction

**Solutions**:

1. **Decrease n-hops**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --n-hops 1 \  # Smaller neighborhood
       ...
   ```

2. **Increase min-degree**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --min-degree 3 \  # More aggressive
       ...
   ```

3. **Disable preserve-test-edges**:
   ```bash
   python filter_training_by_proximity_pyg.py \
       --no-preserve-test-edges \  # Strict filtering
       ...
   ```

---

### Error: `IndexError: list index out of range`

**Cause**: Empty test file or malformed input

**Solution**:
```bash
# Check test file is not empty
wc -l test.txt

# Check format (should be tab-separated)
head test.txt

# Ensure proper format:
# entity1\trelation\tentity2
```

---

### Error: `ValueError: invalid literal for int()`

**Cause**: Non-numeric entity/relation IDs

**Solution**: This script works with string IDs! No changes needed. If you see this error, check input file format:

```bash
# Correct format (tab-separated)
CURIE:123\trelates_to\tCURIE:456

# Incorrect format
CURIE:123 relates_to CURIE:456  # ← No tabs!
```

---

## Verification Steps

### 1. Check Input Files

```bash
# Training file
head -5 train.txt
wc -l train.txt

# Test file
head -5 test.txt
wc -l test.txt

# Format should be:
# head_entity<TAB>relation<TAB>tail_entity
```

### 2. Check Output

```bash
# Output file created
ls -lh results/filtered/train_filtered.txt

# Output is not empty
wc -l results/filtered/train_filtered.txt

# Output format correct
head -5 results/filtered/train_filtered.txt
```

### 3. Check Cache

```bash
# Cache file exists
ls -lh train_graph.pkl

# Cache size reasonable (should be ~3-4x training data size)
du -h train_graph.pkl train.txt
```

---

## Performance Tuning

### For Maximum Speed

```bash
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --cache train_graph.pkl \      # Enable caching
    --n-hops 1 \                   # Minimal neighborhood
    --min-degree 3 \               # Aggressive filtering
    --single-triple                # Process one triple (for testing)
```

### For Maximum Recall (Keep More)

```bash
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 3 \                   # Larger neighborhood
    --min-degree 1 \               # Keep more edges
    --preserve-test-edges          # Keep direct influences
```

### For Balanced Performance

```bash
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --cache train_graph.pkl \      # Enable caching
    --n-hops 2 \                   # Standard (default)
    --min-degree 2 \               # Standard (default)
    --preserve-test-edges          # Keep direct influences (default)
```

---

## Getting Help

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Version

```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
python -c "import torch; print(torch.__version__)"
```

### Minimal Test Case

```bash
# Create minimal test files
echo -e "A\trel1\tB\nB\trel2\tC\nC\trel3\tD" > test_train.txt
echo -e "B\trel2\tC" > test_test.txt

# Run filter
python filter_training_by_proximity_pyg.py \
    --train test_train.txt \
    --test test_test.txt \
    --output test_output.txt \
    --n-hops 1

# Check output
cat test_output.txt
```

---

## Summary

| Issue | Quick Fix |
|-------|-----------|
| Directory not found | **Fixed!** Now auto-creates directories |
| PyG not installed | `pip install torch-geometric` |
| Cache invalidation | Normal behavior - rebuilds automatically |
| Out of memory | Enable `--cache`, reduce `--n-hops` |
| Too slow | Use `--cache train_graph.pkl` |
| Too few triples | Increase `--n-hops`, decrease `--min-degree` |
| Too many triples | Decrease `--n-hops`, increase `--min-degree` |
| Empty output | Check input format (tab-separated) |

**Most common issue**: ~~Directory not found~~ → **Fixed in latest version!**
