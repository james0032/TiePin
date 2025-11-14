# Fixed score_only.py to Handle PyTorch Regular Checkpoint Format

## Date: 2025-11-14

## Issue

score_only.py failed to load PyTorch regular checkpoint files (*.pt) because it couldn't find model parameters in the nested checkpoint structure.

**Error log**:
```
Checkpoint top-level keys: ['state_dict', 'version']
Checkpoint is a plain state_dict
Total keys in state_dict: 2
All state_dict keys:
  state_dict: <class 'collections.OrderedDict'>
  version: <class 'int'>
Could not determine model architecture parameters
embedding_dim=None, output_channels=None
```

## Root Cause

PyTorch regular checkpoints use a nested structure:

```python
checkpoint = {
    'state_dict': OrderedDict([
        ('entity_embeddings.weight', tensor(...)),
        ('conv.weight', tensor(...)),
        ...
    ]),
    'version': 1,
    'epoch': 10,
    'optimizer_state_dict': {...},
    ...
}
```

The code was treating the outer dict as the state_dict, so it only saw keys like `['state_dict', 'version']` instead of the actual model parameters inside `checkpoint['state_dict']`.

## Solution

Add logic to detect and extract the inner `state_dict` from nested checkpoint format.

**Before** (lines 229-235):
```python
elif isinstance(checkpoint, dict):
    # Just a state_dict - infer parameters from tensor shapes
    state_dict = checkpoint
    logger.info("Checkpoint is a plain state_dict")
else:
    logger.error(f"Unknown checkpoint format")
    return
```

**After** (lines 229-243):
```python
elif isinstance(checkpoint, dict):
    # Check if this is a nested structure with state_dict inside
    if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
        # PyTorch regular checkpoint format: {'state_dict': OrderedDict(...), 'version': int, ...}
        state_dict = checkpoint['state_dict']
        logger.info("Checkpoint has nested 'state_dict' key (PyTorch format)")
        logger.info(f"  Outer keys: {list(checkpoint.keys())}")
        logger.info(f"  Inner state_dict has {len(state_dict)} parameter tensors")
    else:
        # Just a state_dict - infer parameters from tensor shapes
        state_dict = checkpoint
        logger.info("Checkpoint is a plain state_dict")
else:
    logger.error(f"Unknown checkpoint format")
    return
```

## Checkpoint Format Detection Logic

The code now handles three checkpoint formats:

### 1. PyKEEN Format (with 'config' key)
```python
checkpoint = {
    'config': {'embedding_dim': 200, 'output_channels': 32, ...},
    'model_state_dict': OrderedDict(...),
    'optimizer_state_dict': {...},
    ...
}
```
**Detection**: `'config' in checkpoint`
**Action**: Load parameters from config directly

### 2. PyTorch Regular Format (nested 'state_dict')
```python
checkpoint = {
    'state_dict': OrderedDict([
        ('entity_embeddings.weight', tensor(...)),
        ('conv.weight', tensor(...)),
        ...
    ]),
    'version': 1,
    'epoch': 10,
    ...
}
```
**Detection**: `'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict)`
**Action**: Extract `checkpoint['state_dict']` and infer parameters from tensor shapes

### 3. Plain state_dict (just parameters)
```python
checkpoint = OrderedDict([
    ('entity_embeddings.weight', tensor(...)),
    ('conv.weight', tensor(...)),
    ...
])
```
**Detection**: No 'config' or 'state_dict' keys, just parameter tensors
**Action**: Use checkpoint directly as state_dict and infer parameters

## Expected Output

### Before (Failed)
```
Checkpoint top-level keys: ['state_dict', 'version']
Checkpoint is a plain state_dict
Total keys in state_dict: 2
All state_dict keys:
  state_dict: <class 'collections.OrderedDict'>
  version: <class 'int'>
ERROR - Could not determine model architecture parameters
```

### After (Success)
```
Checkpoint top-level keys: ['state_dict', 'version']
Checkpoint has nested 'state_dict' key (PyTorch format)
  Outer keys: ['state_dict', 'version']
  Inner state_dict has 12 parameter tensors
================================================================================
Inferring model parameters from state_dict...
================================================================================
Total keys in state_dict: 12
All state_dict keys:
  entity_embeddings.weight: torch.Size([50000, 200])
  relation_embeddings.weight: torch.Size([100, 200])
  conv1.weight: torch.Size([32, 1, 3, 3])
  ...
  ✓ Inferred embedding_dim=200 from entity_embeddings.weight
  ✓ Inferred output_channels=32 from conv1.weight
```

## Common PyTorch Checkpoint Patterns

### Training Checkpoint (Most Common)
```python
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),  # or 'state_dict'
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'version': 1
}, 'checkpoint.pt')
```

### Model Only
```python
torch.save(model.state_dict(), 'model.pt')
```

### With Custom Metadata
```python
torch.save({
    'state_dict': model.state_dict(),
    'config': {'embedding_dim': 200, ...},
    'training_args': {...},
    'version': 2
}, 'model_with_config.pt')
```

## Testing

Test with different checkpoint formats:

### Test 1: PyTorch Regular Format
```bash
python score_only.py \
    --model-path checkpoint.pt \
    --test test.txt \
    --node-dict node_dict.txt \
    --rel-dict rel_dict.txt
```

Expected: Successfully extracts state_dict and infers parameters

### Test 2: Plain state_dict
```bash
python score_only.py \
    --model-path model_state_dict.pt \
    --test test.txt \
    --node-dict node_dict.txt \
    --rel-dict rel_dict.txt
```

Expected: Uses checkpoint directly as state_dict

### Test 3: PyKEEN Format
```bash
python score_only.py \
    --model-path pykeen_model.pkl \
    --test test.txt \
    --node-dict node_dict.txt \
    --rel-dict rel_dict.txt
```

Expected: Loads config from checkpoint

## Related Changes

This fix works together with the earlier AttributeError fix:
- [SCORE_ONLY_SHAPE_FIX.md](SCORE_ONLY_SHAPE_FIX.md) - Added defensive `.shape` checks

Together, these fixes make score_only.py robust to:
1. ✓ Nested checkpoint structures
2. ✓ Non-tensor values in state_dict
3. ✓ Different checkpoint formats (PyKEEN, PyTorch, plain)

## Files Modified

- **score_only.py**: Lines 229-243
  - Added nested state_dict detection
  - Extract inner state_dict from PyTorch format
  - Log outer and inner structure information
