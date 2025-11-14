# Fixed run_tracin.py to Handle PyTorch Checkpoints

## Date: 2025-11-14

## Overview

Applied the same fixes from score_only.py to run_tracin.py to handle PyTorch regular checkpoint format and prevent AttributeError when accessing `.shape` on non-tensor values.

## Issues Fixed

### 1. Nested PyTorch Checkpoint Format
**Lines 168-182**: Added detection for nested `state_dict` key

### 2. AttributeError on .shape Access
**Lines 189-194**: Added defensive check for entity embeddings inference
**Lines 197-202**: Added defensive check for output channels inference
**Lines 250-260**: Added defensive check in fallback configuration search

## Changes Made

### Fix 1: PyTorch Format Detection (Lines 168-182)

**Before**:
```python
elif isinstance(checkpoint, dict):
    # Just a state_dict - infer parameters from tensor shapes
    state_dict = checkpoint
    logger.info("Checkpoint is a plain state_dict")
```

**After**:
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
```

### Fix 2: Entity Embeddings Inference (Lines 189-194)

**Before**:
```python
if 'entity_representations.0._embeddings.weight' in state_dict:
    inferred_dim = state_dict['entity_representations.0._embeddings.weight'].shape[1]
    logger.info(f"  Inferred embedding_dim={inferred_dim} from entity embeddings")
    embedding_dim = inferred_dim
```

**After**:
```python
if 'entity_representations.0._embeddings.weight' in state_dict:
    value = state_dict['entity_representations.0._embeddings.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 2:
        inferred_dim = value.shape[1]
        logger.info(f"  Inferred embedding_dim={inferred_dim} from entity embeddings")
        embedding_dim = inferred_dim
```

### Fix 3: Output Channels Inference (Lines 197-202)

**Before**:
```python
if 'interaction.hr2d.2.weight' in state_dict:
    inferred_channels = state_dict['interaction.hr2d.2.weight'].shape[0]
    logger.info(f"  Inferred output_channels={inferred_channels} from conv layer")
    output_channels = inferred_channels
```

**After**:
```python
if 'interaction.hr2d.2.weight' in state_dict:
    value = state_dict['interaction.hr2d.2.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 1:
        inferred_channels = value.shape[0]
        logger.info(f"  Inferred output_channels={inferred_channels} from conv layer")
        output_channels = inferred_channels
```

### Fix 4: Fallback Configuration Search (Lines 250-260)

**Before**:
```python
if 'interaction.hr1d.0.weight' in test_model.state_dict():
    test_size = test_model.state_dict()['interaction.hr1d.0.weight'].shape[1]
    if test_size == expected_hr1d_in:
        logger.info(f"  ✓ Found: h={h}, w={w}")
        model = test_model
        model.load_state_dict(state_dict)
        model.eval()
        found = True
        break
```

**After**:
```python
if 'interaction.hr1d.0.weight' in test_model.state_dict():
    value = test_model.state_dict()['interaction.hr1d.0.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 2:
        test_size = value.shape[1]
        if test_size == expected_hr1d_in:
            logger.info(f"  ✓ Found: h={h}, w={w}")
            model = test_model
            model.load_state_dict(state_dict)
            model.eval()
            found = True
            break
```

## Comparison with score_only.py

### Similarities
- Both needed PyTorch nested format detection
- Both needed defensive `.shape` checks
- Both have fallback configuration search

### Differences
- **run_tracin.py**: Simpler fallback - just tries h,w combinations and checks hr1d size
- **score_only.py**: Complex fallback - uses convolution formula to calculate expected size first

run_tracin.py doesn't have the convolution formula calculation that score_only.py had, so it didn't need those specific fixes.

## Checkpoint Format Support

run_tracin.py now handles three checkpoint formats:

### 1. PyKEEN Format (with 'config' key)
```python
checkpoint = {
    'config': {'embedding_dim': 200, 'output_channels': 32, ...},
    'model_state_dict': OrderedDict(...),
    ...
}
```

### 2. PyTorch Regular Format (nested 'state_dict')
```python
checkpoint = {
    'state_dict': OrderedDict([
        ('entity_representations.0._embeddings.weight', tensor(...)),
        ('interaction.hr2d.2.weight', tensor(...)),
        ...
    ]),
    'version': 1,
    'epoch': 10,
    ...
}
```

### 3. Plain state_dict
```python
checkpoint = OrderedDict([
    ('entity_representations.0._embeddings.weight', tensor(...)),
    ('interaction.hr2d.2.weight', tensor(...)),
    ...
])
```

## Testing

Test with PyTorch regular checkpoint:

```bash
python run_tracin.py \
    --model-path checkpoint.pt \
    --train train.txt \
    --test test.txt \
    --node-dict node_dict.txt \
    --rel-dict rel_dict.txt \
    --mode test \
    --output-dir tracin_results
```

**Expected output**:
```
Checkpoint has nested 'state_dict' key (PyTorch format)
  Outer keys: ['state_dict', 'version', 'epoch']
  Inner state_dict has 12 parameter tensors
Inferring model parameters from state_dict...
  Inferred embedding_dim=200 from entity embeddings
  Inferred output_channels=32 from conv layer
Model loaded successfully
```

## Files Modified

- **run_tracin.py**: Lines 168-182, 189-194, 197-202, 250-260
  - Added nested state_dict detection
  - Added defensive `.shape` checks at 3 locations
  - Prevents AttributeError on non-tensor values

## Related Fixes

This completes the series of fixes applied to both score_only.py and run_tracin.py:

### score_only.py
1. [SCORE_ONLY_SHAPE_FIX.md](SCORE_ONLY_SHAPE_FIX.md) - Defensive `.shape` checks
2. [SCORE_ONLY_PYTORCH_FORMAT.md](SCORE_ONLY_PYTORCH_FORMAT.md) - Nested state_dict handling
3. [SCORE_ONLY_CONV_FORMULA_FIX.md](SCORE_ONLY_CONV_FORMULA_FIX.md) - Correct convolution formula
4. [SCORE_ONLY_INDENTATION_FIX.md](SCORE_ONLY_INDENTATION_FIX.md) - Correct indentation in fallback search

### run_tracin.py
1. **This fix** - PyTorch format and defensive `.shape` checks

Both scripts are now fully compatible with PyTorch regular checkpoint format!
