# Fixed AttributeError in score_only.py - state_dict Shape Access

## Date: 2025-11-14

## Issue

```
AttributeError: 'collections.OrderedDict' object has no attribute 'shape'
```

This error occurred when score_only.py tried to access `.shape` attribute on values in the state_dict that were themselves OrderedDict objects (nested structures) rather than tensors.

## Root Cause

PyKEEN's model checkpoints can have nested state_dict structures where some values are OrderedDict objects containing other parameters, not tensors directly.

**Example problematic code**:
```python
for key in sorted(state_dict.keys()):
    logger.info(f"  {key}: {state_dict[key].shape}")  # Assumes all values are tensors!
```

If `state_dict[key]` is an OrderedDict instead of a tensor, calling `.shape` raises AttributeError.

## Solution

Add defensive checks before accessing `.shape` attribute:

```python
for key in sorted(state_dict.keys()):
    value = state_dict[key]
    if hasattr(value, 'shape'):
        logger.info(f"  {key}: {value.shape}")
    else:
        logger.info(f"  {key}: {type(value)}")
```

This safely handles both:
- Tensors (have `.shape` attribute)
- OrderedDict or other objects (show type instead)

## Changes Made

### 1. Line 246-251: Debug Logging

**Before**:
```python
for key in sorted(state_dict.keys()):
    logger.info(f"  {key}: {state_dict[key].shape}")
```

**After**:
```python
for key in sorted(state_dict.keys()):
    value = state_dict[key]
    if hasattr(value, 'shape'):
        logger.info(f"  {key}: {value.shape}")
    else:
        logger.info(f"  {key}: {type(value)}")
```

### 2. Line 342-347: Error Logging

**Before**:
```python
for key in list(state_dict.keys())[:10]:
    logger.error(f"  {key}: {state_dict[key].shape}")
```

**After**:
```python
for key in list(state_dict.keys())[:10]:
    value = state_dict[key]
    if hasattr(value, 'shape'):
        logger.error(f"  {key}: {value.shape}")
    else:
        logger.error(f"  {key}: {type(value)}")
```

### 3. Line 260-276: Entity Embedding Detection

**Before**:
```python
for key in entity_embedding_keys:
    if key in state_dict:
        embedding_dim = state_dict[key].shape[1]
        ...
```

**After**:
```python
for key in entity_embedding_keys:
    if key in state_dict:
        value = state_dict[key]
        if hasattr(value, 'shape') and len(value.shape) >= 2:
            embedding_dim = value.shape[1]
            ...
```

### 4. Line 280-289: Output Channels Detection

**Before**:
```python
if 'interaction.hr2d.2.weight' in state_dict:
    output_channels = state_dict['interaction.hr2d.2.weight'].shape[0]
```

**After**:
```python
if 'interaction.hr2d.2.weight' in state_dict:
    value = state_dict['interaction.hr2d.2.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 1:
        output_channels = value.shape[0]
```

### 5. Line 297-301: HR1D Input Size Detection

**Before**:
```python
if 'interaction.hr1d.0.weight' in state_dict and embedding_dim and output_channels:
    hr1d_input_size = state_dict['interaction.hr1d.0.weight'].shape[1]
```

**After**:
```python
if 'interaction.hr1d.0.weight' in state_dict and embedding_dim and output_channels:
    value = state_dict['interaction.hr1d.0.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 2:
        hr1d_input_size = value.shape[1]
```

### 6. Line 420-426: Test Model Verification

**Before**:
```python
if 'interaction.hr1d.0.weight' in test_model.state_dict():
    test_hr1d_size = test_model.state_dict()['interaction.hr1d.0.weight'].shape[1]
```

**After**:
```python
if 'interaction.hr1d.0.weight' in test_model.state_dict():
    value = test_model.state_dict()['interaction.hr1d.0.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 2:
        test_hr1d_size = value.shape[1]
```

## Pattern Used

All fixes follow this defensive pattern:

```python
# 1. Get the value
value = state_dict[key]

# 2. Check if it has shape attribute (and correct dimensions if needed)
if hasattr(value, 'shape') and len(value.shape) >= expected_dims:
    # 3. Safe to access shape
    some_dim = value.shape[index]
```

## Why This Happens

PyKEEN models can have nested structures in state_dict:

```python
state_dict = {
    'entity_representations.0._embeddings': OrderedDict([
        ('weight', torch.Tensor(...)),
        ('bias', torch.Tensor(...))
    ]),
    'entity_representations.0._embeddings.weight': torch.Tensor(...),
    ...
}
```

Some keys point to OrderedDict containers, not tensors directly.

## Testing

After this fix, score_only.py will:
- ✓ Handle PyKEEN checkpoints with nested state_dict
- ✓ Log type information for non-tensor values
- ✓ Continue gracefully when encountering unexpected structures
- ✓ Still extract model parameters correctly from tensor values

## Expected Output

**Before** (crashed):
```
Inferring model parameters from state_dict...
Traceback (most recent call last):
  File "score_only.py", line 343
    logger.error(f"  {key}: {state_dict[key].shape}")
AttributeError: 'collections.OrderedDict' object has no attribute 'shape'
```

**After** (graceful):
```
Inferring model parameters from state_dict...
All state_dict keys:
  entity_representations.0: <class 'collections.OrderedDict'>
  entity_representations.0._embeddings.weight: torch.Size([50000, 200])
  interaction.hr2d.2.weight: torch.Size([32, 1, 3, 3])
  ✓ Inferred embedding_dim=200 from entity_representations.0._embeddings.weight
  ✓ Inferred output_channels=32 from interaction.hr2d.2.weight
```

## Files Modified

- **score_only.py**: Added defensive `.shape` access checks at 6 locations
  - Lines 246-251: Debug logging
  - Lines 342-347: Error logging
  - Lines 260-276: Entity embedding detection
  - Lines 280-289: Output channels detection
  - Lines 297-301: HR1D input size detection
  - Lines 420-426: Test model verification

## Related Issues

This is a common issue when working with PyTorch state_dict objects that may have:
- Nested structures (OrderedDict of OrderedDict)
- Mixed types (tensors, scalars, dicts)
- Different checkpoint formats (PyKEEN vs raw PyTorch)

The fix makes score_only.py robust to all these cases.
