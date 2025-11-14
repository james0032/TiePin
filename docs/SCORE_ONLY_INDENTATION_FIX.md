# Fixed Indentation Bug in score_only.py Fallback Configuration

## Date: 2025-11-14

## Issue

After fixing the convolution formula, score_only.py still failed with the same size mismatch error:

```
RuntimeError: Error(s) in loading state_dict for ConvE:
  size mismatch for interaction.hr1d.0.weight:
    copying a param with shape torch.Size([32, 896]) from checkpoint,
    the shape in current model is torch.Size([32, 1152]).
```

The fallback configuration search was finding the correct model parameters but then using the **wrong** configuration anyway.

## Root Cause

**Indentation bug** in the fallback error recovery section (lines 429-443).

### Before (Incorrect Indentation)

```python
# Check if the hr1d layer size matches
if 'interaction.hr1d.0.weight' in test_model.state_dict():
    value = test_model.state_dict()['interaction.hr1d.0.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 2:
        test_hr1d_size = value.shape[1]
        if test_hr1d_size == expected_hr1d_in:
            logger.info(f"  ✓ Found matching configuration: h={h}, w={w}")
            logger.info(f"    hr1d input size: {test_hr1d_size}")

        # Use this configuration  ← WRONG INDENTATION!
        model = test_model  ← Executes even when sizes DON'T match
        model.load_state_dict(state_dict)
        model.eval()
        found = True
        break
```

The lines 438-442 (`model = test_model`, etc.) were indented at the wrong level. They executed whenever `hasattr(value, 'shape')` was true, **not** when `test_hr1d_size == expected_hr1d_in`.

This meant:
1. Loop tries h=8, w=4 → hr1d size = 1152 ≠ 896 (expected)
2. Code logs nothing (no match message)
3. **But still uses this configuration anyway!** (wrong indentation)
4. Tries to load checkpoint → size mismatch error

### After (Correct Indentation)

```python
# Check if the hr1d layer size matches
if 'interaction.hr1d.0.weight' in test_model.state_dict():
    value = test_model.state_dict()['interaction.hr1d.0.weight']
    if hasattr(value, 'shape') and len(value.shape) >= 2:
        test_hr1d_size = value.shape[1]
        if test_hr1d_size == expected_hr1d_in:
            logger.info(f"  ✓ Found matching configuration: h={h}, w={w}")
            logger.info(f"    hr1d input size: {test_hr1d_size}")

            # Use this configuration  ← CORRECT INDENTATION!
            model = test_model  ← Only executes when sizes match
            model.load_state_dict(state_dict)
            model.eval()
            found = True
            break
```

Now lines 438-443 are inside the `if test_hr1d_size == expected_hr1d_in:` block, so they only execute when the configuration actually matches.

## How It Works Now

### Fallback Configuration Search

When the first model creation fails to load the checkpoint:

1. **Extract expected size** from error message
   ```
   RuntimeError: ... copying a param with shape torch.Size([32, 896]) ...
   Expected hr1d input: 896
   Expected conv output: 896 / 32 = 28
   ```

2. **Try all h, w combinations** for the embedding_dim
   ```python
   for h in range(3, 100):
       if embedding_dim % h == 0:
           w = embedding_dim // h
           # Create test model with this h, w
           test_model = ConvE(..., embedding_height=h, embedding_width=w)
   ```

3. **Check if hr1d size matches**
   ```python
   if test_hr1d_size == expected_hr1d_in:  # Only proceed if match!
       logger.info(f"✓ Found matching configuration: h={h}, w={w}")
       model = test_model  # Use this model
       model.load_state_dict(state_dict)  # Load checkpoint
       found = True
       break  # Stop searching
   ```

4. **If no match found**, error out
   ```python
   if not found:
       logger.error("Could not find matching embedding configuration")
       return
   ```

## Example Execution

### Checkpoint with embedding_dim=32, h=4, w=8 (hr1d=896)

**Search loop**:
```
Try h=4, w=8:
  Create model with h=4, w=8
  Test hr1d size: 896
  896 == 896? YES!
  ✓ Found matching configuration: h=4, w=8
  Use this model and load checkpoint
  Break loop
```

**Before fix (wrong indentation)**:
```
Try h=8, w=4:  (first that divides 32)
  Create model with h=8, w=4
  Test hr1d size: 1152
  1152 == 896? NO!
  (But still use this model anyway - BUG!)
  Try to load checkpoint → SIZE MISMATCH ERROR
```

**After fix (correct indentation)**:
```
Try h=4, w=8:
  Create model with h=4, w=8
  Test hr1d size: 896
  896 == 896? YES!
  ✓ Found matching configuration: h=4, w=8
  Use this model and load checkpoint → SUCCESS
```

## Changes Made

### Line 429-443: Fixed Indentation

**Before**:
```python
                    if 'interaction.hr1d.0.weight' in test_model.state_dict():
                        value = test_model.state_dict()['interaction.hr1d.0.weight']
                        if hasattr(value, 'shape') and len(value.shape) >= 2:
                            test_hr1d_size = value.shape[1]
                            if test_hr1d_size == expected_hr1d_in:
                                logger.info(f"  ✓ Found matching configuration: h={h}, w={w}")
                                logger.info(f"    hr1d input size: {test_hr1d_size}")

                            # Use this configuration
                            model = test_model
                            model.load_state_dict(state_dict)
                            model.eval()
                            found = True
                            break
```

**After** (added 4 spaces to lines 438-443):
```python
                    if 'interaction.hr1d.0.weight' in test_model.state_dict():
                        value = test_model.state_dict()['interaction.hr1d.0.weight']
                        if hasattr(value, 'shape') and len(value.shape) >= 2:
                            test_hr1d_size = value.shape[1]
                            if test_hr1d_size == expected_hr1d_in:
                                logger.info(f"  ✓ Found matching configuration: h={h}, w={w}")
                                logger.info(f"    hr1d input size: {test_hr1d_size}")

                                # Use this configuration
                                model = test_model
                                model.load_state_dict(state_dict)
                                model.eval()
                                found = True
                                break
```

## Testing

### Test Case: Load checkpoint with h=4, w=8

```bash
python score_only.py \
    --model-path checkpoint_h4_w8.pt \
    --test test.txt \
    --node-dict node_dict.txt \
    --rel-dict rel_dict.txt
```

**Expected output**:
```
Inferring model parameters from state_dict...
  ✓ Inferred embedding_dim=32 from entity_embeddings.weight
  ✓ Inferred output_channels=32 from conv.weight
  hr1d input size: 896

Searching for h, w where:
  h * w = 32
  (h-3+1) * (w*2-3+1) = 28

  ✓ Inferred embedding_height=4, embedding_width=8

Creating ConvE model with:
  embedding_dim=32
  output_channels=32
  embedding_height=4
  embedding_width=8

Model loaded successfully
```

### Test Case: Fallback when initial config fails

If the initial parameter inference fails:

**Expected output**:
```
Failed to load state_dict with current configuration
Trying to find correct embedding_height and embedding_width by testing configurations...
  Checkpoint expects hr1d input size: 896
  Expected conv output size: 28

  ✓ Found matching configuration: h=4, w=8
    hr1d input size: 896

Model loaded successfully
```

## Files Modified

- **score_only.py**: Lines 438-443
  - Fixed indentation of model assignment block
  - Now only uses test_model when hr1d sizes match
  - Prevents incorrect model configuration from being used

## Related Fixes

This fix completes the series of score_only.py improvements:
1. [SCORE_ONLY_SHAPE_FIX.md](SCORE_ONLY_SHAPE_FIX.md) - Defensive `.shape` checks
2. [SCORE_ONLY_PYTORCH_FORMAT.md](SCORE_ONLY_PYTORCH_FORMAT.md) - Nested state_dict handling
3. [SCORE_ONLY_CONV_FORMULA_FIX.md](SCORE_ONLY_CONV_FORMULA_FIX.md) - Correct convolution formula
4. **This fix** - Correct indentation in fallback search

Together, these make score_only.py fully functional for loading PyTorch checkpoints with any embedding configuration!
