# Checkpoint Compatibility: train_pytorch.py ↔ TracIn Analysis

**Date**: 2025-11-20
**Status**: ✅ Fully Compatible

## Overview

The checkpoint format saved by `train_pytorch.py` is now fully compatible with TracIn analysis tools (`run_tracin.py`, `batch_tracin_with_filtering.py`).

## Checkpoint Format

### Saved by train_pytorch.py

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,  # Model weights
    'optimizer_state_dict': OrderedDict,  # Optimizer state
    'model_config': {
        'num_entities': int,
        'num_relations': int,
        'embedding_dim': int,
        'embedding_height': int,
        'embedding_width': int,
        'output_channels': int,  # ✅ Now included
    },
    'metrics': {  # Optional
        'train_loss': float,
        'valid_loss': float,
        'valid_mrr': float,
        'valid_hits@1': float,
        'valid_hits@3': float,
        'valid_hits@10': float,
    }
}
```

### File Location

Checkpoints are saved in:
```
{output_dir}/checkpoints/checkpoint_epoch_{epoch:07d}.pt
```

Example:
```
models/conve/checkpoints/checkpoint_epoch_0000016.pt
```

## Loading Compatibility

### run_tracin.py

`run_tracin.py` has been updated to support both checkpoint formats:

1. **train_pytorch.py format** (current):
   ```python
   config = checkpoint.get('model_config', {})  # ✅ Primary format
   ```

2. **Legacy format** (backward compatibility):
   ```python
   config = checkpoint.get('config', {})  # ✅ Fallback
   ```

**Code (lines 170-171)**:
```python
# Check both 'model_config' (train_pytorch.py format) and 'config' (legacy format)
config = checkpoint.get('model_config', checkpoint.get('config', {}))
```

### Model Reconstruction

The TracIn tools reconstruct the ConvE model using the saved configuration:

```python
model = ConvE(
    triples_factory=train_triples,
    embedding_dim=config['embedding_dim'],
    output_channels=config['output_channels'],
    embedding_height=config['embedding_height'],
    embedding_width=config['embedding_width']
)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Changes Made

### 1. train_pytorch.py (Lines 65, 425)

**Added `output_channels` storage**:
```python
# Line 65: Store output_channels as class attribute
self.output_channels = output_channels  # Store for checkpoint saving

# Line 425: Include in checkpoint config
'output_channels': model.output_channels,
```

### 2. run_tracin.py (Line 171)

**Updated config loading**:
```python
# Check both 'model_config' (train_pytorch.py format) and 'config' (legacy format)
config = checkpoint.get('model_config', checkpoint.get('config', {}))
```

## Verification

To verify checkpoint compatibility:

```bash
# Train a model with train_pytorch.py
python train_pytorch.py \
    --train train.txt \
    --valid valid.txt \
    --test test.txt \
    --entity-to-id node_dict.txt \
    --relation-to-id rel_dict.txt \
    --output-dir models/conve \
    --num-epochs 20 \
    --use-mixed-precision

# Run TracIn analysis using the checkpoint
python run_tracin.py \
    --model-path models/conve/checkpoints/checkpoint_epoch_0000016.pt \
    --test test_subset.txt \
    --train train.txt \
    --entity-to-id node_dict.txt \
    --relation-to-id rel_dict.txt \
    --use-optimized-tracin \
    --use-mixed-precision
```

## Expected Behavior

### Successful Loading

```
INFO - Loading model from models/conve/checkpoints/checkpoint_epoch_0000016.pt...
INFO - Extracted model_state_dict with 12 keys
INFO - Loaded config: embedding_dim=200, output_channels=32
INFO - Creating model with: embedding_dim=200, output_channels=32
INFO - Model loaded successfully
```

### Fallback to Inference

If `model_config` is missing (legacy checkpoints), the system will infer parameters from the state_dict:

```
INFO - No config in checkpoint, will infer parameters...
INFO - Inferring model parameters from state_dict...
INFO -   Inferred embedding_dim=200 from entity embeddings
INFO -   Inferred output_channels=32 from conv layer
```

## Files Involved

1. ✅ `train_pytorch.py` - Saves checkpoints with `model_config`
2. ✅ `run_tracin.py` - Loads checkpoints for TracIn analysis
3. ✅ `batch_tracin_with_filtering.py` - Calls `run_tracin.py` with checkpoint path
4. ✅ `tracin.py` - Receives model object (no checkpoint loading)
5. ✅ `tracin_optimized.py` - Receives model object (no checkpoint loading)

## Integration with Snakemake

Both `Snakefile` and `Snakefile_from_step3` have been updated to use `train_pytorch.py`:

```python
rule train_model:
    """
    Train ConvE knowledge graph embedding model using train_pytorch.py
    This generates checkpoint files compatible with TracIn analysis
    """
    shell:
        """
        python train_pytorch.py \
            --train {input.train} \
            ...
            --device {params.device} \
            {params.use_mixed_precision}
        """
```

Checkpoints are automatically saved in:
```
{BASE_DIR}/models/conve/checkpoints/
```

## Conclusion

✅ **Checkpoint format is fully compatible**
✅ **Backward compatibility maintained** (legacy format supported)
✅ **All model parameters saved and loaded correctly**
✅ **TracIn analysis works with train_pytorch.py checkpoints**
✅ **Snakemake pipelines updated to use train_pytorch.py**

The system is production-ready for TracIn analysis using checkpoints from `train_pytorch.py`.
