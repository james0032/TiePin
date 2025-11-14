# Fixed Checkpoint Saving - Switched to PyKEEN's CheckpointTrainingCallback

## Date: 2025-11-13

## Issue

The custom `MultiCheckpointCallback` class was not saving checkpoints during training because PyKEEN's training loop doesn't pass `training_loop` in the kwargs to the `post_epoch()` method.

**Symptom**: No checkpoint files were being created in the checkpoint directory during training.

**Root Cause**: The custom callback tried to access `kwargs.get('training_loop')` which was always `None`, causing the checkpoint save code to be skipped.

## Solution

Replaced the custom `MultiCheckpointCallback` with PyKEEN's built-in `CheckpointTrainingCallback`, which properly integrates with PyKEEN's training loop.

## Changes Made

### 1. Updated Imports (train.py:19-26)

**Before**:
```python
import torch
from pykeen.datasets import Dataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop, TrainingCallback
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
```

**After**:
```python
import torch
from pykeen.datasets import Dataset
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.training import SLCWATrainingLoop
from pykeen.training.callbacks import CheckpointTrainingCallback
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
```

**Changes**:
- ✅ Removed `TrainingCallback` import (no longer needed)
- ✅ Added `CheckpointTrainingCallback` import from `pykeen.training.callbacks`

### 2. Removed Custom MultiCheckpointCallback Class (train.py:38-72)

**Deleted entire class**:
```python
class MultiCheckpointCallback(TrainingCallback):
    """Custom callback to save checkpoints every N epochs with unique names."""

    def __init__(self, checkpoint_dir, checkpoint_frequency=2):
        # ... (73 lines removed)
```

**Reason**: No longer needed - using PyKEEN's built-in implementation instead.

### 3. Updated Checkpoint Callback Initialization (train.py:206-220)

**Before**:
```python
# Create checkpoint callback
if checkpoint_dir is None:
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
checkpoint_callback = MultiCheckpointCallback(
    checkpoint_dir=checkpoint_dir,
    checkpoint_frequency=checkpoint_frequency
)
```

**After**:
```python
# Create checkpoint callback using PyKEEN's built-in CheckpointTrainingCallback
if checkpoint_dir is None:
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')

# Create checkpoint directory if it doesn't exist
os.makedirs(checkpoint_dir, exist_ok=True)

logger.info(f"Checkpoint configuration: saving every {checkpoint_frequency} epochs to {checkpoint_dir}")

checkpoint_callback = CheckpointTrainingCallback(
    root=checkpoint_dir,
    schedule="every",
    schedule_kwargs=dict(frequency=checkpoint_frequency),
    name_template="checkpoint_epoch_{epoch:07d}.pt"
)
```

**Key differences**:
- ✅ Uses PyKEEN's `CheckpointTrainingCallback` instead of custom class
- ✅ Explicitly creates checkpoint directory with `os.makedirs()`
- ✅ Uses `schedule="every"` with `frequency` parameter for "every N epochs" behavior
- ✅ Uses `name_template` to customize checkpoint filenames

## How PyKEEN's CheckpointTrainingCallback Works

### Constructor Parameters

```python
CheckpointTrainingCallback(
    schedule: HintOrType[CheckpointSchedule] = None,
    schedule_kwargs: OptionalKwargs = None,
    keeper: HintOrType[CheckpointKeeper] = None,
    keeper_kwargs: OptionalKwargs = None,
    root: pathlib.Path | str | None = None,
    name_template: str = "checkpoint_{epoch:07d}.pt"
)
```

### Schedule Options

**"every" schedule**: Checkpoint at regular epoch intervals
```python
schedule="every",
schedule_kwargs=dict(frequency=N)  # Save every N epochs
```

**"explicit" schedule**: Checkpoint at specific epochs
```python
schedule="explicit",
schedule_kwargs=dict(epochs=[10, 20, 50, 100])
```

**"union" schedule**: Combine multiple schedules
```python
schedule="union",
schedule_kwargs=dict(
    schedules=[schedule1, schedule2]
)
```

### Keeper Options

By default (`keeper=None`), all checkpoints are kept. You can configure retention policies:

```python
# Keep only last N checkpoints
keeper="last",
keeper_kwargs=dict(n=5)

# Keep best N checkpoints based on validation metric
keeper="best",
keeper_kwargs=dict(n=3)
```

## Configuration

### config.yaml Parameters

```yaml
# Checkpoint configuration
checkpoint_frequency: 2  # Save checkpoint every N epochs
checkpoint_dir: "/workspace/data/robokop/CGGD_alltreat/models/conve/checkpoints"
```

### Command-line Arguments

```bash
python train.py \
    --checkpoint-frequency 2 \
    --checkpoint-dir /path/to/checkpoints \
    ...
```

## Checkpoint File Format

### Filename Pattern

With `name_template="checkpoint_epoch_{epoch:07d}.pt"`:
- Epoch 2: `checkpoint_epoch_0000002.pt`
- Epoch 4: `checkpoint_epoch_0000004.pt`
- Epoch 20: `checkpoint_epoch_0000020.pt`

### File Contents

PyKEEN's checkpoints contain:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'loss': float,
    # ... other PyKEEN-specific metadata
}
```

## Checkpoint Directory Structure

```
/workspace/data/robokop/CGGD_alltreat/models/conve/
├── checkpoints/
│   ├── checkpoint_epoch_0000002.pt
│   ├── checkpoint_epoch_0000004.pt
│   ├── checkpoint_epoch_0000006.pt
│   └── ...
├── trained_model.pkl
├── config.json
└── test_results.json
```

## Benefits of PyKEEN's CheckpointTrainingCallback

1. **Properly integrated**: Works correctly with PyKEEN's training loop
2. **Well-tested**: Part of PyKEEN's core functionality
3. **Flexible scheduling**: Supports multiple checkpoint strategies
4. **Retention policies**: Can automatically clean up old checkpoints
5. **Consistent format**: Uses PyKEEN's standard checkpoint format

## Testing

To verify checkpoints are being saved:

```bash
# Run training with checkpointing enabled
python train.py \
    --train data/train.txt \
    --valid data/valid.txt \
    --test data/test.txt \
    --entity-to-id data/node_dict.txt \
    --relation-to-id data/rel_dict.txt \
    --output-dir models/conve \
    --checkpoint-frequency 2 \
    --checkpoint-dir models/conve/checkpoints \
    --num-epochs 10

# Check for checkpoint files
ls -lh models/conve/checkpoints/
# Should see: checkpoint_epoch_0000002.pt, checkpoint_epoch_0000004.pt, etc.
```

## Loading Checkpoints

To resume training from a checkpoint:

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_0000010.pt')

# Access checkpoint data
epoch = checkpoint['epoch']
model_state = checkpoint['model_state_dict']
optimizer_state = checkpoint['optimizer_state_dict']
loss = checkpoint['loss']

# Load into model
model.load_state_dict(model_state)
optimizer.load_state_dict(optimizer_state)
```

## Summary

**Problem**: Custom `MultiCheckpointCallback` didn't work because it relied on `training_loop` being in kwargs, which PyKEEN doesn't provide.

**Solution**: Use PyKEEN's built-in `CheckpointTrainingCallback` with `schedule="every"` and `frequency=N` for "every N epochs" behavior.

**Result**: Checkpoints are now properly saved during training with filenames like `checkpoint_epoch_0000002.pt`.

## Related Documentation

- [PyKEEN Checkpoint Tutorial](https://pykeen.readthedocs.io/en/stable/tutorial/checkpoints.html)
- [PyKEEN Callbacks API](https://pykeen.readthedocs.io/en/stable/_modules/pykeen/training/callbacks.html)
- [PyKEEN Checkpoint Schedules](https://pykeen.readthedocs.io/en/stable/reference/checkpoints.html)
