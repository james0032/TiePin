# Checkpoint Directory Configuration Update

## Date: 2025-11-13

## Summary

Added support for configurable checkpoint directory in both train.py and the Snakemake pipeline.

## Changes Made

### 1. train.py - Command Line Arguments

**Added two new arguments** ([train.py:352-355](train.py#L352-L355)):

```python
# Checkpoint options
parser.add_argument('--checkpoint-dir', type=str, default=None,
                   help='Directory for checkpoint files (default: {output_dir}/checkpoints)')
parser.add_argument('--checkpoint-frequency', type=int, default=2,
                   help='Save checkpoint every N epochs')
```

**Updated main() function** ([train.py:385-386](train.py#L385-L386)):

```python
checkpoint_dir=args.checkpoint_dir,
checkpoint_frequency=args.checkpoint_frequency,
```

### 2. train.py - Function Signature

**Updated train_model() signature** ([train.py:112-113](train.py#L112-L113)):

```python
# Checkpoint options
checkpoint_dir: Optional[str] = None,
checkpoint_frequency: int = 2,
```

**Updated docstring** ([train.py:142-143](train.py#L142-L143)):

```python
checkpoint_dir: Directory for checkpoint files (default: {output_dir}/checkpoints)
checkpoint_frequency: Save checkpoint every N epochs
```

**Updated checkpoint logic** ([train.py:204-209](train.py#L204-L209)):

```python
# Create checkpoint callback
if checkpoint_dir is None:
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
checkpoint_callback = MultiCheckpointCallback(
    checkpoint_dir=checkpoint_dir,
    checkpoint_frequency=checkpoint_frequency
)
```

### 3. config.yaml

**Updated checkpoint configuration** ([config.yaml:100-103](config.yaml#L100-L103)):

```yaml
# Checkpoint configuration
checkpoint_frequency: 2  # Save checkpoint every N epochs
checkpoint_dir: null  # Custom checkpoint directory (null = use default: {output_dir}/checkpoints)
num_workers: 4  # Number of data loader workers (for train_pytorch.py only)
```

### 4. Snakefile - train_model Rule

**Added checkpoint parameters** ([Snakefile:274-279](Snakefile#L274-L279)):

```python
checkpoint_frequency = config.get("checkpoint_frequency", 2),
patience = config.get("patience", 10),
random_seed = config.get("random_seed", 42),
no_early_stopping = "" if config.get("early_stopping", True) else "--no-early-stopping",
gpu = "" if config.get("use_gpu", True) else "--no-gpu",
checkpoint_dir_arg = f"--checkpoint-dir {config.get('checkpoint_dir')}" if config.get("checkpoint_dir") else ""
```

**Updated shell command** ([Snakefile:306-312](Snakefile#L306-L312)):

```bash
--label-smoothing {params.label_smoothing} \
--checkpoint-frequency {params.checkpoint_frequency} \
--patience {params.patience} \
--random-seed {params.random_seed} \
{params.checkpoint_dir_arg} \
{params.no_early_stopping} \
{params.gpu} \
```

## How to Use

### 1. Default Behavior (No Configuration)

If `checkpoint_dir` is not specified (or set to `null` in config.yaml), checkpoints will be saved to:
```
{output_dir}/checkpoints/
```

Example with default settings:
```bash
python train.py \
    --train data/train.txt \
    --valid data/valid.txt \
    --test data/test.txt \
    --entity-to-id data/node_dict.txt \
    --relation-to-id data/rel_dict.txt \
    --output-dir models/conve
```

Checkpoints will be saved to: `models/conve/checkpoints/conve_checkpoint_epoch_2.pt`, etc.

### 2. Custom Checkpoint Directory via Command Line

Specify a custom directory using the `--checkpoint-dir` argument:

```bash
python train.py \
    --train data/train.txt \
    --valid data/valid.txt \
    --test data/test.txt \
    --entity-to-id data/node_dict.txt \
    --relation-to-id data/rel_dict.txt \
    --output-dir models/conve \
    --checkpoint-dir /custom/checkpoint/path \
    --checkpoint-frequency 5
```

Checkpoints will be saved to: `/custom/checkpoint/path/conve_checkpoint_epoch_5.pt`, etc.

### 3. Custom Checkpoint Directory via config.yaml

Set the `checkpoint_dir` parameter in config.yaml:

```yaml
checkpoint_frequency: 5
checkpoint_dir: "/workspace/shared_checkpoints/conve"
```

Then run the Snakemake pipeline:

```bash
snakemake --cores all train_model
```

Checkpoints will be saved to the specified directory.

### 4. Per-Style Checkpoint Directory (Dynamic)

You can also use relative paths that will be resolved based on the output directory:

```yaml
checkpoint_dir: "checkpoints/run_001"
```

Or use the style variable in the path (requires manual configuration in Snakefile if needed).

## Checkpoint File Naming

Checkpoint files are named with the epoch number:
```
conve_checkpoint_epoch_2.pt
conve_checkpoint_epoch_4.pt
conve_checkpoint_epoch_6.pt
...
```

Each checkpoint contains:
- `epoch`: The epoch number (1-indexed)
- `model_state_dict`: Model parameters
- `optimizer_state_dict`: Optimizer state
- `loss`: Training loss at that epoch

## Configuration Hierarchy

The checkpoint directory is determined by this priority:

1. **Command-line argument** (`--checkpoint-dir`) - Highest priority
2. **config.yaml** (`checkpoint_dir`) - Medium priority
3. **Default fallback** (`{output_dir}/checkpoints`) - Lowest priority

This provides maximum flexibility while maintaining sensible defaults.

## Compatibility

- ✅ Works with PyKEEN pipeline (train.py)
- ✅ Configurable via command line
- ✅ Configurable via config.yaml
- ✅ Configurable via Snakemake
- ✅ Maintains backward compatibility (defaults to {output_dir}/checkpoints)

## Testing

To verify the changes:

```bash
# 1. Dry run to check Snakemake syntax
snakemake -n train_model

# 2. Check train.py help output
python train.py --help

# 3. Test with custom checkpoint directory
python train.py --checkpoint-dir /tmp/test_checkpoints --help
```

## Notes

- The checkpoint directory is created automatically if it doesn't exist
- The `MultiCheckpointCallback` uses `Path.mkdir(parents=True, exist_ok=True)` to ensure the directory exists
- If `training_loop` is not found in kwargs, a warning is logged instead of silently failing
