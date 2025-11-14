# Added Skip Evaluation Option

## Date: 2025-11-13

## Change Summary

Added `--skip-evaluation` flag to train.py to allow users to completely bypass the evaluation step after training.

## Why This Change?

The user asked: "in train.py, is there any setting to not run evaluation at all?"

While train.py already has error handling to prevent Snakemake from removing the model if evaluation fails, there was no way to completely skip evaluation. This is useful when:

1. **Training large models** - Evaluation can be time-consuming
2. **Running experiments** - User may want to evaluate later with different settings
3. **Debugging** - Focus only on training without evaluation overhead
4. **Batch processing** - Train multiple models first, evaluate all of them later with Step 6

## Changes Made

### 1. train.py - Added skip_evaluation Parameter

**Function signature** (lines 116-123):
```python
def train_model(
    # ... other parameters ...
    # Other options
    use_gpu: bool = True,
    random_seed: int = 42,
    early_stopping: bool = True,
    patience: int = 10,
    track_gradients: bool = False,
    skip_evaluation: bool = False  # ← New parameter
):
```

**Docstring** (line 153):
```python
    skip_evaluation: Whether to skip evaluation on test set after training
```

**Evaluation logic** (lines 291-328):
```python
# Evaluate on test set with detailed metrics
if skip_evaluation:
    logger.info("Skipping evaluation (skip_evaluation=True)")
    logger.info("You can run evaluation separately using score_only.py")
else:
    # Wrap in try-except to prevent Snakemake from removing model if evaluation fails
    try:
        logger.info("Evaluating on test set...")
        test_results_path = os.path.join(output_dir, 'test_results.json')
        detailed_results = evaluate_model(...)
        # Print final results...
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error("Model training completed successfully, but evaluation encountered an error.")
        logger.error(f"Model has been saved to: {output_dir}")
        logger.error("You can run evaluation separately using score_only.py")
        # Don't re-raise - allow train.py to exit successfully
```

**Command-line argument** (lines 394-395):
```python
parser.add_argument('--skip-evaluation', action='store_true',
                   help='Skip evaluation on test set after training')
```

**Main function** (line 430):
```python
train_model(
    # ... other arguments ...
    skip_evaluation=args.skip_evaluation
)
```

### 2. config.yaml - Added skip_evaluation Setting

**Lines 109-111**:
```yaml
# Skip evaluation on test set after training (default: false)
# Set to true to skip evaluation and only save the trained model
skip_evaluation: false
```

### 3. Snakefile - Pass skip_evaluation to train.py

**Params section** (line 281):
```python
skip_evaluation = "--skip-evaluation" if config.get("skip_evaluation", False) else ""
```

**Shell command** (line 313):
```bash
python train.py \
    --train {input.train} \
    --valid {input.valid} \
    --test {input.test} \
    # ... other arguments ...
    {params.checkpoint_dir_arg} \
    {params.no_early_stopping} \
    {params.skip_evaluation} \  # ← New flag
    {params.gpu} \
    2>&1 | tee {log}
```

**Updated outputs** (lines 255-260):
```python
output:
    # PyKEEN outputs - model is saved in directory structure
    model_dir = directory(f"{BASE_DIR}/models/conve"),
    config_out = f"{BASE_DIR}/models/conve/config.json"
    # Note: test_results.json is created only if skip_evaluation=false
    # Use evaluate_model rule for separate evaluation
```

**Why remove test_results.json from outputs?**
- When `skip_evaluation=true`, the file won't be created
- Snakemake would fail if an expected output file doesn't exist
- The `evaluate_model` rule (Step 6) handles evaluation separately anyway

## How to Use

### Option 1: Command-line Flag

```bash
python train.py \
    --train train.txt \
    --valid valid.txt \
    --test test.txt \
    --entity-to-id node_dict.txt \
    --relation-to-id rel_dict.txt \
    --output-dir models/conve \
    --skip-evaluation  # ← Add this flag
```

### Option 2: config.yaml Setting

```yaml
# Skip evaluation on test set after training
skip_evaluation: true
```

Then run:
```bash
snakemake --cores 1 train_model
```

### Option 3: Evaluate Later with Step 6

After training with `skip_evaluation=true`, run evaluation separately:

```bash
snakemake --cores 1 evaluate_model
```

This uses score_only.py to evaluate the trained model.

## Expected Output

### With skip_evaluation=false (Default)

```
Training completed!
Saved model to /workspace/data/robokop/CGGD_alltreat/models/conve

Evaluating on test set...
============================================================
Final Test Results:
============================================================
Mean Rank: 1234.56
Mean Reciprocal Rank: 0.1234
Hits@1: 0.0500
Hits@3: 0.1234
Hits@5: 0.2000
Hits@10: 0.3456
============================================================
```

### With skip_evaluation=true

```
Training completed!
Saved model to /workspace/data/robokop/CGGD_alltreat/models/conve

Skipping evaluation (skip_evaluation=True)
You can run evaluation separately using score_only.py
```

## Benefits

1. **Faster training workflow** - No need to wait for evaluation
2. **Flexibility** - Evaluate later with different settings or on different test sets
3. **Batch processing** - Train multiple models without evaluation overhead
4. **Debug-friendly** - Focus on training issues without evaluation noise
5. **Consistent with existing patterns** - Follows same pattern as `--no-early-stopping`, `--no-gpu`, etc.

## Files Modified

1. **train.py**:
   - Added `skip_evaluation` parameter to `train_model()` function
   - Added docstring for new parameter
   - Wrapped evaluation in `if not skip_evaluation:` block
   - Added `--skip-evaluation` command-line argument
   - Passed `args.skip_evaluation` to `train_model()`

2. **config.yaml**:
   - Added `skip_evaluation: false` setting under "Evaluation Configuration"

3. **Snakefile**:
   - Added `skip_evaluation` param that converts config setting to `--skip-evaluation` flag
   - Added `{params.skip_evaluation}` to shell command
   - Removed `test_results.json` from outputs (conditional file)
   - Added comment explaining test_results.json is only created when skip_evaluation=false

## Related Features

- **Error handling** - Even when `skip_evaluation=false`, errors in evaluation don't cause model deletion (see previous fix)
- **Step 6 (evaluate_model)** - Separate evaluation step using score_only.py for post-training evaluation
- **Early stopping** - Uses `--no-early-stopping` flag pattern
- **GPU usage** - Uses `--no-gpu` flag pattern

## Testing

### Test 1: Skip Evaluation via Config

1. Set in config.yaml:
   ```yaml
   skip_evaluation: true
   ```

2. Run training:
   ```bash
   snakemake --cores 1 train_model
   ```

3. Expected: Model saved, no evaluation, logs show "Skipping evaluation"

4. Verify no test_results.json:
   ```bash
   ls models/conve/test_results.json
   # Should not exist
   ```

### Test 2: Skip Evaluation via Command-line

1. Run directly:
   ```bash
   python train.py \
       --train train.txt \
       --valid valid.txt \
       --test test.txt \
       --entity-to-id node_dict.txt \
       --relation-to-id rel_dict.txt \
       --output-dir models/conve \
       --num-epochs 2 \
       --skip-evaluation
   ```

2. Expected: Model saved, no evaluation

### Test 3: Evaluate Later with Step 6

1. Train with skip_evaluation=true
2. Run evaluation separately:
   ```bash
   snakemake --cores 1 evaluate_model
   ```

3. Expected: score_only.py evaluates the model and creates results

### Test 4: Default Behavior (No Skip)

1. Set in config.yaml:
   ```yaml
   skip_evaluation: false
   ```

2. Run training:
   ```bash
   snakemake --cores 1 train_model
   ```

3. Expected: Model saved, evaluation runs, test_results.json created
