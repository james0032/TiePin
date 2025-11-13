# Snakefile Update Summary

## Date: 2025-11-13

## Changes Made

### 1. Backup Files Created
- `Snakefile.backup` - Backup of original Snakefile using train_pytorch.py
- `config.yaml.backup` - Backup of original configuration

### 2. Updated train_model Rule (Step 5)

**Changed from:** `train_pytorch.py` (pure PyTorch implementation)
**Changed to:** `train.py` (PyKEEN implementation with custom checkpoint callback)

#### Key Differences:

**Removed arguments:**
- `--checkpoint-dir` - PyKEEN uses callback-based checkpointing
- `--num-workers` - Not applicable to PyKEEN pipeline

**Added arguments:**
- `--patience` - For early stopping
- `--no-early-stopping` - Flag to disable early stopping

**Output changes:**
- **Before (train_pytorch.py):**
  - `best_model.pt` - Best model checkpoint
  - `final_model.pt` - Final epoch model
  - `training_history.json` - Training metrics per epoch
  - `config.json` - Training configuration
  - `test_results.json` - Test evaluation results

- **After (train.py - PyKEEN):**
  - `models/conve/` directory structure created by PyKEEN
  - `config.json` - Training configuration
  - `test_results.json` - Test evaluation results (integrated)
  - Checkpoint files in `checkpoints/` subdirectory via callback

### 3. Disabled Rules

#### Step 6: evaluate_model (DISABLED)
- **Reason:** PyKEEN's train.py performs evaluation automatically
- **Output:** test_results.json is now created during training
- **Alternative:** Use train_pytorch.py if you need score_only.py functionality

#### Step 7: tracin_analysis (DISABLED)
- **Reason:** TracIn requires checkpoint format from train_pytorch.py
- **Alternative:** Use train_pytorch.py if you need TracIn analysis

### 4. Updated rule all Outputs

**Removed:**
- `best_model.pt`
- `final_model.pt`
- `training_history.json` (not produced by PyKEEN)
- `results/evaluation/test_scores.json` (evaluation now integrated)
- `results/evaluation/test_scores_ranked.json`
- TracIn analysis outputs

**Kept:**
- `config.json` - Training configuration
- `test_results.json` - Test evaluation (now from train.py)

## Advantages of PyKEEN Implementation

1. **Integrated Framework:** Uses PyKEEN's robust training pipeline
2. **Built-in Features:** Early stopping, evaluation, model management
3. **Custom Callbacks:** Checkpoint saving with epoch numbers
4. **Better Abstractions:** Less manual tensor operations
5. **Reproducibility:** Standardized PyKEEN format

## Disadvantages of PyKEEN Implementation

1. **Less Control:** Cannot easily modify training loop internals
2. **No TracIn:** TracIn analysis requires pure PyTorch checkpoints
3. **Different Outputs:** Model saved in PyKEEN directory structure
4. **Callback Limitations:** Custom checkpointing may not have all features

## How to Switch Back to Pure PyTorch

If you need TracIn analysis or more control:

1. **Restore from backup:**
   ```bash
   cp Snakefile.backup Snakefile
   cp config.yaml.backup config.yaml
   ```

2. **Or manually edit Snakefile:**
   - Change line 290: `python train.py` → `python train_pytorch.py`
   - Add back `--checkpoint-dir` and `--num-workers` arguments
   - Remove `--patience` and `--no-early-stopping` arguments
   - Update outputs to include `best_model.pt`, `final_model.pt`, `training_history.json`
   - Re-enable evaluate_model and tracin_analysis rules

## Configuration Compatibility

Both implementations use the same config.yaml parameters:
- ✅ All model architecture parameters (embedding_dim, etc.)
- ✅ Training hyperparameters (num_epochs, batch_size, etc.)
- ✅ Dropout and regularization settings
- ✅ checkpoint_frequency (used by callback in train.py)
- ⚠️ patience - used differently (PyKEEN early stopping vs custom)

## Testing

To verify the changes work:

```bash
# Dry run to check dependencies
snakemake -n

# Run specific rule
snakemake -n train_model

# Run full pipeline (dry run)
snakemake -n --cores all
```

## Notes

- The PyKEEN callback implementation was fixed to address a critical bug where `logger` was undefined
- Checkpoints are now saved to `{output_dir}/checkpoints/conve_checkpoint_epoch_N.pt`
- Warning added if training_loop not found in callback kwargs
