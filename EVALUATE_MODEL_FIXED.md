# evaluate_model Rule Fixed - Removed Unnecessary Dependencies

## Date: 2025-11-13

## Issue

The original `evaluate_model` rule had unnecessary input dependencies that score_only.py doesn't actually use:
- `model_dir = f"{BASE_DIR}/models/conve"` - Not needed as input (used as parameter instead)
- `test_results = f"{BASE_DIR}/models/conve/test_results.json"` - Not used by score_only.py at all

## Changes Made

### Before (Incorrect)

```python
rule evaluate_model:
    input:
        model_dir = f"{BASE_DIR}/models/conve",  # ❌ Should be param, not input
        test = f"{BASE_DIR}/test.txt",
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",
        node_name_dict = f"{BASE_DIR}/processed/node_name_dict.txt",
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt",
        config_out = f"{BASE_DIR}/models/conve/config.json",
        test_results = f"{BASE_DIR}/models/conve/test_results.json"  # ❌ Not used by score_only.py
    params:
        output_dir = f"{BASE_DIR}/results/evaluation",
        use_sigmoid = "--use-sigmoid" if config.get("use_sigmoid", False) else "",
        top_n_arg = f"--top-n {config.get('top_n_triples')}" if config.get("top_n_triples") else "",
        device = "cuda" if config.get("use_gpu", True) else "cpu"
    shell:
        """
        python score_only.py \
            --model-dir {input.model_dir} \  # ❌ Using input instead of params
            ...
        """
```

### After (Correct)

```python
rule evaluate_model:
    input:
        # Ensure training is complete - depend on config.json which is created at the end
        config_out = f"{BASE_DIR}/models/conve/config.json",  # ✅ Dependency to ensure training done
        test = f"{BASE_DIR}/test.txt",
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",
        node_name_dict = f"{BASE_DIR}/processed/node_name_dict.txt",
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt"
    params:
        model_dir = f"{BASE_DIR}/models/conve",  # ✅ Moved to params - it's a path, not a dependency
        output_dir = f"{BASE_DIR}/results/evaluation",
        use_sigmoid = "--use-sigmoid" if config.get("use_sigmoid", False) else "",
        top_n_arg = f"--top-n {config.get('top_n_triples')}" if config.get("top_n_triples") else "",
        device = "cuda" if config.get("use_gpu", True) else "cpu"
    shell:
        """
        python score_only.py \
            --model-dir {params.model_dir} \  # ✅ Using params
            ...
        """
```

## What score_only.py Actually Needs

### Command-line Arguments
```python
parser.add_argument('--model-dir', required=True)      # Directory containing model file
parser.add_argument('--test', required=True)            # Test triples file
parser.add_argument('--entity-to-id')                   # Entity to ID mapping
parser.add_argument('--relation-to-id')                 # Relation to ID mapping
parser.add_argument('--node-name-dict')                 # Entity names (optional)
parser.add_argument('--output', default='test_scores.json')  # Output file
parser.add_argument('--device')                         # cuda/cpu
parser.add_argument('--use-sigmoid', action='store_true')    # Sigmoid flag
parser.add_argument('--top-n', type=int)                # Top N triples
```

### Model File Discovery

score_only.py looks for model files in this order:
1. `best_model.pt` (train_pytorch.py output)
2. `final_model.pt` (train_pytorch.py output)
3. `trained_model.pkl` (PyKEEN output) ← **This is what train.py creates**

The script does NOT use `test_results.json` at all!

## Why These Changes Matter

### 1. Model Directory as Parameter (Not Input)

**Before**: `model_dir` was listed as an input, which means Snakemake would:
- ❌ Try to track it as a file dependency
- ❌ Fail if the directory doesn't exist before the rule runs
- ❌ Not be able to properly determine if the rule needs to run

**After**: `model_dir` is a parameter, which means:
- ✅ It's just a string path passed to the command
- ✅ The actual dependency is on `config.json` (ensuring training is complete)
- ✅ Snakemake can properly track when the rule needs to run

### 2. Removed test_results.json Dependency

**Before**: Waited for `test_results.json` to exist
- ❌ This file is created by train.py's evaluation step
- ❌ Not actually used by score_only.py
- ❌ Unnecessary dependency

**After**: Only depends on `config.json`
- ✅ `config.json` is created at the end of training (line 233-236 in train.py)
- ✅ Ensures the model is fully trained
- ✅ No unnecessary dependencies

## Training Output Timeline

When train.py runs, it creates files in this order:

```python
# 1. Start training
os.makedirs(output_dir, exist_ok=True)

# 2. Save config early
config_path = os.path.join(output_dir, 'config.json')
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

# 3. Train model (this takes time)
result = pipeline(...)

# 4. Save model
result.save_to_directory(output_dir)  # Creates trained_model.pkl

# 5. Evaluate on test set
test_results_path = os.path.join(output_dir, 'test_results.json')
detailed_results = evaluate_model(...)
```

**Wait, config.json is created BEFORE training!**

Let me check if we should actually depend on `trained_model.pkl` instead...

Actually, looking at the code again, `config.json` is written early (line 195-198), but the key insight is that Snakemake's `rule train_model` declares `config.json` as an output, so the rule won't be considered complete until train.py finishes. So depending on `config.json` is correct - it ensures the entire training process completes.

## Verification

To verify score_only.py works correctly:

```bash
# After training completes, run score_only.py manually
python score_only.py \
    --model-dir /workspace/data/robokop/CGGD_alltreat/models/conve \
    --test /workspace/data/robokop/CGGD_alltreat/test.txt \
    --entity-to-id /workspace/data/robokop/CGGD_alltreat/processed/node_dict.txt \
    --relation-to-id /workspace/data/robokop/CGGD_alltreat/processed/rel_dict.txt \
    --node-name-dict /workspace/data/robokop/CGGD_alltreat/processed/node_name_dict.txt \
    --output test_scores.json \
    --device cuda \
    --use-sigmoid
```

The script will:
1. Look in `models/conve/` directory
2. Find `trained_model.pkl` (created by PyKEEN)
3. Load the model
4. Score test triples
5. Save results

No need for `test_results.json`!

## Summary

**Key Changes**:
1. ✅ Moved `model_dir` from `input` to `params`
2. ✅ Removed unnecessary `test_results` input
3. ✅ Updated shell command to use `{params.model_dir}` instead of `{input.model_dir}`
4. ✅ Cleaner dependency tracking - only depends on what's actually needed

**Result**: More accurate Snakemake dependency graph and clearer understanding of what score_only.py requires.
