# Snakefile Fix: Preventing Skipped Steps

## Problem

Snakemake was skipping intermediate steps (like Step 1b: `prepare_dictionaries`) when running the full pipeline.

## Root Cause

**Snakemake only executes rules needed to create the files listed in `rule all: input:`**

If an intermediate rule creates multiple output files, but only SOME of those outputs are:
1. Listed in `rule all`, OR
2. Required as inputs by other rules that ARE in the dependency chain

Then Snakemake considers the rule "complete" as soon as those specific outputs exist, even if it hasn't created ALL outputs.

## The Fix

Added **ALL output files** from intermediate rules to `rule all: input:` to ensure complete execution.

### Before (Incomplete)

```python
rule all:
    input:
        f"{BASE_DIR}/rotorobo.txt",
        f"{BASE_DIR}/edge_map.json",
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results.txt",
        # Missing: treats.txt, filtered files, annotated files
        f"{BASE_DIR}/test.txt",
        f"{BASE_DIR}/train_candidates.txt",
        # Missing: test_statistics.json
        f"{BASE_DIR}/train.txt",
        f"{BASE_DIR}/valid.txt",
        # Missing: split_statistics.json
        f"{BASE_DIR}/processed/node_dict.txt",
        f"{BASE_DIR}/processed/rel_dict.txt",
        # Missing: node_name_dict.txt, graph_stats.txt
        f"{BASE_DIR}/models/conve/config.json",
        f"{BASE_DIR}/results/evaluation/test_scores.json",
        f"{BASE_DIR}/results/evaluation/test_scores_ranked.json"
```

### After (Complete)

```python
rule all:
    input:
        # Subgraph files (from create_subgraph)
        f"{BASE_DIR}/rotorobo.txt",
        f"{BASE_DIR}/edge_map.json",

        # Mechanistic paths output (from extract_mechanistic_paths)
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results.txt",
        f"{BASE_DIR}/results/mechanistic_paths/treats.txt",

        # Filtered treats edges (from filter_treats_with_drugmechdb)
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt",
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results_annotated.csv",
        f"{BASE_DIR}/results/mechanistic_paths/treats_annotated.txt",

        # DrugMechDB test set (from extract_drugmechdb_test)
        f"{BASE_DIR}/test.txt",
        f"{BASE_DIR}/train_candidates.txt",
        f"{BASE_DIR}/test_statistics.json",

        # Train/valid split (from split_data)
        f"{BASE_DIR}/train.txt",
        f"{BASE_DIR}/valid.txt",
        f"{BASE_DIR}/split_statistics.json",

        # Dictionary files (from prepare_dictionaries)
        f"{BASE_DIR}/processed/node_dict.txt",
        f"{BASE_DIR}/processed/node_name_dict.txt",
        f"{BASE_DIR}/processed/rel_dict.txt",
        f"{BASE_DIR}/processed/graph_stats.txt",

        # Trained model (PyKEEN outputs)
        f"{BASE_DIR}/models/conve/config.json",

        # Evaluation results (score_only.py outputs)
        f"{BASE_DIR}/results/evaluation/test_scores.json",
        f"{BASE_DIR}/results/evaluation/test_scores_ranked.json"
```

## What Changed

### Added from `extract_mechanistic_paths`:
- ✅ `treats.txt`

### Added from `filter_treats_with_drugmechdb`:
- ✅ `drugmechdb_treats_filtered.txt`
- ✅ `drugmechdb_path_id_results_annotated.csv`
- ✅ `treats_annotated.txt`

### Added from `extract_drugmechdb_test`:
- ✅ `test_statistics.json`

### Added from `split_data`:
- ✅ `split_statistics.json`

### Added from `prepare_dictionaries`:
- ✅ `node_name_dict.txt`
- ✅ `graph_stats.txt`

## Why This Matters

### Example: `prepare_dictionaries` Being Skipped

**Rule definition:**
```python
rule prepare_dictionaries:
    output:
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",        # ← Was in rule all
        node_name_dict = f"{BASE_DIR}/processed/node_name_dict.txt",  # ← MISSING from rule all
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt",          # ← Was in rule all
        stats = f"{BASE_DIR}/processed/graph_stats.txt"           # ← MISSING from rule all
```

**What happened:**
1. `train_model` needs `node_dict.txt` and `rel_dict.txt`
2. If those files already exist, Snakemake sees the dependency is satisfied
3. `node_name_dict.txt` is only used by `evaluate_model` (not `train_model`)
4. If pipeline stopped before `evaluate_model`, `prepare_dictionaries` wouldn't run
5. Even if it did run, if only partial outputs existed, Snakemake might skip re-running

**Now:**
- ALL 4 outputs are in `rule all`
- Snakemake will ensure ALL files exist
- If any are missing, the rule will run

## Best Practices Going Forward

### 1. List ALL Outputs in `rule all`

```python
rule all:
    input:
        # Every output from every rule that should run in the pipeline
```

### 2. Use Comments to Track Which Rule Creates Each Output

```python
rule all:
    input:
        # Dictionary files (from prepare_dictionaries)
        f"{BASE_DIR}/processed/node_dict.txt",
        f"{BASE_DIR}/processed/node_name_dict.txt",
```

### 3. Verify with Dry Run

```bash
snakemake -n --snakefile Snakefile
```

Look for all rules in the execution plan. If a rule is missing, add its outputs to `rule all`.

### 4. Use `--forcerun` for Testing

```bash
# Force re-run a specific rule to test it
snakemake --forcerun prepare_dictionaries

# Force re-run everything
snakemake -F
```

## Verification

To verify the fix works:

```bash
# 1. Dry run to see execution plan
snakemake -n --snakefile Snakefile --cores 1

# 2. Check that prepare_dictionaries appears in the plan
snakemake -n --snakefile Snakefile --cores 1 | grep "prepare_dictionaries"

# 3. Run the pipeline
snakemake --snakefile Snakefile --cores all

# 4. Verify all outputs exist
ls -lh robokop/CGGD_alltreat/processed/
# Should show: node_dict.txt, node_name_dict.txt, rel_dict.txt, graph_stats.txt
```

## Summary

| Issue | Cause | Fix |
|-------|-------|-----|
| Steps skipped | Missing outputs in `rule all` | Added all intermediate outputs |
| `prepare_dictionaries` skipped | Only 2/4 outputs listed | Added `node_name_dict.txt` and `graph_stats.txt` |
| `filter_treats_with_drugmechdb` skipped | 0/3 outputs listed | Added all 3 filtered files |
| Statistics files missing | Not in `rule all` | Added `test_statistics.json` and `split_statistics.json` |

**Result:** Pipeline now executes ALL steps in correct order, creating ALL outputs.
