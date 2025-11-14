# Step 6 (Evaluate Model) Added Back to Snakefile

## Date: 2025-11-13

## Summary

Re-enabled Step 6 (evaluate_model) in the Snakefile using score_only.py to score test triples after PyKEEN training completes.

## Changes Made

### 1. Added evaluate_model Rule (Snakefile:319-360)

**New rule**: `evaluate_model`

**Purpose**: Score test triples using the trained ConvE model with score_only.py

**Inputs**:
- `model_dir`: `{BASE_DIR}/models/conve` - Directory containing trained model
- `test`: `{BASE_DIR}/test.txt` - Test triples file
- `node_dict`: `{BASE_DIR}/processed/node_dict.txt` - Entity to ID mapping
- `node_name_dict`: `{BASE_DIR}/processed/node_name_dict.txt` - Entity name mapping
- `rel_dict`: `{BASE_DIR}/processed/rel_dict.txt` - Relation to ID mapping
- `config_out`: `{BASE_DIR}/models/conve/config.json` - Ensures training is complete
- `test_results`: `{BASE_DIR}/models/conve/test_results.json` - Ensures training is complete

**Outputs**:
- `test_scores.json` - Test scores in original order (JSON format)
- `test_scores.csv` - Test scores in original order (CSV format)
- `test_scores_ranked.json` - Test scores ranked by score (JSON format)
- `test_scores_ranked.csv` - Test scores ranked by score (CSV format)

**Parameters from config.yaml**:
- `use_sigmoid`: Whether to apply sigmoid to convert scores to probabilities
- `top_n_triples`: Optional - output top N highest scoring triples to TSV
- `use_gpu`: Device selection (cuda vs cpu)

**Shell command**:
```bash
python score_only.py \
    --model-dir {input.model_dir} \
    --test {input.test} \
    --entity-to-id {input.node_dict} \
    --relation-to-id {input.rel_dict} \
    --node-name-dict {input.node_name_dict} \
    --output {output.scores_json} \
    --device {params.device} \
    {params.use_sigmoid} \
    {params.top_n_arg}
```

### 2. Updated rule all (Snakefile:45-52)

**Added evaluation outputs**:
```python
# Evaluation results (score_only.py outputs)
f"{BASE_DIR}/results/evaluation/test_scores.json",
f"{BASE_DIR}/results/evaluation/test_scores_ranked.json"
```

**Updated comment**: Changed from "Evaluation and TracIn analysis are disabled" to "TracIn analysis is disabled" since evaluation is now enabled.

## How It Works

### Execution Flow

1. **Training completes** (Step 5): train.py runs and saves model to `{BASE_DIR}/models/conve/`
2. **PyKEEN evaluation** (built into train.py): Saves `test_results.json` with ranking-based metrics
3. **score_only.py evaluation** (Step 6): Runs after training, provides detailed per-triple scores

### Key Features

**score_only.py advantages**:
- **Fast**: No expensive ranking computation (unlike PyKEEN's evaluator)
- **Detailed output**: Per-triple scores with entity names
- **Multiple formats**: JSON and CSV in both original and ranked order
- **Top-N output**: Optional TSV file with top N triples (no header)

**Runs AFTER training**:
- Depends on `config.json` and `test_results.json` from train.py
- Ensures model training is complete before scoring

### score_only.py vs PyKEEN's test_results.json

| Feature | score_only.py | PyKEEN test_results.json |
|---------|---------------|--------------------------|
| **Format** | JSON, CSV, ranked | JSON only |
| **Metrics** | Per-triple scores | Aggregated metrics (MRR, Hits@K) |
| **Entity names** | ✅ Included | ❌ Not included |
| **Ranking** | ✅ Sorted by score | ❌ Original order |
| **Speed** | ⚡ Fast (no ranking) | 🐢 Slow (computes ranks) |
| **Use case** | Individual predictions | Overall model performance |

## Configuration Options

### config.yaml Parameters

```yaml
# Use sigmoid to convert scores to probabilities [0, 1]
use_sigmoid: true

# Output top N highest scoring triples to a separate TSV file
top_n_triples: null  # Set to a number like 100 to enable

# Device for evaluation
use_gpu: true
```

### Output Files

With default config:

1. **test_scores.json**: All test triples with scores in original order
   ```json
   [
     {
       "head_id": 0,
       "head_label": "MESH:D001249",
       "head_name": "Aspirin",
       "relation_id": 2,
       "relation_label": "biolink:treats",
       "tail_id": 15,
       "tail_label": "MESH:D006331",
       "tail_name": "Heart Disease",
       "score": 0.8542
     },
     ...
   ]
   ```

2. **test_scores.csv**: Same as JSON but in CSV format

3. **test_scores_ranked.json**: Sorted by score (descending) with rank
   ```json
   [
     {
       "rank": 1,
       "head_id": 42,
       "head_label": "MESH:D008687",
       "head_name": "Metformin",
       "relation_id": 2,
       "relation_label": "biolink:treats",
       "tail_id": 123,
       "tail_label": "MESH:D003920",
       "tail_name": "Diabetes Mellitus",
       "score": 0.9823
     },
     ...
   ]
   ```

4. **test_scores_ranked.csv**: Same as ranked JSON but in CSV format

5. **test_scores_top100.txt** (if `top_n_triples: 100`): TSV format, no header
   ```
   MESH:D008687	biolink:treats	MESH:D003920
   MESH:D001249	biolink:treats	MESH:D006331
   ...
   ```

## Compatibility

### Works With:
- ✅ PyKEEN-trained models (train.py)
- ✅ Pure PyTorch models (train_pytorch.py)
- ✅ Both checkpoint formats (.pt and .pkl)
- ✅ Models with different architectures (auto-detects from state_dict)

### Model File Detection:
score_only.py automatically looks for models in this order:
1. `best_model.pt` (train_pytorch.py output)
2. `final_model.pt` (train_pytorch.py output)
3. `trained_model.pkl` (PyKEEN output)

## Running the Pipeline

### Run Complete Pipeline (with evaluation):
```bash
snakemake --cores all
```

### Run Only Evaluation (after training):
```bash
snakemake --cores 1 evaluate_model
```

### Run Specific Steps:
```bash
# Train model
snakemake --cores 4 train_model

# Evaluate model
snakemake --cores 1 evaluate_model
```

## Example Log Output

```
2025-11-13 10:30:15 - INFO - Loading model from /workspace/data/robokop/CGGD_alltreat/models/conve/trained_model.pkl...
2025-11-13 10:30:16 - INFO - Loading test triples from /workspace/data/robokop/CGGD_alltreat/test.txt...
2025-11-13 10:30:16 - INFO - Loaded 1,234 test triples
2025-11-13 10:30:16 - INFO - Loaded 456 entity names
2025-11-13 10:30:17 - INFO - Model loaded successfully
2025-11-13 10:30:17 - INFO - Scoring mode: probabilities (0-1)
2025-11-13 10:30:17 - INFO - Scoring test triples (this will be FAST - no ranking computation)...
2025-11-13 10:30:18 - INFO - ✓ Done! Scored 1,234 triples
2025-11-13 10:30:18 - INFO - ✓ Results saved to:
2025-11-13 10:30:18 - INFO -   - Original order JSON: test_scores.json
2025-11-13 10:30:18 - INFO -   - Original order CSV: test_scores.csv
2025-11-13 10:30:18 - INFO -   - Ranked by score JSON: test_scores_ranked.json
2025-11-13 10:30:18 - INFO -   - Ranked by score CSV: test_scores_ranked.csv
```

## Notes

- **Evaluation is now enabled by default** when running the full pipeline
- **Does NOT replace PyKEEN's evaluation** - both run and provide complementary information
- **PyKEEN test_results.json** provides overall metrics (MRR, Hits@K)
- **score_only.py outputs** provide per-triple predictions with entity names
- **Fast execution**: Typically completes in seconds even for large test sets
- **TracIn analysis** still requires train_pytorch.py (checkpoint format incompatibility)

## Relationship to Other Steps

```
Step 5 (train_model) → train.py
  ↓ Produces: trained_model.pkl, config.json, test_results.json

Step 6 (evaluate_model) → score_only.py
  ↓ Produces: test_scores.json, test_scores_ranked.json, etc.

Step 7 (tracin_analysis) → DISABLED (requires train_pytorch.py)
```

## Future Enhancements

Potential improvements:
1. Add support for scoring validation set as well
2. Generate visualization plots from ranked scores
3. Add statistical analysis of score distributions
4. Support batch processing for very large test sets
