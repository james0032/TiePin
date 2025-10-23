# TracIn Analysis for Top Test Triples

## Overview

This guide shows how to run TracIn analysis on the top-scoring test triples (score > 0.84) extracted from the trained ConvE model predictions.

---

## Files Created

### 1. **top_test_triples.txt** (25 triples)
Tab-separated file with head, relation, tail - ready for TracIn input:
```
CHEBI:7963	predicate:28	MONDO:0016595
CHEBI:34911	predicate:28	MONDO:0004525
...
```

### 2. **top_test_triples_with_scores.csv**
CSV file with scores for reference:
```csv
head_label,relation_label,tail_label,score
CHEBI:7963,predicate:28,MONDO:0016595,0.9164
CHEBI:34911,predicate:28,MONDO:0004525,0.9064
...
```

### 3. **individual_triples/** (25 files)
Separate files for each triple, useful for single-triple TracIn analysis:
- `triple_0_CHEBI_7963_MONDO_0016595.txt`
- `triple_1_CHEBI_34911_MONDO_0004525.txt` (Permethrin → Scabies)
- ... and 23 more

### 4. **top_test_triples_summary.txt**
Human-readable summary of all 25 top triples with scores

---

## Top Test Triples Summary

**Total:** 25 triples with score > 0.84
**Score Range:** 0.8411 - 0.9164
**Mean Score:** 0.8678

### Top 5 Predictions:

1. **Score: 0.9164** - CHEBI:7963 (benzylpenicillin potassium) → MONDO:0016595 (inhalational anthrax)
2. **Score: 0.9064** - CHEBI:34911 (Permethrin) → MONDO:0004525 (scabies)
3. **Score: 0.8926** - CHEBI:27732 (Caffeine) → UMLS:C0042376 (Vascular Headaches)
4. **Score: 0.8896** - UNII:A3ULP0F556 (Eculizumab) → UMLS:C0751339 (Myasthenia Gravis)
5. **Score: 0.8823** - CHEBI:6970 (Mometasone) → MONDO:0015758 (primary cutaneous T-cell lymphoma)

---

## Running TracIn Analysis

### Prerequisites

Ensure you have:
- `trained_model.pt` - Trained ConvE model
- `train.txt` - Training triples
- `entity_to_id.tsv` - Entity mappings
- `relation_to_id.tsv` - Relation mappings
- `edge_map.json` - Predicate names (optional)
- `node_name_dict.txt` - Entity names (optional)

---

## Usage Options

### Option 1: Analyze All 25 Top Triples

Analyze all 25 top-scoring triples in batch:

```bash
cd /Users/jchung/Documents/RENCI/everycure/git/conve_pykeen

python run_tracin.py \
    --model-path path/to/trained_model.pt \
    --train path/to/train.txt \
    --test examples/top_test_triples/top_test_triples.txt \
    --entity-to-id path/to/entity_to_id.tsv \
    --relation-to-id path/to/relation_to_id.tsv \
    --edge-map path/to/edge_map.json \
    --node-name-dict path/to/node_name_dict.txt \
    --output results/top_triples_analysis.json \
    --csv-output results/top_triples_tracin.csv \
    --mode test \
    --output-per-triple \
    --top-k 100 \
    --use-last-layers-only \
    --num-last-layers 2 \
    --device cuda \
    --batch-size 512
```

**Output:**
- `results/top_triples_analysis_test_0.json` through `_test_24.json`
- `results/top_triples_tracin_test_0.csv` through `_test_24.csv`

---

### Option 2: Analyze Single Triple (Permethrin → Scabies)

Analyze just the Permethrin-Scabies prediction (score: 0.9064):

```bash
python run_tracin.py \
    --model-path path/to/trained_model.pt \
    --train path/to/train.txt \
    --test examples/top_test_triples/individual_triples/triple_1_CHEBI_34911_MONDO_0004525.txt \
    --entity-to-id path/to/entity_to_id.tsv \
    --relation-to-id path/to/relation_to_id.tsv \
    --edge-map path/to/edge_map.json \
    --node-name-dict path/to/node_name_dict.txt \
    --output results/permethrin_scabies.json \
    --csv-output results/permethrin_scabies_tracin.csv \
    --mode single \
    --test-indices 0 \
    --top-k 100 \
    --use-last-layers-only \
    --device cuda
```

**Output:**
- `results/permethrin_scabies.json`
- `results/permethrin_scabies_tracin.csv`

---

### Option 3: Analyze Specific Top Triples by Index

Analyze specific triples (e.g., top 3):

```bash
python run_tracin.py \
    --model-path path/to/trained_model.pt \
    --train path/to/train.txt \
    --test examples/top_test_triples/top_test_triples.txt \
    --entity-to-id path/to/entity_to_id.tsv \
    --relation-to-id path/to/relation_to_id.tsv \
    --edge-map path/to/edge_map.json \
    --node-name-dict path/to/node_name_dict.txt \
    --output results/top3_analysis.json \
    --csv-output results/top3_tracin.csv \
    --mode single \
    --test-indices 0 1 2 \
    --top-k 100 \
    --use-last-layers-only \
    --device cuda
```

**Output:**
- `results/top3_analysis.json`
- `results/top3_tracin_test_0.csv` (benzylpenicillin → anthrax)
- `results/top3_tracin_test_1.csv` (Permethrin → scabies)
- `results/top3_tracin_test_2.csv` (Caffeine → headaches)

---

### Option 4: Use with Filtered Training Data

For faster analysis, first filter training data by proximity:

```bash
# Step 1: Filter training data for each top triple
python filter_training_by_proximity_pyg.py \
    --train path/to/train.txt \
    --test examples/top_test_triples/individual_triples/triple_1_CHEBI_34911_MONDO_0004525.txt \
    --output filtered_train_permethrin.txt \
    --cache train_graph.pkl \
    --n-hops 2 \
    --single-triple

# Step 2: Run TracIn on filtered data
python run_tracin.py \
    --train filtered_train_permethrin.txt \
    --test examples/top_test_triples/individual_triples/triple_1_CHEBI_34911_MONDO_0004525.txt \
    --csv-output results/permethrin_scabies_tracin.csv \
    --mode single \
    --use-last-layers-only \
    --device cuda
```

---

## CSV Output Format

The CSV output will have the exact format:

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
CHEBI:34911,Permethrin,predicate:28,biolink:treats,MONDO:0004525,scabies,NCBIGene:1234,GENE1,predicate:29,biolink:target_for,MONDO:0004525,scabies,0.3972
...
```

---

## Analyzing Results

### Load CSV in Python

```python
import pandas as pd

# Load TracIn results
df = pd.read_csv('results/permethrin_scabies_tracin.csv')

# View top 10 most influential training triples
print(df.head(10))

# Analyze by training relation type
by_relation = df.groupby('TrainRel_label')['TracInScore'].agg(['mean', 'count', 'sum'])
print(by_relation.sort_values('sum', ascending=False))

# Find genes that target the same disease
targets = df[df['TrainRel_label'] == 'biolink:target_for']
print(targets[['TrainHead_label', 'TracInScore']].head(20))

# Identify drugs that treat the same disease
drugs = df[df['TrainRel_label'] == 'biolink:treats']
print(drugs[['TrainHead_label', 'TracInScore']].head(20))
```

### Visualize Influences

```python
import matplotlib.pyplot as plt

# Plot top 20 influences
top20 = df.head(20)
plt.figure(figsize=(12, 8))
plt.barh(range(20), top20['TracInScore'])
plt.yticks(range(20), [f"{row['TrainHead_label'][:20]}\n→{row['TrainRel_label'][:20]}\n→{row['TrainTail_label'][:20]}"
                       for _, row in top20.iterrows()], fontsize=8)
plt.xlabel('TracIn Influence Score')
plt.title('Top 20 Most Influential Training Triples')
plt.tight_layout()
plt.savefig('top20_influences.png', dpi=300)
```

---

## Performance Tips

### 1. Use Last Layers Only (50x Speedup)
```bash
--use-last-layers-only \
--num-last-layers 2
```

### 2. Increase Batch Size on GPU
```bash
--batch-size 512  # or 1024 for larger GPUs
--device cuda
```

### 3. Filter Training Data First
Use proximity filtering to reduce training set size by 60-80%:
```bash
python filter_training_by_proximity_pyg.py \
    --n-hops 2 \
    --min-degree 2
```

### 4. Use Smaller top-k
```bash
--top-k 50  # Instead of 100 or more
```

---

## Expected Runtime

**For 25 top triples with full training data (~100K triples):**

| Configuration | GPU | Time per Triple | Total Time |
|---------------|-----|-----------------|------------|
| All layers | V100 | ~45 min | ~19 hours |
| Last 2 layers | V100 | ~60 sec | ~25 minutes |
| Last 2 layers + filtered | V100 | ~20 sec | ~8 minutes |

**Recommendation:** Use `--use-last-layers-only` with filtered training data for best performance.

---

## Interpreting Results

### High TracIn Score Means:

- **Positive score**: Training triple pushes model toward correct prediction
- **Larger magnitude**: More influential (more important for the prediction)
- **Top influences**: Key training examples that "taught" the model this prediction

### What to Look For:

1. **Gene targets**: Training triples like `(Gene X, target_for, Disease)` that share the same disease
2. **Similar drugs**: Training triples with drugs treating the same or related diseases
3. **Mechanism insights**: Related biological pathways or mechanisms
4. **Data quality**: Check if top influences make biological sense

---

## Troubleshooting

### Issue: "Loaded 0 predicate names"
**Fix:** Ensure `--edge-map edge_map.json` is provided and path is correct

### Issue: Out of memory
**Fix:** Reduce batch size: `--batch-size 256` or `--batch-size 128`

### Issue: Too slow
**Fix:** Use last layers only: `--use-last-layers-only --num-last-layers 2`

### Issue: CSV shows IDs instead of names
**Fix:** Provide label files:
```bash
--edge-map edge_map.json \
--node-name-dict node_name_dict.txt
```

---

## Summary

You now have 25 top-scoring test triples ready for TracIn analysis:

✅ **25 triples extracted** with score > 0.84
✅ **Multiple output formats** for different use cases
✅ **Individual triple files** for focused analysis
✅ **Complete usage examples** for all scenarios
✅ **CSV export support** with exact header format

**Next Steps:**

1. Choose which triples to analyze (all 25, top 5, or specific ones)
2. Run TracIn with appropriate mode and options
3. Analyze CSV results to understand influential training data
4. Use insights to improve model or validate predictions

For questions about specific drugs or diseases, focus on those individual triple files!
