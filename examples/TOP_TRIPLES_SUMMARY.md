# Top Test Triples Extraction - Summary

## What Was Done

Extracted the **top 25 test triples** with prediction scores > 0.84 from the trained ConvE model results for TracIn influence analysis.

---

## Input File

**Source:** [20251017_trained_test_scores_ranked.csv](20251017_trained_test_scores_ranked.csv)
- Total predictions: 950
- Filtered to: **25 triples** with score > 0.84

---

## Output Files Created

### Directory Structure

```
examples/
├── top_test_triples/
│   ├── top_test_triples.txt                    # All 25 triples (tab-separated)
│   ├── top_test_triples_with_scores.csv        # With scores for reference
│   ├── top_test_triples_summary.txt            # Human-readable summary
│   └── individual_triples/                      # 25 individual files
│       ├── triple_0_CHEBI_7963_MONDO_0016595.txt
│       ├── triple_1_CHEBI_34911_MONDO_0004525.txt  (Permethrin → Scabies)
│       └── ... (23 more files)
```

---

## Top 10 Test Triples

| Rank | Score  | Drug | Disease | File |
|------|--------|------|---------|------|
| 1 | 0.9164 | CHEBI:7963 (benzylpenicillin potassium) | MONDO:0016595 (inhalational anthrax) | triple_0_... |
| 2 | 0.9064 | CHEBI:34911 (Permethrin) | MONDO:0004525 (scabies) | triple_1_... |
| 3 | 0.8926 | CHEBI:27732 (Caffeine) | UMLS:C0042376 (Vascular Headaches) | triple_2_... |
| 4 | 0.8896 | UNII:A3ULP0F556 (Eculizumab) | UMLS:C0751339 (Myasthenia Gravis) | triple_3_... |
| 5 | 0.8823 | CHEBI:6970 (Mometasone) | MONDO:0015758 (primary cutaneous T-cell lymphoma) | triple_4_... |
| 6 | 0.8779 | CHEBI:7963 (benzylpenicillin potassium) | MONDO:0001316 (streptococcal meningitis) | triple_5_... |
| 7 | 0.8697 | CHEBI:31348 (Capecitabine) | MONDO:0021063 (colon cancer) | triple_6_... |
| 8 | 0.8695 | CHEBI:4911 (Etoposide) | MONDO:0002334 (hematopoietic neoplasm) | triple_7_... |
| 9 | 0.8672 | UNII:37CQ2C7X93 (Canakinumab) | MONDO:0007727 (familial periodic fever) | triple_8_... |
| 10 | 0.8672 | CHEBI:6970 (Mometasone) | UMLS:C0149922 (Lichen Simplex Chronicus) | triple_9_... |

---

## Statistics

**Filtered Dataset:**
- Total triples: 25
- Score range: 0.8411 - 0.9164
- Mean score: 0.8678
- Median score: 0.8661

**Original Dataset:**
- Total predictions: 950
- Score range: 0.0011 - 0.9164
- Mean score: 0.6237
- Top 25 represent: 2.6% of all predictions

---

## Files for TracIn Analysis

### 1. All Triples Together
**File:** `top_test_triples/top_test_triples.txt`

**Format:**
```
CHEBI:7963	predicate:28	MONDO:0016595
CHEBI:34911	predicate:28	MONDO:0004525
...
```

**Use for:** Batch analysis of all 25 triples

**Command:**
```bash
python run_tracin.py \
    --test top_test_triples/top_test_triples.txt \
    --mode test \
    --output-per-triple \
    --csv-output results/top_triples.csv
```

---

### 2. Individual Triple Files
**Directory:** `top_test_triples/individual_triples/`

**Files:** 25 separate .txt files, one per triple

**Use for:** Focused analysis on specific predictions

**Example - Permethrin → Scabies:**
```bash
python run_tracin.py \
    --test top_test_triples/individual_triples/triple_1_CHEBI_34911_MONDO_0004525.txt \
    --mode single \
    --csv-output results/permethrin_scabies_tracin.csv
```

---

### 3. Reference CSV with Scores
**File:** `top_test_triples/top_test_triples_with_scores.csv`

**Format:**
```csv
head_label,relation_label,tail_label,score
CHEBI:7963,predicate:28,MONDO:0016595,0.9164
CHEBI:34911,predicate:28,MONDO:0004525,0.9064
```

**Use for:** Reference and selecting which triples to analyze

---

## Quick Start Examples

### Example 1: Analyze Permethrin → Scabies (Top #2)

```bash
python run_tracin.py \
    --model-path model.pt \
    --train train.txt \
    --test top_test_triples/individual_triples/triple_1_CHEBI_34911_MONDO_0004525.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output permethrin_scabies.json \
    --csv-output permethrin_scabies_tracin.csv \
    --mode single \
    --test-indices 0 \
    --top-k 100 \
    --use-last-layers-only \
    --device cuda
```

---

### Example 2: Analyze Top 5 Triples

```bash
python run_tracin.py \
    --model-path model.pt \
    --train train.txt \
    --test top_test_triples/top_test_triples.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output top5.json \
    --csv-output top5_tracin.csv \
    --mode single \
    --test-indices 0 1 2 3 4 \
    --top-k 100 \
    --use-last-layers-only \
    --device cuda
```

**Output:**
- `top5_tracin_test_0.csv` through `top5_tracin_test_4.csv`

---

### Example 3: Batch Process All 25 Triples

```bash
python run_tracin.py \
    --model-path model.pt \
    --train train.txt \
    --test top_test_triples/top_test_triples.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --edge-map edge_map.json \
    --node-name-dict node_name_dict.txt \
    --output all_top_triples.json \
    --csv-output all_top_triples_tracin.csv \
    --mode test \
    --output-per-triple \
    --max-test-triples 25 \
    --top-k 50 \
    --use-last-layers-only \
    --device cuda \
    --batch-size 512
```

**Output:**
- 25 JSON files: `all_top_triples_test_0.json` through `_test_24.json`
- 25 CSV files: `all_top_triples_tracin_test_0.csv` through `_test_24.csv`

---

## Scripts Created

### 1. extract_top_test_triples.py
**Purpose:** Extract top triples from score-ranked predictions

**Usage:**
```bash
python extract_top_test_triples.py
```

**Configuration:** Edit these variables in the script:
```python
score_threshold = 0.84  # Minimum score
output_dir = Path('top_test_triples')  # Output directory
```

### 2. plot_score_histogram.py
**Purpose:** Visualize score distribution

**Usage:**
```bash
python plot_score_histogram.py
```

**Output:**
- `20251017_trained_test_scores_histogram.png`
- `20251017_trained_test_scores_histogram_log.png`

---

## Understanding the Triples

### All 25 triples use relation: `predicate:28`

From your edge_map.json, `predicate:28` likely maps to **`biolink:treats`**

This means all 25 top predictions are drug-disease treatment relationships:
- Drug (head) → treats → Disease (tail)

### Drug Categories Represented:

- **Antibiotics:** benzylpenicillin, Invanz
- **Cancer drugs:** Capecitabine, Etoposide, Cisplatin
- **Immunotherapy:** Rituximab, Pembrolizumab, Canakinumab
- **Antiparasitics:** Permethrin
- **Steroids:** Mometasone, Fluticasone
- **Stimulants:** Caffeine, Lisdexamfetamine
- **Cardiovascular:** pravastatin
- **Others:** Various therapeutic agents

### Disease Categories:

- Cancers (colon, lung, brain, etc.)
- Infections (anthrax, meningitis, scabies)
- Autoimmune (rheumatoid arthritis, pemphigus)
- Neurological (ADHD, epilepsy, headaches)
- Others

---

## Next Steps

1. **Choose triples to analyze:**
   - Start with top 5 for initial analysis
   - Focus on specific drug-disease pairs of interest
   - Or batch process all 25 if computational resources allow

2. **Run TracIn analysis:**
   - Use examples from TRACIN_TOP_TRIPLES_GUIDE.md
   - Start with `--use-last-layers-only` for speed
   - Consider filtering training data first

3. **Analyze results:**
   - Examine top influential training triples
   - Look for biological mechanisms
   - Validate against known literature
   - Identify novel insights

4. **Iterate:**
   - Refine analysis based on initial findings
   - Adjust score threshold if needed
   - Compare influences across different predictions

---

## Documentation

- **[TRACIN_TOP_TRIPLES_GUIDE.md](TRACIN_TOP_TRIPLES_GUIDE.md)** - Complete TracIn usage guide
- **[TOP_TRIPLES_SUMMARY.md](TOP_TRIPLES_SUMMARY.md)** - This file
- **[top_test_triples_summary.txt](top_test_triples/top_test_triples_summary.txt)** - Human-readable triple list

---

## Summary

✅ **Extracted 25 top triples** with score > 0.84
✅ **Created multiple file formats** for different use cases
✅ **Individual files** for focused analysis
✅ **Complete documentation** and usage examples
✅ **Ready for TracIn analysis** with exact format

**The top triples represent the model's most confident predictions and are ideal candidates for TracIn influence analysis to understand what training data drove these high-confidence predictions.**
