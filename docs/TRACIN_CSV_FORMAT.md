# TracIn CSV Output Format Guide

## Overview

Export TracIn influence analysis results to CSV format with both CURIEs and human-readable labels.

---

## Output Format

### CSV Columns

```
TestHead, TestHead_label, TestRel, TestRel_label, TestTail, TestTail_label,
TrainHead, TrainHead_label, TrainRel, TrainRel_label, TrainTail, TrainTail_label,
TracInScore
```

### Example Output

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore
UNII:U59UGK3IPC,Ublituximab,predicate:27,biolink:treats,MONDO:0005314,relapsing-remitting multiple sclerosis,NCBIGene:3455,IFNAR2,predicate:29,biolink:target_for,MONDO:0005314,relapsing-remitting multiple sclerosis,0.3972736001014709
UNII:U59UGK3IPC,Ublituximab,predicate:27,biolink:treats,MONDO:0005314,relapsing-remitting multiple sclerosis,NCBIGene:8698,S1PR4,predicate:29,biolink:target_for,MONDO:0005314,relapsing-remitting multiple sclerosis,0.3969393968582153
```

---

## Usage

### Quick Start

```bash
python tracin_to_csv.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --relation-labels relation_labels.txt \
    --output results/tracin_influences.csv \
    --top-k 100
```

### Input Files Required

#### 1. Test Triple File (`test_triple.txt`)

Single line with tab-separated triple:

```
UNII:U59UGK3IPC	predicate:27	MONDO:0005314
```

#### 2. Entity-to-ID Mapping (`entity_to_id.tsv`)

```
UNII:U59UGK3IPC	0
MONDO:0005314	1
NCBIGene:3455	2
```

#### 3. Relation-to-ID Mapping (`relation_to_id.tsv`)

```
predicate:27	0
predicate:29	1
```

#### 4. Entity Labels (`node_name_dict.txt`) - Optional

```
Ublituximab	0
relapsing-remitting multiple sclerosis	1
IFNAR2	2
```

#### 5. Relation Labels (`relation_labels.txt`) - Optional

```
biolink:treats	0
biolink:target_for	1
```

---

## Python API Usage

### Method 1: Using TracInAnalyzer

```python
from tracin import TracInAnalyzer
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory
import torch

# Load data and model
train_triples = TriplesFactory.from_path(...)
model = ConvE(...)
model.load_state_dict(torch.load('model.pt'))

# Create analyzer
analyzer = TracInAnalyzer(
    model=model,
    device='cuda',
    use_last_layers_only=True,
    num_last_layers=2
)

# Compute influences
test_triple = (drug_id, relation_id, disease_id)
influences = analyzer.compute_influences_for_test_triple(
    test_triple=test_triple,
    training_triples=train_triples,
    top_k=100
)

# Save to CSV
analyzer.save_influences_to_csv(
    test_triple=test_triple,
    influences=influences,
    output_path='results/tracin.csv',
    id_to_entity={0: 'UNII:U59UGK3IPC', ...},
    id_to_relation={0: 'predicate:27', ...},
    entity_labels={0: 'Ublituximab', ...},  # Optional
    relation_labels={0: 'biolink:treats', ...}  # Optional
)
```

### Method 2: Using Standalone Script

```bash
python tracin_to_csv.py \
    --model-path trained_model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --relation-labels relation_labels.txt \
    --output results/tracin_influences.csv \
    --top-k 100 \
    --device cuda
```

---

## Options

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model-path` | Path to trained model (.pt file) |
| `--train` | Path to training triples file |
| `--test-triple` | Path to file with single test triple |
| `--entity-to-id` | Path to entity_to_id.tsv mapping |
| `--relation-to-id` | Path to relation_to_id.tsv mapping |
| `--output` | Output CSV file path |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--entity-labels` | None | Path to entity labels file |
| `--relation-labels` | None | Path to relation labels file |
| `--top-k` | 100 | Number of top influences |
| `--learning-rate` | 0.001 | Learning rate used during training |
| `--device` | auto | Device to run on (cpu/cuda) |
| `--use-last-layers-only` | True | Use last layers for speed |
| `--num-last-layers` | 2 | Number of last layers to track |

---

## Label Files Format

### Option 1: Tab-Separated (Recommended)

```
label<TAB>index
```

Example (`node_name_dict.txt`):
```
Ublituximab	0
relapsing-remitting multiple sclerosis	1
IFNAR2	2
```

### Option 2: Line-by-Line (Index = Line Number)

```
label1
label2
label3
```

Example:
```
Ublituximab
relapsing-remitting multiple sclerosis
IFNAR2
```

### Option 3: JSON Edge Map (For Qualified Relations)

If your relations are stored as JSON with qualifiers (like `edge_map.json`), you can use the JSON file directly:

```json
{
  "{\"object_aspect_qualifier\": \"activity\", \"object_direction_qualifier\": \"decreased\", \"predicate\": \"biolink:affects\", \"subject_aspect_qualifier\": \"\", \"subject_direction_qualifier\": \"\"}": "predicate:0",
  "{\"object_aspect_qualifier\": \"\", \"object_direction_qualifier\": \"\", \"predicate\": \"biolink:coexpressed_with\", \"subject_aspect_qualifier\": \"\", \"subject_direction_qualifier\": \"\"}": "predicate:1",
  "{\"object_aspect_qualifier\": \"\", \"object_direction_qualifier\": \"\", \"predicate\": \"biolink:treats\", \"subject_aspect_qualifier\": \"\", \"subject_direction_qualifier\": \"\"}": "predicate:27"
}
```

**Usage:**
```bash
python tracin_to_csv.py \
    --model-path model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --relation-labels edge_map.json \    # Use JSON file here
    --output tracin_results.csv \
    --top-k 100
```

**Output:** The system will automatically extract the `"predicate"` field from the JSON, so the CSV will show:
```csv
TestRel,TestRel_label,...
predicate:27,biolink:treats,...
```

Instead of the full JSON string, only the clean predicate name is displayed.

---

## Reading CSV in Python

```python
import pandas as pd

# Load CSV
df = pd.read_csv('results/tracin_influences.csv')

# Access columns
print(df['TestHead'])           # CURIE
print(df['TestHead_label'])     # Human-readable name
print(df['TracInScore'])        # Influence score

# Filter top influences
top_10 = df.head(10)

# Find specific training patterns
targets = df[df['TrainRel_label'] == 'biolink:target_for']

# Group by training relation
by_relation = df.groupby('TrainRel_label')['TracInScore'].mean()
```

---

## Reading CSV in R

```r
library(tidyverse)

# Load CSV
df <- read_csv('results/tracin_influences.csv')

# Access columns
df$TestHead
df$TestHead_label
df$TracInScore

# Filter top influences
top_10 <- df %>% head(10)

# Find specific training patterns
targets <- df %>% filter(TrainRel_label == 'biolink:target_for')

# Group by training relation
by_relation <- df %>%
  group_by(TrainRel_label) %>%
  summarize(mean_score = mean(TracInScore))
```

---

## Complete Example

### Step 1: Prepare Test Triple File

```bash
# Create test triple file
echo -e "UNII:U59UGK3IPC\tpredicate:27\tMONDO:0005314" > test_triple.txt
```

### Step 2: Run TracIn Analysis

```bash
python tracin_to_csv.py \
    --model-path models/trained_model.pt \
    --train data/train.txt \
    --test-triple test_triple.txt \
    --entity-to-id data/entity_to_id.tsv \
    --relation-to-id data/relation_to_id.tsv \
    --entity-labels data/node_name_dict.txt \
    --relation-labels data/relation_labels.txt \
    --output results/ublituximab_ms_tracin.csv \
    --top-k 100 \
    --device cuda
```

### Step 3: Analyze Results

```python
import pandas as pd

# Load results
df = pd.read_csv('results/ublituximab_ms_tracin.csv')

# Show top 10 influences
print("\nTop 10 Most Influential Training Triples:")
print(df[['TrainHead_label', 'TrainRel_label', 'TrainTail_label', 'TracInScore']].head(10))

# Analyze by relation type
print("\nInfluence by Relation Type:")
relation_analysis = df.groupby('TrainRel_label').agg({
    'TracInScore': ['mean', 'count']
}).round(4)
print(relation_analysis)

# Find target genes
print("\nTop Target Genes:")
targets = df[df['TrainRel_label'] == 'biolink:target_for']
print(targets[['TrainHead_label', 'TracInScore']].head(10))
```

---

## Output Description

### Columns Explained

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `TestHead` | CURIE | Test triple head entity ID | `UNII:U59UGK3IPC` |
| `TestHead_label` | String | Human-readable name | `Ublituximab` |
| `TestRel` | CURIE | Test triple relation ID | `predicate:27` |
| `TestRel_label` | String | Human-readable relation | `biolink:treats` |
| `TestTail` | CURIE | Test triple tail entity ID | `MONDO:0005314` |
| `TestTail_label` | String | Human-readable name | `relapsing-remitting multiple sclerosis` |
| `TrainHead` | CURIE | Training triple head entity ID | `NCBIGene:3455` |
| `TrainHead_label` | String | Human-readable name | `IFNAR2` |
| `TrainRel` | CURIE | Training triple relation ID | `predicate:29` |
| `TrainRel_label` | String | Human-readable relation | `biolink:target_for` |
| `TrainTail` | CURIE | Training triple tail entity ID | `MONDO:0005314` |
| `TrainTail_label` | String | Human-readable name | `relapsing-remitting multiple sclerosis` |
| `TracInScore` | Float | Influence score (higher = more influential) | `0.3972736` |

### Score Interpretation

- **Positive scores**: Training triple pushes model toward correct prediction
- **Larger magnitude**: More influential (important for the prediction)
- **Sorted descending**: Most influential triples appear first

---

## Integration with Proximity Filtering

Combine with proximity filtering for faster analysis:

```bash
# Step 1: Filter training data
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test_triple.txt \
    --output train_filtered.txt \
    --cache train_graph.pkl \
    --n-hops 2 \
    --single-triple

# Step 2: Run TracIn on filtered data
python tracin_to_csv.py \
    --model-path model.pt \
    --train train_filtered.txt \    # Use filtered data!
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --output results/tracin.csv \
    --top-k 100
```

---

## Files Created

1. **[tracin.py](tracin.py)** - Added `save_influences_to_csv()` method
2. **[tracin_to_csv.py](tracin_to_csv.py)** - Standalone CSV export script
3. **[TRACIN_CSV_FORMAT.md](TRACIN_CSV_FORMAT.md)** - This guide

---

## Summary

### Quick Command

```bash
python tracin_to_csv.py \
    --model-path model.pt \
    --train train.txt \
    --test-triple test_triple.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --entity-labels node_name_dict.txt \
    --relation-labels relation_labels.txt \
    --output tracin_results.csv \
    --top-k 100
```

### Output Format

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,
TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,
TracInScore
```

**Your exact format is supported!** 🎯
