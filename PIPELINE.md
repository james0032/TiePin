# Snakemake Pipeline for ConvE PyKEEN

This document describes the automated Snakemake pipeline for the complete ConvE PyKEEN workflow, from data extraction to TracIn analysis.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Pipeline Steps](#pipeline-steps)
- [Output Files](#output-files)
- [Utility Commands](#utility-commands)
- [Troubleshooting](#troubleshooting)

## Overview

The Snakemake pipeline automates the following workflow:

1. **Extract Mechanistic Paths** - Extract drug-disease mechanistic paths from DrugMechDB
2. **Create Subgraph** - Build ROBOKOP subgraph from edges file
3. **Prepare Dictionaries** - Generate node and relation dictionaries
4. **Split Data** - Split into train/validation/test sets
5. **Preprocess Data** - Convert to PyKEEN format
6. **Train Model** - Train ConvE knowledge graph embedding model
7. **Evaluate Model** - Evaluate on test set with detailed metrics
8. **TracIn Analysis** (Optional) - Analyze training data influence

## Installation

### 1. Install Dependencies

```bash
# Install all requirements including snakemake
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check snakemake installation
snakemake --version

# Should output: 7.0.0 or higher
```

## Configuration

All pipeline parameters are configured in `config.yaml`. Edit this file to customize the pipeline:

### Essential Configuration

```yaml
# Input edges file
edges_file: "roboedges.jsonl.gz"

# Data split ratios
train_ratio: 0.8
valid_ratio: 0.1

# Model parameters
embedding_dim: 200
embedding_height: 10
embedding_width: 20
num_epochs: 100
batch_size: 256
learning_rate: 0.001
```

### Optional Features

```yaml
# Run old mechanistic path method (slower)
run_old_method: false

# Run TracIn analysis (very slow, computationally expensive)
run_tracin: false

# Use GPU for training
use_gpu: true
```

See `config.yaml` for all available options with detailed comments.

## Running the Pipeline

### Run Complete Pipeline

```bash
# Run all steps with maximum parallelization
snakemake --cores all

# Run with specific number of cores
snakemake --cores 4

# Dry run to see what will be executed
snakemake -n

# Dry run with detailed reasoning
snakemake -n -r
```

### Run Specific Steps

```bash
# Run only mechanistic path extraction
snakemake results/mechanistic_paths/drugmechdb_path_id_results.txt --cores 1

# Run only up to model training
snakemake models/conve/trained_model.pkl --cores all

# Run only evaluation
snakemake results/evaluation/test_results.json --cores 1
```

### Visualize Pipeline

```bash
# Generate DAG visualization
snakemake --dag | dot -Tpdf > pipeline_dag.pdf

# Generate rule graph
snakemake --rulegraph | dot -Tpdf > pipeline_rules.pdf
```

## Pipeline Steps

### Step 1: Extract Mechanistic Paths

**Input:**
- `roboedges.jsonl.gz` (or path specified in config)

**Output:**
- `results/mechanistic_paths/drugmechdb_path_id_results.txt`
- `results/mechanistic_paths/treats_mechanistic_paths.json` (if run_old_method=true)

**What it does:**
Extracts drug-disease mechanistic paths from DrugMechDB using pre-defined `drugmechdb_path_id` values.

### Step 2: Create ROBOKOP Subgraph

**Input:**
- Edges file

**Output:**
- `data/raw/rotorobo.txt` - Subgraph triples
- `data/raw/edge_map.json` - Edge predicate mapping

**What it does:**
Creates a subgraph from the full knowledge graph for downstream processing.

### Step 3: Prepare Dictionaries

**Input:**
- `data/raw/rotorobo.txt`
- `data/raw/edge_map.json`

**Output:**
- `data/processed/node_dict.txt` - Entity to index mapping
- `data/processed/rel_dict.txt` - Relation to index mapping
- `data/processed/node_name_dict.txt` - Entity names
- `data/processed/graph_stats.txt` - Graph statistics

**What it does:**
Generates dictionary files mapping entities and relations to integer indices.

### Step 4: Split Data

**Input:**
- `data/raw/rotorobo.txt`

**Output:**
- `data/raw/train.tsv`
- `data/raw/valid.tsv`
- `data/raw/test.tsv`

**What it does:**
Splits the graph into train/validation/test sets based on configured ratios.

### Step 5: Preprocess Data

**Input:**
- Train/valid/test TSV files
- Dictionary files
- Edge map

**Output:**
- `data/processed/train.txt`
- `data/processed/valid.txt`
- `data/processed/test.txt`
- `data/processed/train_entity_to_id.tsv`
- `data/processed/train_relation_to_id.tsv`

**What it does:**
Converts data to PyKEEN format while preserving dictionary indices.

### Step 6: Train ConvE Model

**Input:**
- Preprocessed train/valid/test files
- Entity and relation mappings

**Output:**
- `models/conve/trained_model.pkl` - Trained model
- `models/conve/config.json` - Training configuration
- `models/conve/losses.tsv` - Training losses

**What it does:**
Trains the ConvE knowledge graph embedding model using configured hyperparameters.

**GPU Usage:**
- Automatically uses GPU if available (controlled by `use_gpu` in config)
- Set `use_gpu: false` for CPU-only training

### Step 7: Evaluate Model

**Input:**
- Trained model
- Test data

**Output:**
- `results/evaluation/test_results.json` - Detailed results
- `results/evaluation/test_results.csv` - Results in CSV format

**What it does:**
Evaluates the model on the test set, computing:
- Mean Rank (MR)
- Mean Reciprocal Rank (MRR)
- Hits@1, Hits@3, Hits@10
- Per-triple predictions with scores and rankings

### Step 8: TracIn Analysis (Optional)

**Input:**
- Trained model
- Test data

**Output:**
- `results/tracin/tracin_analysis_{batch}.json`

**What it does:**
Computes training data influence on test predictions using the TracIn method.

**Note:** This is computationally very expensive. Enable only if needed by setting `run_tracin: true` in config.

## Output Files

All outputs are organized in the following directory structure:

```
conve_pykeen/
├── data/
│   ├── raw/
│   │   ├── rotorobo.txt
│   │   ├── edge_map.json
│   │   ├── train.tsv
│   │   ├── valid.tsv
│   │   └── test.tsv
│   └── processed/
│       ├── node_dict.txt
│       ├── rel_dict.txt
│       ├── train.txt
│       ├── valid.txt
│       ├── test.txt
│       ├── train_entity_to_id.tsv
│       └── train_relation_to_id.tsv
├── models/
│   └── conve/
│       ├── trained_model.pkl
│       ├── config.json
│       └── losses.tsv
├── results/
│   ├── mechanistic_paths/
│   │   └── drugmechdb_path_id_results.txt
│   ├── evaluation/
│   │   ├── test_results.json
│   │   └── test_results.csv
│   └── tracin/
│       └── tracin_analysis_*.json
└── logs/
    ├── extract_mechanistic_paths.log
    ├── create_subgraph.log
    ├── prepare_dictionaries.log
    ├── split_data.log
    ├── preprocess_data.log
    ├── train_model.log
    ├── evaluate_model.log
    └── tracin_analysis_*.log
```

## Utility Commands

### Clean Commands

```bash
# Remove all generated files (complete clean)
snakemake clean

# Remove only models and results (keep preprocessed data)
snakemake clean_models

# Remove only analysis results (keep models and data)
snakemake clean_results
```

### Check What Would Run

```bash
# Dry run to see pending steps
snakemake -n

# Show reasons for re-running rules
snakemake -n -r
```

### Force Re-run

```bash
# Force re-run all steps
snakemake --forceall --cores all

# Force re-run specific step
snakemake --forcerun train_model --cores all
```

### Generate Reports

```bash
# Generate HTML report with statistics
snakemake --report report.html

# Generate detailed execution timeline
snakemake --forceall --cores all --stats stats.json
```

## Troubleshooting

### Pipeline Fails on Specific Step

**Check the log file:**
```bash
# Logs are in logs/ directory
cat logs/train_model.log
```

**Re-run just that step:**
```bash
# Force re-run the failing rule
snakemake --forcerun <rule_name> --cores 1
```

### Out of Memory Error

**Solution 1: Reduce batch size**
Edit `config.yaml`:
```yaml
batch_size: 128  # Reduce from 256
```

**Solution 2: Use CPU instead of GPU**
Edit `config.yaml`:
```yaml
use_gpu: false
```

### Missing Input File

**Check if edges file exists:**
```bash
ls -lh roboedges.jsonl.gz
```

**Update path in config.yaml:**
```yaml
edges_file: "/full/path/to/edges.jsonl.gz"
```

### Pipeline Stuck or Hanging

**Check running processes:**
```bash
# See what's currently running
ps aux | grep python
```

**Kill and restart:**
```bash
# Kill stuck processes
pkill -f snakemake

# Restart with verbose output
snakemake --cores all --verbose
```

### Embedding Dimension Error

Ensure `embedding_dim = embedding_height × embedding_width`:

```yaml
# Valid configurations:
embedding_dim: 200
embedding_height: 10
embedding_width: 20  # 10 × 20 = 200 ✓

# Invalid:
embedding_dim: 200
embedding_height: 10
embedding_width: 15  # 10 × 15 = 150 ✗
```

### GPU Not Available

If GPU is not available but `use_gpu: true`:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

Set to CPU mode in `config.yaml`:
```yaml
use_gpu: false
```

## Advanced Usage

### Running on HPC/SLURM

Create a cluster configuration file `cluster.yaml`:

```yaml
__default__:
  partition: "general"
  time: "2:00:00"
  mem: "16G"
  cores: 1

train_model:
  partition: "gpu"
  time: "24:00:00"
  mem: "32G"
  cores: 4
  gres: "gpu:1"
```

Run with SLURM:
```bash
snakemake --cluster "sbatch -p {cluster.partition} -t {cluster.time} \
  --mem={cluster.mem} -c {cluster.cores}" \
  --cluster-config cluster.yaml \
  --jobs 10
```

### Customizing Individual Steps

You can override specific parameters without editing `config.yaml`:

```bash
# Override config values from command line
snakemake --cores all --config num_epochs=200 batch_size=128
```

### Parallel TracIn Batches

To run TracIn analysis in parallel batches, edit `config.yaml`:

```yaml
run_tracin: true
tracin_batches: 4  # Run 4 parallel batches
```

This will create 4 separate TracIn jobs that can run in parallel.

## Performance Tips

1. **Use GPU** - Training is 10-100x faster on GPU
2. **Adjust batch size** - Larger batches (up to memory limit) are faster
3. **Skip TracIn** - Only run if needed, it's very expensive
4. **Parallel execution** - Use `--cores all` for maximum parallelization
5. **Cache results** - Snakemake automatically caches completed steps

## Example Workflows

### Quick Test Run

```yaml
# config.yaml - minimal setup for testing
num_epochs: 10
batch_size: 128
run_tracin: false
run_old_method: false
```

```bash
snakemake --cores all
```

### Full Production Run

```yaml
# config.yaml - full production setup
num_epochs: 200
batch_size: 512
run_tracin: true
tracin_batches: 4
use_gpu: true
```

```bash
snakemake --cores all --use-conda
```

### Re-train with Different Hyperparameters

```bash
# Clean only models
snakemake clean_models

# Re-train with new config
snakemake models/conve/trained_model.pkl --cores all
```

## Getting Help

For issues or questions:

1. Check logs in `logs/` directory
2. Review configuration in `config.yaml`
3. See main README.md for detailed documentation
4. Run with `--verbose` for detailed output

```bash
snakemake --cores all --verbose
```
