# Pipeline Quick Start Guide

## Overview

This Snakemake pipeline uses a **style-based directory structure**. All outputs are organized under `robokop/{style}/` where `style` determines the graph filtering strategy.

## Configuration

Edit `config.yaml` to set your parameters:

```yaml
# Input files
node_file: "robokop/nodes.jsonl"
edges_file: "robokop/edges.jsonl"

# Filtering style - determines output directory
style: "CGGD_alltreat"  # Options: original, CD, CCGGDD, CGGD, rCD, keepall, CGGD_alltreat
```

### Available Styles

- **original**: Remove subclass_of and CAID edges
- **CD**: Chemical-Disease edges only
- **CCGGDD**: Chemical-Chemical, Gene-Gene, Disease-Disease edges
- **CGGD**: Chemical-Gene-Disease edges
- **rCD**: Remove Chemical-Disease edges
- **keepall**: Keep all edges (no filtering)
- **CGGD_alltreat**: CGGD edges + all 'treats' relationships (recommended)

## Directory Structure

All outputs for a given style are organized under `robokop/{style}/`:

```
robokop/CGGD_alltreat/           # Example with style="CGGD_alltreat"
├── rotorobo.txt                  # Subgraph triples
├── edge_map.json                 # Edge predicate mapping
├── train.tsv                     # Training data
├── valid.tsv                     # Validation data
├── test.tsv                      # Test data
├── processed/
│   ├── node_dict.txt             # Entity to index mapping
│   ├── rel_dict.txt              # Relation to index mapping
│   ├── train.txt                 # Preprocessed training data
│   ├── valid.txt                 # Preprocessed validation data
│   ├── test.txt                  # Preprocessed test data
│   ├── train_entity_to_id.tsv    # Entity mappings
│   └── train_relation_to_id.tsv  # Relation mappings
├── models/
│   └── conve/
│       ├── trained_model.pkl     # Trained model
│       ├── config.json           # Training config
│       └── losses.tsv            # Training losses
├── results/
│   ├── mechanistic_paths/
│   │   └── drugmechdb_path_id_results.txt
│   ├── evaluation/
│   │   ├── test_results.json
│   │   └── test_results.csv
│   └── tracin/
│       └── tracin_analysis_*.json
└── logs/
    ├── create_subgraph.log
    ├── prepare_dictionaries.log
    ├── split_data.log
    ├── preprocess_data.log
    ├── train_model.log
    └── evaluate_model.log
```

## Running the Pipeline

### Basic Usage

```bash
# Run complete pipeline with default config
snakemake --cores all

# Run with custom config file
snakemake --cores all --configfile my_config.yaml

# Dry run to see what will execute
snakemake -n
```

### Running Different Styles

To run the pipeline with different filtering styles, just change the `style` parameter in `config.yaml`:

```bash
# Edit config.yaml and set style: "CD"
snakemake --cores all

# Or override from command line
snakemake --cores all --config style=CD
```

This will create outputs in `robokop/CD/` directory.

### Running Multiple Styles in Parallel

To compare different filtering strategies, you can run them in separate sessions:

```bash
# Terminal 1: Run CGGD_alltreat style
snakemake --cores 4 --config style=CGGD_alltreat

# Terminal 2: Run CD style
snakemake --cores 4 --config style=CD

# Terminal 3: Run CGGD style
snakemake --cores 4 --config style=CGGD
```

Each will create its own directory under `robokop/`.

## Common Commands

### View Pipeline DAG

```bash
# Generate pipeline visualization
snakemake --dag | dot -Tpdf > pipeline.pdf
```

### Run Specific Steps

```bash
# Run only up to subgraph creation
snakemake robokop/CGGD_alltreat/rotorobo.txt --cores 1

# Run only up to model training
snakemake robokop/CGGD_alltreat/models/conve/trained_model.pkl --cores all

# Run only evaluation
snakemake robokop/CGGD_alltreat/results/evaluation/test_results.json --cores 1
```

### Clean Commands

```bash
# Remove all outputs for current style (from config.yaml)
snakemake clean

# Remove only models and results (keep preprocessed data)
snakemake clean_models

# Remove only analysis results (keep models and data)
snakemake clean_results

# Remove ALL style directories
snakemake clean_all_styles
```

## Key Parameters

### Essential Parameters (config.yaml)

```yaml
# Graph filtering
style: "CGGD_alltreat"

# Data split
train_ratio: 0.8
valid_ratio: 0.1

# Model architecture
embedding_dim: 200
embedding_height: 10
embedding_width: 20

# Training
num_epochs: 100
batch_size: 256
learning_rate: 0.001
```

### Optional Features

```yaml
# Run old mechanistic path method (slower)
run_old_method: false

# Run TracIn analysis (very computationally expensive)
run_tracin: false

# Use GPU
use_gpu: true
```

## Example Workflows

### Quick Test Run

```yaml
# config.yaml
style: "CGGD_alltreat"
num_epochs: 10
batch_size: 128
run_tracin: false
```

```bash
snakemake --cores all
```

### Production Run

```yaml
# config.yaml
style: "CGGD_alltreat"
num_epochs: 200
batch_size: 512
use_gpu: true
```

```bash
snakemake --cores all
```

### Compare Multiple Filtering Strategies

```bash
# Run pipeline for each style
for style in CGGD_alltreat CD CGGD; do
    snakemake --cores 4 --config style=$style
done

# Compare results
ls -lh robokop/*/results/evaluation/test_results.json
```

## Troubleshooting

### Check Logs

Logs are organized by style:

```bash
# View logs for current style
cat robokop/CGGD_alltreat/logs/train_model.log

# Check all logs
ls robokop/CGGD_alltreat/logs/
```

### Out of Memory

Edit `config.yaml`:
```yaml
batch_size: 128  # Reduce from 256
use_gpu: false   # Use CPU if GPU OOM
```

### Missing Input Files

Ensure your input files exist:
```bash
ls -lh robokop/nodes.jsonl
ls -lh robokop/edges.jsonl
```

Update paths in `config.yaml` if needed.

## Tips

1. **Start with a small test**: Use `num_epochs: 10` for quick testing
2. **Use GPU**: Training is much faster on GPU
3. **Check dry run first**: Always run `snakemake -n` before full execution
4. **Monitor progress**: Check logs in `robokop/{style}/logs/`
5. **Compare styles**: Run different styles to see which works best for your use case

## Getting Help

- Full documentation: See [PIPELINE.md](PIPELINE.md)
- Main README: See [README.md](README.md)
- Check logs: `robokop/{style}/logs/*.log`
