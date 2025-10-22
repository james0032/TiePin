# TracIn Quick Start Guide

## Find Test Triple Index

To find the row number (0-based) for a specific triple in your test set:

```bash
# Example: Find "CHEBI:34911  predicate:28  MONDO:0004525"
grep -n "CHEBI:34911[[:space:]]predicate:28[[:space:]]MONDO:0004525" test.txt

# Output: 857:CHEBI:34911	predicate:28	MONDO:0004525
# Line 857 = Index 856 (0-based)
```

## Basic Command Template

```bash
python run_tracin.py \
  --model-path <path_to_model.pt> \
  --train <train_triples.txt> \
  --test <test_triples.txt> \
  --entity-to-id <entity_to_id.tsv> \
  --relation-to-id <relation_to_id.tsv> \
  --output <output.json> \
  --mode single \
  --test-indices <index> \
  --top-k 20 \
  --device cuda \
  --batch-size 512 \
  --learning-rate 0.001
```

## Real Example (Permethrin-Scabies)

```bash
python run_tracin.py \
  --model-path examples/conve_model.pt \
  --train examples/train.txt \
  --test examples/test.txt \
  --entity-to-id examples/entity_to_id.tsv \
  --relation-to-id examples/relation_to_id.tsv \
  --edge-map examples/edge_map.json \
  --node-name-dict examples/node_name_dict.txt \
  --output results/permethrin_scabies_856.json \
  --mode single \
  --test-indices 856 \
  --top-k 50 \
  --device cuda \
  --batch-size 1024 \
  --learning-rate 0.001
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | test | `single`, `test`, or `self` |
| `--test-indices` | [0] | Test triple indices to analyze (single mode) |
| `--top-k` | 10 | Number of top influential triples to return |
| `--batch-size` | 256 | Training batch size (larger=faster on GPU) |
| `--device` | cpu | `cpu` or `cuda` |
| `--max-test-triples` | None | Limit number of test triples (for speed) |
| `--output-per-triple` | False | Save separate JSON per test triple |

## Batch Size Recommendations

| Hardware | Recommended Batch Size |
|----------|----------------------|
| CPU | 64-128 |
| GPU 8GB | 256-512 |
| GPU 16GB | 512-1024 |
| GPU 24GB+ | 1024-2048 |

## Performance Estimates

With 16M training edges on GPU:
- **batch_size=256**: ~10-15 hours per test triple
- **batch_size=512**: ~5-10 hours per test triple
- **batch_size=1024**: ~2-5 hours per test triple

## Common Issues

### Out of Memory
```bash
# Reduce batch size
--batch-size 128
```

### Slow Performance
```bash
# Use GPU and larger batch
--device cuda --batch-size 1024
```

### Wrong Test Triple
```bash
# Double-check index with grep
grep -n "YOUR_TRIPLE_PATTERN" test.txt
```

## Output Format

Single mode produces JSON like:
```json
[
  {
    "test_triple": [1446771, 21, 72754],
    "test_triple_index": 856,
    "test_head": 1446771,
    "test_head_label": "CHEBI:34911",
    "test_head_name": "permethrin",
    "test_relation": 21,
    "test_relation_label": "predicate:28",
    "test_relation_name": "treats",
    "test_tail": 72754,
    "test_tail_label": "MONDO:0004525",
    "test_tail_name": "scabies",
    "self_influence": 0.12345,
    "influences": [
      {
        "train_head": 123,
        "train_relation": 5,
        "train_tail": 456,
        "influence": 0.98765,
        "train_head_label": "CHEBI:...",
        "train_relation_label": "predicate:...",
        "train_tail_label": "MONDO:..."
      }
    ]
  }
]
```

## Testing Before Full Run

Test with a small subset first:
```bash
# Limit to first 100 training triples (edit train.txt temporarily)
head -100 train.txt > train_small.txt

python run_tracin.py \
  --train train_small.txt \
  --test test.txt \
  ... \
  --test-indices 856 \
  --batch-size 50
```

## Monitor Progress

Use `watch` to monitor GPU usage:
```bash
# In another terminal
watch -n 1 nvidia-smi
```

Look for:
- GPU utilization near 100%
- Memory usage stable
- No error messages
