# How to Set --preserve-test-edges Flag

## Overview

The `--preserve-test-edges` flag controls whether edges containing test entities are always kept, even if they fail degree filtering.

---

## Command Line Usage

### Option 1: Preserve Test Edges (Default)

```bash
# Explicit - always keep edges with test entities
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --preserve-test-edges

# Or omit (default behavior)
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt
```

**Behavior**: Edges containing test head or tail are ALWAYS kept, regardless of degree

### Option 2: Don't Preserve (Strict Filtering)

```bash
# Apply strict degree filtering to ALL edges
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --no-preserve-test-edges      # ← Use this flag
```

**Behavior**: ALL edges (including test edges) must pass degree threshold

---

## Python API Usage

### Option 1: Preserve Test Edges (Default)

```python
from filter_training_by_proximity_pyg import ProximityFilterPyG

filter_obj = ProximityFilterPyG(training_triples)

# Explicit
filtered = filter_obj.filter_for_single_test_triple(
    test_triple=(1, 0, 2),
    n_hops=2,
    min_degree=2,
    preserve_test_entity_edges=True  # Default
)

# Or omit (same behavior)
filtered = filter_obj.filter_for_single_test_triple(
    test_triple=(1, 0, 2),
    n_hops=2,
    min_degree=2
)
```

### Option 2: Don't Preserve (Strict Filtering)

```python
filtered = filter_obj.filter_for_single_test_triple(
    test_triple=(1, 0, 2),
    n_hops=2,
    min_degree=2,
    preserve_test_entity_edges=False  # Strict filtering
)
```

---

## Comparison

### Example Graph

```
Test triple: (1, r, 2)

    0 ───── 1 ───── 2 ───── 3
    │       │       │       │
    4       5       6       7
  (deg=1) (deg=2) (deg=2) (deg=1)
```

### With `--preserve-test-edges` (Default)

```bash
python filter...py --preserve-test-edges --min-degree 2
```

**Kept edges**:
- ✓ (0, 1) - Entity 1 is in test triple
- ✓ (1, 2) - Both entities in test triple
- ✓ (2, 3) - Entity 2 is in test triple
- ✓ (1, 5) - Entity 1 is in test triple
- ✓ (2, 6) - Entity 2 is in test triple
- ✗ (0, 4) - Neither entity in test triple, node 4 has degree 1
- ✗ (3, 7) - Neither entity in test triple, node 7 has degree 1

**Result**: 5 edges kept (preserves direct influences)

### With `--no-preserve-test-edges` (Strict)

```bash
python filter...py --no-preserve-test-edges --min-degree 2
```

**Kept edges**:
- ✗ (0, 1) - Node 0 has degree < 2 (even though 1 is in test)
- ✓ (1, 2) - Both nodes have degree ≥ 2
- ✗ (2, 3) - Node 3 has degree < 2 (even though 2 is in test)
- ✓ (1, 5) - Both nodes have degree ≥ 2
- ✓ (2, 6) - Both nodes have degree ≥ 2
- ✗ (0, 4) - Node 0 and 4 have degree < 2
- ✗ (3, 7) - Node 3 and 7 have degree < 2

**Result**: 3 edges kept (strict degree filtering)

---

## When to Use Each Option

### Use `--preserve-test-edges` (Default) When:

✅ **You want to keep direct influences**
  - Edges directly involving test entities are important
  - Even if they lead to low-degree nodes

✅ **Analyzing relationship patterns**
  - Understanding direct connections matters
  - Example: drug → side_effect relationships

✅ **Conservative filtering**
  - Don't want to remove potentially important edges
  - Better to keep more than remove too much

### Use `--no-preserve-test-edges` When:

✅ **You want pure graph structure filtering**
  - Apply same rules to all edges uniformly
  - No special treatment for test entities

✅ **Very aggressive filtering**
  - Minimize training set as much as possible
  - Only keep well-connected subgraph

✅ **Benchmarking graph structure**
  - Testing impact of graph connectivity
  - Comparing different filtering strategies

---

## Examples

### Example 1: Drug Discovery (Recommended: Preserve)

```bash
# Keep direct drug-disease relationships even if low-degree
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 2 \
    --min-degree 2 \
    --preserve-test-edges  # Keep direct drug effects
```

**Rationale**: Direct drug-disease edges are valuable even if rare

### Example 2: Entity Type Prediction (Optional: Strict)

```bash
# Only keep well-connected entities
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 1 \
    --min-degree 3 \
    --no-preserve-test-edges  # Pure structure-based
```

**Rationale**: Entity types determined by overall connectivity

### Example 3: Experiment Comparison

```bash
# Run 1: With preservation
python filter_training_by_proximity_pyg.py \
    --train train.txt --test test.txt \
    --output filtered_preserve.txt \
    --preserve-test-edges

# Run 2: Without preservation
python filter_training_by_proximity_pyg.py \
    --train train.txt --test test.txt \
    --output filtered_strict.txt \
    --no-preserve-test-edges

# Compare results
wc -l filtered_preserve.txt filtered_strict.txt
```

---

## Python API: Full Control

```python
from filter_training_by_proximity_pyg import ProximityFilterPyG

filter_obj = ProximityFilterPyG(training_triples)

test_triple = (1, 0, 2)

# Conservative (keep more)
filtered_conservative = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=3,
    min_degree=1,
    preserve_test_entity_edges=True
)

# Balanced (recommended)
filtered_balanced = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=2,
    min_degree=2,
    preserve_test_entity_edges=True  # Default
)

# Aggressive (remove more)
filtered_aggressive = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=2,
    min_degree=3,
    preserve_test_entity_edges=False
)

print(f"Conservative: {len(filtered_conservative)} triples")
print(f"Balanced:     {len(filtered_balanced)} triples")
print(f"Aggressive:   {len(filtered_aggressive)} triples")
```

---

## Summary

### Set to False (Strict Filtering)

```bash
# Command line
--no-preserve-test-edges

# Python
preserve_test_entity_edges=False
```

### Set to True (Default, Recommended)

```bash
# Command line
--preserve-test-edges
# Or omit (default)

# Python
preserve_test_entity_edges=True
# Or omit (default)
```

### Quick Reference Table

| Command Line Flag | Python Parameter | Behavior |
|-------------------|------------------|----------|
| `--preserve-test-edges` | `preserve_test_entity_edges=True` | Keep test edges (default) |
| `--no-preserve-test-edges` | `preserve_test_entity_edges=False` | Strict filtering |
| (omitted) | (omitted) | Keep test edges (default) |

**Recommendation**: Use default (`--preserve-test-edges`) unless you have a specific reason to use strict filtering.
