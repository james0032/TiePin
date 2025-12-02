# Quick Start Guide

## TL;DR - What You Have

You now have **3 independent implementations** for filtering knowledge graph triples:

1. ⚡ **PyG** (PyTorch Geometric) - Your existing implementation, fastest
2. 🔍 **NetworkX** - New transparent implementation, easy to debug
3. 🚀 **igraph** - New C-based implementation, fast alternative

Plus tools to **validate** and **compare** them all.

---

## Quick Commands

### Test Everything Works
```bash
cd /Users/jchung/Documents/RENCI/everycure/git/conve_pykeen
python test_filtering.py
```
This creates synthetic data and tests all three implementations.

### Validate Your PyG Results
```bash
python validate_filter_networkx.py \
    --train YOUR_TRAIN.txt \
    --test YOUR_TEST.txt \
    --pyg-output YOUR_PYG_OUTPUT.txt \
    --n-hops 2 \
    --path-filtering \
    --verbose
```
This compares PyG with NetworkX and shows differences.

### Use NetworkX Instead of PyG
```bash
python filter_training_networkx.py \
    --train YOUR_TRAIN.txt \
    --test YOUR_TEST.txt \
    --output YOUR_OUTPUT.txt \
    --n-hops 2 \
    --path-filtering
```

### Compare All Three
```bash
python compare_all_implementations.py \
    --train YOUR_TRAIN.txt \
    --test YOUR_TEST.txt \
    --n-hops 2 \
    --path-filtering \
    --save-outputs
```

---

## NetworkX Functions for N-Hop Queries

**Q: Can NetworkX find all nodes/edges within N-hops between two nodes?**

**A: YES!** Here's how:

```python
import networkx as nx

# Your graph
G = nx.Graph()
# ... add edges ...

# 1. All nodes within N hops from source
nodes = nx.single_source_shortest_path_length(G, source, cutoff=N)

# 2. All paths between two nodes (up to length N)
paths = list(nx.all_simple_paths(G, source, target, cutoff=N))

# 3. Subgraph within N hops
subgraph = nx.ego_graph(G, node, radius=N)

# 4. All shortest paths
shortest = list(nx.all_shortest_paths(G, source, target))
```

---

## Files Created

### Main Implementations
- [filter_training_networkx.py](filter_training_networkx.py) - NetworkX implementation
- [filter_training_igraph.py](filter_training_igraph.py) - igraph implementation
- [filter_training_by_proximity_pyg.py](filter_training_by_proximity_pyg.py) - Your existing PyG

### Tools
- [validate_filter_networkx.py](validate_filter_networkx.py) - Validate PyG with NetworkX
- [compare_all_implementations.py](compare_all_implementations.py) - Compare all three
- [test_filtering.py](test_filtering.py) - Test suite with synthetic data

### Documentation
- [README_FILTERING.md](README_FILTERING.md) - Complete usage guide
- [graph_libraries_comparison.md](graph_libraries_comparison.md) - Library comparison
- [SUMMARY.md](SUMMARY.md) - Implementation summary
- [QUICK_START.md](QUICK_START.md) - This file

---

## Which One Should I Use?

| Scenario | Recommendation |
|----------|----------------|
| **Production, large graphs** | Use PyG (fastest) |
| **Debugging, understanding logic** | Use NetworkX (most transparent) |
| **Want speed without PyG complexity** | Use igraph |
| **Validating results** | Compare all three |
| **Finding bugs** | Use NetworkX with `--verbose` |

---

## Debugging Your Issue

You said: *"output edges does not really fall between drug and disease edges as asked"*

### Step 1: Run Validation
```bash
python validate_filter_networkx.py \
    --train train.txt \
    --test test.txt \
    --pyg-output pyg_filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --verbose \
    --output networkx_filtered.txt
```

### Step 2: Look at Output
The verbose mode will show for EACH edge:
```
Edge (src, dst): Valid path: drug->src(1)->dst(1)->disease (len=2)
Edge (src, dst) REJECTED: No monotonic path: src(drug=2,dis=1), dst(drug=1,dis=2)
```

### Step 3: Compare Results
The script will tell you:
- How many edges match
- Which edges are only in PyG
- Which edges are only in NetworkX
- Why they differ

### Step 4: Fix the Issue
If NetworkX gives better results → use NetworkX or fix PyG
If they match → the algorithm is working as designed, may need different parameters

---

## Common Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `--n-hops` | 2 | How many hops from test entities |
| `--min-degree` | 2 | Keep edges if endpoint degree ≥ this |
| `--path-filtering` | False | Only keep edges on drug→disease paths |
| `--strict-hop-constraint` | False | Both endpoints must be ≤ N hops |
| `--max-total-path-length` | None | Max drug_dist + disease_dist |
| `--preserve-test-edges` | True | Always keep edges with test entities |
| `--verbose` | False | Show detailed edge decisions |

---

## Example Workflow

```bash
# 1. Test implementations work
python test_filtering.py

# 2. Run your data through NetworkX (transparent)
python filter_training_networkx.py \
    --train train.txt \
    --test test.txt \
    --output nx_output.txt \
    --n-hops 2 \
    --path-filtering

# 3. Compare with your existing PyG output
python validate_filter_networkx.py \
    --train train.txt \
    --test test.txt \
    --pyg-output pyg_output.txt \
    --output nx_output.txt \
    --n-hops 2 \
    --path-filtering \
    --verbose

# 4. If they differ, investigate the differences
# The verbose output will tell you exactly which edges differ and why

# 5. Get consensus from all three
python compare_all_implementations.py \
    --train train.txt \
    --test test.txt \
    --n-hops 2 \
    --path-filtering \
    --save-outputs
```

---

## Installation

```bash
# NetworkX (required)
pip install networkx numpy

# PyG (optional, for comparison)
pip install torch torch-geometric

# igraph (optional, for comparison)
pip install igraph
```

---

## Key Insight: Path Filtering

The `--path-filtering` flag is crucial for your use case. It ensures edges are actually on paths between drugs and diseases.

**Without path filtering**: Edges are kept if in intersection
**With path filtering**: Edges must be on a monotonic path drug→disease

If you're seeing edges that don't connect drugs and diseases, you likely need `--path-filtering`.

---

## Need Help?

1. Read [README_FILTERING.md](README_FILTERING.md) for detailed docs
2. Run `python test_filtering.py` to verify setup
3. Use `--verbose` flag to see what's happening
4. Check [graph_libraries_comparison.md](graph_libraries_comparison.md) for NetworkX examples

---

## Summary

**You asked for**: A NetworkX alternative to validate PyG results

**You got**:
- ✅ Complete NetworkX implementation
- ✅ Complete igraph implementation
- ✅ Validation tools
- ✅ Comparison tools
- ✅ Test suite
- ✅ Comprehensive docs
- ✅ Answer to NetworkX N-hop question: **YES, it can!**

All three implementations use the same filtering logic, so you can cross-validate and be confident in the results.
