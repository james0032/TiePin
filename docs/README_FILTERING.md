# Knowledge Graph Triple Filtering Implementations

This directory contains multiple implementations for filtering training triples based on N-hop proximity to test triples. This is useful for knowledge graph embedding tasks where you want to focus on relevant subgraphs.

## Overview

We provide **three independent implementations** of the same filtering algorithm:

1. **PyTorch Geometric (PyG)** - Fastest, GPU-compatible, production-ready
2. **NetworkX** - Most transparent, easiest to debug, best for validation
3. **igraph** - Fast C-based alternative, good balance of speed and usability

## Files

### Core Implementations

- **`filter_training_by_proximity_pyg.py`** - PyTorch Geometric implementation (fastest)
- **`filter_training_networkx.py`** - NetworkX implementation (most transparent)
- **`filter_training_igraph.py`** - igraph implementation (C-based, fast)

### Validation & Comparison

- **`validate_filter_networkx.py`** - Validate PyG results using NetworkX
- **`compare_all_implementations.py`** - Run all three and compare results

### Documentation

- **`graph_libraries_comparison.md`** - Detailed comparison of Python graph libraries
- **`README_FILTERING.md`** - This file

## Quick Start

### Installation

```bash
# For PyG implementation (fastest)
pip install torch torch-geometric numpy

# For NetworkX implementation (recommended for validation)
pip install networkx numpy

# For igraph implementation (optional)
pip install igraph numpy
```

### Basic Usage

#### 1. PyTorch Geometric (Production Use)

```bash
python filter_training_by_proximity_pyg.py \
    --train data/train.txt \
    --test data/test.txt \
    --output data/train_filtered.txt \
    --n-hops 2 \
    --min-degree 2 \
    --path-filtering
```

#### 2. NetworkX (Validation & Debugging)

```bash
python filter_training_networkx.py \
    --train data/train.txt \
    --test data/test.txt \
    --output data/train_filtered_nx.txt \
    --n-hops 2 \
    --min-degree 2 \
    --path-filtering
```

#### 3. Compare All Implementations

```bash
python compare_all_implementations.py \
    --train data/train.txt \
    --test data/test.txt \
    --n-hops 2 \
    --path-filtering \
    --implementations pyg networkx igraph \
    --save-outputs
```

## Filtering Modes

### 1. Intersection Mode (Default)

Keeps edges where both endpoints are within N-hops of BOTH drug and disease nodes.

```bash
python filter_training_networkx.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 2
```

### 2. Path Filtering Mode (Strictest)

Only keeps edges that lie on valid paths between drug and disease nodes.

```bash
python filter_training_networkx.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 2 \
    --path-filtering
```

**Path Filtering Logic:**
- An edge (src, dst) is kept if it's on a monotonic path from drug → disease
- Direction 1: drug → src → dst → disease (src closer to drug, dst closer to disease)
- Direction 2: drug → dst → src → disease (dst closer to drug, src closer to disease)

### 3. Strict Hop Constraint

Enforces that BOTH endpoints of each edge are within N-hops from test entities.

```bash
python filter_training_networkx.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 2 \
    --strict-hop-constraint
```

### 4. Maximum Path Length

Limits total path length (drug_dist + disease_dist) when using path filtering.

```bash
python filter_training_networkx.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 2 \
    --path-filtering \
    --max-total-path-length 3
```

## Parameters Explained

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--train` | Required | Path to training triples file (TSV format) |
| `--test` | Required | Path to test triples file (TSV format) |
| `--output` | Required | Path to save filtered training triples |
| `--n-hops` | 2 | Number of hops from test entities |
| `--min-degree` | 2 | Minimum degree threshold for keeping edges |

### Filtering Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--preserve-test-edges` | True | Always keep edges containing test entities |
| `--no-preserve-test-edges` | False | Apply strict filtering to all edges |
| `--strict-hop-constraint` | False | Both endpoints must be within N-hops |
| `--path-filtering` | False | Only keep edges on drug-disease paths |
| `--max-total-path-length` | None | Maximum path length (for path filtering) |

### Other Options

| Parameter | Description |
|-----------|-------------|
| `--single-triple` | Filter for first test triple only (debugging) |
| `--cache` | (PyG only) Cache graph for reuse |
| `--verbose` | (NetworkX validation) Print detailed diagnostics |

## File Format

### Input Format (TSV)

Triples should be in tab-separated format:

```
head_entity    relation    tail_entity
DRUGBANK:DB00001    treats    MESH:D001249
CHEMBL:CHEMBL1234    interacts_with    UNIPROT:P12345
```

### Output Format

Same TSV format as input, containing only filtered triples.

## Use Cases

### Use Case 1: Validate PyG Results

You've run the PyG filter but want to verify it's working correctly:

```bash
# 1. Run PyG filter
python filter_training_by_proximity_pyg.py \
    --train train.txt \
    --test test.txt \
    --output train_pyg.txt \
    --n-hops 2 \
    --path-filtering

# 2. Validate with NetworkX
python validate_filter_networkx.py \
    --train train.txt \
    --test test.txt \
    --pyg-output train_pyg.txt \
    --n-hops 2 \
    --path-filtering \
    --verbose
```

### Use Case 2: Debug Path Filtering

You want to see which edges are being kept/rejected:

```bash
python validate_filter_networkx.py \
    --train train.txt \
    --test test.txt \
    --output train_nx.txt \
    --n-hops 2 \
    --path-filtering \
    --verbose
```

This will print detailed diagnostics for each edge decision.

### Use Case 3: Compare All Implementations

Ensure all three implementations agree:

```bash
python compare_all_implementations.py \
    --train train.txt \
    --test test.txt \
    --n-hops 2 \
    --path-filtering \
    --save-outputs \
    --output-dir comparison_results
```

This will:
- Run all three implementations
- Compare results
- Show timing information
- Save outputs from each
- Flag any discrepancies

### Use Case 4: Production Filtering

For production use with large graphs:

```bash
# Use PyG with caching for best performance
python filter_training_by_proximity_pyg.py \
    --train large_train.txt \
    --test large_test.txt \
    --output large_train_filtered.txt \
    --n-hops 2 \
    --min-degree 2 \
    --path-filtering \
    --cache graph_cache.pkl
```

## Performance Comparison

For a knowledge graph with ~100K triples:

| Implementation | Graph Build | Filtering | Total Time | Memory |
|---------------|-------------|-----------|------------|---------|
| **PyG** | 0.5-1s | 0.2-0.8s | **1-2s** | Medium |
| **NetworkX** | 1-2s | 2-5s | **3-7s** | High |
| **igraph** | 0.5-1s | 0.5-2s | **1-3s** | Low |

### Recommendations by Graph Size

- **Small graphs (<100K triples)**: Use **NetworkX** for transparency
- **Medium graphs (100K-1M triples)**: Use **PyG** or **igraph**
- **Large graphs (>1M triples)**: Use **PyG** with GPU acceleration

## Algorithm Explanation

### High-Level Flow

1. **Build Graph**: Create undirected graph from training triples
2. **Compute Neighborhoods**:
   - Find all nodes within N-hops of drug nodes (test heads)
   - Find all nodes within N-hops of disease nodes (test tails)
3. **Intersection**: Keep only nodes reachable from BOTH drugs and diseases
4. **Filter Edges**:
   - Keep if either endpoint has degree ≥ min_degree
   - OR if edge connects to test entity (if preserve_test_edges=True)
   - AND (if path_filtering) edge is on a valid drug→disease path
5. **Return Triples**: Map filtered edges back to original triples

### Path Filtering Logic

For an edge (src, dst) to be on a valid path:

```
drug_dist[src] ≤ drug_dist[dst] AND disease_dist[dst] ≤ disease_dist[src]
  → src is closer to drug, dst is closer to disease
  → Monotonic progression: drug → src → dst → disease

OR

drug_dist[dst] ≤ drug_dist[src] AND disease_dist[src] ≤ disease_dist[dst]
  → dst is closer to drug, src is closer to disease
  → Monotonic progression: drug → dst → src → disease
```

This ensures edges form part of a shortest path between drugs and diseases.

## Troubleshooting

### Problem: Results don't match between implementations

**Solution**: Run comparison script with verbose output:

```bash
python compare_all_implementations.py \
    --train train.txt \
    --test test.txt \
    --n-hops 2 \
    --path-filtering \
    --save-outputs
```

Check the comparison output to see which triples differ.

### Problem: Too few triples after filtering

**Possible causes**:
1. `--path-filtering` is too strict → Try without path filtering first
2. `--n-hops` is too small → Increase to 3 or 4
3. `--min-degree` is too high → Decrease to 1 or 0

**Debug**:
```bash
python validate_filter_networkx.py \
    --train train.txt \
    --test test.txt \
    --n-hops 2 \
    --verbose
```

Look at rejection reasons in the output.

### Problem: Too many triples after filtering

**Possible causes**:
1. Not using path filtering → Add `--path-filtering`
2. `--n-hops` is too large → Decrease to 1 or 2
3. `--preserve-test-edges` keeping too many → Try `--no-preserve-test-edges`

### Problem: NetworkX too slow

**Solution**: Switch to PyG or igraph:

```bash
# Use igraph (faster than NetworkX, easier than PyG)
python filter_training_igraph.py \
    --train train.txt \
    --test test.txt \
    --output filtered.txt \
    --n-hops 2 \
    --path-filtering
```

## Advanced Topics

### Custom Graph Analysis

You can use the filter classes in your own scripts:

```python
from filter_training_networkx import NetworkXProximityFilter
import numpy as np

# Load your data
train_triples = np.array([[0, 0, 1], [1, 0, 2], ...])  # [head, rel, tail]
test_triples = np.array([[0, 0, 2], ...])

# Create filter
filter_obj = NetworkXProximityFilter(train_triples)

# Access the NetworkX graph directly
graph = filter_obj.graph
print(f"Graph has {graph.number_of_nodes()} nodes")

# Compute custom metrics
import networkx as nx
diameter = nx.diameter(graph) if nx.is_connected(graph) else float('inf')
print(f"Graph diameter: {diameter}")

# Filter triples
filtered = filter_obj.filter_for_multiple_test_triples(
    test_triples,
    n_hops=2,
    path_filtering=True
)
```

### Finding All Paths

To find all paths between drug and disease nodes:

```python
import networkx as nx
from filter_training_networkx import NetworkXProximityFilter

# Create filter and get graph
filter_obj = NetworkXProximityFilter(train_triples)
G = filter_obj.graph

# Find all simple paths up to length 4
drug_node = 123
disease_node = 456

paths = list(nx.all_simple_paths(G, drug_node, disease_node, cutoff=4))
print(f"Found {len(paths)} paths")

for path in paths[:10]:  # Show first 10
    print(f"Path: {' -> '.join(map(str, path))}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{kg_proximity_filter,
  title={Knowledge Graph Proximity Filter},
  author={Your Name},
  year={2024},
  url={https://github.com/yourorg/yourrepo}
}
```

## Contributing

To add a new implementation:

1. Implement the same interface (filter_for_multiple_test_triples)
2. Add to comparison script
3. Test against existing implementations
4. Update this README

## License

[Your License Here]

## Support

For issues or questions:
- Check the troubleshooting section
- Review [graph_libraries_comparison.md](graph_libraries_comparison.md)
- Open an issue on GitHub
