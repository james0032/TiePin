# Python Graph Libraries for N-Hop Path Finding

## Overview
This document compares Python graph libraries that can be used to find all nodes and edges within N-hops between two designated nodes, as an alternative or validation for the PyTorch Geometric (PyG) implementation.

---

## 1. **NetworkX** (Recommended for Validation)

**Installation:** `pip install networkx`

**Pros:**
- Pure Python, easy to understand and debug
- Excellent documentation
- Rich set of graph algorithms
- No compilation required
- Perfect for validation and prototyping

**Cons:**
- Slower than compiled libraries (C++/Rust backends)
- Higher memory usage for large graphs

### Key Functions for N-Hop Path Finding:

```python
import networkx as nx

# Create graph
G = nx.Graph()

# 1. Get all nodes within N hops from a node
def get_n_hop_neighborhood(G, node, n_hops):
    """Get all nodes within n hops of a node."""
    return nx.single_source_shortest_path_length(G, node, cutoff=n_hops)

# 2. Get all shortest paths between two nodes
def get_all_shortest_paths(G, source, target):
    """Get all shortest paths."""
    return list(nx.all_shortest_paths(G, source, target))

# 3. Get all simple paths up to length N
def get_all_paths_up_to_n(G, source, target, max_length):
    """Get all simple paths up to max_length."""
    return list(nx.all_simple_paths(G, source, target, cutoff=max_length))

# 4. Get N-hop ego graph (subgraph within N hops)
def get_ego_graph(G, node, radius):
    """Get subgraph within radius hops of node."""
    return nx.ego_graph(G, node, radius=radius)

# 5. Check path existence and length
def check_path(G, source, target):
    """Check if path exists and get its length."""
    if nx.has_path(G, source, target):
        return nx.shortest_path_length(G, source, target)
    return None

# 6. Get nodes at exact distance N
def get_nodes_at_distance_n(G, node, distance):
    """Get nodes exactly N hops away."""
    # NetworkX doesn't have a direct function, but you can compute:
    all_distances = nx.single_source_shortest_path_length(G, node)
    return [n for n, d in all_distances.items() if d == distance]
```

### Example: Find all edges on paths between drug and disease

```python
import networkx as nx

def find_edges_on_drug_disease_paths(G, drug_nodes, disease_nodes, max_hops):
    """Find all edges on paths between drugs and diseases within max_hops."""

    edges_on_paths = set()

    for drug in drug_nodes:
        for disease in disease_nodes:
            # Check if path exists
            if not nx.has_path(G, drug, disease):
                continue

            # Get shortest path length
            path_length = nx.shortest_path_length(G, drug, disease)

            # Skip if too long
            if path_length > max_hops * 2:
                continue

            # Get all simple paths up to max length
            paths = nx.all_simple_paths(G, drug, disease, cutoff=max_hops*2)

            for path in paths:
                # Extract edges from path
                for i in range(len(path) - 1):
                    edges_on_paths.add((path[i], path[i+1]))

    return edges_on_paths
```

---

## 2. **igraph** (Fast Alternative)

**Installation:** `pip install igraph`

**Pros:**
- C core with Python bindings (very fast)
- Memory efficient
- Excellent for large-scale graphs
- Well-documented
- Good community support

**Cons:**
- Slightly different API from NetworkX
- Requires compilation (may have installation issues)

### Key Functions:

```python
import igraph as ig

# Create graph
g = ig.Graph()

# 1. Get shortest paths
def get_shortest_paths(g, source, target):
    """Get all shortest paths between source and target."""
    return g.get_all_shortest_paths(source, target)

# 2. Get neighborhood within N hops
def get_n_hop_neighborhood(g, vertex, order):
    """Get vertices within 'order' hops."""
    return g.neighborhood(vertex, order=order)

# 3. Get subgraph within N hops
def get_subgraph_n_hops(g, vertex, order):
    """Get subgraph induced by N-hop neighborhood."""
    vertices = g.neighborhood(vertex, order=order)
    return g.subgraph(vertices)

# 4. Get all simple paths (with length constraint)
def get_all_simple_paths(g, source, target, cutoff):
    """Get all simple paths up to cutoff length."""
    # Note: igraph doesn't have built-in all_simple_paths with cutoff
    # You need to implement using BFS or DFS
    pass

# 5. Calculate distances
def get_distances(g, source, target):
    """Get shortest path distance."""
    return g.shortest_paths(source, target)[0][0]

# Example: Convert from NetworkX
def networkx_to_igraph(nx_graph):
    """Convert NetworkX graph to igraph."""
    g = ig.Graph.from_networkx(nx_graph)
    return g
```

---

## 3. **graph-tool** (Highest Performance)

**Installation:** `conda install -c conda-forge graph-tool` (conda only, complex pip install)

**Pros:**
- C++ core with Python bindings (fastest)
- Extremely memory efficient
- Best performance for very large graphs (millions of nodes)
- OpenMP parallelization support

**Cons:**
- Difficult installation (conda recommended)
- Steeper learning curve
- Less intuitive API
- Smaller community than NetworkX

### Key Functions:

```python
import graph_tool.all as gt

# Create graph
g = gt.Graph(directed=False)

# 1. Get shortest distance
def get_shortest_distance(g, source, target):
    """Get shortest path distance."""
    dist = gt.shortest_distance(g, source, target)
    return dist

# 2. Get shortest path
def get_shortest_path(g, source, target):
    """Get shortest path as list of vertices."""
    vlist, elist = gt.shortest_path(g, source, target)
    return vlist

# 3. Get distances from source to all nodes
def get_all_distances_from_source(g, source, max_dist):
    """Get distances from source to all reachable nodes."""
    dist_map = gt.shortest_distance(g, source, max_dist=max_dist)
    return dist_map

# 4. Pseudo-diameter (useful for graph analysis)
def get_pseudo_diameter(g):
    """Get pseudo-diameter of graph."""
    diameter, ends = gt.pseudo_diameter(g)
    return diameter
```

---

## 4. **rustworkx** (Rust-based, PyG's alternative)

**Installation:** `pip install rustworkx`

**Pros:**
- Rust backend (very fast, memory safe)
- Similar performance to igraph
- Easier installation than graph-tool
- Good integration with quantum computing libraries (Qiskit)

**Cons:**
- Smaller community
- Less mature than NetworkX/igraph
- API still evolving

### Key Functions:

```python
import rustworkx as rx

# Create graph
g = rx.PyGraph()

# 1. Get shortest path
def get_shortest_path(g, source, target):
    """Get shortest path."""
    return rx.dijkstra_shortest_paths(g, source, target)

# 2. Get distances from source
def get_distances_from_source(g, source):
    """Get distances from source to all nodes."""
    return rx.dijkstra_shortest_path_lengths(g, source)

# 3. Get all shortest paths
def get_all_shortest_paths(g, source, target):
    """Get all shortest paths."""
    return rx.all_shortest_paths(g, source, target)

# 4. BFS search
def bfs_search(g, source, max_depth):
    """BFS traversal with max depth."""
    # Use bfs_successors with depth limit
    pass
```

---

## 5. **Comparison Table**

| Library | Speed | Memory | Installation | Learning Curve | N-hop Support | Best For |
|---------|-------|--------|--------------|----------------|---------------|----------|
| **NetworkX** | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ | Validation, prototyping, small graphs |
| **igraph** | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | Medium-large graphs, production |
| **graph-tool** | ★★★★★ | ★★★★★ | ★☆☆☆☆ | ★★☆☆☆ | ★★★☆☆ | Very large graphs, research |
| **rustworkx** | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | Modern Python projects |
| **PyTorch Geometric** | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | Deep learning, GPU acceleration |

---

## 6. **Recommendation for Your Use Case**

### For Validation (Priority 1):
**Use NetworkX** - It's already implemented in `validate_filter_networkx.py`

**Why:**
- Transparent, easy to debug
- Rich API for path analysis
- Can easily compare results with PyG
- Good for understanding what's going wrong

### For Production Alternative (Priority 2):
**Use igraph**

**Why:**
- Fast enough for your graph size
- Easy installation
- Good balance of speed and ease of use
- Can handle larger graphs if needed

### Example igraph implementation:

```python
import igraph as ig

def filter_with_igraph(train_triples, test_triples, n_hops):
    """Filter using igraph."""

    # Build igraph graph
    edges = [(int(h), int(t)) for h, r, t in train_triples]
    g = ig.Graph(edges=edges, directed=False)

    # Get drug and disease nodes
    drug_nodes = [int(h) for h, r, t in test_triples]
    disease_nodes = [int(t) for h, r, t in test_triples]

    # Get N-hop neighborhoods
    drug_neighborhoods = set()
    for drug in drug_nodes:
        drug_neighborhoods.update(g.neighborhood(drug, order=n_hops))

    disease_neighborhoods = set()
    for disease in disease_nodes:
        disease_neighborhoods.update(g.neighborhood(disease, order=n_hops))

    # Find intersection
    intersection = drug_neighborhoods & disease_neighborhoods

    # Get subgraph
    subgraph = g.subgraph(list(intersection))

    return subgraph
```

---

## 7. **Specific Function: All Paths Between Two Node Sets**

Here's a NetworkX function to find ALL paths between drug and disease nodes:

```python
def find_all_paths_between_sets(G, source_set, target_set, max_length):
    """
    Find all simple paths between any node in source_set and any node in target_set.

    Args:
        G: NetworkX graph
        source_set: Set of source nodes (e.g., drugs)
        target_set: Set of target nodes (e.g., diseases)
        max_length: Maximum path length to consider

    Returns:
        Dictionary mapping (source, target) -> list of paths
    """
    all_paths = {}

    for source in source_set:
        for target in target_set:
            if source == target:
                continue

            try:
                # Get all simple paths up to max_length
                paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))

                if paths:
                    all_paths[(source, target)] = paths

            except nx.NetworkXNoPath:
                continue

    return all_paths


def extract_edges_from_paths(all_paths):
    """Extract all unique edges that appear in any path."""
    edges_on_paths = set()

    for (source, target), paths in all_paths.items():
        for path in paths:
            # Extract edges from this path
            for i in range(len(path) - 1):
                edges_on_paths.add((min(path[i], path[i+1]), max(path[i], path[i+1])))

    return edges_on_paths
```

---

## 8. **Performance Expectations**

For a knowledge graph with ~100K triples:

| Library | Graph Build | N-hop Query (single) | N-hop Query (batch) | Path Enumeration |
|---------|-------------|---------------------|---------------------|------------------|
| NetworkX | 1-2s | 10-50ms | 0.5-2s | Slow (seconds-minutes) |
| igraph | 0.5-1s | 5-20ms | 0.2-1s | Fast (milliseconds-seconds) |
| graph-tool | 0.3-0.7s | 2-10ms | 0.1-0.5s | Very fast (milliseconds) |
| PyG | 0.5-1s | 5-15ms | 0.2-0.8s | Fast (milliseconds-seconds) |

---

## 9. **Next Steps**

1. **Use the provided `validate_filter_networkx.py`** to validate your PyG results
2. **If NetworkX is too slow**, consider implementing with **igraph**
3. **If you need path enumeration**, NetworkX's `all_simple_paths` is the easiest option
4. **For production**, stick with PyG but validate critical logic with NetworkX

---

## 10. **Testing the Validation Script**

```bash
# Run NetworkX validation
python validate_filter_networkx.py \
    --train train.txt \
    --test test.txt \
    --output train_filtered_nx.txt \
    --pyg-output train_filtered_pyg.txt \
    --n-hops 2 \
    --min-degree 2 \
    --path-filtering \
    --verbose

# This will:
# 1. Filter using NetworkX
# 2. Load PyG results
# 3. Compare and show differences
# 4. Print detailed diagnostics
```
