# Proximity Filter Algorithm Analysis

## Date: 2025-11-14

## Current Algorithm: UNION of N-Hop Neighborhoods

### What It Does

The current `filter_training_by_proximity_pyg.py` uses a **UNION** approach:

```python
# Lines 429-436: Collect ALL test entities
test_entities = set()
for h, r, t in test_triples:
    test_entities.add(int(h))  # Add all drugs
    test_entities.add(int(t))  # Add all diseases

# Lines 444-450: Get n-hop neighborhood around ALL entities at once
subset_nodes, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=test_entity_list,  # Pass ALL drugs and diseases together
    num_hops=n_hops,
    edge_index=self.edge_index,
    ...
)
```

### Visualization

With n_hops=2, the current algorithm includes:

```
Drug1 ----> X ----> Y          Z <---- W <---- Disease1
       (1 hop)  (2 hops)  (no path)  (2 hops) (1 hop)
```

**Nodes included**:
- Drug1 (test entity)
- X (1 hop from Drug1)
- Y (2 hops from Drug1)
- Z (unconnected, but 2 hops from Disease1)
- W (1 hop from Disease1)
- Disease1 (test entity)

**Problem**: Z is included even though there's NO path from Drug1 to Disease1 through Z.

### Why This Is Wrong for Drug-Disease Paths

The UNION approach includes:
1. ✓ Nodes near drugs
2. ✓ Nodes near diseases
3. ✗ Nodes that are NOT on paths connecting drugs to diseases

This defeats the purpose of proximity filtering for drug repurposing, where we want to:
- Keep edges that could form mechanistic paths from drugs to diseases
- Remove irrelevant edges that are far from both drugs AND diseases

## Desired Algorithm: INTERSECTION of N-Hop Neighborhoods

### What It Should Do

Use an **INTERSECTION** approach to focus on potential paths:

1. Compute n-hop neighborhoods from **drugs** (heads)
2. Compute n-hop neighborhoods from **diseases** (tails)
3. Keep only edges where **BOTH** endpoints are reachable from BOTH a drug AND a disease

### Visualization

```
Drug1 ----> X ----> M <---- W <---- Disease1
       (1 hop)  (2 hops)  (2 hops) (1 hop)
```

**With intersection approach**:
- X is 1 hop from Drug1, and also reachable from Disease1 (3 hops)
- M is 2 hops from Drug1, and 2 hops from Disease1
- W is 3 hops from Drug1, and 1 hop from Disease1
- Edge (X, M) is kept because both X and M are within n_hops from drugs AND diseases
- Edge (M, W) is kept because both M and W are within n_hops from drugs AND diseases

**Nodes excluded**:
- Y that's 2 hops from Drug1 but far from any disease (>n_hops)
- Z that's 2 hops from Disease1 but far from any drug (>n_hops)

## Implementation Strategy

### Option 1: Separate Head/Tail Neighborhoods (Recommended)

```python
# Separate drug and disease entities
drug_entities = set(test_triples[:, 0])  # heads
disease_entities = set(test_triples[:, 2])  # tails

# Get n-hop neighborhoods separately
drug_neighborhood = k_hop_subgraph(node_idx=list(drug_entities), num_hops=n_hops, ...)
disease_neighborhood = k_hop_subgraph(node_idx=list(disease_entities), num_hops=n_hops, ...)

# Convert to sets of nodes
drug_nodes = set(drug_neighborhood[0].tolist())
disease_nodes = set(disease_neighborhood[0].tolist())

# INTERSECTION: Keep edges where BOTH endpoints are in both neighborhoods
for edge in edges:
    src, dst = edge
    if (src in drug_nodes and src in disease_nodes and
        dst in drug_nodes and dst in disease_nodes):
        keep_edge(edge)
```

### Option 2: Compute Distances from Each Set

More precise but slower:

```python
# Compute shortest distances from drugs to all nodes
drug_distances = compute_hop_distances(drug_entities, max_hops=n_hops)

# Compute shortest distances from diseases to all nodes
disease_distances = compute_hop_distances(disease_entities, max_hops=n_hops)

# Keep edges where BOTH endpoints are within n_hops of BOTH drug and disease sets
for edge in edges:
    src, dst = edge
    if (drug_distances[src] <= n_hops and disease_distances[src] <= n_hops and
        drug_distances[dst] <= n_hops and disease_distances[dst] <= n_hops):
        keep_edge(edge)
```

## Expected Impact

### Before (UNION)
- Includes ~50-70% of training edges (depends on n_hops)
- Many irrelevant edges far from drug-disease paths

### After (INTERSECTION)
- Includes ~20-40% of training edges (more selective)
- Focuses on edges that could participate in drug-disease paths
- Better signal-to-noise ratio for drug repurposing

### Example with n_hops=2

**UNION approach**:
- Nodes reachable: Drug nodes ∪ Disease nodes ∪ (nodes within 2 hops of either)
- Result: Very large subgraph

**INTERSECTION approach**:
- Nodes reachable: (nodes within 2 hops of drugs) ∩ (nodes within 2 hops of diseases)
- Result: Only nodes that could be on paths between drugs and diseases
- Much more focused on relevant mechanistic paths

## Biological Motivation

For drug repurposing, we care about **mechanistic paths**:

```
Drug --> Target --> Pathway --> Phenotype --> Disease
```

**UNION approach problems**:
- Includes proteins far from diseases (only near drugs)
- Includes phenotypes far from drugs (only near diseases)
- Dilutes the signal with irrelevant biology

**INTERSECTION approach benefits**:
- Focuses on intermediate nodes (targets, pathways, phenotypes) that connect drugs and diseases
- Keeps edges that participate in complete drug → disease paths
- Better aligns with mechanistic reasoning

## Next Steps

1. Implement Option 1 (separate neighborhoods + intersection)
2. Add flag `--use-intersection` to preserve backward compatibility
3. Update default to use intersection for drug-disease filtering
4. Compare filtering statistics before/after
5. Update documentation

## Related Files

- [filter_training_by_proximity_pyg.py](filter_training_by_proximity_pyg.py) - Current implementation
- Lines 402-508: `filter_for_multiple_test_triples()` method
- Lines 429-436: Test entity collection (union)
- Lines 444-450: k_hop_subgraph call (operates on union)
