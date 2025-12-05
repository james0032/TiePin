# Filter Changes to create_robokop_subgraph.py

## Summary

Added four new filters to the `clean_baseline_kg` function to improve knowledge graph quality by removing irrelevant and low-quality edges.

---

## New Filters

### Filter 1: Non-Human Reactome Edges

**Purpose:** Remove reactions from non-human species in Reactome database.

**Logic:**
- Check if subject or object has prefix `REACT:`
- Extract species code from the ID format: `REACT:R-XXX-...` where XXX is the species code
- If species code is not `HSA` (Homo sapiens), remove the edge
- Examples of non-human codes: `DDI` (dog), `PFA` (malaria), `BTA` (cow)

**Code:**
```python
if subject.startswith("REACT:") or obj.startswith("REACT:"):
    # Split by '-' and check if the species code is not "HSA"
    if len(parts) >= 2 and parts[1] != "HSA":
        return True  # Remove edge
```

---

### Filter 2: BindingDB Affinity Filter

**Purpose:** Keep only high-affinity binding interactions from BindingDB.

**Logic:**
- If primary knowledge source is `infores:bindingdb`
- Check the `attributes` field for an attribute with `attribute_type_id == "affinity"`
- **Keep** edge only if affinity is not None AND > 7
- **Remove** edge if:
  - Affinity attribute is missing (None)
  - Affinity value ≤ 7

**Code:**
```python
if source == "infores:bindingdb":
    attributes = edge.get("attributes", [])
    affinity_value = None

    for attr in attributes:
        if attr.get("attribute_type_id") == "affinity":
            affinity_value = attr.get("value")
            break

    if affinity_value is None or affinity_value <= 7:
        return True  # Remove edge
```

**Note:** Higher affinity values (> 7) indicate stronger binding, which are more biologically relevant.

---

### Filter 3: NCIT Subclass Ontology Edges

**Purpose:** Remove internal NCIT ontology hierarchy edges to focus on biological relationships.

**Logic:**
- If predicate is `biolink:subclass_of`
- AND both subject and object have prefix `NCIT:`
- Then remove the edge

**Code:**
```python
if predicate == "biolink:subclass_of":
    if subject.startswith("NCIT:") and obj.startswith("NCIT:"):
        return True  # Remove edge
```

**Rationale:** NCIT subclass relationships are ontological hierarchies that don't represent direct biological interactions. These can create noise in the knowledge graph for drug repurposing tasks.

---

### Filter 4: Low-Degree Nodes

**Purpose:** Remove edges connected to nodes with very few connections (low degree), as they are likely to be poorly connected or less informative.

**Implementation:**

1. **Pre-compute low-degree nodes** (before edge filtering):
   ```python
   def compute_low_degree_nodes(edges_file, min_degree=2):
       """Count degree of each node and identify those with degree < min_degree"""
       degree_count = defaultdict(int)

       # Count occurrences of each node
       for edge in edges_file:
           degree_count[edge["subject"]] += 1
           degree_count[edge["object"]] += 1

       # Return nodes with degree < min_degree
       return {node for node, degree in degree_count.items() if degree < min_degree}
   ```

2. **Filter edges during processing**:
   ```python
   if low_degree_nodes is not None:
       if subject in low_degree_nodes or obj in low_degree_nodes:
           return True  # Remove edge
   ```

**Parameters:**
- Default `min_degree = 2` (removes nodes with degree 0 or 1)
- Configurable via `--min-degree` command-line argument

**Rationale:** Nodes with very few connections are often:
- Data quality issues (orphaned nodes)
- Very specific entities with limited biological relevance
- Noise that doesn't contribute to meaningful graph structure

---

## Updated Statistics Logging

The `log_clean_baseline_kg_stats()` function now tracks and reports:

```
Total edges processed: X
  Filtered by predicate: X (X%)
  Filtered by source: X (X%)
  Filtered by Reactome (non-human): X (X%)       ← NEW
  Filtered by BindingDB (affinity): X (X%)      ← NEW
  Filtered by NCIT subclass_of: X (X%)          ← NEW
  Filtered by low degree: X (X%)                ← NEW
  Kept edges: X (X%)
```

---

## Command-Line Usage

### Basic Usage (default min_degree=2)
```bash
python create_robokop_subgraph.py --style clean_baseline
```

### Custom Minimum Degree Threshold
```bash
# Remove nodes with degree < 3
python create_robokop_subgraph.py --style clean_baseline --min-degree 3

# Remove nodes with degree < 5
python create_robokop_subgraph.py --style clean_baseline --min-degree 5

# Don't filter by degree (set to 0)
python create_robokop_subgraph.py --style clean_baseline --min-degree 0
```

### Full Example with All Options
```bash
python create_robokop_subgraph.py \
    --style clean_baseline \
    --node-file robokop/nodes.jsonl \
    --edges-file robokop/edges.jsonl \
    --outdir output/clean_baseline_deg3 \
    --min-degree 3 \
    --log-level INFO
```

---

## Code Changes Summary

### New Functions
1. `compute_low_degree_nodes(edges_file, min_degree=2)` - Pre-computes set of low-degree nodes

### Modified Functions
1. `clean_baseline_kg(edge, typemap, low_degree_nodes=None)` - Added 4 new filters
2. `log_clean_baseline_kg_stats()` - Added statistics for new filters
3. `create_robokop_input(...)` - Added `min_degree` parameter and low-degree node computation
4. `parse_arguments()` - Added `--min-degree` argument

### New Imports
```python
import igraph as ig
from collections import defaultdict
```

---

## Performance Considerations

### Two-Pass Processing for clean_baseline Style

When using `--style clean_baseline`, the script now makes **two passes** through the data:

1. **First pass** (degree computation): Read all edges to count node degrees
2. **Second pass** (filtering): Apply all filters including low-degree filter

**Impact:**
- Slightly longer processing time (~2x slower for very large graphs)
- Better memory efficiency (only stores node degree counts, not full graph)
- More comprehensive filtering results

**Optimization:** The degree computation is optimized with:
- Dictionary-based counting (O(1) lookup)
- Progress logging every 100K edges
- Set-based storage of low-degree nodes for O(1) membership testing

---

## Filter Order and Priority

Filters are applied in this order (early filters can short-circuit later ones):

1. ✅ Reactome species filter
2. ✅ BindingDB affinity filter
3. ✅ NCIT subclass filter
4. ✅ Low-degree node filter
5. ✅ Predicate filter (original)
6. ✅ Source filter (original)

**Rationale:** More specific filters (1-4) are checked first to avoid unnecessary computation for edges that will be filtered anyway.

---

## Testing Recommendations

### Test Filter 1 (Reactome)
```bash
# Search for REACT edges in output to verify only HSA remain
grep "REACT:" output/clean_baseline/rotorobo.txt | head -20
```

### Test Filter 2 (BindingDB)
```bash
# Count BindingDB edges (should only be high-affinity)
grep "infores:bindingdb" logs/clean_baseline.log
```

### Test Filter 3 (NCIT subclass)
```bash
# Verify no NCIT-NCIT subclass edges remain
# (Check in logs that some were filtered)
grep "NCIT subclass" logs/clean_baseline.log
```

### Test Filter 4 (Low degree)
```bash
# Check statistics for number of low-degree nodes filtered
grep "low degree" logs/clean_baseline.log
```

---

## Example Output Log

```
================================================================================
CLEAN_BASELINE_KG FILTERING STATISTICS
================================================================================
Total edges processed: 10,000,000
  Filtered by predicate: 1,234,567 (12.35%)
  Filtered by source: 456,789 (4.57%)
  Filtered by Reactome (non-human): 123,456 (1.23%)
  Filtered by BindingDB (affinity): 78,901 (0.79%)
  Filtered by NCIT subclass_of: 45,678 (0.46%)
  Filtered by low degree: 2,345,678 (23.46%)
  Kept edges: 5,715,931 (57.16%)
```

---

## Notes

1. **Filter Independence:** Each filter is independent and tracks its own statistics
2. **Cumulative Filtering:** An edge removed by one filter won't be counted by subsequent filters
3. **Degree Calculation:** Node degree is calculated from the **original** edge file before any filtering
4. **Memory Efficiency:** Low-degree nodes are stored as a set (not the full graph) for O(1) lookup
5. **Backward Compatibility:** Other styles (CGGD, CCGGDD, etc.) are unaffected by these changes
