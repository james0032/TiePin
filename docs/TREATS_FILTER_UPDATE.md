# Biolink:Treats Edge Protection Update

## Overview

This document describes the changes made to `create_robokop_subgraph.py` to protect all `biolink:treats` edges and their associated nodes from filtering when using the `clean_baseline` style.

## Changes Made

### 1. New Function: `collect_treats_edge_nodes()`

**Location:** Lines 91-130

**Purpose:** Collects all node IDs (both subject and object) that appear in `biolink:treats` edges.

**How it works:**
- Scans through the entire edges file
- Identifies edges where `predicate == "biolink:treats"`
- Collects both subject and object node IDs from these edges
- Returns a set of unique node IDs that should be protected from low-degree filtering

**Output:**
```
INFO - Collecting nodes from biolink:treats edges...
INFO - Found X biolink:treats edges
INFO - Collected Y unique nodes from treats edges
```

### 2. Updated Function: `compute_low_degree_nodes()`

**Location:** Lines 133-182

**Changes:**
- Added new parameter: `protected_nodes` (optional set of node IDs)
- Modified the low-degree node identification logic to exclude protected nodes
- Updated filtering logic: `if degree < min_degree and node not in protected_nodes`

**Example:**
```python
low_degree_nodes = compute_low_degree_nodes(
    edges_file,
    min_degree=2,
    protected_nodes=treats_nodes  # Nodes from treats edges won't be filtered
)
```

**Output:**
```
INFO - Computing low-degree nodes (degree < 2)...
INFO - Found X nodes with degree < 2 out of Y total nodes
INFO - Protected Z nodes from low-degree filtering
INFO - Low-degree node percentage: XX.XX%
```

### 3. Updated Function: `clean_baseline_kg()`

**Location:** Lines 185-296

**Changes:**

#### a. Added new counter
- `clean_baseline_kg.kept_treats_edges = 0` (line 201)

#### b. Added TOP FILTER (lines 203-210)
```python
# TOP FILTER: Always keep biolink:treats edges
# These edges and their nodes are protected from all filtering
if predicate == "biolink:treats":
    clean_baseline_kg.kept_treats_edges += 1
    clean_baseline_kg.kept_edges += 1
    return False  # Don't filter this edge
```

**Why this is the "top filter":**
- Placed BEFORE all other filtering logic
- Ensures treats edges are NEVER filtered regardless of:
  - Source (even if from filtered sources)
  - Reactome species
  - BindingDB affinity
  - NCIT subclass_of
  - Low-degree nodes
  - Any other filter criteria

### 4. Updated Function: `log_clean_baseline_kg_stats()`

**Location:** Lines 298-339

**Changes:**
- Added reporting line for protected treats edges (line 310):
```python
logger.info(f"  Kept biolink:treats edges (protected): {clean_baseline_kg.kept_treats_edges:,} ...")
```

**Example output:**
```
================================================================================
CLEAN_BASELINE_KG FILTERING STATISTICS
================================================================================
Total edges processed: 1,000,000
  Kept biolink:treats edges (protected): 5,432 (0.54%)
  Filtered by predicate: 234,567 (23.46%)
  Filtered by source: 123,456 (12.35%)
  ...
```

### 5. Updated Function: `create_robokop_input()`

**Location:** Lines 471-475

**Changes:**
```python
if style == "clean_baseline":
    # First collect nodes from biolink:treats edges (protected from filtering)
    treats_nodes = collect_treats_edge_nodes(edges_file)
    # Then compute low-degree nodes, excluding treats nodes
    low_degree_nodes = compute_low_degree_nodes(edges_file, min_degree=min_degree, protected_nodes=treats_nodes)
```

**Workflow:**
1. Collect all node IDs from biolink:treats edges
2. Pass these node IDs to `compute_low_degree_nodes()` as protected
3. Protected nodes won't be included in the low_degree_nodes set
4. Result: Treats edges are kept, and their nodes won't be filtered

## How It Works (Complete Flow)

### Step 1: Collect Treats Nodes
```
edges.jsonl
    ↓
collect_treats_edge_nodes()
    ↓
Set of node IDs from treats edges
```

### Step 2: Compute Low-Degree Nodes (with protection)
```
edges.jsonl + protected_nodes (from Step 1)
    ↓
compute_low_degree_nodes()
    ↓
Set of low-degree nodes (EXCLUDING protected nodes)
```

### Step 3: Filter Edges
```
For each edge in edges.jsonl:
    ↓
clean_baseline_kg()
    ↓
Is predicate == "biolink:treats"?
    YES → KEEP (return False immediately)
    NO  → Continue with other filters
        ↓
    Is subject or object in low_degree_nodes?
        YES → FILTER OUT (return True)
        NO  → Continue with other filters
        ↓
    ... (other filters)
```

## Example Scenarios

### Scenario 1: Treats Edge with Low-Degree Node

**Input:**
```
Edge: CHEMBL:123 --treats--> MONDO:456
Degree(CHEMBL:123) = 1  # Would normally be filtered
Degree(MONDO:456) = 1   # Would normally be filtered
```

**Output:**
- ✅ Edge is KEPT (predicate is biolink:treats)
- ✅ CHEMBL:123 is protected (appears in treats edge)
- ✅ MONDO:456 is protected (appears in treats edge)

### Scenario 2: Non-Treats Edge with Treats Node

**Input:**
```
Edge: CHEMBL:123 --interacts_with--> PUBCHEM:789
Degree(CHEMBL:123) = 1  # Protected (appears in treats edge from Scenario 1)
Degree(PUBCHEM:789) = 1  # NOT protected
```

**Output:**
- ❌ Edge is FILTERED OUT (PUBCHEM:789 is low-degree and not protected)
- Even though CHEMBL:123 is protected, the edge has another low-degree node

### Scenario 3: Treats Edge from Filtered Source

**Input:**
```
Edge: CHEMBL:999 --treats--> MONDO:888
Source: infores:text-mining-provider-targeted  # Normally filtered source
```

**Output:**
- ✅ Edge is KEPT (treats edges bypass ALL filters)
- Even though the source would normally be filtered, treats edges are protected

## Testing Recommendations

### Test 1: Verify Treats Edges Are Kept
```bash
python src/create_robokop_subgraph.py \
    --style clean_baseline \
    --edges-file test_data/edges.jsonl \
    --node-file test_data/nodes.jsonl \
    --min-degree 2 \
    --log-level INFO

# Check output for:
# - "Kept biolink:treats edges (protected): X"
# - Verify X > 0 if treats edges exist in input
```

### Test 2: Verify Treats Nodes Are Protected
```bash
# Extract treats node IDs from input
grep "biolink:treats" edges.jsonl | jq -r '.subject, .object' | sort -u > treats_nodes.txt

# Check output file for these node IDs
grep -f treats_nodes.txt output/rotorobo.txt

# Should find these nodes in output even if they have low degree
```

### Test 3: Check Statistics Output
```bash
# Look for the statistics section in logs
grep -A 20 "CLEAN_BASELINE_KG FILTERING STATISTICS" logs/output.log

# Verify:
# - "Kept biolink:treats edges (protected): X (Y%)"
# - "Protected Z nodes from low-degree filtering"
```

## Benefits

1. **Preserves Important Therapeutic Relationships**
   - All biolink:treats edges are kept regardless of other criteria
   - Critical for drug repurposing and mechanistic pathway analysis

2. **Protects Related Nodes**
   - Drugs and diseases in treats relationships won't be removed for low degree
   - Prevents loss of potentially important entities

3. **Transparent Reporting**
   - Logs show exactly how many treats edges were protected
   - Statistics clearly indicate the impact of protection

4. **Minimal Performance Impact**
   - Single pass through edges to collect treats nodes
   - Set-based lookups are O(1) for protection checks

## Migration Notes

### For Existing Users

No changes needed to existing code or workflows. The new functionality:
- Only affects `clean_baseline` style
- Automatically protects treats edges
- No new command-line arguments required
- Backward compatible with existing configs

### For New Workflows

To use this feature:
```bash
python src/create_robokop_subgraph.py \
    --style clean_baseline \
    --min-degree 2  # or any other value
```

The treats edge protection is **automatic** when using `clean_baseline` style.

## Summary

| Component | Change | Impact |
|-----------|--------|--------|
| `collect_treats_edge_nodes()` | NEW | Identifies nodes to protect |
| `compute_low_degree_nodes()` | MODIFIED | Excludes protected nodes |
| `clean_baseline_kg()` | MODIFIED | Keeps all treats edges |
| `log_clean_baseline_kg_stats()` | MODIFIED | Reports protected edges |
| `create_robokop_input()` | MODIFIED | Orchestrates protection flow |

**Total lines changed:** ~70 lines
**New functions:** 1
**Modified functions:** 4
**Breaking changes:** None
