# Make_test.py Binning Analysis - Node Overlap Evaluation

## Executive Summary

**❌ NOT EXHAUSTIVE** - The many-to-many binning strategy is **NOT done in an exhaustive way**. There **IS potential node overlap** between the bins, specifically:

- ✅ **1-to-1 bin**: No overlap (by definition)
- ⚠️ **1-to-N bin**: Can have node overlap with N-to-M bin
- ⚠️ **N-to-1 bin**: Can have node overlap with N-to-M bin
- ❌ **N-to-M bin**: Can share nodes with 1-to-N and N-to-1 bins

## Critical Issue

### The Problem: Incomplete N-to-M Detection

The current logic categorizes edges as follows (lines 166-178):

```python
if subj_count == 1 and obj_count == 1:
    # True 1-to-1 edge
    one_to_one.append(edge)
elif subj_count > 1 and obj_count == 1:
    # N-to-1 pattern (multiple subjects to same object)
    n_to_one_groups[obj].append(edge)
elif subj_count == 1 and obj_count > 1:
    # 1-to-N pattern (same subject to multiple objects)
    one_to_n_groups[subject].append(edge)
else:
    # Both subject and object appear multiple times (N-to-M pattern)
    many_to_many.append(edge)
```

### Issue Explanation

**This categorization is edge-centric, not node-centric.**

An edge is classified as N-to-M **only if BOTH** its subject AND object appear multiple times. However, this **does NOT** ensure that nodes appearing in N-to-M edges are excluded from other bins.

### Concrete Example

Consider this scenario:

```
Edge 1: Drug_A treats Disease_X  (Drug_A appears 2x, Disease_X appears 1x)
Edge 2: Drug_A treats Disease_Y  (Drug_A appears 2x, Disease_Y appears 1x)
Edge 3: Drug_B treats Disease_X  (Drug_B appears 1x, Disease_X appears 1x)
Edge 4: Drug_A treats Disease_Z  (Drug_A appears 2x, Disease_Z appears 2x)
Edge 5: Drug_C treats Disease_Z  (Drug_C appears 1x, Disease_Z appears 2x)
```

**Counts**:
- Drug_A: 3 occurrences (edges 1, 2, 4)
- Drug_B: 1 occurrence (edge 3)
- Drug_C: 1 occurrence (edge 5)
- Disease_X: 2 occurrences (edges 1, 3)
- Disease_Y: 1 occurrence (edge 2)
- Disease_Z: 2 occurrences (edges 4, 5)

**Current Categorization**:

| Edge | Subject Count | Object Count | Category | Reason |
|------|---------------|--------------|----------|---------|
| Edge 1: Drug_A → Disease_X | 3 | 2 | **N-to-1** ❌ | subj>1, obj=2>1 → **WRONG! Should be N-to-M** |
| Edge 2: Drug_A → Disease_Y | 3 | 1 | **1-to-N** | subj=3>1, obj=1 |
| Edge 3: Drug_B → Disease_X | 1 | 2 | **1-to-N** | subj=1, obj=2>1 |
| Edge 4: Drug_A → Disease_Z | 3 | 2 | **N-to-M** ✓ | subj>1, obj>1 |
| Edge 5: Drug_C → Disease_Z | 1 | 2 | **1-to-N** | subj=1, obj>1 |

Wait, let me recalculate with the actual code logic:

**Actual Categorization** (following code):

```python
# Edge 1: Drug_A (count=3) → Disease_X (count=2)
if subj_count > 1 and obj_count == 1:  # 3 > 1 and 2 == 1? NO
elif subj_count == 1 and obj_count > 1:  # 1 == 1 and 2 > 1? NO
else:  # Both > 1
    → N-to-M ✓ CORRECT

# Edge 2: Drug_A (count=3) → Disease_Y (count=1)
if subj_count > 1 and obj_count == 1:  # 3 > 1 and 1 == 1? YES
    → N-to-1 (grouped by object Disease_Y)

# Edge 3: Drug_B (count=1) → Disease_X (count=2)
if subj_count == 1 and obj_count > 1:  # 1 == 1 and 2 > 1? YES
    → 1-to-N (grouped by subject Drug_B)

# Edge 4: Drug_A (count=3) → Disease_Z (count=2)
else:  # subj=3>1, obj=2>1
    → N-to-M ✓ CORRECT

# Edge 5: Drug_C (count=1) → Disease_Z (count=2)
if subj_count == 1 and obj_count > 1:  # 1 == 1 and 2 > 1? YES
    → 1-to-N (grouped by subject Drug_C)
```

**Node Overlap**:

- **Drug_A**: Appears in N-to-1 bin (Edge 2) AND N-to-M bin (Edge 1, Edge 4) ❌
- **Disease_X**: Appears in 1-to-N bin (Edge 3) AND N-to-M bin (Edge 1) ❌
- **Disease_Z**: Appears in 1-to-N bin (Edge 5) AND N-to-M bin (Edge 4) ❌

**Result**: **NODE OVERLAP EXISTS!**

## Detailed Analysis

### What the Code Actually Does

#### Lines 127-129: Count Calculation
```python
# Count occurrences of subjects and objects in treats edges
subject_counts = Counter(edge[0] for edge in treats_edges)
object_counts = Counter(edge[2] for edge in treats_edges)
```

**What this counts**: For each node, how many times it appears as a subject or object in ALL treats edges.

#### Lines 161-178: Edge Categorization
```python
for edge in treats_edges:
    subject, predicate, obj = edge
    subj_count = subject_counts[subject]  # Total occurrences of this subject
    obj_count = object_counts[obj]        # Total occurrences of this object

    if subj_count == 1 and obj_count == 1:
        one_to_one.append(edge)
    elif subj_count > 1 and obj_count == 1:
        n_to_one_groups[obj].append(edge)
    elif subj_count == 1 and obj_count > 1:
        one_to_n_groups[subject].append(edge)
    else:
        # subj_count > 1 AND obj_count > 1
        many_to_many.append(edge)
```

**Key Problem**: An edge is categorized based on its own subject/object counts, but nodes can appear in edges with different categorizations.

### Why Overlap Occurs

#### Scenario 1: Subject in Multiple Categories

```
Drug_A treats Disease_1  (Drug_A count=3, Disease_1 count=1) → N-to-1 bin
Drug_A treats Disease_2  (Drug_A count=3, Disease_2 count=1) → N-to-1 bin
Drug_A treats Disease_3  (Drug_A count=3, Disease_3 count=2) → N-to-M bin
Drug_B treats Disease_3  (Drug_B count=1, Disease_3 count=2) → 1-to-N bin
```

**Node Overlap**:
- Drug_A appears in both N-to-1 and N-to-M bins
- Disease_3 appears in both 1-to-N and N-to-M bins

#### Scenario 2: Object in Multiple Categories

```
Drug_1 treats Disease_X  (Drug_1 count=1, Disease_X count=3) → 1-to-N bin
Drug_2 treats Disease_X  (Drug_2 count=1, Disease_X count=3) → 1-to-N bin
Drug_3 treats Disease_X  (Drug_3 count=2, Disease_X count=3) → N-to-M bin
Drug_3 treats Disease_Y  (Drug_3 count=2, Disease_Y count=1) → N-to-1 bin
```

**Node Overlap**:
- Disease_X appears in both 1-to-N and N-to-M bins
- Drug_3 appears in both N-to-1 and N-to-M bins

### Visual Representation

```
Bins as they currently exist:

┌─────────────┐
│  1-to-1     │  Drug_X → Disease_X
│             │  (Drug_X count=1, Disease_X count=1)
└─────────────┘  ✓ No overlap possible (both nodes appear only once)

┌─────────────┐
│  1-to-N     │  Drug_A → Disease_1, Disease_2, Disease_3
│             │  (Drug_A count=3, each Disease count=1)
│             │  Drug_B → Disease_4 (count=2)
└─────────────┘  ⚠️ Drug_A might also appear in N-to-M bin
                 ⚠️ Disease nodes might appear in N-to-M bin

┌─────────────┐
│  N-to-1     │  Drug_1, Drug_2, Drug_3 → Disease_Y
│             │  (each Drug count=1, Disease_Y count=3)
│             │  Drug_4, Drug_5 → Disease_Z (count=2)
└─────────────┘  ⚠️ Disease_Y might also appear in N-to-M bin
                 ⚠️ Drug nodes might appear in N-to-M bin

┌─────────────┐
│  N-to-M     │  Drug_A → Disease_Z
│             │  (Drug_A count=3, Disease_Z count=2)
│             │  Drug_B → Disease_Y
│             │  (Drug_B count=2, Disease_Y count=3)
└─────────────┘  ❌ Shares Drug_A with 1-to-N bin
                 ❌ Shares Disease_Y with N-to-1 bin
                 ❌ Shares Disease_Z with 1-to-N bin
```

## Is N-to-M Exhaustive?

### Question: Does N-to-M bin contain ALL edges where both nodes participate in multiple relationships?

**Answer**: **NO, it's not exhaustive.**

### What N-to-M bin actually contains:

```python
# N-to-M bin contains edges where:
# - The specific edge's subject appears in >1 treats edge, AND
# - The specific edge's object appears in >1 treats edge
```

### What N-to-M bin SHOULD contain (for true exhaustiveness):

```python
# N-to-M bin should contain ALL edges involving nodes that:
# - Ever appear in an edge where both subject and object have count > 1
# - Form connected components of multi-relationship nodes
```

### Example of Non-Exhaustiveness

```
Edge A: Drug_1 (count=2) → Disease_1 (count=1)  → Categorized as N-to-1
Edge B: Drug_1 (count=2) → Disease_2 (count=2)  → Categorized as N-to-M
Edge C: Drug_2 (count=1) → Disease_2 (count=2)  → Categorized as 1-to-N
```

**Analysis**:
- Edge A is in N-to-1 bin
- Edge B is in N-to-M bin
- Edge C is in 1-to-N bin

But Drug_1 and Disease_2 both appear in Edge B (N-to-M). Therefore:
- Edge A (containing Drug_1) should arguably be in N-to-M bin too
- Edge C (containing Disease_2) should arguably be in N-to-M bin too

**Current behavior**: N-to-M bin only gets Edge B, missing A and C.

## Impact on Train/Test Split

### Leakage Risk

The current sampling strategy (lines 293-320) samples N-to-M edges by subject:

```python
# Group N-to-M edges by subject
nm_by_subject = defaultdict(list)
for edge in many_to_many:
    nm_by_subject[edge[0]].append(edge)

# Shuffle subjects and sample complete groups
nm_subjects = list(nm_by_subject.keys())
random.shuffle(nm_subjects)

sampled_many_to_many = []
for subject in nm_subjects:
    edges = nm_by_subject[subject]
    sampled_many_to_many.extend(edges)  # All edges for this subject
```

**Problem**: This only groups N-to-M edges by their subject. But:

1. **Subject overlap**: If Drug_A appears in both N-to-1 and N-to-M bins:
   - N-to-M sampling might select Drug_A's N-to-M edges for test
   - N-to-1 sampling might select Drug_A's N-to-1 edges for train
   - **LEAKAGE**: Drug_A edges split across train and test

2. **Object overlap**: If Disease_Z appears in both 1-to-N and N-to-M bins:
   - 1-to-N sampling might select edges pointing to Disease_Z for test
   - N-to-M sampling might select different edges pointing to Disease_Z for train
   - **LEAKAGE**: Disease_Z edges split across train and test

### Example Leakage Scenario

```
Edges in dataset:
1. Drug_A treats Disease_1 (Drug_A count=3, Disease_1 count=1) → N-to-1 bin
2. Drug_A treats Disease_2 (Drug_A count=3, Disease_2 count=1) → N-to-1 bin
3. Drug_A treats Disease_3 (Drug_A count=3, Disease_3 count=2) → N-to-M bin
4. Drug_B treats Disease_3 (Drug_B count=1, Disease_3 count=2) → 1-to-N bin
```

**Sampling outcome** (possible):
- N-to-1 sampling: Selects Disease_1 group → Edges 1, 2 go to **TEST**
- N-to-M sampling: Selects Drug_A → Edge 3 goes to **TEST**
- 1-to-N sampling: Selects Drug_B → Edge 4 goes to **TEST**

**Result**:
- Drug_A appears in test set (edges 1, 2, 3) ✓ Good
- Disease_3 appears in test set (edges 3, 4) ✓ Good

**BUT**, if different random sampling occurs:
- N-to-1 sampling: Selects Disease_1 group → Edges 1, 2 go to **TEST**
- N-to-M sampling: Does NOT select Drug_A → Edge 3 goes to **TRAIN**
- 1-to-N sampling: Does NOT select Drug_B → Edge 4 goes to **TRAIN**

**Result**:
- Drug_A in TEST (edges 1, 2) AND TRAIN (edge 3) ❌ **LEAKAGE!**
- Disease_3 in TRAIN (edges 3, 4) ✓ Okay (not in test)

## Verification: Node Overlap Check

### What to check:

```python
# After categorization, check if nodes appear in multiple bins

# Extract all nodes from each bin
nodes_1to1 = set()
for s, p, o in one_to_one:
    nodes_1to1.add(s)
    nodes_1to1.add(o)

nodes_1toN = set()
for edges in one_to_n_groups.values():
    for s, p, o in edges:
        nodes_1toN.add(s)
        nodes_1toN.add(o)

nodes_Nto1 = set()
for edges in n_to_one_groups.values():
    for s, p, o in edges:
        nodes_Nto1.add(s)
        nodes_Nto1.add(o)

nodes_NtoM = set()
for s, p, o in many_to_many:
    nodes_NtoM.add(s)
    nodes_NtoM.add(o)

# Check overlaps
overlap_1toN_NtoM = nodes_1toN & nodes_NtoM
overlap_Nto1_NtoM = nodes_Nto1 & nodes_NtoM
overlap_1to1_others = nodes_1to1 & (nodes_1toN | nodes_Nto1 | nodes_NtoM)

print(f"1-to-N ∩ N-to-M: {len(overlap_1toN_NtoM)} nodes")
print(f"N-to-1 ∩ N-to-M: {len(overlap_Nto1_NtoM)} nodes")
print(f"1-to-1 ∩ Others: {len(overlap_1to1_others)} nodes (should be 0)")
```

**Expected result**:
- `overlap_1toN_NtoM` > 0 (overlap exists)
- `overlap_Nto1_NtoM` > 0 (overlap exists)
- `overlap_1to1_others` = 0 (no overlap)

## Recommended Fix

### Option 1: True Connected Component Approach

Group ALL edges that share nodes with any N-to-M edge into a single "complex" bin:

```python
def find_connected_components(treats_edges, subject_counts, object_counts):
    """Find connected components where nodes participate in multiple relationships."""

    # Step 1: Identify N-to-M edges (both subject and object have count > 1)
    nm_edges = [
        (s, p, o) for s, p, o in treats_edges
        if subject_counts[s] > 1 and object_counts[o] > 1
    ]

    # Step 2: Find all nodes involved in N-to-M edges
    nm_nodes = set()
    for s, p, o in nm_edges:
        nm_nodes.add(s)
        nm_nodes.add(o)

    # Step 3: Collect ALL edges involving these nodes
    complex_edges = [
        (s, p, o) for s, p, o in treats_edges
        if s in nm_nodes or o in nm_nodes
    ]

    # Step 4: Simple edges are those NOT in complex
    complex_edge_set = set(complex_edges)
    simple_edges = [e for e in treats_edges if e not in complex_edge_set]

    # Step 5: Re-categorize simple edges
    simple_1to1 = []
    simple_1toN = defaultdict(list)
    simple_Nto1 = defaultdict(list)

    for s, p, o in simple_edges:
        sc = subject_counts[s]
        oc = object_counts[o]

        if sc == 1 and oc == 1:
            simple_1to1.append((s, p, o))
        elif sc > 1 and oc == 1:
            simple_Nto1[o].append((s, p, o))
        elif sc == 1 and oc > 1:
            simple_1toN[s].append((s, p, o))
        # Note: sc > 1 and oc > 1 should not occur here (already in complex)

    return simple_1to1, simple_1toN, simple_Nto1, complex_edges
```

**Advantages**:
- ✅ No node overlap between simple and complex bins
- ✅ True leakage prevention
- ✅ All connected edges sampled together

**Disadvantages**:
- ⚠️ Complex bin might become very large
- ⚠️ Less granular control over sampling

### Option 2: Strict Node-Level Binning

Assign each node to exactly one bin, then categorize edges based on node bins:

```python
def strict_node_binning(treats_edges, subject_counts, object_counts):
    """Assign each node to a single bin based on its complexity."""

    # Classify nodes by their participation pattern
    node_to_bin = {}

    for node in set(list(subject_counts.keys()) + list(object_counts.keys())):
        sc = subject_counts.get(node, 0)
        oc = object_counts.get(node, 0)

        if sc + oc > 2:  # Node appears in multiple edges (as subject or object)
            node_to_bin[node] = 'complex'
        elif sc + oc == 2:
            node_to_bin[node] = 'simple'
        else:  # sc + oc == 1
            node_to_bin[node] = 'singleton'

    # Categorize edges based on node bins
    complex_edges = []
    simple_edges = []
    singleton_edges = []

    for s, p, o in treats_edges:
        s_bin = node_to_bin[s]
        o_bin = node_to_bin[o]

        if 'complex' in [s_bin, o_bin]:
            complex_edges.append((s, p, o))
        elif 'simple' in [s_bin, o_bin]:
            simple_edges.append((s, p, o))
        else:
            singleton_edges.append((s, p, o))

    return singleton_edges, simple_edges, complex_edges
```

**Advantages**:
- ✅ Each node in exactly one category
- ✅ No overlap
- ✅ Clear assignment rules

**Disadvantages**:
- ⚠️ Less fine-grained than current 4-bin approach
- ⚠️ Loses distinction between 1-to-N and N-to-1

### Option 3: Accept Overlap but Track It

Keep current approach but explicitly track and report overlaps:

```python
def sample_with_overlap_tracking(one_to_one, one_to_n_groups, n_to_one_groups, many_to_many):
    """Sample edges while tracking node overlaps."""

    # ... existing sampling logic ...

    # Track which nodes appear in which bins
    node_bins = defaultdict(set)

    for s, p, o in test_edges:
        # Determine which bin this edge came from
        if (s, p, o) in one_to_one:
            node_bins[s].add('1-to-1')
            node_bins[o].add('1-to-1')
        # ... check other bins ...

    # Report overlaps
    overlapping_nodes = {node: bins for node, bins in node_bins.items() if len(bins) > 1}

    logger.warning(f"Found {len(overlapping_nodes)} nodes appearing in multiple bins")

    return test_edges, remaining_treats, overlapping_nodes
```

**Advantages**:
- ✅ Minimal code changes
- ✅ Provides visibility into overlaps
- ✅ Can warn users

**Disadvantages**:
- ❌ Doesn't fix the overlap problem
- ❌ Still has potential leakage risk

## Conclusion

### Answers to Your Questions:

1. **Is many-to-many done in an exhaustive way?**
   - **NO**. The N-to-M bin only contains edges where BOTH the subject and object have count > 1. It does not exhaustively include all edges involving nodes that participate in multi-relationship patterns.

2. **Do 1-to-1, 1-to-N, and N-to-1 bins have any nodes overlapped with N-to-M bin?**
   - **YES**. Nodes can appear in multiple bins:
     - A subject appearing in both N-to-1 edges (with count=1 objects) and N-to-M edges (with count>1 objects)
     - An object appearing in both 1-to-N edges (with count=1 subjects) and N-to-M edges (with count>1 subjects)

### Recommendations:

1. **Immediate**: Add overlap detection code to verify the extent of overlap in your actual data

2. **Short-term**: Implement Option 3 (track and report overlaps) to understand the issue's impact

3. **Long-term**: Consider Option 1 (connected components) or Option 2 (strict node binning) to ensure true leakage prevention

### Risk Assessment:

**Current Risk Level**: **MEDIUM-HIGH**

- Risk depends on the actual overlap percentage in your data
- If overlap is <5%, impact may be minimal
- If overlap is >20%, significant leakage risk exists
- Recommendation: Run overlap analysis on your actual rotorobo.txt to quantify risk

### Verification Script:

I can provide a script to analyze your actual data and report exact overlap statistics if needed.
