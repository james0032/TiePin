# Enhanced Debugging for make_test_with_drugmechdb_treat.py

## Date: 2025-11-13

## Issue

The script is reporting that it removed 0 test edges when it should remove 750 edges:

```json
{
  "test_edges_removed_from_rotorobo": 0,
  "train_candidate_edges": 18602343,
  "test_edges": 750
}
```

This means the matching logic is not finding the test edges in rotorobo.txt.

## Debug Enhancements Added

### 1. Enhanced load_filtered_treats_edges() - Lines 91-124

**Added logging**:
- Log the header line to verify TSV format
- Log the first data line (raw) to see actual content
- Log the parsed columns to verify correct splitting
- Log the first edge tuple with explicit field labels

**Purpose**: Identify if the filtered TSV is being parsed correctly and what format the entity IDs are in.

### 2. Enhanced Edge Map Loading - Lines 350-360

**Added logging**:
- Log the specific treats predicate IDs found
- More descriptive error message if no treats predicates found
- Help text pointing to edge_map.json format issue

**Purpose**: Verify that treats_predicates set is not empty and contains expected predicate IDs.

### 3. New Debug Section Before Edge Removal - Lines 387-412

**Added comprehensive debugging**:
```python
# Debug: Show sample edges from test set and rotorobo
logger.info("=" * 80)
logger.info("DEBUG: Sample edges comparison")
logger.info("=" * 80)
logger.info(f"Sample test edge (from filtered TSV): {test_edges[0]}")
logger.info(f"Sample rotorobo edge: {all_edges[0]}")

# Check if any rotorobo edges have treats predicates
treats_edges_in_rotorobo = [e for e in all_edges if e[1] in treats_predicates]
logger.info(f"Total treats edges in rotorobo.txt: {len(treats_edges_in_rotorobo)}")
if treats_edges_in_rotorobo:
    logger.info(f"Sample treats edge from rotorobo: {treats_edges_in_rotorobo[0]}")

# Check if first test edge exists in rotorobo
if test_edges:
    first_test_subj, first_test_pred, first_test_obj = test_edges[0]
    logger.info(f"Looking for first test edge in rotorobo:")
    logger.info(f"  Test edge: ({first_test_subj}, {first_test_pred}, {first_test_obj})")
    matching = [(s, p, o) for s, p, o in all_edges if s == first_test_subj and o == first_test_obj]
    if matching:
        logger.info(f"  Found {len(matching)} edge(s) with same subject/object in rotorobo:")
        for s, p, o in matching[:3]:
            logger.info(f"    ({s}, {p}, {o}) - predicate in treats? {p in treats_predicates}")
    else:
        logger.info(f"  NOT FOUND: No edge with subject={first_test_subj} and object={first_test_obj}")
logger.info("=" * 80)
```

**Purpose**:
- Compare edge formats between filtered TSV and rotorobo.txt
- Count how many treats edges exist in rotorobo.txt
- Check if the first test edge can be found in rotorobo.txt
- Verify if matching edges have predicates in treats_predicates set

## Potential Issues to Diagnose

### Issue 1: Entity ID Mismatch

**Symptom**: "NOT FOUND: No edge with subject=X and object=Y"

**Cause**: The filtered TSV uses different entity IDs than rotorobo.txt

**Example**:
- Filtered TSV: `MESH:D001234`
- rotorobo.txt: `UMLS:C0001234`

**Solution**: Need to normalize entity IDs or use a mapping file

### Issue 2: Empty treats_predicates Set

**Symptom**: "No treats predicates found in edge map"

**Cause**: edge_map.json format is different than expected

**Example edge_map.json format expected**:
```json
{
  "{\"predicate\": \"biolink:treats\", \"source\": \"DrugBank\"}": "R001",
  "{\"predicate\": \"biolink:treats\", \"source\": \"ChEMBL\"}": "R002"
}
```

**Solution**: Check edge_map.json format and update parsing logic if needed

### Issue 3: No treats Edges in rotorobo.txt

**Symptom**: "Total treats edges in rotorobo.txt: 0"

**Cause**: rotorobo.txt doesn't contain any edges with treats predicates, or the predicate IDs are different

**Solution**:
- Verify rotorobo.txt was created with treats edges
- Check if style filter removed all treats edges
- Verify predicate ID mapping

### Issue 4: Column Order Mismatch

**Symptom**: First data line shows unexpected values in subject/predicate/object positions

**Example**:
```
Header line: Drug\tPredicate\tDisease\tPair_Exists
First data line: MESH:D001234\tbiolink:treats\tMESH:D005678\tTrue
Columns: ['MESH:D001234', 'biolink:treats', 'MESH:D005678', 'True']
```

If columns are in different order than expected:
```
Header line: Predicate\tDrug\tDisease\tPair_Exists
First data line: biolink:treats\tMESH:D001234\tMESH:D005678\tTrue
```

**Solution**: Update column indices in load_filtered_treats_edges()

## Expected Debug Output

### Successful Case

```
Loading filtered treats edges from drugmechdb_treats_filtered.txt
Header line: Drug	Predicate	Disease	Pair_Exists
First data line: MESH:D001234	biolink:treats	MONDO:0005148	True
  Columns: ['MESH:D001234', 'biolink:treats', 'MONDO:0005148', 'True']
Loaded 3964 filtered treats edges
First edge tuple: (subject='MESH:D001234', predicate='biolink:treats', object='MONDO:0005148')

Found 5 treats predicate IDs: {'R001', 'R002', 'R003', 'R004', 'R005'}
Treats predicate IDs: {'R001', 'R002', 'R003', 'R004', 'R005'}

================================================================================
DEBUG: Sample edges comparison
================================================================================
Sample test edge (from filtered TSV): ('MESH:D001234', 'biolink:treats', 'MONDO:0005148')
Sample rotorobo edge: ('MESH:D001234', 'R001', 'MONDO:0005148')
Total treats edges in rotorobo.txt: 12543
Sample treats edge from rotorobo: ('MESH:D001234', 'R001', 'MONDO:0005148')
Looking for first test edge in rotorobo:
  Test edge: (MESH:D001234, biolink:treats, MONDO:0005148)
  Found 1 edge(s) with same subject/object in rotorobo:
    (MESH:D001234, R001, MONDO:0005148) - predicate in treats? True
================================================================================

Removed 750 test edges from rotorobo.txt
Train candidates: 18601593 edges
```

### Failed Case (Entity ID Mismatch)

```
Sample test edge (from filtered TSV): ('DRUGBANK:DB00001', 'biolink:treats', 'MONDO:0005148')
Sample rotorobo edge: ('MESH:D001234', 'R001', 'MONDO:0005148')
Total treats edges in rotorobo.txt: 12543
Sample treats edge from rotorobo: ('MESH:D001234', 'R001', 'MONDO:0005148')
Looking for first test edge in rotorobo:
  Test edge: (DRUGBANK:DB00001, biolink:treats, MONDO:0005148)
  NOT FOUND: No edge with subject=DRUGBANK:DB00001 and object=MONDO:0005148
```

**Issue**: Filtered TSV uses DRUGBANK IDs, but rotorobo.txt uses MESH IDs

## Running the Script with Debug Output

```bash
python src/make_test_with_drugmechdb_treat.py \
  --input-dir /workspace/data/robokop/CGGD_alltreat \
  --filtered-tsv /workspace/data/robokop/CGGD_alltreat/results/mechanistic_paths/drugmechdb_treats_filtered.txt \
  --test-pct 0.20 \
  --log-level DEBUG
```

The debug output will help identify:
1. ✅ What format the filtered TSV uses
2. ✅ What treats predicate IDs were found
3. ✅ How many treats edges exist in rotorobo.txt
4. ✅ Whether test edges can be found in rotorobo.txt
5. ✅ What the entity ID mismatch is (if any)

## Next Steps After Debugging

Based on the debug output, the fix will likely be one of:

1. **Entity ID normalization**: Add a mapping step to convert entity IDs
2. **Edge map format fix**: Update find_treats_predicates() parsing logic
3. **Column reordering**: Update load_filtered_treats_edges() column indices
4. **Predicate mapping**: Map biolink:treats to the correct predicate IDs

## Files Modified

- [make_test_with_drugmechdb_treat.py](src/make_test_with_drugmechdb_treat.py)
  - Lines 91-124: Enhanced load_filtered_treats_edges()
  - Lines 350-360: Enhanced edge map error handling
  - Lines 387-412: Added comprehensive debug section

## Related Issues

- Original issue: Test edges not being removed (0 removed instead of 750)
- Previous fix attempt: Changed matching from full triples to (subject, object) pairs
- Root cause: Still under investigation with enhanced debugging
