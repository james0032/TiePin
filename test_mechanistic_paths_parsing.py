#!/usr/bin/env python3
"""Test script to verify mechanistic paths CSV parsing."""

import tempfile
from pathlib import Path
from add_ground_truth_column import load_mechanistic_paths

# Create test CSV file with the exact format from your example
test_csv_content = """Drug,Disease,Intermediate_Nodes,drugmechdb_path_id
CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445, HGNC.FAMILY:862, NCBITaxon:5052]",DB00582_MESH_D055744_1
CHEBI:17154,MONDO:0019975,[CHEBI:12345],PATH_002
CHEBI:99999,HP:0001234,[],PATH_003
CHEBI:88888,MONDO:0001111,"[GO:0001234, GO:0005678, HGNC:9999]",PATH_004
"""

# Create temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write(test_csv_content)
    temp_file = f.name

try:
    print("Testing mechanistic paths CSV parsing...")
    print(f"Test file: {temp_file}")
    print()

    # Load the mechanistic paths
    paths = load_mechanistic_paths(temp_file)

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)

    # Check results
    assert len(paths) == 4, f"Expected 4 paths, got {len(paths)}"

    # Test case 1: Path with 4 intermediate nodes
    key1 = ('CHEBI:10023', 'HP:0020103')
    assert key1 in paths, f"Missing path for {key1}"
    nodes1 = paths[key1]
    assert len(nodes1) == 4, f"Expected 4 nodes for {key1}, got {len(nodes1)}: {nodes1}"
    assert nodes1 == ['GO:0006696', 'GO:0030445', 'HGNC.FAMILY:862', 'NCBITaxon:5052'], \
        f"Wrong nodes for {key1}: {nodes1}"
    print(f"✓ Test 1 passed: {key1}")
    print(f"  Intermediate nodes: {nodes1}")

    # Test case 2: Path with 1 intermediate node
    key2 = ('CHEBI:17154', 'MONDO:0019975')
    assert key2 in paths, f"Missing path for {key2}"
    nodes2 = paths[key2]
    assert len(nodes2) == 1, f"Expected 1 node for {key2}, got {len(nodes2)}: {nodes2}"
    assert nodes2 == ['CHEBI:12345'], f"Wrong nodes for {key2}: {nodes2}"
    print(f"✓ Test 2 passed: {key2}")
    print(f"  Intermediate nodes: {nodes2}")

    # Test case 3: Empty path
    key3 = ('CHEBI:99999', 'HP:0001234')
    assert key3 in paths, f"Missing path for {key3}"
    nodes3 = paths[key3]
    assert len(nodes3) == 0, f"Expected 0 nodes for {key3}, got {len(nodes3)}: {nodes3}"
    print(f"✓ Test 3 passed: {key3}")
    print(f"  Intermediate nodes: {nodes3} (empty list)")

    # Test case 4: Path with 3 intermediate nodes
    key4 = ('CHEBI:88888', 'MONDO:0001111')
    assert key4 in paths, f"Missing path for {key4}"
    nodes4 = paths[key4]
    assert len(nodes4) == 3, f"Expected 3 nodes for {key4}, got {len(nodes4)}: {nodes4}"
    assert nodes4 == ['GO:0001234', 'GO:0005678', 'HGNC:9999'], \
        f"Wrong nodes for {key4}: {nodes4}"
    print(f"✓ Test 4 passed: {key4}")
    print(f"  Intermediate nodes: {nodes4}")

    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe mechanistic paths CSV parsing is working correctly.")
    print("The script correctly:")
    print("  - Handles CSV format with quoted columns")
    print("  - Parses bracketed lists: [NODE1, NODE2, NODE3]")
    print("  - Strips whitespace from node names")
    print("  - Handles empty lists: []")
    print("  - Preserves node names with special characters (e.g., HGNC.FAMILY:862)")

finally:
    # Clean up temp file
    Path(temp_file).unlink()
    print(f"\nCleaned up temp file: {temp_file}")
