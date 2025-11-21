#!/usr/bin/env python3
"""Test script to verify intermediate nodes parsing with different column names."""

import csv
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from add_ground_truth_column import load_mechanistic_paths

def test_column_name_variants():
    """Test that the script handles different column name formats."""

    test_cases = [
        {
            'name': 'Underscore format (Intermediate_Nodes)',
            'header': 'Drug,Disease,Intermediate_Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445]",DB00582_MESH_D055744_1'
        },
        {
            'name': 'Bracket format ([Intermediate Nodes])',
            'header': 'Drug,Disease,[Intermediate Nodes],drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445]",DB00582_MESH_D055744_1'
        },
        {
            'name': 'Space format (Intermediate Nodes)',
            'header': 'Drug,Disease,Intermediate Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445]",DB00582_MESH_D055744_1'
        },
        {
            'name': 'Empty intermediate nodes',
            'header': 'Drug,Disease,Intermediate_Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,[],DB00582_MESH_D055744_1'
        },
    ]

    all_passed = True

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"  Header: {test_case['header']}")

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(test_case['header'] + '\n')
            f.write(test_case['row'] + '\n')
            temp_path = f.name

        try:
            # Try to load
            paths = load_mechanistic_paths(temp_path)

            # Check result
            expected_key = ('CHEBI:10023', 'HP:0020103')
            if expected_key in paths:
                nodes = paths[expected_key]
                print(f"  ✓ PASS: Found {len(nodes)} intermediate nodes: {nodes}")
            else:
                print(f"  ✗ FAIL: Expected key {expected_key} not found in paths")
                all_passed = False

        except Exception as e:
            print(f"  ✗ FAIL: Exception raised: {e}")
            all_passed = False
        finally:
            # Clean up
            Path(temp_path).unlink()

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests PASSED")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(test_column_name_variants())
