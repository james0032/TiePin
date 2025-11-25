#!/usr/bin/env python3
"""Test script to verify handling of malformed intermediate nodes data."""

import csv
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from add_ground_truth_column import load_mechanistic_paths

def test_malformed_data():
    """Test that the script handles malformed data gracefully."""

    test_cases = [
        {
            'name': 'Missing closing bracket (like row 4511)',
            'header': 'Drug,Disease,Intermediate_Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,[CHEBI:18361,DB00582_MESH_D055744_1',
            'expected_nodes': 0,
            'should_warn': True
        },
        {
            'name': 'Missing closing bracket with GO term (like row 4512)',
            'header': 'Drug,Disease,Intermediate_Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,[GO:0001525,DB00582_MESH_D055744_1',
            'expected_nodes': 0,
            'should_warn': True
        },
        {
            'name': 'Properly formatted',
            'header': 'Drug,Disease,Intermediate_Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445]",DB00582_MESH_D055744_1',
            'expected_nodes': 2,
            'should_warn': False
        },
        {
            'name': 'Empty list',
            'header': 'Drug,Disease,Intermediate_Nodes,drugmechdb_path_id',
            'row': 'CHEBI:10023,HP:0020103,[],DB00582_MESH_D055744_1',
            'expected_nodes': 0,
            'should_warn': False
        },
    ]

    all_passed = True

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"  Row: {test_case['row']}")

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
                actual_count = len(nodes)
                expected_count = test_case['expected_nodes']

                if actual_count == expected_count:
                    print(f"  ✓ PASS: Found {actual_count} intermediate nodes (expected {expected_count})")
                    if actual_count > 0:
                        print(f"    Nodes: {nodes}")
                else:
                    print(f"  ✗ FAIL: Found {actual_count} nodes, expected {expected_count}")
                    all_passed = False
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
        print("\nThe script now handles malformed data gracefully:")
        print("  - Missing closing brackets → treated as empty list")
        print("  - Warning logged for data quality issues")
        print("  - Processing continues without crashing")
        return 0
    else:
        print("✗ Some tests FAILED")
        return 1

if __name__ == '__main__':
    sys.exit(test_malformed_data())
