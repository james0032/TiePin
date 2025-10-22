"""
Simple test for JSON predicate extraction without requiring PyKEEN.
"""

import json


def extract_predicate_from_json(relation_str: str) -> str:
    """Extract predicate value from JSON-formatted relation string.

    Args:
        relation_str: Either a JSON string like '{"predicate": "biolink:affects", ...}'
                     or a simple string like 'predicate:27'

    Returns:
        Extracted predicate value (e.g., 'biolink:affects') or original string if not JSON
    """
    try:
        # Try to parse as JSON
        relation_obj = json.loads(relation_str)
        # Extract the predicate field
        if isinstance(relation_obj, dict) and 'predicate' in relation_obj:
            return relation_obj['predicate']
    except (json.JSONDecodeError, TypeError):
        # Not JSON, return as-is
        pass
    return relation_str


def test_extract_predicate():
    """Test the predicate extraction function."""

    test_cases = [
        # Case 1: Full JSON string from edge_map.json
        (
            '{"object_aspect_qualifier": "activity", "object_direction_qualifier": "decreased", "predicate": "biolink:affects", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}',
            "biolink:affects"
        ),
        # Case 2: Simpler JSON
        (
            '{"predicate": "biolink:treats"}',
            "biolink:treats"
        ),
        # Case 3: Not JSON - should return as-is
        (
            "predicate:27",
            "predicate:27"
        ),
        # Case 4: JSON without predicate field - should return as-is
        (
            '{"relation": "biolink:treats"}',
            '{"relation": "biolink:treats"}'
        ),
    ]

    print("Testing JSON predicate extraction...")
    print("=" * 80)

    all_passed = True
    for i, (input_str, expected_output) in enumerate(test_cases, 1):
        try:
            result = extract_predicate_from_json(input_str)
            passed = result == expected_output
            all_passed = all_passed and passed

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\nTest {i}: {status}")
            if len(input_str) > 80:
                print(f"  Input:    {input_str[:77]}...")
            else:
                print(f"  Input:    {input_str}")
            print(f"  Expected: {expected_output}")
            print(f"  Got:      {result}")

        except Exception as e:
            all_passed = False
            print(f"\nTest {i}: ✗ ERROR")
            print(f"  Input:    {input_str[:80]}...")
            print(f"  Error:    {e}")

    print("\n" + "=" * 80)
    return all_passed


def test_edge_map_format():
    """Test edge_map.json format parsing."""

    print("\n\nTesting edge_map.json format...")
    print("=" * 80)

    # Sample edge_map.json structure
    sample_edge_map = {
        '{"object_aspect_qualifier": "activity", "object_direction_qualifier": "decreased", "predicate": "biolink:affects", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}': "predicate:0",
        '{"object_aspect_qualifier": "", "object_direction_qualifier": "", "predicate": "biolink:coexpressed_with", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}': "predicate:1",
        '{"object_aspect_qualifier": "", "object_direction_qualifier": "", "predicate": "biolink:treats", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}': "predicate:27",
    }

    print("\nSample edge_map.json:")
    print(json.dumps(sample_edge_map, indent=2)[:300] + "...")

    # Simulate loading and converting
    relation_labels = {}
    for json_str, predicate_id in sample_edge_map.items():
        # Extract the number from "predicate:X"
        if predicate_id.startswith('predicate:'):
            idx = int(predicate_id.split(':')[1])
            relation_labels[idx] = json_str

    print(f"\nCreated mapping: {len(relation_labels)} relations")

    # Test extraction
    print("\nExtracting predicates:")
    expected_results = {
        0: "biolink:affects",
        1: "biolink:coexpressed_with",
        27: "biolink:treats"
    }

    all_correct = True
    for idx in sorted(relation_labels.keys()):
        json_str = relation_labels[idx]
        predicate = extract_predicate_from_json(json_str)
        expected = expected_results.get(idx, "UNKNOWN")

        passed = predicate == expected
        all_correct = all_correct and passed

        status = "✓" if passed else "✗"
        print(f"  {status} Index {idx}: {predicate}")

    print("\n" + "=" * 80)
    return all_correct


def demo_csv_output():
    """Demonstrate what the CSV output would look like."""

    print("\n\nDemonstrating CSV output format...")
    print("=" * 80)

    # Simulate having these mappings
    id_to_relation = {
        27: 'predicate:27'
    }

    relation_labels = {
        27: '{"object_aspect_qualifier": "", "object_direction_qualifier": "", "predicate": "biolink:treats", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}'
    }

    # Get relation ID and label
    test_r = 27
    test_r_id = id_to_relation.get(test_r, f'UNKNOWN_{test_r}')

    # Extract predicate from JSON
    if test_r in relation_labels:
        test_r_label = extract_predicate_from_json(relation_labels[test_r])
    else:
        test_r_label = extract_predicate_from_json(test_r_id)

    print("\nInput:")
    print(f"  Relation index: {test_r}")
    print(f"  Relation ID: {test_r_id}")
    print(f"  Relation label (JSON): {relation_labels[test_r][:80]}...")

    print("\nCSV output columns:")
    print(f"  TestRel: {test_r_id}")
    print(f"  TestRel_label: {test_r_label}")

    print("\nExpected CSV format:")
    print("TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,...")
    print(f"UNII:U59UGK3IPC,Ublituximab,{test_r_id},{test_r_label},MONDO:0005314,multiple sclerosis,...")

    print("\n" + "=" * 80)
    return True


if __name__ == '__main__':
    print("JSON Predicate Extraction Tests")
    print("=" * 80)

    test1_passed = test_extract_predicate()
    test2_passed = test_edge_map_format()
    demo_csv_output()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Predicate extraction test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Edge map format test:      {'✓ PASS' if test2_passed else '✗ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✓✓✓ All tests passed! ✓✓✓")
        print("\nThe CSV output will correctly show:")
        print("  TestRel: predicate:27")
        print("  TestRel_label: biolink:treats")
        print("\nInstead of the full JSON string.")
        exit(0)
    else:
        print("\n✗✗✗ Some tests failed ✗✗✗")
        exit(1)
