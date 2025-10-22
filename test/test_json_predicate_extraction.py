"""
Test script to verify JSON predicate extraction works correctly.
"""

import json
from tracin import TracInAnalyzer


def test_extract_predicate_from_json():
    """Test the _extract_predicate_from_json method."""

    # Create a dummy analyzer (we just need the method)
    analyzer = TracInAnalyzer(
        model=None,  # We won't actually use the model
        device='cpu'
    )

    # Test cases
    test_cases = [
        # Case 1: Full JSON string
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
            result = analyzer._extract_predicate_from_json(input_str)
            passed = result == expected_output
            all_passed = all_passed and passed

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\nTest {i}: {status}")
            print(f"  Input:    {input_str[:80]}{'...' if len(input_str) > 80 else ''}")
            print(f"  Expected: {expected_output}")
            print(f"  Got:      {result}")

        except Exception as e:
            all_passed = False
            print(f"\nTest {i}: ✗ ERROR")
            print(f"  Input:    {input_str[:80]}{'...' if len(input_str) > 80 else ''}")
            print(f"  Error:    {e}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")

    return all_passed


def test_edge_map_json_loading():
    """Test loading edge_map.json format."""

    print("\n\nTesting edge_map.json loading...")
    print("=" * 80)

    # Create a sample edge_map.json
    sample_edge_map = {
        '{"object_aspect_qualifier": "activity", "object_direction_qualifier": "decreased", "predicate": "biolink:affects", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}': "predicate:0",
        '{"object_aspect_qualifier": "", "object_direction_qualifier": "", "predicate": "biolink:coexpressed_with", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}': "predicate:1",
        '{"object_aspect_qualifier": "", "object_direction_qualifier": "", "predicate": "biolink:treats", "subject_aspect_qualifier": "", "subject_direction_qualifier": ""}': "predicate:27",
    }

    # Save to temporary file
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_edge_map, f)
        temp_path = f.name

    try:
        # Load using the load_labels function
        from tracin_to_csv import load_labels

        relation_labels = load_labels(temp_path)

        print(f"\nLoaded {len(relation_labels)} relation labels")
        print("\nSample mappings (index -> JSON string):")
        for idx in sorted(relation_labels.keys())[:3]:
            json_str = relation_labels[idx]
            print(f"  {idx}: {json_str[:80]}{'...' if len(json_str) > 80 else ''}")

        # Now test extraction
        analyzer = TracInAnalyzer(model=None, device='cpu')

        print("\nExtracted predicates:")
        for idx in sorted(relation_labels.keys()):
            json_str = relation_labels[idx]
            predicate = analyzer._extract_predicate_from_json(json_str)
            print(f"  {idx}: {predicate}")

        # Verify expected predicates
        expected_predicates = {
            0: "biolink:affects",
            1: "biolink:coexpressed_with",
            27: "biolink:treats"
        }

        all_correct = True
        for idx, expected in expected_predicates.items():
            actual = analyzer._extract_predicate_from_json(relation_labels[idx])
            if actual != expected:
                print(f"✗ FAIL: Index {idx} expected '{expected}', got '{actual}'")
                all_correct = False

        if all_correct:
            print("\n✓ All predicates extracted correctly!")
            return True
        else:
            print("\n✗ Some predicates were not extracted correctly")
            return False

    finally:
        # Clean up temp file
        os.unlink(temp_path)


if __name__ == '__main__':
    test1_passed = test_extract_predicate_from_json()
    test2_passed = test_edge_map_json_loading()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Predicate extraction test: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Edge map JSON test:        {'✓ PASS' if test2_passed else '✗ FAIL'}")

    if test1_passed and test2_passed:
        print("\n✓✓✓ All tests passed! ✓✓✓")
        exit(0)
    else:
        print("\n✗✗✗ Some tests failed ✗✗✗")
        exit(1)
