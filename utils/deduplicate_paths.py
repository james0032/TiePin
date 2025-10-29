#!/usr/bin/env python3
"""
Deduplicate drug-disease pairs and merge intermediate nodes.
"""
import csv
import sys
from collections import defaultdict
from typing import Dict, Set


def parse_intermediate_nodes(node_str: str) -> Set[str]:
    """Parse the intermediate nodes string into a set of nodes.

    Handles both simple format: [NODE1, NODE2, NODE3]
    and nested format: [NODE1, NODE2, [NODE3, NODE4, NODE5]]
    """
    node_str = node_str.strip()

    # Remove outer brackets if present
    if node_str.startswith('[') and node_str.endswith(']'):
        node_str = node_str[1:-1]

    nodes = set()
    current_node = ""
    bracket_depth = 0

    for char in node_str:
        if char == '[':
            bracket_depth += 1
            # Don't include the bracket itself
        elif char == ']':
            bracket_depth -= 1
            # Don't include the bracket itself
        elif char == ',' and bracket_depth == 0:
            # We're at a top-level comma, so save the current node
            if current_node.strip():
                nodes.add(current_node.strip())
            current_node = ""
        else:
            current_node += char

    # Add the last node
    if current_node.strip():
        nodes.add(current_node.strip())

    return nodes


def format_intermediate_nodes(nodes: Set[str]) -> str:
    """Format a set of nodes back into the bracketed list format."""
    sorted_nodes = sorted(nodes)
    return '[' + ', '.join(sorted_nodes) + ']'


def deduplicate_paths(input_file: str, output_file: str):
    """
    Deduplicate drug-disease pairs and merge intermediate nodes.

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    # Dictionary to store drug-disease pairs and their intermediate nodes
    pair_to_nodes: Dict[tuple, Set[str]] = defaultdict(set)

    # Read the input file
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            # Handle rows with varying lengths due to unquoted commas in intermediate nodes
            if len(row) < 3:
                print(f"Warning: Skipping invalid row (too few fields): {row}", file=sys.stderr)
                continue
            elif len(row) == 3:
                # Properly formatted row
                drug = row[0].strip()
                disease = row[1].strip()
                intermediate_nodes_str = row[2].strip()
            else:
                # Row has more than 3 fields - merge everything after field 2 as intermediate nodes
                drug = row[0].strip()
                disease = row[1].strip()
                # Join all remaining fields back together with commas
                intermediate_nodes_str = ','.join(row[2:]).strip()

            # Parse intermediate nodes
            nodes = parse_intermediate_nodes(intermediate_nodes_str)

            # Add to the union of nodes for this drug-disease pair
            pair_to_nodes[(drug, disease)].update(nodes)

    # Write the deduplicated results
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Drug', 'Disease', '[Intermediate Nodes]'])

        for (drug, disease), nodes in sorted(pair_to_nodes.items()):
            formatted_nodes = format_intermediate_nodes(nodes)
            writer.writerow([drug, disease, formatted_nodes])

    print(f"Processed {len(pair_to_nodes)} unique drug-disease pairs")
    print(f"Output written to: {output_file}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python deduplicate_paths.py <input_file> <output_file>")
        print("\nExample:")
        print("  python deduplicate_paths.py input.csv output.csv")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    deduplicate_paths(input_file, output_file)


if __name__ == '__main__':
    main()
