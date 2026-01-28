"""
Map CURIEs to preferred CURIEs using the nodes.jsonl file.

For each CURIE in the input file:
1. Check if the CURIE is used as "id" in nodes.jsonl -> use it as preferred
2. If not, check if CURIE is in "equivalent_identifiers" list -> use the "id" as preferred
3. If not found in nodes.jsonl -> fill in "not exist"

Input: ec_moa_pairs_with_curies.csv (Drug, Drug_CURIE, Disease, Disease_CURIE)
Output: ec_moa_pairs_with_curies.csv with added prefered_Drug_CURIE and prefered_Disease_CURIE columns
"""

import argparse
import pandas as pd
import json
from pathlib import Path


def load_nodes_jsonl(nodes_path: str) -> tuple[set, dict]:
    """
    Load nodes.jsonl and build lookup structures.

    Returns:
        Tuple of:
        - set of all node IDs
        - dict mapping equivalent_identifiers to their primary ID
    """
    print(f"Loading nodes from: {nodes_path}")

    node_ids = set()
    equiv_to_primary = {}

    with open(nodes_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i:,} nodes...")

            try:
                node = json.loads(line.strip())
                node_id = node.get("id")

                if node_id:
                    node_ids.add(node_id)

                    # Map equivalent identifiers to this primary ID
                    equiv_ids = node.get("equivalent_identifiers", [])
                    for equiv_id in equiv_ids:
                        if equiv_id != node_id:
                            equiv_to_primary[equiv_id] = node_id

            except json.JSONDecodeError:
                continue

    print(f"  Total nodes: {len(node_ids):,}")
    print(f"  Total equivalent mappings: {len(equiv_to_primary):,}")

    return node_ids, equiv_to_primary


def get_preferred_curie(
    curie: str,
    node_ids: set,
    equiv_to_primary: dict
) -> str:
    """
    Get the preferred CURIE for a given CURIE.

    Args:
        curie: The input CURIE
        node_ids: Set of all primary node IDs
        equiv_to_primary: Dict mapping equivalent IDs to primary IDs

    Returns:
        The preferred CURIE, or "not exist" if not found
    """
    if pd.isna(curie) or curie is None:
        return "not exist"

    curie = str(curie).strip()

    # Check if CURIE is a primary node ID
    if curie in node_ids:
        return curie

    # Check if CURIE is an equivalent identifier
    if curie in equiv_to_primary:
        return equiv_to_primary[curie]

    return "not exist"


def main():
    parser = argparse.ArgumentParser(
        description="Map CURIEs to preferred CURIEs using nodes.jsonl"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input CSV file with Drug_CURIE and Disease_CURIE columns"
    )
    parser.add_argument(
        "-n", "--nodes",
        type=str,
        default="/Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/robokop/nodes.jsonl",
        help="Path to nodes.jsonl file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: overwrite input file)"
    )

    args = parser.parse_args()

    # Paths
    nodes_path = Path(args.nodes)
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    # Load nodes.jsonl
    node_ids, equiv_to_primary = load_nodes_jsonl(str(nodes_path))

    # Load input CSV
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")

    # Map Drug CURIEs to preferred
    print("\nMapping Drug CURIEs to preferred...")
    df["prefered_Drug_CURIE"] = df["Drug_CURIE"].apply(
        lambda x: get_preferred_curie(x, node_ids, equiv_to_primary)
    )

    # Map Disease CURIEs to preferred
    print("Mapping Disease CURIEs to preferred...")
    df["prefered_Disease_CURIE"] = df["Disease_CURIE"].apply(
        lambda x: get_preferred_curie(x, node_ids, equiv_to_primary)
    )

    # Reorder columns
    df = df[["Drug", "Drug_CURIE", "prefered_Drug_CURIE", "Disease", "Disease_CURIE", "prefered_Disease_CURIE"]]

    # Save output
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Save triple format output (Drug_CURIE, predicate:38, Disease_CURIE)
    # Filter out rows with "not exist" in either CURIE
    valid_df = df[
        (df["prefered_Drug_CURIE"] != "not exist") &
        (df["prefered_Disease_CURIE"] != "not exist")
    ]

    # Create triple format: Drug_CURIE \t predicate:38 \t Disease_CURIE
    triple_output_path = output_path.parent / (output_path.stem + "_triples.txt")
    with open(triple_output_path, 'w') as f:
        for _, row in valid_df.iterrows():
            f.write(f"{row['prefered_Drug_CURIE']}\tpredicate:38\t{row['prefered_Disease_CURIE']}\n")

    print(f"Saved triples to: {triple_output_path}")
    print(f"  Valid triples (no 'not exist'): {len(valid_df):,}/{len(df):,}")

    # Print statistics
    drug_found = (df["prefered_Drug_CURIE"] != "not exist").sum()
    disease_found = (df["prefered_Disease_CURIE"] != "not exist").sum()

    print(f"\n=== Mapping Statistics ===")
    print(f"Drugs found in nodes.jsonl:    {drug_found:,}/{len(df):,} ({100*drug_found/len(df):.1f}%)")
    print(f"Diseases found in nodes.jsonl: {disease_found:,}/{len(df):,} ({100*disease_found/len(df):.1f}%)")

    # Show samples of "not exist"
    not_exist_drugs = df[df["prefered_Drug_CURIE"] == "not exist"]["Drug_CURIE"].dropna().unique()
    not_exist_diseases = df[df["prefered_Disease_CURIE"] == "not exist"]["Disease_CURIE"].dropna().unique()

    if len(not_exist_drugs) > 0:
        print(f"\nDrug CURIEs not found ({len(not_exist_drugs)}):")
        for curie in not_exist_drugs[:10]:
            print(f"  - {curie}")
        if len(not_exist_drugs) > 10:
            print(f"  ... and {len(not_exist_drugs) - 10} more")

    if len(not_exist_diseases) > 0:
        print(f"\nDisease CURIEs not found ({len(not_exist_diseases)}):")
        for curie in not_exist_diseases[:10]:
            print(f"  - {curie}")
        if len(not_exist_diseases) > 10:
            print(f"  ... and {len(not_exist_diseases) - 10} more")

    # Show sample output
    print("\nSample output (first 5 rows):")
    print(df.head().to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
