"""
Analyze how many nodes from a drug/disease list are present in the node dictionary (graph).

This script:
1. Loads node_dict.txt to get all nodes in the knowledge graph
2. Loads the drug/disease CSV file
3. Checks which nodes from the CSV are in the graph
4. Adds a new column 'in_graph' indicating presence
5. Saves results to a new file with statistics
"""

import argparse
import pandas as pd
from pathlib import Path


def load_node_dict(node_dict_path: str) -> set:
    """Load node dictionary and return set of node IDs."""
    nodes = set()
    with open(node_dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                nodes.add(parts[0])
    return nodes


def analyze_nodes_in_graph(
    node_dict_path: str,
    drug_dis_csv_path: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    Analyze which nodes from drug/disease list are in the graph.

    Args:
        node_dict_path: Path to node_dict.txt
        drug_dis_csv_path: Path to drug/disease CSV file
        output_path: Path for output file (optional)

    Returns:
        DataFrame with added 'in_graph' column
    """
    # Load node dictionary
    print(f"Loading node dictionary from: {node_dict_path}")
    graph_nodes = load_node_dict(node_dict_path)
    print(f"  Total nodes in graph: {len(graph_nodes):,}")

    # Load drug/disease CSV
    print(f"\nLoading drug/disease list from: {drug_dis_csv_path}")
    df = pd.read_csv(drug_dis_csv_path)
    print(f"  Total entries in CSV: {len(df):,}")

    # Check which nodes are in the graph
    df['in_graph'] = df['id'].isin(graph_nodes)

    # Calculate statistics
    total = len(df)
    in_graph = df['in_graph'].sum()
    not_in_graph = total - in_graph

    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total entries:     {total:,}")
    print(f"In graph:          {in_graph:,} ({100*in_graph/total:.1f}%)")
    print(f"Not in graph:      {not_in_graph:,} ({100*not_in_graph/total:.1f}%)")

    # Breakdown by type
    drugs = df[df['is_drug'] == True]
    diseases = df[df['is_disease'] == True]

    drugs_in_graph = drugs['in_graph'].sum()
    diseases_in_graph = diseases['in_graph'].sum()

    print(f"\n--- Breakdown by Type ---")
    print(f"Drugs:    {drugs_in_graph:,}/{len(drugs):,} in graph ({100*drugs_in_graph/len(drugs):.1f}%)")
    print(f"Diseases: {diseases_in_graph:,}/{len(diseases):,} in graph ({100*diseases_in_graph/len(diseases):.1f}%)")

    # Show examples of nodes not in graph
    not_in_graph_df = df[df['in_graph'] == False]
    if len(not_in_graph_df) > 0:
        print(f"\n--- Examples of Nodes NOT in Graph (first 10) ---")
        for _, row in not_in_graph_df.head(10).iterrows():
            node_type = "drug" if row['is_drug'] else "disease"
            print(f"  {row['id']} ({node_type})")

    # Save output
    if output_path is None:
        input_path = Path(drug_dis_csv_path)
        output_path = input_path.parent / f"{input_path.stem}_with_in_graph{input_path.suffix}"

    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Check which nodes from drug/disease list are in the knowledge graph'
    )
    parser.add_argument(
        '--node-dict', '-n',
        type=str,
        default='../data/clean_baseline/node_dict.txt',
        help='Path to node_dict.txt'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../data/clean_baseline/drug_dis_list_subclass_edge_removed_baseline.csv',
        help='Path to drug/disease CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path for output file (default: input_with_in_graph.csv)'
    )

    args = parser.parse_args()

    analyze_nodes_in_graph(
        node_dict_path=args.node_dict,
        drug_dis_csv_path=args.input,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
