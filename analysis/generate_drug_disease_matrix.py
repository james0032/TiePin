"""
Generate all drug-disease pairs from nodes that are in the graph.

Takes drugs (is_drug=True, in_graph=True) and pairs them with all diseases
(is_disease=True, in_graph=True) to create a matrix of potential treats edges.

Output format (TSV):
    drug_id    predicate:38    disease_id
"""

import pandas as pd
from itertools import product


def generate_matrix(
    input_path: str = "../data/clean_baseline/drug_dis_list_with_in_graph.csv",
    output_path: str = "../data/clean_baseline/matrix.txt",
    predicate: str = "predicate:38"
):
    """
    Generate drug-disease pair matrix.

    Args:
        input_path: Path to CSV with is_drug, is_disease, id, in_graph columns
        output_path: Path for output TSV file
        predicate: Predicate string to use (default: predicate:38 for treats)
    """
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # Filter drugs that are in graph
    drugs_in_graph = df[(df['is_drug'] == True) & (df['in_graph'] == True)]['id'].tolist()
    print(f"  Drugs in graph: {len(drugs_in_graph):,}")

    # Filter diseases that are in graph
    diseases_in_graph = df[(df['is_disease'] == True) & (df['in_graph'] == True)]['id'].tolist()
    print(f"  Diseases in graph: {len(diseases_in_graph):,}")

    # Calculate total pairs
    total_pairs = len(drugs_in_graph) * len(diseases_in_graph)
    print(f"\n  Total pairs to generate: {len(drugs_in_graph):,} x {len(diseases_in_graph):,} = {total_pairs:,}")

    # Generate all pairs
    print(f"\nGenerating pairs...")
    with open(output_path, 'w') as f:
        for drug in drugs_in_graph:
            for disease in diseases_in_graph:
                f.write(f"{drug}\t{predicate}\t{disease}\n")

    print(f"\nOutput saved to: {output_path}")
    print(f"Total lines written: {total_pairs:,}")


if __name__ == '__main__':
    generate_matrix()
