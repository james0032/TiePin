"""
================================================================================
Find Mechanistic Paths from DrugMechDB
================================================================================

This script extracts mechanistic paths from DrugMechDB that explain drug-disease
treatment relationships (biolink:treats edges).

OVERVIEW:
---------
1. Extracts all 'biolink:treats' edges from the input edges file
2. Builds a graph from DrugMechDB edges only
3. For each drug-disease pair from treats edges, finds all DrugMechDB paths
   that connect them, using the pre-defined drugmechdb_path_id
4. Outputs results showing Drug, Disease, Intermediate Nodes, and drugmechdb_path_id

REQUIRED INPUT FILES:
--------------------
- edges_file: JSONL file (optionally gzipped) containing graph edges
  * Must contain 'biolink:treats' edges
  * Must contain DrugMechDB edges with 'drugmechdb_path_id' property
  * Format: One JSON object per line with fields:
    - subject: source node ID
    - object: target node ID
    - predicate: relationship type (e.g., 'biolink:treats')
    - primary_knowledge_source: source database (e.g., 'infores:drugmechdb')
    - drugmechdb_path_id: list of path IDs (for DrugMechDB edges only)

OUTPUT FILES:
-------------
1. drugmechdb_path_id_results.txt (NEW METHOD - recommended)
   - Format: Drug,Disease,Intermediate_Nodes,drugmechdb_path_id
   - Each row represents one pre-defined DrugMechDB path connecting a drug-disease pair

2. treats.txt (NEW METHOD - for use with add_pair_exists_column.py)
   - Format: Drug\tPredicate\tDisease (tab-separated, rotorobo.txt format)
   - All biolink:treats edges extracted from the input file
   - Can be used as input to add_pair_exists_column.py

3. treats_mechanistic_paths.json (OLD METHOD - for reference)
   - Full JSON output with all mechanistic paths found via graph search

4. treats_mechanistic_paths.txt (OLD METHOD)
   - Text format with one row per path found

5. dedup_treats_mechanistic_paths.txt (OLD METHOD)
   - Deduplicated drug-disease pairs with all intermediate nodes merged

COMMAND LINE ARGUMENTS:
-----------------------
--edges_file        : Path to input edges JSONL file (default: roboedges.jsonl.gz)
--max_length        : Maximum path length to search (default: 5)
--output_dir        : Directory for output files (default: current directory)
--run_old_method    : Run old graph search method in addition to path_id method (default: False)

EXAMPLE USAGE:
--------------
python find_mechanistic_paths.py --edges_file /path/to/edges.jsonl.gz --max_length 5

python find_mechanistic_paths.py --edges_file edges.jsonl --output_dir ./results/

DEPENDENCIES:
-------------
- gzip, json, collections (standard library)
- torch, torch_geometric (for graph operations)
- networkx (for path finding)

================================================================================
"""

import gzip
import json
import argparse
import os
from collections import defaultdict
import torch
from torch_geometric.data import Data
import networkx as nx

def open_edges_file(edges_file):
    """Open edges file, handling both gzipped and plain text files"""
    if edges_file.endswith('.gz'):
        return gzip.open(edges_file, 'rt')
    else:
        return open(edges_file, 'r')

def extract_treats_edges(edges_file='roboedges.jsonl.gz'):
    """Extract all edges with 'biolink:treats' predicate"""

    treats_edges = []
    total_lines = 0

    print("Step 1: Extracting 'biolink:treats' edges...")
    print(f"  Reading from: {edges_file}")
    with open_edges_file(edges_file) as f:
        for line in f:
            total_lines += 1
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} lines... Found {len(treats_edges)} treats edges")
            
            record = json.loads(line)
            if record.get('predicate') == 'biolink:treats':
                treats_edges.append({
                    'subject': record['subject'],
                    'object': record['object'],
                    'source': record.get('primary_knowledge_source', 'unknown')
                })
    
    print(f"  Found {len(treats_edges)} 'biolink:treats' edges")
    return treats_edges

def build_drugmechdb_graph(edges_file='roboedges.jsonl.gz'):
    """Build graph from DrugMechDB edges only and extract path_id information"""

    edges = []
    node_to_idx = {}
    node_idx = 0
    edge_attributes = []
    path_id_map = defaultdict(lambda: {'edges': [], 'nodes': set()})

    print("\nStep 2: Building DrugMechDB graph...")
    print(f"  Reading from: {edges_file}")
    with open_edges_file(edges_file) as f:
        for line in f:
            record = json.loads(line)

            if record.get('primary_knowledge_source') == 'infores:drugmechdb':
                subj = record['subject']
                obj = record['object']
                pred = record.get('predicate', 'unknown')
                path_ids = record.get('drugmechdb_path_id', [])

                if subj not in node_to_idx:
                    node_to_idx[subj] = node_idx
                    node_idx += 1
                if obj not in node_to_idx:
                    node_to_idx[obj] = node_idx
                    node_idx += 1

                edges.append([node_to_idx[subj], node_to_idx[obj]])
                edge_attr = {
                    'predicate': pred,
                    'subject_id': subj,
                    'object_id': obj,
                    'path_ids': path_ids
                }
                edge_attributes.append(edge_attr)

                # Store edges and nodes for each path_id
                for path_id in path_ids:
                    path_id_map[path_id]['edges'].append({
                        'subject': subj,
                        'object': obj,
                        'predicate': pred
                    })
                    path_id_map[path_id]['nodes'].add(subj)
                    path_id_map[path_id]['nodes'].add(obj)

    print(f"  DrugMechDB graph: {len(node_to_idx)} nodes, {len(edges)} edges")
    print(f"  Found {len(path_id_map)} unique drugmechdb_path_ids")

    idx_to_node = {v: k for k, v in node_to_idx.items()}
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, idx_to_node, edge_attributes, node_to_idx, dict(path_id_map)

def build_edge_lookup(edge_index, edge_attributes):
    """Build lookup for edge attributes"""
    edge_lookup = {}
    edges = edge_index.t().numpy()
    
    for i, (src, dst) in enumerate(edges):
        edge_lookup[(src, dst)] = edge_attributes[i]
    
    return edge_lookup

def find_mechanistic_paths_for_treats_edge(treats_edge, G, idx_to_node, node_to_idx, edge_lookup, max_length=5):
    """Find all DrugMechDB paths explaining a single treats edge"""
    
    drug = treats_edge['subject']
    disease = treats_edge['object']
    
    # Check if both nodes exist in DrugMechDB graph
    if drug not in node_to_idx or disease not in node_to_idx:
        return []
    
    drug_idx = node_to_idx[drug]
    disease_idx = node_to_idx[disease]
    
    # Find all simple paths from drug to disease
    paths = []
    try:
        nx_paths = nx.all_simple_paths(G, drug_idx, disease_idx, cutoff=max_length)
        
        for path in nx_paths:
            # Build path with edge attributes
            path_with_attrs = []
            for j in range(len(path) - 1):
                src_idx = path[j]
                dst_idx = path[j + 1]
                
                edge_attr = edge_lookup.get((src_idx, dst_idx), {
                    'predicate': 'unknown',
                    'subject_id': idx_to_node[src_idx],
                    'object_id': idx_to_node[dst_idx]
                })
                
                path_with_attrs.append({
                    'source_node': idx_to_node[src_idx],
                    'target_node': idx_to_node[dst_idx],
                    'predicate': edge_attr['predicate']
                })
            
            paths.append({
                'length': len(path),
                'nodes': [idx_to_node[node] for node in path],
                'edges': path_with_attrs
            })
    except nx.NetworkXNoPath:
        pass
    
    return paths

def extract_paths_by_path_id(path_id_map, treats_edges):
    """Extract drug-disease pairs from treats_edges and find their drugmechdb_path_ids

    Args:
        path_id_map: Dictionary mapping path_id to edges and nodes
        treats_edges: List of treats edges containing drug-disease pairs

    Returns:
        List of results with drug, disease, intermediate_nodes, and drugmechdb_path_id
    """

    print("\nStep 3: Extracting paths based on drugmechdb_path_id for treats edges...")

    # Create a set of (drug, disease) pairs from treats_edges for quick lookup
    treats_pairs = set()
    for edge in treats_edges:
        treats_pairs.add((edge['subject'], edge['object']))

    print(f"  Processing {len(treats_pairs)} unique drug-disease pairs from treats edges")

    # Build a reverse mapping: (drug, disease) -> list of path_ids
    pair_to_path_ids = defaultdict(list)

    for path_id, path_data in path_id_map.items():
        # Get all nodes in this path
        all_nodes = path_data['nodes']

        # Check which treats pair this path_id belongs to
        for drug, disease in treats_pairs:
            # Check if both drug and disease are in the path nodes
            if drug in all_nodes and disease in all_nodes:
                pair_to_path_ids[(drug, disease)].append({
                    'path_id': path_id,
                    'nodes': all_nodes,
                    'edges': path_data['edges']
                })

    # Build results
    results = []
    pairs_with_paths = 0

    for drug, disease in sorted(treats_pairs):
        path_ids_list = pair_to_path_ids.get((drug, disease), [])

        if path_ids_list:
            pairs_with_paths += 1

        # For each path_id connecting this drug-disease pair
        for path_info in path_ids_list:
            all_nodes = path_info['nodes']
            path_id = path_info['path_id']

            # Identify intermediate nodes (exclude drug and disease)
            intermediate_nodes = []
            for node in all_nodes:
                if node != drug and node != disease:
                    intermediate_nodes.append(node)

            results.append({
                'drug': drug,
                'disease': disease,
                'intermediate_nodes': sorted(intermediate_nodes),
                'drugmechdb_path_id': path_id,
                'edges': path_info['edges']
            })

    print(f"  Found {len(results)} paths for {pairs_with_paths}/{len(treats_pairs)} drug-disease pairs")
    print(f"  Drug-disease pairs without paths: {len(treats_pairs) - pairs_with_paths}")

    return results

def find_all_mechanistic_paths(treats_edges, edge_index, idx_to_node, node_to_idx, edge_attributes, max_length=5):
    """Find DrugMechDB mechanistic paths for all treats edges"""

    print("\nStep 3: Finding DrugMechDB mechanistic paths for each treats edge...")

    # Build NetworkX graph and edge lookup
    G = nx.DiGraph()
    edges = edge_index.t().numpy()
    G.add_edges_from(edges)
    edge_lookup = build_edge_lookup(edge_index, edge_attributes)

    print(f"  DrugMechDB graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # For each treats edge, find mechanistic paths
    results = []
    treats_with_paths = 0
    total_paths_found = 0

    for i, treats_edge in enumerate(treats_edges):
        if i % 10 == 0:
            print(f"  Processing treats edge {i}/{len(treats_edges)}...")

        paths = find_mechanistic_paths_for_treats_edge(
            treats_edge, G, idx_to_node, node_to_idx, edge_lookup, max_length
        )

        if paths:
            treats_with_paths += 1
            total_paths_found += len(paths)

        results.append({
            'treats_edge': treats_edge,
            'num_mechanistic_paths': len(paths),
            'mechanistic_paths': paths
        })

    print(f"\n  Treats edges with mechanistic paths: {treats_with_paths}/{len(treats_edges)}")
    print(f"  Total mechanistic paths found: {total_paths_found}")

    return results

def summarize_path_id_results(path_id_results):
    """Summarize the path_id-based mechanistic paths found"""

    print(f"\n{'='*70}")
    print("DRUGMECHDB PATH_ID SUMMARY")
    print(f"{'='*70}")

    if not path_id_results:
        print("No paths found.")
        return

    # Overall statistics
    total_paths = len(path_id_results)
    unique_pairs = len(set((r['drug'], r['disease']) for r in path_id_results))
    paths_with_intermediates = sum(1 for r in path_id_results if r['intermediate_nodes'])

    print(f"Total drugmechdb_path_ids: {total_paths}")
    print(f"Unique drug-disease pairs: {unique_pairs}")
    print(f"Paths with intermediate nodes: {paths_with_intermediates}")
    print(f"Paths without intermediate nodes (direct): {total_paths - paths_with_intermediates}")

    if unique_pairs > 0:
        avg_paths_per_pair = total_paths / unique_pairs
        print(f"Average paths per drug-disease pair: {avg_paths_per_pair:.2f}")

    # Intermediate node statistics
    all_intermediates = []
    for r in path_id_results:
        all_intermediates.extend(r['intermediate_nodes'])

    if all_intermediates:
        unique_intermediates = set(all_intermediates)
        print(f"\nIntermediate node statistics:")
        print(f"  Total intermediate node occurrences: {len(all_intermediates)}")
        print(f"  Unique intermediate nodes: {len(unique_intermediates)}")

        # Path length distribution
        path_lengths = [len(r['intermediate_nodes']) + 2 for r in path_id_results]  # +2 for drug and disease
        length_dist = defaultdict(int)
        for length in path_lengths:
            length_dist[length] += 1

        print(f"\nPath length distribution:")
        for length in sorted(length_dist.keys()):
            print(f"  Length {length}: {length_dist[length]} paths")

    # Show examples
    print(f"\nSample paths:")
    examples_shown = 0
    for r in path_id_results:
        if examples_shown >= 5:
            break

        print(f"\n  Path ID: {r['drugmechdb_path_id']}")
        print(f"    Drug: {r['drug']}")
        print(f"    Disease: {r['disease']}")
        if r['intermediate_nodes']:
            print(f"    Intermediate nodes: {', '.join(r['intermediate_nodes'])}")
        else:
            print(f"    Intermediate nodes: (none - direct path)")
        print(f"    Edges in path:")
        for edge in r['edges']:
            print(f"      {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")

        examples_shown += 1

def summarize_results(results):
    """Summarize the mechanistic paths found (OLD METHOD)"""

    print(f"\n{'='*70}")
    print("MECHANISTIC PATH SUMMARY (OLD METHOD)")
    print(f"{'='*70}")
    
    # Overall statistics
    total_treats = len(results)
    treats_with_paths = sum(1 for r in results if r['num_mechanistic_paths'] > 0)
    total_paths = sum(r['num_mechanistic_paths'] for r in results)
    
    print(f"Total treats edges: {total_treats}")
    print(f"Treats edges with DrugMechDB paths: {treats_with_paths}")
    print(f"Treats edges WITHOUT DrugMechDB paths: {total_treats - treats_with_paths}")
    print(f"Total mechanistic paths: {total_paths}")
    
    if treats_with_paths > 0:
        avg_paths = total_paths / treats_with_paths
        print(f"Average paths per treats edge (when paths exist): {avg_paths:.2f}")
    
    # Path length distribution
    all_paths = []
    for r in results:
        all_paths.extend(r['mechanistic_paths'])
    
    if all_paths:
        lengths = [p['length'] for p in all_paths]
        length_dist = defaultdict(int)
        for l in lengths:
            length_dist[l] += 1
        
        print(f"\nMechanistic path length distribution:")
        for length in sorted(length_dist.keys()):
            print(f"  Length {length}: {length_dist[length]} paths")
        
        # Predicate patterns
        predicate_sequences = []
        for path in all_paths:
            pred_seq = tuple(edge['predicate'] for edge in path['edges'])
            predicate_sequences.append(pred_seq)
        
        pred_counts = defaultdict(int)
        for seq in predicate_sequences:
            pred_counts[seq] += 1
        
        print(f"\nTop 10 predicate patterns in mechanistic paths:")
        for seq, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            pred_str = ' → '.join(seq)
            print(f"  {pred_str}: {count} paths")
    
    # Show examples
    print(f"\nSample treats edges with their mechanistic paths:")
    examples_shown = 0
    for r in results:
        if r['num_mechanistic_paths'] > 0 and examples_shown < 5:
            print(f"\n  Treats edge:")
            print(f"    {r['treats_edge']['subject']} --[biolink:treats]--> {r['treats_edge']['object']}")
            print(f"    Source: {r['treats_edge']['source']}")
            print(f"    DrugMechDB mechanistic paths found: {r['num_mechanistic_paths']}")
            
            # Show first mechanistic path
            if r['mechanistic_paths']:
                path = r['mechanistic_paths'][0]
                print(f"    Example mechanistic path (length {path['length']}):")
                for edge in path['edges']:
                    print(f"      {edge['source_node']} --[{edge['predicate']}]--> {edge['target_node']}")
            
            examples_shown += 1

def save_results(results, output_file='treats_mechanistic_paths.json', output_dir='.'):
    """Save results to JSON file"""

    output_path = os.path.join(output_dir, output_file)
    print(f"\n{'='*70}")
    print(f"Saving results to {output_path}...")

    with open(output_path, 'w') as f:
        json.dump({
            'total_treats_edges': len(results),
            'treats_edges_with_paths': sum(1 for r in results if r['num_mechanistic_paths'] > 0),
            'total_mechanistic_paths': sum(r['num_mechanistic_paths'] for r in results),
            'results': results
        }, f, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved to {output_path} ({file_size_mb:.2f} MB)")

def save_results_txt(results, output_txt='treats_mechanistic_paths.txt', output_dir='.'):
    """Save mechanistic paths to a .txt file in the format:
       Drug, Disease, [list of intermediate nodes in the mechanistic path]
    """
    output_path = os.path.join(output_dir, output_txt)
    print(f"\n{'='*70}")
    print(f"Saving simplified results to {output_path}...")

    with open(output_path, 'w') as f:
        f.write("Drug,Disease,[Intermediate Nodes]\n")
        for r in results:
            drug = r['treats_edge']['subject']
            disease = r['treats_edge']['object']
            for path in r['mechanistic_paths']:
                nodes = path['nodes']
                if len(nodes) > 2:
                    intermediates = nodes[1:-1]  # exclude drug and disease
                else:
                    intermediates = []
                intermediates_str = "[" + ", ".join(intermediates) + "]"
                f.write(f"{drug},{disease},{intermediates_str}\n")

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"✓ Saved to {output_path} ({file_size_kb:.2f} KB)")


def save_results_txt_deduplicated(results, output_txt='dedup_treats_mechanistic_paths.txt', output_dir='.'):
    """Save deduplicated mechanistic paths to a .txt file.

    For each unique (drug, disease) pair, merges all intermediate nodes
    from all mechanistic paths into a single list (union).

    Format: Drug,Disease,[Intermediate Nodes]
    """
    output_path = os.path.join(output_dir, output_txt)
    print(f"\n{'='*70}")
    print(f"Saving deduplicated results to {output_path}...")

    # Build dictionary: (drug, disease) -> set of all intermediate nodes
    pair_to_intermediates = {}

    for r in results:
        drug = r['treats_edge']['subject']
        disease = r['treats_edge']['object']
        pair = (drug, disease)

        if pair not in pair_to_intermediates:
            pair_to_intermediates[pair] = set()

        # Collect all intermediate nodes from all paths for this pair
        for path in r['mechanistic_paths']:
            nodes = path['nodes']
            if len(nodes) > 2:
                intermediates = nodes[1:-1]  # exclude drug and disease
                pair_to_intermediates[pair].update(intermediates)

    # Write deduplicated results
    with open(output_path, 'w') as f:
        f.write("Drug,Disease,[Intermediate Nodes]\n")

        # Sort by drug, then disease for consistent output
        for (drug, disease) in sorted(pair_to_intermediates.keys()):
            intermediates = sorted(pair_to_intermediates[(drug, disease)])
            intermediates_str = "[" + ", ".join(intermediates) + "]"
            f.write(f"{drug},{disease},{intermediates_str}\n")

    file_size_kb = os.path.getsize(output_path) / 1024

    total_pairs = len(pair_to_intermediates)
    pairs_with_intermediates = sum(1 for nodes in pair_to_intermediates.values() if nodes)

    print(f"✓ Saved to {output_path} ({file_size_kb:.2f} KB)")
    print(f"  Total unique drug-disease pairs: {total_pairs}")
    print(f"  Pairs with intermediate nodes: {pairs_with_intermediates}")

def save_path_id_results(results, output_txt='drugmechdb_path_id_results.txt', output_dir='.'):
    """Save results grouped by drugmechdb_path_id to a .txt file.

    Format: Drug,Disease,[Intermediate Nodes],drugmechdb_path_id
    """
    output_path = os.path.join(output_dir, output_txt)
    print(f"\n{'='*70}")
    print(f"Saving path_id-based results to {output_path}...")

    with open(output_path, 'w') as f:
        f.write("Drug,Disease,Intermediate_Nodes,drugmechdb_path_id\n")

        for r in results:
            drug = r['drug']
            disease = r['disease']
            intermediates = r['intermediate_nodes']
            path_id = r['drugmechdb_path_id']

            # Format intermediate nodes as a list string
            intermediates_str = "[" + ", ".join(intermediates) + "]" if intermediates else "[]"

            f.write(f"{drug},{disease},{intermediates_str},{path_id}\n")

    file_size_kb = os.path.getsize(output_path) / 1024

    print(f"✓ Saved to {output_path} ({file_size_kb:.2f} KB)")
    print(f"  Total paths: {len(results)}")
    print(f"  Paths with intermediate nodes: {sum(1 for r in results if r['intermediate_nodes'])}")

def save_treats_edges_tsv(treats_edges, output_tsv='treats.txt', output_dir='.'):
    """Save treats edges in rotorobo.txt format (tab-separated).

    Format: subject\tpredicate\tobject
    This matches the format of rotorobo.txt for use with add_pair_exists_column.py
    """
    output_path = os.path.join(output_dir, output_tsv)
    print(f"\n{'='*70}")
    print(f"Saving treats edges to {output_path} (rotorobo.txt format)...")

    with open(output_path, 'w') as f:
        # Write header (optional, matches typical TSV format)
        f.write("Drug\tPredicate\tDisease\n")

        # Write each treats edge as tab-separated values
        for edge in treats_edges:
            drug = edge['subject']
            disease = edge['object']
            predicate = 'biolink:treats'
            f.write(f"{drug}\t{predicate}\t{disease}\n")

    file_size_kb = os.path.getsize(output_path) / 1024

    print(f"✓ Saved to {output_path} ({file_size_kb:.2f} KB)")
    print(f"  Total treats edges: {len(treats_edges)}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Find mechanistic paths from DrugMechDB for drug-disease treatment relationships',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python find_mechanistic_paths.py --edges_file /path/to/edges.jsonl.gz
  python find_mechanistic_paths.py --edges_file edges.jsonl --max_length 5 --output_dir ./results/
  python find_mechanistic_paths.py --edges_file edges.jsonl --run_old_method
        """
    )

    parser.add_argument(
        '--edges_file',
        type=str,
        default='roboedges.jsonl.gz',
        help='Path to input edges JSONL file (can be gzipped). Default: roboedges.jsonl.gz'
    )

    parser.add_argument(
        '--max_length',
        type=int,
        default=5,
        help='Maximum path length to search for mechanistic paths (only used with --run_old_method). Default: 5'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save output files. Default: current directory'
    )

    parser.add_argument(
        '--run_old_method',
        action='store_true',
        help='Run old graph search method in addition to path_id method. This will generate additional output files.'
    )

    return parser.parse_args()

def main(edges_file='roboedges.jsonl.gz', max_length=5, output_dir='.', run_old_method=False):
    """Main function to extract and analyze mechanistic paths

    Args:
        edges_file: Path to input edges JSONL file
        max_length: Maximum path length to search (only used if run_old_method is True)
        output_dir: Directory to save output files
        run_old_method: Whether to run the old graph search method
    """
    # Create output directory if it doesn't exist
    if output_dir != '.' and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"\n{'='*70}")
    print("FINDING MECHANISTIC PATHS FROM DRUGMECHDB")
    print(f"{'='*70}")
    print(f"Input file: {edges_file}")
    print(f"Output directory: {output_dir}")
    print(f"Run old method: {run_old_method}")
    if run_old_method:
        print(f"Max path length: {max_length}")
    print(f"{'='*70}\n")

    # Step 1: Extract all treats edges
    treats_edges = extract_treats_edges(edges_file)

    # Step 2: Build DrugMechDB graph and extract path_id information
    edge_index, idx_to_node, edge_attributes, node_to_idx, path_id_map = build_drugmechdb_graph(edges_file)

    # Step 3: Extract paths based on drugmechdb_path_id for treats edges
    path_id_results = extract_paths_by_path_id(path_id_map, treats_edges)

    # Step 4: Summarize path_id results
    summarize_path_id_results(path_id_results)

    # Step 5: Save path_id-based results
    save_path_id_results(path_id_results, output_dir=output_dir)

    # Step 6: Save treats edges in rotorobo.txt format
    save_treats_edges_tsv(treats_edges, output_dir=output_dir)

    # Optional: Run old method if requested
    results = None
    if run_old_method:
        print(f"\n{'='*70}")
        print("RUNNING OLD METHOD (graph search)")
        print(f"{'='*70}\n")

        # Find mechanistic paths for each treats edge (OLD METHOD)
        results = find_all_mechanistic_paths(
            treats_edges, edge_index, idx_to_node, node_to_idx, edge_attributes, max_length=max_length
        )

        # Summarize old method results
        summarize_results(results)

        # Save old method results
        save_results(results, output_dir=output_dir)
        save_results_txt(results, output_dir=output_dir)
        save_results_txt_deduplicated(results, output_dir=output_dir)

    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")

    return path_id_results, results

if __name__ == "__main__":
    args = parse_args()
    path_id_results, results = main(
        edges_file=args.edges_file,
        max_length=args.max_length,
        output_dir=args.output_dir,
        run_old_method=args.run_old_method
    )