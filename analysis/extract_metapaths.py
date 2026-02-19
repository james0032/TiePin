#!/usr/bin/env python3
"""
Extract k-hop metapaths from TracIn CSV files.

For each input CSV, extracts connecting paths between test head and tail entities
at 2-hop, 3-hop, and 4-hop distances, computes metapath schemas, and outputs
results in JSON, CSV, and TXT formats.

Usage:
    python extract_metapaths.py <csv_file> [--output-dir <dir>] [--max-edges <n>]
    python extract_metapaths.py <csv_file> --hops 2 3 4 [--output-dir <dir>]

Input CSV columns:
    TestHead, TestHead_label, TestRel, TestRel_label, TestTail, TestTail_label,
    TrainHead, TrainHead_label, TrainRel, TrainRel_label, TrainTail, TrainTail_label,
    TracInScore, SelfInfluence
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


# ============================================================================
# Entity type mapping
# ============================================================================

TYPE_MAPPING = {
    'CHEBI': 'C',           # Chemical
    'DRUGBANK': 'C',
    'PUBCHEM.COMPOUND': 'C',
    'UNII': 'C',
    'MESH': 'C',
    'NCBIGENE': 'G',        # Gene
    'HGNC': 'G',
    'ENSEMBL': 'G',
    'MONDO': 'D',            # Disease
    'DOID': 'D',
    'HP': 'D',               # Phenotype (treating as Disease)
    'EFO': 'D',
    'UMLS': 'D',
    'GO': 'P',               # Biological Process
    'UBERON': 'A',            # Anatomy
    'CL': 'Ct',              # Cell type
}

ENTITY_TYPE_LEGEND = {
    'C': 'Chemical/Drug',
    'G': 'Gene',
    'D': 'Disease/Phenotype',
    'P': 'Biological Process',
    'A': 'Anatomy',
    'Ct': 'Cell Type',
}


def get_entity_type(curie):
    """Determine entity type abbreviation from CURIE prefix."""
    if ':' not in curie:
        return 'Unknown'
    prefix = curie.split(':')[0].upper()
    return TYPE_MAPPING.get(prefix, prefix[:2])


# ============================================================================
# Data loading and graph construction
# ============================================================================

def load_test_triple_data(csv_file, max_edges=None):
    """Load test triple CSV data. Returns list of row dicts."""
    data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_edges and i >= max_edges:
                break
            data.append(row)
    print(f"Loaded {len(data)} training edges")
    return data


def build_graph_from_data(data):
    """
    Build graph structure from CSV data.

    Returns:
        (nodes dict, links list, test_edge_info dict)
    """
    nodes = {}
    links = []
    test_edge_info = None

    for row in data:
        if test_edge_info is None:
            test_edge_info = {
                'head': row['TestHead'],
                'head_label': row['TestHead_label'],
                'tail': row['TestTail'],
                'tail_label': row['TestTail_label'],
                'relation': row['TestRel_label'],
            }

        score = float(row['TracInScore'])

        for curie, label in [(row['TrainHead'], row['TrainHead_label']),
                             (row['TrainTail'], row['TrainTail_label'])]:
            if curie not in nodes:
                entity_type = 'test_entity' if curie in [row['TestHead'], row['TestTail']] else 'train_entity'
                nodes[curie] = {'id': curie, 'label': label, 'type': entity_type, 'curie': curie}

        links.append({
            'source': row['TrainHead'],
            'target': row['TrainTail'],
            'type': 'train_edge',
            'relation': row['TrainRel_label'],
            'score': score,
        })

    # Ensure test entities are in nodes
    if test_edge_info:
        for curie, label_key in [(test_edge_info['head'], 'head_label'),
                                  (test_edge_info['tail'], 'tail_label')]:
            if curie not in nodes:
                nodes[curie] = {
                    'id': curie, 'label': test_edge_info[label_key],
                    'type': 'test_entity', 'curie': curie,
                }

        links.append({
            'source': test_edge_info['head'],
            'target': test_edge_info['tail'],
            'type': 'test_edge',
            'relation': test_edge_info['relation'],
            'score': None,
        })

    print(f"Built graph with {len(nodes)} nodes and {len(links)} edges")
    return nodes, links, test_edge_info


# ============================================================================
# Path extraction (BFS)
# ============================================================================

def extract_k_hop_paths(nodes, links, test_edge_info, k=3):
    """
    Extract all k-hop paths between test head and tail using BFS.

    Args:
        nodes: Dictionary of nodes
        links: List of edges
        test_edge_info: Test edge information
        k: Number of hops

    Returns:
        Dictionary with paths and metapath information
    """
    # Build bidirectional adjacency list
    graph = defaultdict(list)
    for link in links:
        source = link['source']
        target = link['target']
        relation = link.get('relation', 'unknown')
        graph[source].append((target, relation))
        graph[target].append((source, relation))

    test_head = test_edge_info['head']
    test_tail = test_edge_info['tail']

    all_paths = []

    # BFS from both test entities
    for start_node in [test_head, test_tail]:
        start_label = "Test Head" if start_node == test_head else "Test Tail"
        queue = deque([(start_node, [(start_node, None)], {start_node})])

        while queue:
            current, path, visited = queue.popleft()

            if len(path) == k + 1:
                all_paths.append({'start': start_label, 'path': path, 'start_node': start_node})
                continue

            if len(path) > k:
                continue

            for neighbor, relation in graph[current]:
                if neighbor not in visited:
                    new_visited = visited | {neighbor}
                    queue.append((neighbor, path + [(neighbor, relation)], new_visited))

    # Format and filter connecting paths
    formatted_paths = []
    connecting_paths = []

    for path_info in all_paths:
        path = path_info['path']
        first_node = path[0][0]
        last_node = path[-1][0]
        is_connecting = (first_node == test_head and last_node == test_tail)

        path_parts = []
        for i, (node_curie, relation) in enumerate(path):
            node_label = nodes.get(node_curie, {}).get('label', node_curie)
            if i == 0:
                path_parts.append(node_label)
            else:
                path_parts.append(relation)
                path_parts.append(node_label)

        path_dict = {
            'path_string': ' -> '.join(path_parts),
            'start_from': path_info['start'],
            'nodes': [nodes.get(nc, {}).get('label', nc) for nc, _ in path],
            'relations': [rel for _, rel in path[1:]],
            'node_curies': [nc for nc, _ in path],
            'connects_test_entities': is_connecting,
        }

        formatted_paths.append(path_dict)
        if is_connecting:
            connecting_paths.append(path_dict)

    # Extract metapaths
    metapath_data = extract_metapaths(connecting_paths, test_head, test_tail)

    return {
        'total_paths': len(formatted_paths),
        'connecting_paths_count': len(connecting_paths),
        'k_hops': k,
        'test_edge': {
            'head': test_edge_info['head_label'],
            'relation': test_edge_info['relation'],
            'tail': test_edge_info['tail_label'],
            'head_curie': test_edge_info['head'],
            'tail_curie': test_edge_info['tail'],
        },
        'all_paths': formatted_paths,
        'connecting_paths': connecting_paths,
        'metapaths': metapath_data,
    }


# ============================================================================
# Metapath extraction
# ============================================================================

def extract_metapaths(connecting_paths, test_head_curie, test_tail_curie):
    """
    Extract metapath schemas from connecting paths and count occurrences.

    Returns:
        Dictionary with metapath counts and examples
    """
    metapath_counter = Counter()
    metapath_examples = defaultdict(list)

    for path in connecting_paths:
        node_curies = path['node_curies']
        relations = path['relations']

        metapath_parts = [get_entity_type(node_curies[0])]
        for i, relation in enumerate(relations):
            metapath_parts.append(relation)
            metapath_parts.append(get_entity_type(node_curies[i + 1]))

        metapath = ' -> '.join(metapath_parts)
        metapath_counter[metapath] += 1

        if len(metapath_examples[metapath]) < 3:
            metapath_examples[metapath].append(path['path_string'])

    metapath_list = []
    for metapath, count in metapath_counter.most_common():
        metapath_list.append({
            'metapath': metapath,
            'count': count,
            'percentage': (count / len(connecting_paths)) * 100 if connecting_paths else 0,
            'examples': metapath_examples[metapath],
        })

    return {
        'total_unique_metapaths': len(metapath_counter),
        'metapaths': metapath_list,
        'entity_type_legend': ENTITY_TYPE_LEGEND,
    }


# ============================================================================
# Output writers
# ============================================================================

def save_json(paths_result, output_path):
    """Save full results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(paths_result, f, indent=2)
    print(f"  JSON saved to: {output_path}")


def save_metapaths_csv(paths_result, output_path):
    """Save metapaths to CSV with columns: Rank, Metapath, Count, Percentage, Example_Path."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Rank', 'Metapath', 'Count', 'Percentage', 'Example_Path'])
        for i, mp in enumerate(paths_result['metapaths']['metapaths'], 1):
            writer.writerow([
                i,
                mp['metapath'],
                mp['count'],
                f"{mp['percentage']:.2f}",
                mp['examples'][0] if mp['examples'] else '',
            ])
    print(f"  Metapaths CSV saved to: {output_path}")


def save_txt(paths_result, output_path):
    """Save full results to text file."""
    k = paths_result['k_hops']
    te = paths_result['test_edge']

    with open(output_path, 'w') as f:
        f.write(f"Test Edge: {te['head']} -> {te['relation']} -> {te['tail']}\n")
        f.write(f"Total {k}-hop paths: {paths_result['total_paths']}\n")
        f.write(f"Paths connecting head to tail: {paths_result['connecting_paths_count']}\n")
        f.write(f"Unique metapaths: {paths_result['metapaths']['total_unique_metapaths']}\n")
        f.write("=" * 80 + "\n\n")

        f.write("ENTITY TYPE LEGEND:\n")
        for abbr, full_name in paths_result['metapaths']['entity_type_legend'].items():
            f.write(f"  {abbr} = {full_name}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        f.write("METAPATHS (sorted by frequency):\n")
        f.write("=" * 80 + "\n\n")
        for i, mp in enumerate(paths_result['metapaths']['metapaths'], 1):
            f.write(f"{i}. {mp['metapath']}\n")
            f.write(f"   Count: {mp['count']} ({mp['percentage']:.2f}%)\n")
            f.write(f"   Examples:\n")
            for ex in mp['examples']:
                f.write(f"     - {ex}\n")
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n\n")
        f.write("ALL CONNECTING PATHS (Head -> Tail):\n")
        f.write("=" * 80 + "\n\n")
        for i, path in enumerate(paths_result['connecting_paths'], 1):
            f.write(f"{i}. {path['path_string']}\n")
            f.write(f"   Nodes: {' -> '.join(path['nodes'])}\n")
            f.write(f"   CURIEs: {' -> '.join(path['node_curies'])}\n\n")

    print(f"  Text file saved to: {output_path}")


def save_metapaths_summary(paths_result, output_path):
    """Save compact metapaths summary."""
    te = paths_result['test_edge']

    with open(output_path, 'w') as f:
        f.write("METAPATH SUMMARY\n")
        f.write(f"Test Edge: {te['head']} -> {te['relation']} -> {te['tail']}\n")
        f.write(f"Total connecting paths: {paths_result['connecting_paths_count']}\n")
        f.write(f"Unique metapaths: {paths_result['metapaths']['total_unique_metapaths']}\n")
        f.write("=" * 100 + "\n\n")

        f.write("ENTITY TYPE LEGEND:\n")
        for abbr, full_name in paths_result['metapaths']['entity_type_legend'].items():
            f.write(f"  {abbr} = {full_name}\n")
        f.write("\n" + "=" * 100 + "\n\n")

        f.write(f"{'Rank':<6} {'Count':<8} {'%':<8} Metapath\n")
        f.write("-" * 100 + "\n")
        for i, mp in enumerate(paths_result['metapaths']['metapaths'], 1):
            f.write(f"{i:<6} {mp['count']:<8} {mp['percentage']:<7.2f} {mp['metapath']}\n")

        f.write("\n" + "=" * 100 + "\n\n")
        f.write("TOP 20 METAPATHS WITH EXAMPLES:\n")
        f.write("=" * 100 + "\n\n")
        for i, mp in enumerate(paths_result['metapaths']['metapaths'][:20], 1):
            f.write(f"{i}. {mp['metapath']}\n")
            f.write(f"   Count: {mp['count']} ({mp['percentage']:.2f}%)\n")
            if mp['examples']:
                f.write(f"   Example: {mp['examples'][0]}\n\n")

    print(f"  Metapaths summary saved to: {output_path}")


# ============================================================================
# Main pipeline
# ============================================================================

def process_csv(csv_file, output_dir, hops=(2, 3, 4), max_edges=None):
    """
    Process a TracIn CSV file: extract paths for each hop count, write outputs.

    Args:
        csv_file: Path to input CSV
        output_dir: Directory for output files
        hops: Tuple of hop counts to extract (default: 2, 3, 4)
        max_edges: Maximum training edges to load
    """
    csv_path = Path(csv_file)
    stem = csv_path.stem
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {csv_file}")
    print("=" * 60)

    # Load data and build graph once
    data = load_test_triple_data(csv_file, max_edges)
    nodes, links, test_edge_info = build_graph_from_data(data)

    for k in hops:
        print(f"\n{'='*60}")
        print(f"Extracting {k}-hop paths...")
        print(f"{'='*60}")

        paths_result = extract_k_hop_paths(nodes, links, test_edge_info, k)

        te = paths_result['test_edge']
        print(f"Test Edge: {te['head']} -> {te['relation']} -> {te['tail']}")
        print(f"Total {k}-hop paths: {paths_result['total_paths']}")
        print(f"Connecting paths: {paths_result['connecting_paths_count']}")
        print(f"Unique metapaths: {paths_result['metapaths']['total_unique_metapaths']}")

        # Print top 10 metapaths
        for i, mp in enumerate(paths_result['metapaths']['metapaths'][:10], 1):
            print(f"  {i}. {mp['metapath']}  (count={mp['count']}, {mp['percentage']:.1f}%)")

        # Write output files per hop count
        prefix = f"{stem}_{k}hop"
        save_json(paths_result, out / f"{prefix}_paths.json")
        save_txt(paths_result, out / f"{prefix}_paths.txt")
        save_metapaths_csv(paths_result, out / f"{prefix}_metapaths.csv")
        save_metapaths_summary(paths_result, out / f"{prefix}_metapaths_summary.txt")

    print(f"\nDone. All outputs in: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract k-hop metapaths from TracIn CSV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_metapaths.py input.csv
  python extract_metapaths.py input.csv --output-dir results/
  python extract_metapaths.py input.csv --hops 2 3 4 --max-edges 5000
        """,
    )
    parser.add_argument('csv_file', help='Path to TracIn CSV file')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Output directory (default: same directory as input CSV)')
    parser.add_argument('--hops', '-k', nargs='+', type=int, default=[2, 3, 4],
                        help='Hop counts to extract (default: 2 3 4)')
    parser.add_argument('--max-edges', '-m', type=int, default=None,
                        help='Maximum number of training edges to load')
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or str(Path(args.csv_file).parent)

    process_csv(args.csv_file, output_dir, hops=tuple(args.hops), max_edges=args.max_edges)


if __name__ == '__main__':
    main()
