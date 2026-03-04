#!/usr/bin/env python3
"""
Extract 2-hop and 3-hop paths from TracIn CSV files.

Builds a directed graph from TracIn training edges, then finds all k-hop
paths connecting the test head to the test tail. Paths are scored using
sum, arithmetic mean, geometric mean, and harmonic mean of edge TracIn scores.

Two input modes:
  1. Raw TracIn CSV  (columns: TrainRel_label) — no qualifier enrichment
  2. Filtered+enriched CSV (columns: TrainRel_label_enriched) — uses enriched labels

Usage:
    # Process specific files
    python extract_tracin_paths.py \
        --input triple_000_tracin.csv triple_001_tracin.csv \
        --k 2 3

    # Process all tracin CSVs in a directory
    python extract_tracin_paths.py \
        --input-dir /path/to/tracin_csvs/ \
        --pattern "*_tracin_filtered_enriched.csv" \
        --k 2 3

    # Filter raw TracIn first, enrich with edge map, then extract
    python extract_tracin_paths.py \
        --input triple_000_tracin.csv \
        --edge-map /path/to/edge_map.json \
        --k 2 3

    # Custom output directory
    python extract_tracin_paths.py \
        --input triple_000_tracin.csv \
        --output-dir /path/to/output/ \
        --k 2 3
"""

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def calculate_path_scores(edge_scores: List[float]) -> Dict[str, float]:
    """Calculate sum, mean, geometric mean, and harmonic mean for a path."""
    if not edge_scores:
        return {"sum": 0.0, "mean": 0.0, "geometric_mean": 0.0, "harmonic_mean": 0.0}

    arr = np.array(edge_scores)
    sum_score = float(np.sum(arr))
    mean_score = float(np.mean(arr))

    # Geometric mean (sign-preserving via absolute values)
    abs_arr = np.abs(arr)
    if np.any(abs_arr == 0):
        geometric_mean = 0.0
    else:
        geometric_mean = float(stats.gmean(abs_arr))
        if np.prod(np.sign(arr)) < 0:
            geometric_mean = -geometric_mean

    # Harmonic mean
    if np.all(arr > 0):
        harmonic_mean = float(stats.hmean(arr))
    elif np.all(arr < 0):
        harmonic_mean = -float(stats.hmean(np.abs(arr)))
    else:
        harmonic_mean = float(stats.hmean(np.abs(arr)))

    return {
        "sum": sum_score,
        "mean": mean_score,
        "geometric_mean": geometric_mean,
        "harmonic_mean": harmonic_mean,
    }


# ---------------------------------------------------------------------------
# Edge-map enrichment helpers
# ---------------------------------------------------------------------------

def load_and_invert_edge_map(edge_map_file: str) -> Dict:
    """Load edge_map.json and invert: predicate_id -> qualifier dict."""
    with open(edge_map_file, "r") as f:
        edge_map = json.load(f)
    inverted = {}
    for json_key, predicate_id in edge_map.items():
        inverted[predicate_id] = json.loads(json_key)
    return inverted


def enrich_relation(predicate: str, predicate_label: str, edge_map: Dict) -> str:
    """Append qualifier info to a predicate label."""
    if predicate not in edge_map:
        return predicate_label
    quals = edge_map[predicate]
    parts = []
    if quals.get("object_direction_qualifier"):
        parts.append(quals["object_direction_qualifier"])
    if quals.get("object_aspect_qualifier"):
        parts.append(quals["object_aspect_qualifier"])
    if parts:
        return f"{predicate_label} ({' '.join(parts)})"
    return predicate_label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tracin_csv(
    csv_file: str,
    filter_by_self_influence: bool = False,
    edge_map: Optional[Dict] = None,
) -> List[dict]:
    """Load a TracIn CSV, optionally filter and enrich.

    Args:
        csv_file: Path to the CSV.
        filter_by_self_influence: If True, keep only rows where
            TracInScore >= SelfInfluence.
        edge_map: If provided, add TrainRel_label_enriched column.

    Returns:
        List of row dicts.
    """
    rows: List[dict] = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    initial = len(rows)

    if filter_by_self_influence:
        rows = [
            r for r in rows
            if float(r["TracInScore"]) >= float(r["SelfInfluence"])
        ]
        logger.info(
            "  Filtered by SelfInfluence: %d -> %d rows",
            initial, len(rows),
        )

    if edge_map is not None:
        for r in rows:
            r["TrainRel_label_enriched"] = enrich_relation(
                r["TrainRel"], r["TrainRel_label"], edge_map
            )
        n_enriched = sum(
            1 for r in rows
            if r["TrainRel_label"] != r["TrainRel_label_enriched"]
        )
        logger.info("  Enriched %d relation labels with qualifiers", n_enriched)

    logger.info("  Loaded %d training edges from %s", len(rows), csv_file)
    return rows


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def build_graph(rows: List[dict]) -> Tuple[dict, list, dict]:
    """Build node dict, edge list, and test-edge info from TracIn rows.

    Returns:
        (nodes, links, test_edge_info)
    """
    nodes: Dict[str, dict] = {}
    links: List[dict] = []
    test_edge_info: Optional[dict] = None

    for row in rows:
        if test_edge_info is None:
            test_edge_info = {
                "head": row["TestHead"],
                "head_label": row["TestHead_label"],
                "tail": row["TestTail"],
                "tail_label": row["TestTail_label"],
                "relation": row["TestRel_label"],
                "relation_id": row["TestRel"],
            }

        score = float(row["TracInScore"])

        # Pick the best available relation label
        rel_label = row.get("TrainRel_label_enriched", row["TrainRel_label"])

        for curie, label in [
            (row["TrainHead"], row["TrainHead_label"]),
            (row["TrainTail"], row["TrainTail_label"]),
        ]:
            if curie not in nodes:
                etype = "test_entity" if curie in (row["TestHead"], row["TestTail"]) else "train_entity"
                nodes[curie] = {"id": curie, "label": label, "type": etype}

        links.append({
            "source": row["TrainHead"],
            "target": row["TrainTail"],
            "relation": rel_label,
            "relation_id": row["TrainRel"],
            "score": score,
            "type": "train_edge",
        })

    # Ensure test entities exist in node set
    if test_edge_info:
        for curie, lbl_key in [
            (test_edge_info["head"], "head_label"),
            (test_edge_info["tail"], "tail_label"),
        ]:
            if curie not in nodes:
                nodes[curie] = {"id": curie, "label": test_edge_info[lbl_key], "type": "test_entity"}

        links.append({
            "source": test_edge_info["head"],
            "target": test_edge_info["tail"],
            "relation": test_edge_info["relation"],
            "relation_id": test_edge_info.get("relation_id", ""),
            "score": None,
            "type": "test_edge",
        })

    logger.info("  Graph: %d nodes, %d edges", len(nodes), len(links))
    return nodes, links, test_edge_info


# ---------------------------------------------------------------------------
# Path extraction (BFS, unidirectional)
# ---------------------------------------------------------------------------

def extract_k_hop_paths(
    nodes: dict,
    links: list,
    test_edge_info: dict,
    k: int = 3,
) -> dict:
    """Extract all k-hop directed paths from test head to test tail.

    Uses BFS up to depth k from both test head and test tail.
    Only paths that start at test head and end at test tail are kept
    as "connecting" paths.
    """
    # Adjacency list + edge score lookup
    # Graph stores (target, relation_label, relation_id) per source
    graph: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    edge_scores: Dict[Tuple[str, str, str], float] = {}

    for link in links:
        src, tgt = link["source"], link["target"]
        rel = link.get("relation", "unknown")
        rel_id = link.get("relation_id", rel)
        sc = link.get("score", 0.0) if link.get("score") is not None else 0.0
        edge_scores[(src, tgt, rel)] = sc
        graph[src].append((tgt, rel, rel_id))

    test_head = test_edge_info["head"]
    test_tail = test_edge_info["tail"]

    all_paths: List[dict] = []

    for start_node in [test_head, test_tail]:
        start_label = "Test Head" if start_node == test_head else "Test Tail"

        # Each path entry: (node_curie, relation_label, relation_id)
        queue: deque = deque([(start_node, [(start_node, None, None)], {start_node})])

        while queue:
            current, path, visited = queue.popleft()

            if len(path) == k + 1:
                all_paths.append({"start": start_label, "path": path, "start_node": start_node})
                continue

            if len(path) > k:
                continue

            for neighbor, rel, rel_id in graph[current]:
                if neighbor not in visited:
                    new_visited = visited | {neighbor}
                    queue.append((neighbor, path + [(neighbor, rel, rel_id)], new_visited))

    # Format and filter
    formatted: List[dict] = []
    connecting: List[dict] = []

    for pinfo in all_paths:
        path = pinfo["path"]  # list of (node_curie, rel_label, rel_id)
        first_node = path[0][0]
        last_node = path[-1][0]
        is_connecting = (first_node == test_head and last_node == test_tail)

        esc: List[float] = []
        for i in range(len(path) - 1):
            cur = path[i][0]
            nxt = path[i + 1][0]
            rel = path[i + 1][1]
            esc.append(edge_scores.get((cur, nxt, rel), 0.0))

        scores = calculate_path_scores(esc)

        # Human-readable path string
        parts: List[str] = []
        for i, (curie, rel, _rel_id) in enumerate(path):
            label = nodes.get(curie, {}).get("label", curie)
            if i == 0:
                parts.append(label)
            else:
                parts.append(rel)
                parts.append(label)

        path_dict = {
            "path_string": " -> ".join(parts),
            "start_from": pinfo["start"],
            "nodes": [nodes.get(c, {}).get("label", c) for c, _, _ in path],
            "relations": [rel for _, rel, _ in path[1:]],
            "relation_ids": [rel_id for _, _, rel_id in path[1:]],
            "node_curies": [c for c, _, _ in path],
            "connects_test_entities": is_connecting,
            "edge_scores": esc,
            "score_sum": scores["sum"],
            "score_mean": scores["mean"],
            "score_geometric_mean": scores["geometric_mean"],
            "score_harmonic_mean": scores["harmonic_mean"],
        }
        formatted.append(path_dict)
        if is_connecting:
            connecting.append(path_dict)

    connecting.sort(key=lambda x: x["score_mean"], reverse=True)

    return {
        "total_paths": len(formatted),
        "connecting_paths_count": len(connecting),
        "k_hops": k,
        "test_edge": {
            "head": test_edge_info["head_label"],
            "relation": test_edge_info["relation"],
            "tail": test_edge_info["tail_label"],
            "head_curie": test_edge_info["head"],
            "tail_curie": test_edge_info["tail"],
        },
        "all_paths": formatted,
        "connecting_paths": connecting,
    }


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    base_name: str,
    k: int,
    result: dict,
) -> Tuple[str, str]:
    """Write JSON and human-readable TXT for one (file, k) pair."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{base_name}_{k}_hop_paths.json"
    txt_path = output_dir / f"{base_name}_{k}_hop_paths.txt"

    # JSON
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # TXT
    te = result["test_edge"]
    with open(txt_path, "w") as f:
        f.write(f"Test Edge: {te['head']} -> {te['relation']} -> {te['tail']}\n")
        f.write(f"Total {k}-hop paths: {result['total_paths']}\n")
        f.write(f"Paths connecting head to tail: {result['connecting_paths_count']}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CONNECTING PATHS (Head -> Tail, sorted by mean):\n")
        f.write("=" * 80 + "\n\n")
        for i, p in enumerate(result["connecting_paths"], 1):
            f.write(f"{i}. {p['path_string']}\n")
            f.write(f"   Nodes:  {' -> '.join(p['nodes'])}\n")
            f.write(f"   CURIEs: {' -> '.join(p['node_curies'])}\n")
            f.write(f"   Scores:\n")
            f.write(f"     Sum:            {p['score_sum']:.6f}\n")
            f.write(f"     Mean:           {p['score_mean']:.6f}\n")
            f.write(f"     Geometric Mean: {p['score_geometric_mean']:.6f}\n")
            f.write(f"     Harmonic Mean:  {p['score_harmonic_mean']:.6f}\n")
            f.write(f"   Edge Scores: {[f'{s:.6f}' for s in p['edge_scores']]}\n\n")

    logger.info("  Saved %s", json_path)
    logger.info("  Saved %s", txt_path)

    # Path-edges CSV — one row per edge in each connecting path.
    # Columns match what join_path_edges_with_gt.py expects.
    csv_path = output_dir / f"{base_name}_{k}_hop_path_edges.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path_rank", "k_hops", "path_string",
            "edge_pos", "TrainHead", "TrainRel", "TrainTail",
            "TrainHead_label", "TrainRel_label", "TrainTail_label",
            "TracInScore", "path_score_mean",
        ])
        for rank, p in enumerate(result["connecting_paths"], 1):
            for ei in range(len(p["node_curies"]) - 1):
                writer.writerow([
                    rank,
                    k,
                    p["path_string"],
                    ei + 1,
                    p["node_curies"][ei],
                    p["relation_ids"][ei],
                    p["node_curies"][ei + 1],
                    p["nodes"][ei],
                    p["relations"][ei],
                    p["nodes"][ei + 1],
                    f"{p['edge_scores'][ei]:.6f}",
                    f"{p['score_mean']:.6f}",
                ])
    logger.info("  Saved %s", csv_path)

    return str(json_path), str(txt_path), str(csv_path)


def save_combined_results(
    output_dir: Path,
    base_name: str,
    test_edge: dict,
    all_k_results: Dict[int, dict],
) -> str:
    """Write a combined TXT+JSON with connecting paths from all k-hop values,
    ranked together by mean of edge TracIn scores.

    Args:
        output_dir: Where to write the files.
        base_name: File stem (without k suffix).
        test_edge: Test edge info dict.
        all_k_results: {k: extract_k_hop_paths result dict}.

    Returns:
        Path to the combined TXT file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all connecting paths, tagging each with its hop count
    combined: List[dict] = []
    for k, result in sorted(all_k_results.items()):
        for p in result["connecting_paths"]:
            entry = dict(p)
            entry["k_hops"] = k
            combined.append(entry)

    # Sort by mean descending
    combined.sort(key=lambda x: x["score_mean"], reverse=True)

    # Counts per k
    counts = {k: res["connecting_paths_count"] for k, res in sorted(all_k_results.items())}

    # --- JSON ---
    json_path = output_dir / f"{base_name}_combined_paths.json"
    with open(json_path, "w") as f:
        json.dump({
            "test_edge": test_edge,
            "connecting_paths_by_k": {str(k): c for k, c in counts.items()},
            "total_connecting_paths": len(combined),
            "sorted_by": "mean",
            "paths": combined,
        }, f, indent=2)

    # --- TXT ---
    txt_path = output_dir / f"{base_name}_combined_paths.txt"
    te = test_edge
    with open(txt_path, "w") as f:
        f.write(f"Test Edge: {te['head']} -> {te['relation']} -> {te['tail']}\n")
        for k, cnt in counts.items():
            f.write(f"{k}-hop connecting paths: {cnt}\n")
        f.write(f"Total connecting paths: {len(combined)}\n")
        f.write("=" * 80 + "\n\n")
        f.write("ALL CONNECTING PATHS (combined k-hops, sorted by mean):\n")
        f.write("=" * 80 + "\n\n")
        for i, p in enumerate(combined, 1):
            f.write(f"{i}. [{p['k_hops']}-hop] {p['path_string']}\n")
            f.write(f"   Nodes:  {' -> '.join(p['nodes'])}\n")
            f.write(f"   CURIEs: {' -> '.join(p['node_curies'])}\n")
            f.write(f"   Scores:\n")
            f.write(f"     Sum:            {p['score_sum']:.6f}\n")
            f.write(f"     Mean:           {p['score_mean']:.6f}\n")
            f.write(f"     Geometric Mean: {p['score_geometric_mean']:.6f}\n")
            f.write(f"     Harmonic Mean:  {p['score_harmonic_mean']:.6f}\n")
            f.write(f"   Edge Scores: {[f'{s:.6f}' for s in p['edge_scores']]}\n\n")

    # Combined path-edges CSV
    csv_path = output_dir / f"{base_name}_combined_path_edges.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "path_rank", "k_hops", "path_string",
            "edge_pos", "TrainHead", "TrainRel", "TrainTail",
            "TrainHead_label", "TrainRel_label", "TrainTail_label",
            "TracInScore", "path_score_mean",
        ])
        for rank, p in enumerate(combined, 1):
            for ei in range(len(p["node_curies"]) - 1):
                writer.writerow([
                    rank,
                    p["k_hops"],
                    p["path_string"],
                    ei + 1,
                    p["node_curies"][ei],
                    p["relation_ids"][ei],
                    p["node_curies"][ei + 1],
                    p["nodes"][ei],
                    p["relations"][ei],
                    p["nodes"][ei + 1],
                    f"{p['edge_scores'][ei]:.6f}",
                    f"{p['score_mean']:.6f}",
                ])

    logger.info("  Saved combined: %s", txt_path)
    logger.info("  Saved combined: %s", json_path)
    logger.info("  Saved combined: %s", csv_path)
    return str(txt_path), str(csv_path)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def process_file(
    csv_file: str,
    k_values: List[int],
    output_dir: Path,
    filter_self_influence: bool,
    edge_map: Optional[Dict],
) -> List[dict]:
    """Process one TracIn CSV for all requested k values."""
    logger.info("Processing: %s", csv_file)

    rows = load_tracin_csv(
        csv_file,
        filter_by_self_influence=filter_self_influence,
        edge_map=edge_map,
    )

    if not rows:
        logger.warning("  No rows after loading/filtering — skipping")
        return []

    nodes, links, test_edge_info = build_graph(rows)

    base = Path(csv_file).stem
    # Strip common suffixes to keep output names clean
    for suffix in ["_tracin_filtered_enriched", "_tracin_filtered", "_tracin"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    summaries = []
    all_k_results: Dict[int, dict] = {}

    for k in k_values:
        logger.info("  Extracting %d-hop paths ...", k)
        result = extract_k_hop_paths(nodes, links, test_edge_info, k)
        all_k_results[k] = result

        te = result["test_edge"]
        logger.info(
            "  Test Edge: %s -> %s -> %s", te["head"], te["relation"], te["tail"],
        )
        logger.info("  Total %d-hop paths: %d", k, result["total_paths"])
        logger.info(
            "  Connecting (head->tail): %d", result["connecting_paths_count"],
        )

        if result["connecting_paths"]:
            logger.info("  Top 5 connecting paths by mean:")
            for i, p in enumerate(result["connecting_paths"][:5], 1):
                logger.info(
                    "    %d. [Mean=%.6f, Sum=%.6f] %s",
                    i,
                    p["score_mean"],
                    p["score_sum"],
                    p["path_string"],
                )

        json_out, txt_out, csv_out = save_results(output_dir, base, k, result)
        summaries.append({
            "file": csv_file,
            "k": k,
            "total_paths": result["total_paths"],
            "connecting_paths": result["connecting_paths_count"],
            "json_output": json_out,
            "txt_output": txt_out,
            "csv_output": csv_out,
            "status": "SUCCESS",
        })

    # Save combined file with all k-hop connecting paths ranked by mean
    if len(k_values) > 1 and all_k_results:
        te = result["test_edge"]
        save_combined_results(output_dir, base, te, all_k_results)

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Extract 2/3-hop paths from TracIn CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", nargs="+", metavar="CSV",
        help="One or more TracIn CSV files",
    )
    group.add_argument(
        "--input-dir", metavar="DIR",
        help="Directory containing TracIn CSV files",
    )
    parser.add_argument(
        "--pattern", default="*_tracin*.csv",
        help="Glob pattern when using --input-dir (default: *_tracin*.csv)",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[2, 3],
        help="Hop values to extract (default: 2 3)",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Output directory (default: same directory as each input file)",
    )
    parser.add_argument(
        "--edge-map", metavar="JSON", default=None,
        help="Path to edge_map.json for qualifier enrichment",
    )
    parser.add_argument(
        "--filter", action="store_true",
        help="Filter rows where TracInScore >= SelfInfluence before path extraction",
    )

    args = parser.parse_args()

    # Collect input files
    if args.input:
        csv_files = args.input
    else:
        input_dir = Path(args.input_dir)
        csv_files = sorted(str(p) for p in input_dir.glob(args.pattern))
        if not csv_files:
            logger.error("No files matching '%s' in %s", args.pattern, args.input_dir)
            return 1
        logger.info("Found %d files matching '%s'", len(csv_files), args.pattern)

    # Load edge map if provided
    edge_map = None
    if args.edge_map:
        logger.info("Loading edge map from %s", args.edge_map)
        edge_map = load_and_invert_edge_map(args.edge_map)
        logger.info("  %d predicate mappings loaded", len(edge_map))

    # Process
    all_summaries: List[dict] = []

    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            logger.error("File not found: %s", csv_file)
            all_summaries.append({
                "file": csv_file, "k": None,
                "total_paths": 0, "connecting_paths": 0,
                "status": "SKIPPED: not found",
            })
            continue

        out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
        try:
            sums = process_file(
                csv_file, args.k, out_dir, args.filter, edge_map,
            )
            all_summaries.extend(sums)
        except Exception:
            logger.exception("Error processing %s", csv_file)
            all_summaries.append({
                "file": csv_file, "k": None,
                "total_paths": 0, "connecting_paths": 0,
                "status": "FAILED",
            })

    # Summary table
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(
        "%-55s %-4s %-8s %-8s %-8s",
        "File", "k", "Total", "Connect", "Status",
    )
    logger.info("-" * 80)
    for s in all_summaries:
        short = Path(s["file"]).stem[:52]
        logger.info(
            "%-55s %-4s %-8s %-8s %-8s",
            short,
            s.get("k", "-"),
            s.get("total_paths", "-"),
            s.get("connecting_paths", "-"),
            s["status"],
        )

    ok = sum(1 for s in all_summaries if s["status"] == "SUCCESS")
    logger.info("-" * 80)
    logger.info("Total: %d runs, %d succeeded, %d failed", len(all_summaries), ok, len(all_summaries) - ok)

    return 0


if __name__ == "__main__":
    sys.exit(main())
