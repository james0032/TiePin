#!/usr/bin/env python3
"""
TracIn analysis pipeline — runs three steps in sequence:

  1. plot_tracin_distribution  — histogram + KDE of TracIn scores
  2. extract_tracin_paths      — 2/3-hop path extraction with scoring
  3. join_path_edges_with_gt   — annotate path edges with ground truth

Each step's output is wired as input to the next step automatically.

Usage:
    # Single TracIn CSV (auto-discovers *_tracin_with_gt.csv for step 3)
    python pipeline/run_tracin_pipeline.py \
        --input data/triple_005_tracin.csv

    # Multiple files
    python pipeline/run_tracin_pipeline.py \
        --input data/triple_005_tracin.csv data/triple_006_tracin.csv

    # All TracIn CSVs in a directory
    python pipeline/run_tracin_pipeline.py \
        --input-dir data/tracin_results/ \
        --pattern "*_tracin.csv"

    # With filtering and edge-map enrichment
    python pipeline/run_tracin_pipeline.py \
        --input data/triple_005_tracin.csv \
        --filter \
        --edge-map data/edge_map.json

    # Explicit ground-truth file
    python pipeline/run_tracin_pipeline.py \
        --input data/triple_005_tracin.csv \
        --gt data/triple_005_tracin_with_gt.csv

    # Custom output directory
    python pipeline/run_tracin_pipeline.py \
        --input data/triple_005_tracin.csv \
        --output-dir results/

    # Skip specific steps
    python pipeline/run_tracin_pipeline.py \
        --input data/triple_005_tracin.csv \
        --skip-plot --skip-gt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the three pipeline steps as libraries
# Adjust sys.path so we can import from the analysis/ directory
_ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from plot_tracin_distribution import plot_tracin_distribution
from extract_tracin_paths import (
    load_and_invert_edge_map,
    load_tracin_csv,
    build_graph,
    extract_k_hop_paths,
    save_results,
    save_combined_results,
)
from join_path_edges_with_gt import join_path_edges_with_gt, extract_top_n_path_edges


def find_gt_file(tracin_csv: Path) -> Optional[Path]:
    """Try to find the corresponding *_tracin_with_gt.csv file.

    Naming convention: if input is ``triple_005_..._tracin.csv``,
    look for ``triple_005_..._tracin_with_gt.csv`` in the same directory.
    If the input itself is already a ``*_tracin_with_gt.csv``, return it.
    """
    stem = tracin_csv.stem

    # If the input IS the _with_gt file, use it directly
    if stem.endswith("_tracin_with_gt"):
        return tracin_csv

    # Strip known suffixes to get the base, then try _with_gt
    # Order matters: longer suffixes first to avoid partial matches
    for suffix in ["_tracin_filtered_enriched", "_tracin_filtered", "_tracin"]:
        if stem.endswith(suffix):
            base = stem[: -len(suffix)]
            candidate = tracin_csv.parent / f"{base}_tracin_with_gt.csv"
            if candidate.exists():
                return candidate
            break

    # Also try just appending _with_gt before .csv
    candidate = tracin_csv.parent / f"{stem}_with_gt.csv"
    if candidate.exists():
        return candidate

    return None


def get_base_name(csv_file: Path) -> str:
    """Strip common TracIn suffixes to get a clean base name."""
    stem = csv_file.stem
    for suffix in ["_tracin_filtered_enriched", "_tracin_filtered", "_tracin_with_gt", "_tracin"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def run_pipeline_for_file(
    tracin_csv: Path,
    output_dir: Path,
    k_values: List[int],
    filter_self_influence: bool,
    edge_map: Optional[Dict],
    gt_file: Optional[Path],
    skip_plot: bool,
    skip_paths: bool,
    skip_gt: bool,
    top_n: int = 20,
):
    """Run the full pipeline for one TracIn CSV file."""
    base = get_base_name(tracin_csv)
    logger.info("=" * 80)
    logger.info("Pipeline: %s", tracin_csv.name)
    logger.info("  Base name: %s", base)
    logger.info("  Output dir: %s", output_dir)
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Plot distribution
    # ------------------------------------------------------------------
    if not skip_plot:
        logger.info("")
        logger.info("--- Step 1: Plot TracIn score distribution ---")
        png_path = output_dir / f"{base}_tracin_distribution.png"
        try:
            plot_tracin_distribution(tracin_csv, png_path)
            logger.info("  Plot saved: %s", png_path)
        except Exception:
            logger.exception("  Step 1 failed")
    else:
        logger.info("--- Step 1: SKIPPED (--skip-plot) ---")

    # ------------------------------------------------------------------
    # Step 2: Extract paths
    # ------------------------------------------------------------------
    combined_path_edges_csv = None
    if not skip_paths:
        logger.info("")
        logger.info("--- Step 2: Extract %s-hop paths ---", "/".join(str(k) for k in k_values))
        try:
            rows = load_tracin_csv(
                str(tracin_csv),
                filter_by_self_influence=filter_self_influence,
                edge_map=edge_map,
            )
            if not rows:
                logger.warning("  No rows after loading/filtering — skipping path extraction")
            else:
                nodes, links, test_edge_info = build_graph(rows)
                all_k_results: Dict[int, dict] = {}

                for k in k_values:
                    logger.info("  Extracting %d-hop paths ...", k)
                    result = extract_k_hop_paths(nodes, links, test_edge_info, k)
                    all_k_results[k] = result

                    logger.info(
                        "  %d-hop: %d total, %d connecting",
                        k, result["total_paths"], result["connecting_paths_count"],
                    )
                    save_results(output_dir, base, k, result)

                # Combined file
                if len(k_values) > 1 and all_k_results:
                    te = list(all_k_results.values())[-1]["test_edge"]
                    _, combined_csv = save_combined_results(output_dir, base, te, all_k_results)
                    combined_path_edges_csv = Path(combined_csv)
                elif len(k_values) == 1:
                    # Single k — use that k's path_edges CSV
                    k = k_values[0]
                    combined_path_edges_csv = output_dir / f"{base}_{k}_hop_path_edges.csv"

        except Exception:
            logger.exception("  Step 2 failed")
    else:
        logger.info("--- Step 2: SKIPPED (--skip-paths) ---")

    # ------------------------------------------------------------------
    # Step 3: Join path edges with ground truth
    # ------------------------------------------------------------------
    if not skip_gt and combined_path_edges_csv is not None:
        # Auto-discover GT file if not provided
        actual_gt = gt_file
        if actual_gt is None:
            actual_gt = find_gt_file(tracin_csv)

        if actual_gt is not None and actual_gt.exists():
            logger.info("")
            logger.info("--- Step 3: Join path edges with ground truth ---")
            logger.info("  GT file: %s", actual_gt)
            logger.info("  Path edges: %s", combined_path_edges_csv)
            try:
                # Top-N path edges (unique edges from top N paths with GT labels)
                if top_n is not None:
                    logger.info("  Top N: %d", top_n)
                    top_n_csv = extract_top_n_path_edges(
                        actual_gt, combined_path_edges_csv, top_n,
                        output_path=output_dir / f"{combined_path_edges_csv.stem.replace('_path_edges', '')}_top{top_n}_path_edges.csv",
                    )
                    logger.info("  Top-N output: %s", top_n_csv)

                # Full join (all path edges with GT labels)
                out_csv = output_dir / f"{combined_path_edges_csv.stem}_with_gt.csv"
                join_path_edges_with_gt(actual_gt, combined_path_edges_csv, out_csv)
                logger.info("  Joined output: %s", out_csv)
            except Exception:
                logger.exception("  Step 3 failed")
        else:
            logger.info("")
            logger.info("--- Step 3: SKIPPED (no ground truth file found) ---")
            if gt_file:
                logger.warning("  Specified GT file not found: %s", gt_file)
    elif skip_gt:
        logger.info("--- Step 3: SKIPPED (--skip-gt) ---")
    elif combined_path_edges_csv is None:
        logger.info("--- Step 3: SKIPPED (no path edges from step 2) ---")

    logger.info("")
    logger.info("Pipeline complete for %s", tracin_csv.name)
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run the TracIn analysis pipeline (plot -> extract paths -> join GT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1. plot_tracin_distribution  — histogram + KDE of TracIn scores
  2. extract_tracin_paths      — 2/3-hop path extraction with scoring
  3. join_path_edges_with_gt   — annotate path edges with ground truth

Output files per input (e.g. triple_005_CHEBI_30411_MONDO_0008228):
  *_tracin_distribution.png         (step 1)
  *_2_hop_paths.json / .txt / .csv  (step 2)
  *_3_hop_paths.json / .txt / .csv  (step 2)
  *_combined_paths.json / .txt      (step 2)
  *_combined_path_edges.csv         (step 2)
  *_combined_path_edges_with_gt.csv (step 3)
        """,
    )

    # Input
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
        "--pattern", default="*_tracin.csv",
        help="Glob pattern when using --input-dir (default: *_tracin.csv)",
    )

    # Output
    parser.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Output directory (default: same directory as each input file)",
    )

    # Step 2 options
    parser.add_argument(
        "--k", nargs="+", type=int, default=[2, 3],
        help="Hop values to extract (default: 2 3)",
    )
    parser.add_argument(
        "--filter", action="store_true",
        help="Filter rows where TracInScore >= SelfInfluence",
    )
    parser.add_argument(
        "--edge-map", metavar="JSON", default=None,
        help="Path to edge_map.json for qualifier enrichment",
    )

    # Step 3 options
    parser.add_argument(
        "--gt", metavar="CSV", default=None,
        help="Explicit ground truth CSV (auto-discovered if not provided)",
    )

    # Skip flags
    parser.add_argument("--skip-plot", action="store_true", help="Skip step 1 (distribution plot)")
    parser.add_argument("--skip-paths", action="store_true", help="Skip step 2 (path extraction)")
    parser.add_argument("--skip-gt", action="store_true", help="Skip step 3 (GT join)")
    parser.add_argument(
        "--top-n", type=int, default=50, metavar="N",
        help="Extract unique edges from the top N paths (default: 20)",
    )

    args = parser.parse_args()

    # Collect input files
    if args.input:
        csv_files = [Path(f) for f in args.input]
    else:
        input_dir = Path(args.input_dir)
        csv_files = sorted(input_dir.glob(args.pattern))
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

    gt_file = Path(args.gt) if args.gt else None

    # Run pipeline for each file
    for csv_file in csv_files:
        if not csv_file.exists():
            logger.error("File not found: %s — skipping", csv_file)
            continue

        out_dir = Path(args.output_dir) if args.output_dir else csv_file.parent

        run_pipeline_for_file(
            tracin_csv=csv_file,
            output_dir=out_dir,
            k_values=args.k,
            filter_self_influence=args.filter,
            edge_map=edge_map,
            gt_file=gt_file,
            skip_plot=args.skip_plot,
            skip_paths=args.skip_paths,
            skip_gt=args.skip_gt,
            top_n=args.top_n,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
