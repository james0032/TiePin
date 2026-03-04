# TracIn Analysis Pipeline
#
# Individual scripts (each has argparse — run with --help):
#   analysis/plot_tracin_distribution.py   # Step 1: Draw distribution plot
#   analysis/extract_tracin_paths.py       # Step 2: Extract 2/3-hop paths by top TracIn
#   analysis/join_path_edges_with_gt.py    # Step 3: Join path edges with ground truth
#
# Pipeline runner (chains all 3 steps, auto-wires outputs to inputs):
#   analysis/pipeline/run_tracin_pipeline.py
#
# Examples:
#   python analysis/pipeline/run_tracin_pipeline.py \
#       --input data/clean_baseline/examples/triple_005_CHEBI_30411_MONDO_0008228_tracin_with_gt.csv
#
#   python analysis/pipeline/run_tracin_pipeline.py \
#       --input-dir data/CGGD_alltreat/ --pattern "*_tracin.csv" --filter --k 2 3
#
#   python analysis/pipeline/run_tracin_pipeline.py \
#       --input data/triple_005_tracin.csv --edge-map data/edge_map.json --filter --output-dir results/
