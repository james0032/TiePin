"""
Snakemake pipeline for ConvE PyKEEN Knowledge Graph Embedding and TracIn Analysis

This pipeline automates the complete workflow:
1. Create ROBOKOP subgraph (with style-based filtering)
1b. Prepare dictionary files
2. Extract mechanistic paths from DrugMechDB
2b. Filter treats edges with DrugMechDB paths (using add_pair_exists_column.py)
3. Extract DrugMechDB test set
4. Split data into train/valid
5. Train ConvE model
6. Evaluate model
7. Run TracIn analysis

Usage:
    snakemake --cores all
    snakemake --cores all --configfile config.yaml
    snakemake -n  # Dry run
"""

configfile: "config.yaml"

# Get style from config to use as base directory
# All outputs will be organized under {BASE_DIR}/{style}/
STYLE = config.get("style", "CGGD_alltreat")
BASE_DIR = f"{config['BASE_DIR']}/{STYLE}"

# Define all final outputs
rule all:
    input:
        # Subgraph files
        f"{BASE_DIR}/rotorobo.txt",
        f"{BASE_DIR}/edge_map.json",
        # Mechanistic paths output (from extract_mechanistic_paths)
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results.txt",
        f"{BASE_DIR}/results/mechanistic_paths/treats.txt",
        # Filtered treats edges (from filter_treats_with_drugmechdb)
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt",
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results_annotated.csv",
        f"{BASE_DIR}/results/mechanistic_paths/treats_annotated.txt",
        # DrugMechDB test set (from extract_drugmechdb_test)
        f"{BASE_DIR}/test.txt",
        f"{BASE_DIR}/train_candidates.txt",
        f"{BASE_DIR}/test_statistics.json",
        # Train/valid split (from split_data)
        f"{BASE_DIR}/train.txt",
        f"{BASE_DIR}/valid.txt",
        f"{BASE_DIR}/split_statistics.json",
        # Dictionary files (from prepare_dictionaries)
        f"{BASE_DIR}/processed/node_dict.txt",
        f"{BASE_DIR}/processed/node_name_dict.txt",
        f"{BASE_DIR}/processed/rel_dict.txt",
        f"{BASE_DIR}/processed/graph_stats.txt",
        # Trained model (PyKEEN outputs)
        f"{BASE_DIR}/models/conve/config.json",
        # Note: test_results.json from train.py is optional (only if skip_evaluation=false)
        # Evaluation results (score_only.py outputs) - use this for reliable evaluation
        f"{BASE_DIR}/results/evaluation/test_scores.json",
        f"{BASE_DIR}/results/evaluation/test_scores_ranked.json"
        # Note: TracIn analysis is disabled when using train.py (PyKEEN)
        # To enable TracIn, use train_pytorch.py instead

# ============================================================================
# Step 1: Create ROBOKOP Subgraph
# ============================================================================

rule create_subgraph:
    """
    Create ROBOKOP subgraph from edges file with specified filtering style
    The output directory is automatically robokop/{style}/ based on --style parameter
    """
    input:
        node_file = config["node_file"],
        edges_file = config["edges_file"]
    output:
        subgraph = f"{BASE_DIR}/rotorobo.txt",
        edge_map = f"{BASE_DIR}/edge_map.json"
    params:
        style = STYLE,
        outdir = BASE_DIR
    log:
        f"{BASE_DIR}/logs/create_subgraph.log"
    shell:
        """
        python src/create_robokop_subgraph.py \
            --node-file {input.node_file} \
            --edges-file {input.edges_file} \
            --style {params.style} \
            --outdir {params.outdir} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 1b: Prepare Dictionary Files
# ============================================================================

rule prepare_dictionaries:
    """
    Generate node_dict and rel_dict from subgraph data
    """
    input:
        subgraph = f"{BASE_DIR}/rotorobo.txt",
        edge_map = f"{BASE_DIR}/edge_map.json",
        nodes_file = config["node_file"]
    output:
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",
        node_name_dict = f"{BASE_DIR}/processed/node_name_dict.txt",
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt",
        stats = f"{BASE_DIR}/processed/graph_stats.txt"
    params:
        output_dir = f"{BASE_DIR}/processed"
    log:
        f"{BASE_DIR}/logs/prepare_dictionaries.log"
    shell:
        """
        python src/prepare_dict.py \
            --input {input.subgraph} \
            --edge-map {input.edge_map} \
            --nodes-file {input.nodes_file} \
            --output-dir {params.output_dir} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 2: Extract Mechanistic Paths from DrugMechDB
# ============================================================================

rule extract_mechanistic_paths:
    """
    Extract drug-disease mechanistic paths from DrugMechDB using path_id
    Note: Depends on Step 1 completing first to ensure sequential execution
    """
    input:
        edges_file = config["edges_file"],
        subgraph = f"{BASE_DIR}/rotorobo.txt"  # Wait for Step 1 to complete
    output:
        path_results = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results.txt",
        treats_tsv = f"{BASE_DIR}/results/mechanistic_paths/treats.txt",
        json_results = f"{BASE_DIR}/results/mechanistic_paths/treats_mechanistic_paths.json" if config.get("run_old_method", False) else []
    params:
        output_dir = f"{BASE_DIR}/results/mechanistic_paths",
        max_length = config.get("max_path_length", 5),
        run_old = "--run_old_method" if config.get("run_old_method", False) else ""
    log:
        f"{BASE_DIR}/logs/extract_mechanistic_paths.log"
    shell:
        """
        python src/find_mechanistic_paths.py \
            --edges_file {input.edges_file} \
            --max_length {params.max_length} \
            --output_dir {params.output_dir} \
            {params.run_old} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 2b: Filter Treats Edges with DrugMechDB Paths
# ============================================================================

rule filter_treats_with_drugmechdb:
    """
    Filter treats edges to only include those with DrugMechDB mechanistic paths
    Uses add_pair_exists_column.py to cross-reference treats.txt with drugmechdb_path_id_results.txt
    """
    input:
        path_results_txt = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results.txt",
        treats_tsv = f"{BASE_DIR}/results/mechanistic_paths/treats.txt"
    output:
        filtered_tsv = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt",
        annotated_csv = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results_annotated.csv",
        annotated_tsv = f"{BASE_DIR}/results/mechanistic_paths/treats_annotated.txt"
    log:
        f"{BASE_DIR}/logs/filter_treats_with_drugmechdb.log"
    shell:
        """
        python utils/add_pair_exists_column.py \
            {input.path_results_txt} \
            {input.treats_tsv} \
            {output.annotated_csv} \
            {output.annotated_tsv} \
            {output.filtered_tsv} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 3: Extract DrugMechDB Test Set
# ============================================================================

rule extract_drugmechdb_test:
    """
    Extract test edges from DrugMechDB-verified treats edges and remove them from rotorobo.txt
    Uses edge_map.json to convert biolink:treats to predicate IDs for matching
    """
    input:
        subgraph = f"{BASE_DIR}/rotorobo.txt",
        edge_map = f"{BASE_DIR}/edge_map.json",  # Required: maps biolink:treats to predicate IDs
        filtered_tsv = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt",
        # Force Step 1b to run (dict files not used in this step but needed for pipeline completeness)
        node_dict = f"{BASE_DIR}/processed/node_dict.txt"
    output:
        test = f"{BASE_DIR}/test.txt",
        train_candidates = f"{BASE_DIR}/train_candidates.txt",
        stats = f"{BASE_DIR}/test_statistics.json"
    params:
        input_dir = BASE_DIR,
        test_pct = config.get("drugmechdb_test_pct", 0.10),
        seed = config.get("random_seed", 42)
    log:
        f"{BASE_DIR}/logs/extract_drugmechdb_test.log"
    shell:
        """
        python src/make_test_with_drugmechdb_treat.py \
            --input-dir {params.input_dir} \
            --filtered-tsv {input.filtered_tsv} \
            --test-pct {params.test_pct} \
            --seed {params.seed} \
            --output-dir {params.input_dir} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 4: Split Data into Train/Valid
# ============================================================================

rule split_data:
    """
    Split train_candidates into training and validation sets
    Note: test.txt is already created in Step 3
    """
    input:
        train_candidates = f"{BASE_DIR}/train_candidates.txt"
    output:
        train = f"{BASE_DIR}/train.txt",
        valid = f"{BASE_DIR}/valid.txt",
        stats = f"{BASE_DIR}/split_statistics.json"
    params:
        output_dir = BASE_DIR,
        train_ratio = config.get("train_ratio", 0.9),
        seed = config.get("random_seed", 42)
    log:
        f"{BASE_DIR}/logs/split_data.log"
    shell:
        """
        python src/train_valid_split.py \
            --input {input.train_candidates} \
            --output-dir {params.output_dir} \
            --train-ratio {params.train_ratio} \
            --seed {params.seed} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 5: Train ConvE Model
# ============================================================================

rule train_model:
    """
    Train ConvE knowledge graph embedding model using PyKEEN
    """
    input:
        train = f"{BASE_DIR}/train.txt",
        valid = f"{BASE_DIR}/valid.txt",
        test = f"{BASE_DIR}/test.txt",
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt"
    output:
        # PyKEEN outputs - model is saved in directory structure
        model_dir = directory(f"{BASE_DIR}/models/conve"),
        config_out = f"{BASE_DIR}/models/conve/config.json"
        # Note: test_results.json is created only if skip_evaluation=false
        # Use evaluate_model rule for separate evaluation
    params:
        output_dir = f"{BASE_DIR}/models/conve",
        num_epochs = config.get("num_epochs", 100),
        batch_size = config.get("batch_size", 256),
        learning_rate = config.get("learning_rate", 0.001),
        embedding_dim = config.get("embedding_dim", 200),
        embedding_height = config.get("embedding_height", 10),
        embedding_width = config.get("embedding_width", 20),
        output_channels = config.get("output_channels", 32),
        kernel_height = config.get("kernel_height", 3),
        kernel_width = config.get("kernel_width", 3),
        input_dropout = config.get("input_dropout", 0.2),
        feature_map_dropout = config.get("feature_map_dropout", 0.2),
        output_dropout = config.get("output_dropout", 0.3),
        label_smoothing = config.get("label_smoothing", 0.1),
        checkpoint_frequency = config.get("checkpoint_frequency", 2),
        patience = config.get("patience", 10),
        random_seed = config.get("random_seed", 42),
        no_early_stopping = "" if config.get("early_stopping", True) else "--no-early-stopping",
        gpu = "" if config.get("use_gpu", True) else "--no-gpu",
        checkpoint_dir_arg = f"--checkpoint-dir {BASE_DIR}/{config.get('checkpoint_dir')}" if config.get("checkpoint_dir") else "",
        skip_evaluation = "--skip-evaluation" if config.get("skip_evaluation", False) else ""
    log:
        f"{BASE_DIR}/logs/train_model.log"
    resources:
        gpu = 1 if config.get("use_gpu", True) else 0
    shell:
        """
        python train.py \
            --train {input.train} \
            --valid {input.valid} \
            --test {input.test} \
            --entity-to-id {input.node_dict} \
            --relation-to-id {input.rel_dict} \
            --output-dir {params.output_dir} \
            --num-epochs {params.num_epochs} \
            --batch-size {params.batch_size} \
            --learning-rate {params.learning_rate} \
            --embedding-dim {params.embedding_dim} \
            --embedding-height {params.embedding_height} \
            --embedding-width {params.embedding_width} \
            --output-channels {params.output_channels} \
            --kernel-height {params.kernel_height} \
            --kernel-width {params.kernel_width} \
            --input-dropout {params.input_dropout} \
            --feature-map-dropout {params.feature_map_dropout} \
            --output-dropout {params.output_dropout} \
            --label-smoothing {params.label_smoothing} \
            --checkpoint-frequency {params.checkpoint_frequency} \
            --patience {params.patience} \
            --random-seed {params.random_seed} \
            {params.checkpoint_dir_arg} \
            {params.no_early_stopping} \
            {params.skip_evaluation} \
            {params.gpu} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 6: Evaluate Model (Score Test Triples)
# ============================================================================

rule evaluate_model:
    """
    Score test triples using trained ConvE model with score_only.py
    This runs AFTER train.py completes and provides detailed scoring
    """
    input:
        # Ensure training is complete - depend on config.json which is created at the end
        config_out = f"{BASE_DIR}/models/conve/config.json",
        test = f"{BASE_DIR}/test.txt",
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",
        node_name_dict = f"{BASE_DIR}/processed/node_name_dict.txt",
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt"
    output:
        scores_json = f"{BASE_DIR}/results/evaluation/test_scores.json",
        scores_csv = f"{BASE_DIR}/results/evaluation/test_scores.csv",
        scores_ranked_json = f"{BASE_DIR}/results/evaluation/test_scores_ranked.json",
        scores_ranked_csv = f"{BASE_DIR}/results/evaluation/test_scores_ranked.csv"
    params:
        model_dir = f"{BASE_DIR}/models/conve",
        output_dir = f"{BASE_DIR}/results/evaluation",
        use_sigmoid = "--use-sigmoid" if config.get("use_sigmoid", False) else "",
        top_n_arg = f"--top-n {config.get('top_n_triples')}" if config.get("top_n_triples") else "",
        device = "cuda" if config.get("use_gpu", True) else "cpu"
    log:
        f"{BASE_DIR}/logs/evaluate_model.log"
    shell:
        """
        mkdir -p {params.output_dir}

        python score_only.py \
            --model-dir {params.model_dir} \
            --test {input.test} \
            --entity-to-id {input.node_dict} \
            --relation-to-id {input.rel_dict} \
            --node-name-dict {input.node_name_dict} \
            --output {output.scores_json} \
            --device {params.device} \
            {params.use_sigmoid} \
            {params.top_n_arg} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 7: TracIn Analysis (DISABLED - requires train_pytorch.py outputs)
# ============================================================================
# NOTE: TracIn analysis requires checkpoint files from train_pytorch.py.
# PyKEEN's train.py uses a different checkpoint format via callbacks.
# To use TracIn, switch back to train_pytorch.py in the train_model rule.

# ============================================================================
# Utility Rules
# ============================================================================

rule clean:
    """
    Remove all generated files for the current style
    """
    params:
        base_dir = BASE_DIR
    shell:
        """
        rm -rf {params.base_dir}
        """

rule clean_models:
    """
    Remove only trained models (keep data preprocessing)
    """
    params:
        models_dir = f"{BASE_DIR}/models",
        eval_dir = f"{BASE_DIR}/results/evaluation",
        tracin_dir = f"{BASE_DIR}/results/tracin"
    shell:
        """
        rm -rf {params.models_dir}
        rm -rf {params.eval_dir}
        rm -rf {params.tracin_dir}
        """

rule clean_results:
    """
    Remove only analysis results (keep models and data)
    """
    params:
        eval_dir = f"{BASE_DIR}/results/evaluation",
        tracin_dir = f"{BASE_DIR}/results/tracin"
    shell:
        """
        rm -rf {params.eval_dir}
        rm -rf {params.tracin_dir}
        """

rule clean_all_styles:
    """
    Remove all robokop directories (all styles)
    """
    shell:
        """
        rm -rf robokop/
        """
