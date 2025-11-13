"""
Snakemake pipeline for ConvE PyKEEN Knowledge Graph Embedding and TracIn Analysis

This pipeline automates the complete workflow:
1. Create ROBOKOP subgraph (with style-based filtering)
2. Extract mechanistic paths from DrugMechDB
2b. Filter treats edges with DrugMechDB paths (using add_pair_exists_column.py)
3. Extract DrugMechDB test set
4. Prepare dictionary files
5. Split data into train/valid/test
6. Preprocess data for PyKEEN
7. Train ConvE model
8. Evaluate model
9. Run TracIn analysis

Usage:
    snakemake --cores all
    snakemake --cores all --configfile config.yaml
    snakemake -n  # Dry run
"""

configfile: "config.yaml"

# Get style from config to use as base directory
# All outputs will be organized under /workspace/data/robokop/{style}/
STYLE = config.get("style", "CGGD_alltreat")
BASE_DIR = f"/workspace/data/robokop/{STYLE}"

# Define all final outputs
rule all:
    input:
        # Subgraph files
        f"{BASE_DIR}/rotorobo.txt",
        f"{BASE_DIR}/edge_map.json",
        # Mechanistic paths output
        f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_path_id_results.txt",
        # DrugMechDB test set
        f"{BASE_DIR}/test.txt",
        f"{BASE_DIR}/train_candidates.txt",
        # Dictionary files
        f"{BASE_DIR}/processed/node_dict.txt",
        f"{BASE_DIR}/processed/rel_dict.txt",
        # Preprocessed data
        f"{BASE_DIR}/processed/train.txt",
        f"{BASE_DIR}/processed/valid.txt",
        f"{BASE_DIR}/processed/test.txt",
        # Trained model
        f"{BASE_DIR}/models/conve/best_model.pt",
        f"{BASE_DIR}/models/conve/final_model.pt",
        f"{BASE_DIR}/models/conve/test_results.json",
        # Evaluation results (score_only.py outputs)
        f"{BASE_DIR}/results/evaluation/test_scores.json",
        f"{BASE_DIR}/results/evaluation/test_scores_ranked.json",
        # TracIn analysis (optional)
        expand(f"{BASE_DIR}/results/tracin/tracin_analysis_{{batch}}.json",
               batch=range(config.get("tracin_batches", 1))) if config.get("run_tracin", False) else []

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
    """
    input:
        subgraph = f"{BASE_DIR}/rotorobo.txt",
        edge_map = f"{BASE_DIR}/edge_map.json",
        filtered_tsv = f"{BASE_DIR}/results/mechanistic_paths/drugmechdb_treats_filtered.txt"
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
# Step 4: Prepare Dictionary Files
# ============================================================================

rule prepare_dictionaries:
    """
    Generate node_dict and rel_dict from subgraph data
    """
    input:
        subgraph = f"{BASE_DIR}/rotorobo.txt",
        edge_map = f"{BASE_DIR}/edge_map.json"
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
            --output-dir {params.output_dir} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 5: Split Data into Train/Valid/Test
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
# Step 6: Preprocess Data for PyKEEN
# ============================================================================

rule preprocess_data:
    """
    Convert data to PyKEEN format while preserving dictionary indices
    """
    input:
        train = f"{BASE_DIR}/train.txt",
        valid = f"{BASE_DIR}/valid.txt",
        test = f"{BASE_DIR}/test.txt",
        node_dict = f"{BASE_DIR}/processed/node_dict.txt",
        rel_dict = f"{BASE_DIR}/processed/rel_dict.txt",
        edge_map = f"{BASE_DIR}/edge_map.json"
    output:
        train_out = f"{BASE_DIR}/processed/train.txt",
        valid_out = f"{BASE_DIR}/processed/valid.txt",
        test_out = f"{BASE_DIR}/processed/test.txt",
        entity_map = f"{BASE_DIR}/processed/train_entity_to_id.tsv",
        relation_map = f"{BASE_DIR}/processed/train_relation_to_id.tsv"
    params:
        output_dir = f"{BASE_DIR}/processed",
        validate = "" if config.get("validate_data", True) else "--no-validate"
    log:
        f"{BASE_DIR}/logs/preprocess_data.log"
    shell:
        """
        python preprocess.py \
            --train {input.train} \
            --valid {input.valid} \
            --test {input.test} \
            --node-dict {input.node_dict} \
            --rel-dict {input.rel_dict} \
            --edge-map {input.edge_map} \
            --output-dir {params.output_dir} \
            {params.validate} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 7: Train ConvE Model
# ============================================================================

rule train_model:
    """
    Train ConvE knowledge graph embedding model using pure PyTorch
    """
    input:
        train = f"{BASE_DIR}/processed/train.txt",
        valid = f"{BASE_DIR}/processed/valid.txt",
        test = f"{BASE_DIR}/processed/test.txt",
        entity_map = f"{BASE_DIR}/processed/train_entity_to_id.tsv",
        relation_map = f"{BASE_DIR}/processed/train_relation_to_id.tsv"
    output:
        best_model = f"{BASE_DIR}/models/conve/best_model.pt",
        final_model = f"{BASE_DIR}/models/conve/final_model.pt",
        config_out = f"{BASE_DIR}/models/conve/config.json",
        test_results = f"{BASE_DIR}/models/conve/test_results.json",
        training_history = f"{BASE_DIR}/models/conve/training_history.json"
    params:
        output_dir = f"{BASE_DIR}/models/conve",
        checkpoint_dir = f"{BASE_DIR}/models/conve/checkpoints",
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
        num_workers = config.get("num_workers", 4),
        random_seed = config.get("random_seed", 42),
        gpu = "" if config.get("use_gpu", True) else "--no-gpu"
    log:
        f"{BASE_DIR}/logs/train_model.log"
    resources:
        gpu = 1 if config.get("use_gpu", True) else 0
    shell:
        """
        python train_pytorch.py \
            --train {input.train} \
            --valid {input.valid} \
            --test {input.test} \
            --entity-to-id {input.entity_map} \
            --relation-to-id {input.relation_map} \
            --output-dir {params.output_dir} \
            --checkpoint-dir {params.checkpoint_dir} \
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
            --num-workers {params.num_workers} \
            --random-seed {params.random_seed} \
            {params.gpu} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 8: Evaluate Model
# ============================================================================

rule evaluate_model:
    """
    Score test triples with ConvE model (without computing rankings)
    """
    input:
        best_model = f"{BASE_DIR}/models/conve/best_model.pt",
        test = f"{BASE_DIR}/processed/test.txt",
        entity_map = f"{BASE_DIR}/processed/train_entity_to_id.tsv",
        relation_map = f"{BASE_DIR}/processed/train_relation_to_id.tsv",
        node_name_dict = f"{BASE_DIR}/processed/node_name_dict.txt"
    output:
        scores_json = f"{BASE_DIR}/results/evaluation/test_scores.json",
        scores_csv = f"{BASE_DIR}/results/evaluation/test_scores.csv",
        ranked_json = f"{BASE_DIR}/results/evaluation/test_scores_ranked.json",
        ranked_csv = f"{BASE_DIR}/results/evaluation/test_scores_ranked.csv"
    params:
        model_dir = f"{BASE_DIR}/models/conve",
        output = f"{BASE_DIR}/results/evaluation/test_scores.json",
        use_sigmoid = "--use-sigmoid" if config.get("use_sigmoid", False) else "",
        top_n = lambda wildcards: f"--top-n {config.get('top_n_triples')}" if config.get('top_n_triples') else ""
    log:
        f"{BASE_DIR}/logs/evaluate_model.log"
    shell:
        """
        python score_only.py \
            --model-dir {params.model_dir} \
            --test {input.test} \
            --entity-to-id {input.entity_map} \
            --relation-to-id {input.relation_map} \
            --node-name-dict {input.node_name_dict} \
            --output {params.output} \
            {params.use_sigmoid} \
            {params.top_n} \
            2>&1 | tee {log}
        """

# ============================================================================
# Step 9: TracIn Analysis (Optional)
# ============================================================================

rule tracin_analysis:
    """
    Run TracIn analysis to compute training data influence on test predictions
    """
    input:
        best_model = f"{BASE_DIR}/models/conve/best_model.pt",
        test = f"{BASE_DIR}/processed/test.txt"
    output:
        analysis = f"{BASE_DIR}/results/tracin/tracin_analysis_{{batch}}.json"
    params:
        model_dir = f"{BASE_DIR}/models/conve",
        output_dir = f"{BASE_DIR}/results/tracin",
        learning_rate = config.get("learning_rate", 0.001),
        top_k = config.get("tracin_top_k", 10),
        max_test = config.get("tracin_max_test_triples", 100),
        batch = "{batch}"
    log:
        f"{BASE_DIR}/logs/tracin_analysis_{{batch}}.log"
    resources:
        gpu = 1 if config.get("use_gpu", True) else 0
    shell:
        """
        python run_tracin_fast.py \
            --model-dir {params.model_dir} \
            --learning-rate {params.learning_rate} \
            --top-k {params.top_k} \
            --max-test-triples {params.max_test} \
            --batch-id {params.batch} \
            --output-dir {params.output_dir} \
            2>&1 | tee {log}
        """

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
