# Given a base edges.jsonl from robokop, filter it and format the results to make the correct input for nn-geometric
# The input format is a tab-delimited file of subject\tpredicate\object\n.
import os
import argparse
import logging
from pathlib import Path

import jsonlines
import json
import igraph as ig
from collections import defaultdict

# Configure logger
logger = logging.getLogger(__name__)

def remove_subclass_and_cid(edge, typemap):
    if edge["predicate"] == "biolink:subclass_of":
        return True
    if edge["subject"].startswith("CAID"):
        return True
    return False

def check_accepted(edge, typemap, accepted):
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in accepted:
        if acc[0] in subj_types and acc[1] in obj_types:
            return False
        if acc[1] in subj_types and acc[0] in obj_types:
            return False
    return True

def check_remove(edge, typemap, remove):
    subj = edge["subject"]
    obj = edge["object"]
    subj_types = typemap.get(subj, set())
    obj_types = typemap.get(obj, set())
    for acc in remove:
        if acc[0] in subj_types and acc[1] in obj_types:
            return True
        if acc[1] in subj_types and acc[0] in obj_types:
            return True
    return False

def remove_CD(edge, typemap):
    remove = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
               ("biolink:DiseaseOrPhenotypicFeature", "biolink:ChemicalEntity")
              ]
    return check_remove(edge, typemap, remove)

def dont_remove(edge, typemap):
    return False

def keep_CD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature") ]
    return check_accepted(edge, typemap, accepted)

def keep_CCGGDD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CGGD(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:subclass_of":
        return True
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:Gene"),
                ("biolink:Gene", "biolink:Gene"),
                ("biolink:Gene", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def compute_low_degree_nodes(edges_file, min_degree=2):
    """
    Build a graph from edges.jsonl and identify nodes with degree < min_degree.

    Parameters
    ----------
    edges_file : str
        Path to edges.jsonl file
    min_degree : int
        Minimum degree threshold (nodes with degree < min_degree will be returned)

    Returns
    -------
    set
        Set of node IDs with degree < min_degree
    """
    logger.info(f"Computing low-degree nodes (degree < {min_degree})...")

    # Count degrees using a dictionary
    degree_count = defaultdict(int)
    edge_count = 0

    with jsonlines.open(edges_file) as reader:
        for edge in reader:
            subject = edge.get("subject", "")
            obj = edge.get("object", "")

            degree_count[subject] += 1
            degree_count[obj] += 1
            edge_count += 1

            if edge_count % 100000 == 0:
                logger.debug(f"Processed {edge_count} edges for degree calculation...")

    # Find nodes with degree < min_degree
    low_degree_nodes = {node for node, degree in degree_count.items() if degree < min_degree}

    logger.info(f"Found {len(low_degree_nodes):,} nodes with degree < {min_degree} out of {len(degree_count):,} total nodes")
    logger.info(f"Low-degree node percentage: {len(low_degree_nodes)/len(degree_count)*100:.2f}%")

    return low_degree_nodes


def clean_baseline_kg(edge, typemap, low_degree_nodes=None):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit

    # Initialize counters if not already present
    if not hasattr(clean_baseline_kg, 'predicate_counter'):
        clean_baseline_kg.predicate_counter = {}
        clean_baseline_kg.source_counter = {}
        clean_baseline_kg.filtered_by_predicate = 0
        clean_baseline_kg.filtered_by_source = 0
        clean_baseline_kg.filtered_by_reactome = 0
        clean_baseline_kg.filtered_by_bindingdb = 0
        clean_baseline_kg.filtered_by_ncit_subclass = 0
        clean_baseline_kg.filtered_by_low_degree = 0
        clean_baseline_kg.kept_edges = 0

    filtered_predicates = ["biolink:in_taxon", "biolink:related_to", "biolink:expressed_in",
                          "biolink:located_in", "biolink:temporally_related_to",
                          "biolink:affects_response_to", "biolink:decreases_response_to",
                          "biolink:increases_response_to"]

    filtered_sources = ["infores:text-mining-provider-targeted", "infores:zfin", "infores:tiga",
                       "infores:flybase", "infores:sgd", "infores:rgd", "infores:mgi"]
    # additional filters: non-human from reactome. text mined kp, affinity <7 from binidngdb. NCIT nodes using subclass_of predicate. Take out degree 1 or degree 2 nodes.
    predicate = edge.get("predicate", "")
    source = edge.get("primary_knowledge_source", "")
    subject = edge.get("subject", "")
    obj = edge.get("object", "")

    # Track predicate
    if predicate not in clean_baseline_kg.predicate_counter:
        clean_baseline_kg.predicate_counter[predicate] = 0
    clean_baseline_kg.predicate_counter[predicate] += 1

    # Track source
    if source not in clean_baseline_kg.source_counter:
        clean_baseline_kg.source_counter[source] = 0
    clean_baseline_kg.source_counter[source] += 1

    # Filter 1: Non-human Reactome edges
    # If subject or object has prefix REACT and middle part is not "HSA", remove edge
    if subject.startswith("REACT:") or obj.startswith("REACT:"):
        # Extract the middle part (species code) from REACT IDs
        # Format is typically REACT:R-XXX-... where XXX is the species code
        subject_is_non_human = False
        object_is_non_human = False

        if subject.startswith("REACT:"):
            # Split by '-' and check if the species code (second part) is not "HSA"
            parts = subject.split("-")
            if len(parts) >= 2 and parts[1] != "HSA":
                subject_is_non_human = True

        if obj.startswith("REACT:"):
            parts = obj.split("-")
            if len(parts) >= 2 and parts[1] != "HSA":
                object_is_non_human = True

        if subject_is_non_human or object_is_non_human:
            clean_baseline_kg.filtered_by_reactome += 1
            return True

    # Filter 2: BindingDB affinity filter
    # If source is bindingdb, only keep if affinity is not None and > 7
    if source == "infores:bindingdb":
        # Check for affinity attribute in edge attributes
        attributes = edge.get("attributes", [])
        affinity_value = None

        for attr in attributes:
            if attr.get("attribute_type_id") == "affinity":
                affinity_value = attr.get("value")
                break

        # If affinity is None or <= 7, filter out the edge
        if affinity_value is None or affinity_value <= 7:
            clean_baseline_kg.filtered_by_bindingdb += 1
            return True

    # Filter 3: NCIT subclass_of edges
    # If predicate is biolink:subclass_of and both subject and object have prefix "NCIT", remove
    if predicate == "biolink:subclass_of":
        if subject.startswith("NCIT:") and obj.startswith("NCIT:"):
            clean_baseline_kg.filtered_by_ncit_subclass += 1
            return True

    # Filter 4: Low-degree nodes
    # If subject or object is in low_degree_nodes, remove edge
    if low_degree_nodes is not None:
        if subject in low_degree_nodes or obj in low_degree_nodes:
            clean_baseline_kg.filtered_by_low_degree += 1
            return True

    # Original filters
    if predicate in filtered_predicates:
        clean_baseline_kg.filtered_by_predicate += 1
        return True
    elif source in filtered_sources:
        clean_baseline_kg.filtered_by_source += 1
        return True

    clean_baseline_kg.kept_edges += 1
    return False

def log_clean_baseline_kg_stats():
    """Log statistics about what clean_baseline_kg has processed."""
    if not hasattr(clean_baseline_kg, 'predicate_counter'):
        logger.info("No statistics available for clean_baseline_kg")
        return

    logger.info("=" * 80)
    logger.info("CLEAN_BASELINE_KG FILTERING STATISTICS")
    logger.info("=" * 80)

    total_processed = sum(clean_baseline_kg.predicate_counter.values())
    logger.info(f"Total edges processed: {total_processed:,}")
    logger.info(f"  Filtered by predicate: {clean_baseline_kg.filtered_by_predicate:,} ({clean_baseline_kg.filtered_by_predicate/total_processed*100:.2f}%)")
    logger.info(f"  Filtered by source: {clean_baseline_kg.filtered_by_source:,} ({clean_baseline_kg.filtered_by_source/total_processed*100:.2f}%)")
    logger.info(f"  Filtered by Reactome (non-human): {clean_baseline_kg.filtered_by_reactome:,} ({clean_baseline_kg.filtered_by_reactome/total_processed*100:.2f}%)")
    logger.info(f"  Filtered by BindingDB (affinity): {clean_baseline_kg.filtered_by_bindingdb:,} ({clean_baseline_kg.filtered_by_bindingdb/total_processed*100:.2f}%)")
    logger.info(f"  Filtered by NCIT subclass_of: {clean_baseline_kg.filtered_by_ncit_subclass:,} ({clean_baseline_kg.filtered_by_ncit_subclass/total_processed*100:.2f}%)")
    logger.info(f"  Filtered by low degree: {clean_baseline_kg.filtered_by_low_degree:,} ({clean_baseline_kg.filtered_by_low_degree/total_processed*100:.2f}%)")
    logger.info(f"  Kept edges: {clean_baseline_kg.kept_edges:,} ({clean_baseline_kg.kept_edges/total_processed*100:.2f}%)")

    logger.info("")
    logger.info("PREDICATE FREQUENCY (Top 20):")
    logger.info(f"  {'Predicate':<60} {'Count':>10} {'%':>8}")
    logger.info(f"  {'-'*60} {'-'*10} {'-'*8}")
    sorted_predicates = sorted(clean_baseline_kg.predicate_counter.items(), key=lambda x: x[1], reverse=True)
    for pred, count in sorted_predicates[:20]:
        percentage = count / total_processed * 100
        logger.info(f"  {pred:<60} {count:>10,} {percentage:>7.2f}%")

    if len(sorted_predicates) > 20:
        logger.info(f"  ... and {len(sorted_predicates) - 20} more predicates")

    logger.info("")
    logger.info("PRIMARY KNOWLEDGE SOURCE FREQUENCY (Top 20):")
    logger.info(f"  {'Source':<60} {'Count':>10} {'%':>8}")
    logger.info(f"  {'-'*60} {'-'*10} {'-'*8}")
    sorted_sources = sorted(clean_baseline_kg.source_counter.items(), key=lambda x: x[1], reverse=True)
    for source, count in sorted_sources[:20]:
        percentage = count / total_processed * 100
        logger.info(f"  {source:<60} {count:>10,} {percentage:>7.2f}%")

    if len(sorted_sources) > 20:
        logger.info(f"  ... and {len(sorted_sources) - 20} more sources")

    logger.info("=" * 80)

def keep_CGGD_alltreat(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:treats":
        return False
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:GeneOrGeneProduct"),
                ("biolink:GeneOrGeneProduct", "biolink:GeneOrGeneProduct"),
                ("biolink:GeneOrGeneProduct", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def keep_CCGGDD_alltreat(edge, typemap):
    # return True if you want to filter this edge out
    # We want to keep edges between chemicals and genes, between genes and disease, and between chemicals and diseases
    # Unfortunately this means that we need a type map... Dangit
    if edge["predicate"] == "biolink:treats":
        return False
    accepted = [ ("biolink:ChemicalEntity", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:ChemicalEntity", "biolink:ChemicalEntity"),
                ("biolink:ChemicalEntity", "biolink:GeneOrGeneProduct"),
                ("biolink:GeneOrGeneProduct", "biolink:GeneOrGeneProduct"),
                ("biolink:DiseaseOrPhenotypicFeature", "biolink:DiseaseOrPhenotypicFeature"),
                ("biolink:GeneOrGeneProduct", "biolink:DiseaseOrPhenotypicFeature")]
    return check_accepted(edge, typemap, accepted)

def pred_trans(edge, edge_map):
    edge_key = {"predicate": edge["predicate"]}
    edge_key["subject_aspect_qualifier"] = edge.get("subject_aspect_qualifier", "")
    edge_key["object_aspect_qualifier"] = edge.get("object_aspect_qualifier", "")
    edge_key["subject_direction_qualifier"] = edge.get("subject_direction_qualifier", "")
    edge_key["object_direction_qualifier"] = edge.get("object_direction_qualifier", "")
    edge_key_string = json.dumps(edge_key, sort_keys=True)
    if edge_key_string not in edge_map:
        edge_map[edge_key_string] = f"predicate:{len(edge_map)}"
    return edge_map[edge_key_string]


def dump_edge_map(edge_map, outdir):
    output_file=f"{outdir}/edge_map.json"
    logger.info(f"Writing edge map to {output_file}")
    with open(output_file, "w") as writer:
        json.dump(edge_map, writer, indent=2)
    logger.info(f"Edge map written with {len(edge_map)} unique predicates")

def create_robokop_input(node_file="robokop/nodes.jsonl", edges_file="robokop/edges.jsonl", style="original", outdir=None, min_degree=2):
    # Determine output directory
    if outdir is None:
        outdir = f"robokop/{style}"

    output_file = f"{outdir}/rotorobo.txt"

    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Node file: {node_file}")
    logger.info(f"  Edges file: {edges_file}")
    logger.info(f"  Style: {style}")
    logger.info(f"  Output directory: {outdir}")
    logger.info(f"  Output file: {output_file}")
    if style == "clean_baseline":
        logger.info(f"  Min degree threshold: {min_degree}")
    logger.info("=" * 80)

    # Select filtering strategy based on style
    if style == "original":
        # This filters the edges by
        # 1) removing all subclass_of and
        # 2) removing all edges with a subject that starts with "CAID"
        remove_edge = remove_subclass_and_cid
        logger.info("Using 'original' style: removing subclass_of and CAID edges")
    elif style == "CD":
        # No subclasses
        # only chemical/disease edges
        remove_edge = keep_CD
        logger.info("Using 'CD' style: keeping only Chemical-Disease edges")
    elif style == "CCGGDD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CCGGDD
        logger.info("Using 'CCGGDD' style: keeping Chemical-Chemical, Gene-Gene, Disease-Disease edges")
    elif style == "CGGD":
        # No subclasses
        # only chemical/disease edges and disease/disease edges
        remove_edge = keep_CGGD
        logger.info("Using 'CGGD' style: keeping Chemical-Gene-Disease edges")
    elif style == "rCD":
        remove_edge = remove_CD
        logger.info("Using 'rCD' style: removing Chemical-Disease edges")
    elif style == "keepall":
        remove_edge = dont_remove
        logger.info("Using 'keepall' style: keeping all edges")
    elif style == "CGGD_alltreat":
        remove_edge = keep_CGGD_alltreat
        logger.info("Using 'CGGD_alltreat' style: keeping CGGD edges plus all 'treats' relationships")
    elif style == "CCGGDD_alltreat":
        remove_edge = keep_CCGGDD_alltreat
        logger.info("Using 'CCGGDD_alltreat' style: keeping CCGGDD edges plus all 'treats' relationships")   
    elif style == "clean_baseline":
        remove_edge = clean_baseline_kg
        logger.info("Using 'clean_baseline' style: Remove almost irrelevant predicates and primary knowledge sources")   
        
    else:
        logger.error(f"Unknown style: {style}")
        logger.error("Valid styles: original, CD, CCGGDD, CGGD, rCD, keepall, CGGD_alltreat")
        raise ValueError(f"Unknown style: {style}")

    # Create output directory if it doesn't exist
    if not os.path.exists(outdir):
        logger.info(f"Creating output directory: {outdir}")
        os.makedirs(outdir)

    # Build type map from nodes file
    logger.info(f"Reading nodes from {node_file}...")
    type_map = {}
    node_count = 0
    with jsonlines.open(node_file) as reader:
        for node in reader:
            type_map[node["id"]] = set(node["category"])
            node_count += 1
            if node_count % 10000 == 0:
                logger.debug(f"Processed {node_count} nodes...")

    logger.info(f"Type map created with {len(type_map)} nodes")

    # Compute low-degree nodes if using clean_baseline style
    low_degree_nodes = None
    if style == "clean_baseline":
        low_degree_nodes = compute_low_degree_nodes(edges_file, min_degree=min_degree)

    # Process edges
    logger.info(f"Processing edges from {edges_file}...")
    edge_map = {}
    total_edges = 0
    filtered_edges = 0
    kept_edges = 0

    with jsonlines.open(edges_file) as reader:
        with open(output_file, "w") as writer:
            for edge in reader:
                total_edges += 1
                if total_edges % 100000 == 0:
                    logger.info(f"Processed {total_edges} edges, kept {kept_edges}, filtered {filtered_edges}")

                # Pass low_degree_nodes to clean_baseline_kg if applicable
                if style == "clean_baseline":
                    if remove_edge(edge, type_map, low_degree_nodes):
                        filtered_edges += 1
                        continue
                else:
                    if remove_edge(edge, type_map):
                        filtered_edges += 1
                        continue

                writer.write(f"{edge['subject']}\t{pred_trans(edge,edge_map)}\t{edge['object']}\n")
                kept_edges += 1

    logger.info(f"Edge processing complete:")
    logger.info(f"  Total edges processed: {total_edges}")
    logger.info(f"  Edges kept: {kept_edges}")
    logger.info(f"  Edges filtered: {filtered_edges}")
    logger.info(f"  Filter rate: {(filtered_edges/total_edges*100):.2f}%")

    # Log detailed statistics if using clean_baseline style
    if style == "clean_baseline":
        log_clean_baseline_kg_stats()

    dump_edge_map(edge_map, outdir)
    logger.info(f"Subgraph creation complete! Output written to {output_file}")

def setup_logging(log_level):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create ROBOKOP subgraph by filtering edges based on node types and predicates.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available styles:
  original       - Remove subclass_of and CAID edges
  CD             - Keep only Chemical-Disease edges
  CCGGDD         - Keep Chemical-Chemical, Gene-Gene, Disease-Disease edges
  CGGD           - Keep Chemical-Gene-Disease edges
  rCD            - Remove Chemical-Disease edges
  keepall        - Keep all edges (no filtering)
  CGGD_alltreat  - Keep CGGD edges plus all 'treats' relationships

Examples:
  python create_robokop_subgraph.py --style CGGD_alltreat
  python create_robokop_subgraph.py --node-file data/nodes.jsonl --edges-file data/edges.jsonl --style CD --outdir output/cd_graph
  python create_robokop_subgraph.py --style keepall --log-level DEBUG
        """
    )

    parser.add_argument(
        '--style',
        type=str,
        default='CGGD_alltreat',
        choices=['original', 'CD', 'CCGGDD', 'CGGD', 'rCD', 'keepall', 'CGGD_alltreat', 'CCGGDD_alltreat', 'clean_baseline'],
        help='Filtering style to apply (default: CGGD_alltreat)'
    )

    parser.add_argument(
        '--node-file',
        type=str,
        default='robokop/nodes.jsonl',
        help='Path to the nodes JSONL file (default: robokop/nodes.jsonl)'
    )

    parser.add_argument(
        '--edges-file',
        type=str,
        default='robokop/edges.jsonl',
        help='Path to the edges JSONL file (default: robokop/edges.jsonl)'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        default=None,
        help='Output directory (default: robokop/{style})'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--min-degree',
        type=int,
        default=2,
        help='Minimum degree threshold for clean_baseline style (nodes with degree < min-degree will be filtered out, default: 2)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    logger.info("Starting ROBOKOP subgraph creation")
    logger.info(f"Python executable: {os.sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        # Validate input files exist
        if not os.path.exists(args.node_file):
            logger.error(f"Node file not found: {args.node_file}")
            return 1

        if not os.path.exists(args.edges_file):
            logger.error(f"Edges file not found: {args.edges_file}")
            return 1

        # Create subgraph
        create_robokop_input(
            node_file=args.node_file,
            edges_file=args.edges_file,
            style=args.style,
            outdir=args.outdir,
            min_degree=args.min_degree
        )

        logger.info("All processing complete!")
        return 0

    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
