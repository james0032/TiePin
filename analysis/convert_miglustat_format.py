"""
Convert miglustat_gaucher_disease_important_edges.csv to the TracIn format.

Source format columns:
- test_triple, test_head, test_relation, test_tail, prediction_score, edge_rank,
  edge_source, edge_relation, edge_target, edge_weight, error

Target format columns:
- TestHead, TestHead_label, TestRel, TestRel_label, TestTail, TestTail_label,
  TrainHead, TrainHead_label, TrainRel, TrainRel_label, TrainTail, TrainTail_label,
  TracInScore, SelfInfluence, ModifiedZScore

Note: Using edge_weight as TracInScore, fabricating CURIE IDs
"""

import pandas as pd
from pathlib import Path


# Mapping of entity names to fabricated CURIE IDs
CURIE_MAP = {
    # Drug
    "Miglustat": "CHEBI:50381",
    # Disease
    "Gaucher disease": "MONDO:0018150",
}

# Predicate mapping
PREDICATE_MAP = {
    "biolink:treats": "predicate:38",
    "biolink:contributes_to": "predicate:11",
    "biolink:target_for": "predicate:40",
    "biolink:gene_associated_with_condition": "predicate:21",
    "biolink:treats_or_applied_or_studied_to_treat": "predicate:42",
    "biolink:has_phenotype": "predicate:23",
    "biolink:causes": "predicate:7",
    "biolink:interacts_with": "predicate:28",
    "biolink:affects": "predicate:2",
    "biolink:related_to": "predicate:39",
}


def get_curie(name: str) -> str:
    """Get or fabricate a CURIE ID for an entity name."""
    if name in CURIE_MAP:
        return CURIE_MAP[name]

    # Fabricate CURIE based on entity type heuristics
    name_lower = name.lower()

    # Check if it looks like a gene (all caps, short)
    if name.isupper() and len(name) <= 10:
        curie = f"NCBIGene:{abs(hash(name)) % 100000}"
    # Check if it looks like a disease
    elif any(kw in name_lower for kw in ["disease", "syndrome", "disorder", "anemia", "osis", "itis"]):
        curie = f"MONDO:{abs(hash(name)) % 1000000:07d}"
    # Check if it looks like a phenotype
    elif any(kw in name_lower for kw in ["pain", "fever", "diarrhea", "nausea", "fatigue", "weakness"]):
        curie = f"HP:{abs(hash(name)) % 1000000:07d}"
    # Default to CHEBI for drugs/chemicals
    else:
        curie = f"CHEBI:{abs(hash(name)) % 100000}"

    # Cache for consistency
    CURIE_MAP[name] = curie
    return curie


def get_predicate_id(predicate: str) -> str:
    """Get predicate ID from predicate name."""
    if predicate in PREDICATE_MAP:
        return PREDICATE_MAP[predicate]
    # Fabricate for unknown predicates
    pred_id = f"predicate:{abs(hash(predicate)) % 100}"
    PREDICATE_MAP[predicate] = pred_id
    return pred_id


def main():
    input_path = Path("/Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain/data/08_reporting/miglustat_gaucher_disease_important_edges.csv")
    output_path = input_path.parent / "miglustat_gaucher_disease_important_edges_tracin_format.csv"

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows")

    # Build new dataframe in target format
    rows = []
    for _, row in df.iterrows():
        # Test triple info
        test_head = row["test_head"]
        test_rel = row["test_relation"]
        test_tail = row["test_tail"]

        # Train/edge info
        train_head = row["edge_source"]
        train_rel = row["edge_relation"]
        train_tail = row["edge_target"]

        # Edge weight as TracInScore (10-fold lower)
        tracin_score = row["edge_weight"] / 10.0

        new_row = {
            "TestHead": get_curie(test_head),
            "TestHead_label": test_head,
            "TestRel": get_predicate_id(test_rel),
            "TestRel_label": test_rel,
            "TestTail": get_curie(test_tail),
            "TestTail_label": test_tail,
            "TrainHead": get_curie(train_head),
            "TrainHead_label": train_head,
            "TrainRel": get_predicate_id(train_rel),
            "TrainRel_label": train_rel,
            "TrainTail": get_curie(train_tail),
            "TrainTail_label": train_tail,
            "TracInScore": tracin_score,
            "SelfInfluence": 0.0,  # Placeholder
            "ModifiedZScore": 0.0,  # Placeholder
        }
        rows.append(new_row)

    # Create output dataframe
    output_df = pd.DataFrame(rows)

    # Save
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Output rows: {len(output_df):,}")

    # Show first few rows
    print("\nFirst 3 rows:")
    print(output_df.head(3).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
