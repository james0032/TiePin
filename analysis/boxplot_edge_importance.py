"""
Create boxplot of edge importance scores categorized by predicate name using seaborn.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # Load file (no header)
    input_path = Path("/Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/gnnexplain/data/08_reporting/Miglustat_1000_edges.csv")

    print(f"Loading: {input_path}")

    # Load CSV without header, use column indices
    # Index 0 = triple_tuple, Index 7 = edge_predicate, Index 9 = edge_importance
    df = pd.read_csv(input_path, header=None)

    # Rename columns we need
    df = df.rename(columns={0: "triple_tuple", 7: "edge_predicate", 9: "edge_importance"})

    # Filter to only Miglustat rows (file contains multiple drug-disease pairs)
    df = df[df["triple_tuple"].str.contains("Miglustat")]

    # Drop any rows with NaN in key columns
    df = df.dropna(subset=["edge_predicate", "edge_importance"])

    # Ensure edge_predicate is string type for categorical plotting
    df["edge_predicate"] = df["edge_predicate"].astype(str)

    print(f"Loaded {len(df):,} rows")
    print(f"\nUnique predicates: {df['edge_predicate'].nunique()}")
    print(f"\nPredicate value counts:")
    print(df['edge_predicate'].value_counts())

    # Create boxplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Order by mean importance score
    predicate_order = (
        df.groupby("edge_predicate")["edge_importance"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Create boxplot with seaborn
    sns.boxplot(
        data=df,
        x="edge_predicate",
        y="edge_importance",
        order=predicate_order,
        palette="Set2",
        ax=ax
    )

    # Customize plot
    ax.set_xlabel("Predicate", fontsize=12)
    ax.set_ylabel("Edge Importance Score", fontsize=12)
    ax.set_title("Edge Importance by Predicate Type\nMiglustat -> biolink:treats -> Gaucher disease", fontsize=14)

    # Rotate x-axis labels for readability (45 degrees)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Add grid
    ax.grid(True, axis='y', alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_dir = input_path.parent
    output_path = output_dir / "Miglustat_1000_edges_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF
    pdf_path = output_dir / "Miglustat_1000_edges_boxplot.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.show()

    # Print statistics per predicate
    print("\n=== Statistics per Predicate ===")
    stats = df.groupby("edge_predicate")["edge_importance"].agg(["count", "mean", "std", "median", "min", "max"])
    stats = stats.sort_values("median", ascending=False)
    print(stats.to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
