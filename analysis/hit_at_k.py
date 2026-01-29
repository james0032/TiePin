"""
Calculate Hit@K metrics for K=[1,100] from sampled test overlap files.

For each disease, rank all candidate drugs by score (descending). Then for
each true drug-disease pair (is_treat=true), check the rank of the true drug.
Hit@K = fraction of true pairs where true drug rank <= K.

Input:
  - scores_matrix_ConvE_sampled_test_overlap.txt
  - scores_matrix_CompGCN_sampled_test_overlap.txt
Output:
  - hit_at_k.csv with K, ConvE_HitAtK, CompGCN_HitAtK columns
  - hit_at_k.png plot with both curves
"""

import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path


def compute_ranks(df: pl.DataFrame, score_col: str, drug_col: str, disease_col: str) -> pl.DataFrame:
    """
    Sort by score descending, then for each disease compute rank as row position
    (1-based) within that disease's group.

    Returns DataFrame with columns: drug, disease, rank
    """
    # Sort entire dataframe by score descending
    sorted_df = df.sort(score_col, descending=True)

    # Assign rank as row number within each disease group (1-based)
    sorted_df = sorted_df.with_columns(
        (pl.col(score_col).cum_count().over(disease_col)).alias("rank")
    )

    # Filter to only true treat pairs
    true_ranks = sorted_df.filter(pl.col("is_treat") == True).select([
        pl.col(drug_col).alias("drug"),
        pl.col(disease_col).alias("disease"),
        pl.col("rank"),
    ])

    return true_ranks


def compute_hit_at_k(ranks: pl.DataFrame, max_k: int = 100) -> list[float]:
    """
    Compute Hit@K for K=1..max_k.
    Returns list of Hit@K values.
    """
    n = len(ranks)
    rank_col = ranks["rank"]

    hit_rates = []
    for k in range(1, max_k + 1):
        hits = (rank_col <= k).sum()
        hit_rates.append(float(hits / n))

    return hit_rates


def print_summary(name: str, ranks: pl.DataFrame, hit_rates: list[float]):
    """Print summary statistics."""
    print(f"\n=== {name} ===")
    print(f"Total true pairs evaluated: {len(ranks):,}")

    for k in [1, 3, 5, 10, 20, 30]:
        if k <= len(hit_rates):
            rate = hit_rates[k - 1]
            hits = int(rate * len(ranks))
            print(f"  Hit@{k:<3d}: {hits:>4d}/{len(ranks)} ({100*rate:.2f}%)")

    rank_col = ranks["rank"]
    print(f"\n  Rank stats - Mean: {rank_col.mean():.1f}, Median: {rank_col.median():.1f}, "
          f"Min: {rank_col.min():.0f}, Max: {rank_col.max():.0f}")


def main():
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline")

    conve_path = base_path / "scores_matrix_ConvE_sampled_test_overlap.txt"
    compgcn_path = base_path / "scores_matrix_CompGCN_sampled_test_overlap.txt"

    max_k = 30

    # Load ConvE
    print(f"Loading ConvE: {conve_path}")
    conve_df = pl.read_csv(str(conve_path))
    print(f"  Rows: {len(conve_df):,}, is_treat=true: {conve_df.filter(pl.col('is_treat') == True).height:,}")

    # Load CompGCN
    print(f"Loading CompGCN: {compgcn_path}")
    compgcn_df = pl.read_csv(str(compgcn_path))
    print(f"  Rows: {len(compgcn_df):,}, is_treat=true: {compgcn_df.filter(pl.col('is_treat') == True).height:,}")

    # Compute ranks
    print("\nComputing ranks...")
    conve_ranks = compute_ranks(conve_df, score_col="score", drug_col="head_label", disease_col="tail_label")
    compgcn_ranks = compute_ranks(compgcn_df, score_col="sigmoid_score", drug_col="head", disease_col="tail")

    # Compute Hit@K
    conve_hits = compute_hit_at_k(conve_ranks, max_k)
    compgcn_hits = compute_hit_at_k(compgcn_ranks, max_k)

    # Print summaries
    print_summary("ConvE", conve_ranks, conve_hits)
    print_summary("CompGCN", compgcn_ranks, compgcn_hits)

    # Save CSV
    ks = list(range(1, max_k + 1))
    results = pl.DataFrame({
        "K": ks,
        "ConvE_HitAtK": conve_hits,
        "CompGCN_HitAtK": compgcn_hits,
    })
    output_csv = base_path / "hit_at_k.csv"
    results.write_csv(str(output_csv))
    print(f"\nSaved to: {output_csv}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ks, conve_hits, linewidth=2, color="steelblue", label="ConvE")
    ax.plot(ks, compgcn_hits, linewidth=2, color="darkorange", label="CompGCN")

    ax.set_xlabel("K", fontsize=12)
    ax.set_ylabel("Hit@K", fontsize=12)
    ax.set_title("Hit@K Curve", fontsize=14)
    ax.set_xlim(1, max_k)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add key markers
    for k in [1, 5, 10, 20, 30]:
        conve_rate = conve_hits[k - 1]
        compgcn_rate = compgcn_hits[k - 1]
        ax.plot(k, conve_rate, 'o', color="steelblue", markersize=5)
        ax.plot(k, compgcn_rate, 'o', color="darkorange", markersize=5)

        ax.annotate(f"{100*conve_rate:.1f}%", (k, conve_rate),
                    textcoords="offset points", xytext=(8, 5), fontsize=8, color="steelblue")
        ax.annotate(f"{100*compgcn_rate:.1f}%", (k, compgcn_rate),
                    textcoords="offset points", xytext=(8, -10), fontsize=8, color="darkorange")

    plt.tight_layout()

    plot_path = base_path / "hit_at_k.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
