"""
Script to combine scores from CGGD_alltreat and CCGGDD_alltreat datasets,
rank by CGGD_alltreat scores, and visualize score distributions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def combine_scores(
    cggd_path: str = "../data/CGGD_alltreat/scores_test.csv",
    ccggdd_path: str = "../data/CCGGDD_alltreat/scores_test.csv"
) -> pd.DataFrame:
    """
    Combine scores from two CSV files and rank by CGGD_alltreat scores.

    Parameters
    ----------
    cggd_path : str
        Path to CGGD_alltreat scores CSV file
    ccggdd_path : str
        Path to CCGGDD_alltreat scores CSV file

    Returns
    -------
    pd.DataFrame
        Combined dataframe ranked by CGGD_alltreat scores (descending)
    """
    # Read both CSV files
    cggd_df = pd.read_csv(cggd_path)
    ccggdd_df = pd.read_csv(ccggdd_path)

    # Add source column to identify which dataset each row came from
    cggd_df['source'] = 'CGGD_alltreat'
    ccggdd_df['source'] = 'CCGGDD_alltreat'

    # Combine the dataframes
    combined_df = pd.concat([cggd_df, ccggdd_df], ignore_index=True)

    # Rank by score column (assuming column is named 'score')
    # If the score column has a different name, adjust accordingly
    score_col = 'score' if 'score' in combined_df.columns else combined_df.columns[0]

    # For CGGD_alltreat rows, use their score for ranking
    # Create a ranking column based on CGGD scores
    cggd_scores = cggd_df.set_index(cggd_df.columns[0])[score_col] if len(cggd_df.columns) > 1 else cggd_df[score_col]

    # Sort by score descending (higher scores first)
    combined_df = combined_df.sort_values(by=score_col, ascending=False).reset_index(drop=True)

    return combined_df


def plot_score_distributions(
    df: pd.DataFrame,
    score_col: str = 'score',
    source_col: str = 'source',
    figsize: tuple = (12, 6),
    save_path: str = None
):
    """
    Plot score distributions for each group using matplotlib with seaborn style.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing scores and source information
    score_col : str
        Name of the score column
    source_col : str
        Name of the source/group column
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get unique groups
    groups = df[source_col].unique()
    colors = sns.color_palette("husl", len(groups))

    # Plot 1: Histogram with KDE
    for idx, group in enumerate(groups):
        group_data = df[df[source_col] == group][score_col]
        axes[0].hist(group_data, alpha=0.6, label=group, bins=30, color=colors[idx], edgecolor='black')

    axes[0].set_xlabel('Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Score Distribution by Group (Histogram)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Box plot
    data_for_boxplot = [df[df[source_col] == group][score_col].values for group in groups]
    bp = axes[1].boxplot(data_for_boxplot, labels=groups, patch_artist=True)

    # Color the box plots
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_xlabel('Group', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Score Distribution by Group (Boxplot)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    # Reset style
    sns.reset_orig()


def plot_kde_comparison(
    df: pd.DataFrame,
    score_col: str = 'score',
    source_col: str = 'source',
    figsize: tuple = (10, 6),
    save_path: str = None
):
    """
    Plot kernel density estimation comparison for score distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing scores and source information
    score_col : str
        Name of the score column
    source_col : str
        Name of the source/group column
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, displays the plot.
    """
    sns.set_style("whitegrid")

    plt.figure(figsize=figsize)

    # Plot KDE for each group
    for group in df[source_col].unique():
        group_data = df[df[source_col] == group][score_col]
        sns.kdeplot(data=group_data, label=group, linewidth=2, fill=True, alpha=0.3)

    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Score Distribution Comparison (KDE)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    sns.reset_orig()


if __name__ == "__main__":
    # Combine the scores
    print("Combining scores from CGGD_alltreat and CCGGDD_alltreat...")
    combined_df = combine_scores(
        cggd_path="../data/CGGD_alltreat/scores_test.csv",
        ccggdd_path="../data/CCGGDD_alltreat/scores_test.csv"
    )

    print(f"Combined dataframe shape: {combined_df.shape}")
    print(f"\nFirst few rows:")
    print(combined_df.head(10))

    # Save combined and ranked data
    output_path = "combined_scores_ranked.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\nCombined and ranked scores saved to {output_path}")

    # Print summary statistics
    print("\nSummary statistics by group:")
    score_col = 'score' if 'score' in combined_df.columns else combined_df.columns[0]
    print(combined_df.groupby('source')[score_col].describe())

    # Plot distributions
    print("\nGenerating distribution plots...")
    plot_score_distributions(
        combined_df,
        score_col=score_col,
        save_path="score_distributions.png"
    )

    plot_kde_comparison(
        combined_df,
        score_col=score_col,
        save_path="score_kde_comparison.png"
    )

    print("\nDone!")
