"""
Visualize and compare TracIn evaluation metrics (MRR and MAP) between datasets.

This script creates comprehensive visualizations comparing the performance of
TracIn scores in identifying ground truth (In_path) edges from DrugMechDB
across different datasets (CCGGDD_alltreat vs CGGD_alltreat).
"""

import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_evaluation_results(ccggdd_path: Path, cggd_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load evaluation results from both datasets.

    Parameters
    ----------
    ccggdd_path : Path
        Path to CCGGDD_alltreat evaluation results CSV
    cggd_path : Path
        Path to CGGD_alltreat evaluation results CSV

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        CCGGDD and CGGD dataframes
    """
    logger.info(f"Loading CCGGDD results from {ccggdd_path}")
    ccggdd_df = pd.read_csv(ccggdd_path)
    ccggdd_df['dataset'] = 'CCGGDD_alltreat'

    logger.info(f"Loading CGGD results from {cggd_path}")
    cggd_df = pd.read_csv(cggd_path)
    cggd_df['dataset'] = 'CGGD_alltreat'

    # Filter out zero values (no ground truth found)
    ccggdd_df_filtered = ccggdd_df[ccggdd_df['coverage'] > 0].copy()
    cggd_df_filtered = cggd_df[cggd_df['coverage'] > 0].copy()

    logger.info(f"CCGGDD: {len(ccggdd_df_filtered)}/{len(ccggdd_df)} triples with ground truth")
    logger.info(f"CGGD: {len(cggd_df_filtered)}/{len(cggd_df)} triples with ground truth")

    return ccggdd_df_filtered, cggd_df_filtered


def plot_mrr_map_comparison(ccggdd_df: pd.DataFrame, cggd_df: pd.DataFrame,
                             output_dir: Path, show_plots: bool = False):
    """
    Create comparison plots for MRR and MAP between datasets.

    Parameters
    ----------
    ccggdd_df : pd.DataFrame
        CCGGDD evaluation results
    cggd_df : pd.DataFrame
        CGGD evaluation results
    output_dir : Path
        Directory to save plots
    show_plots : bool
        Whether to display plots interactively
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Combine data for plotting
    combined_df = pd.concat([ccggdd_df, cggd_df], ignore_index=True)

    # 1. MRR Distribution (Histogram)
    ax1 = axes[0, 0]
    for dataset in ['CCGGDD_alltreat', 'CGGD_alltreat']:
        data = combined_df[combined_df['dataset'] == dataset]['mrr']
        ax1.hist(data, bins=30, alpha=0.6, label=dataset, edgecolor='black')

    ax1.set_xlabel('MRR', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('MRR Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. MAP Distribution (Histogram)
    ax2 = axes[0, 1]
    for dataset in ['CCGGDD_alltreat', 'CGGD_alltreat']:
        data = combined_df[combined_df['dataset'] == dataset]['map']
        ax2.hist(data, bins=30, alpha=0.6, label=dataset, edgecolor='black')

    ax2.set_xlabel('MAP', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('MAP Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. MRR Boxplot
    ax3 = axes[1, 0]
    data_for_boxplot = [
        ccggdd_df['mrr'].values,
        cggd_df['mrr'].values
    ]
    bp1 = ax3.boxplot(data_for_boxplot, labels=['CCGGDD', 'CGGD'],
                       patch_artist=True, showmeans=True)
    colors = sns.color_palette("husl", 2)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax3.set_ylabel('MRR', fontsize=12, fontweight='bold')
    ax3.set_title('MRR Comparison (Boxplot)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add statistical test results
    stat, pval = stats.mannwhitneyu(ccggdd_df['mrr'], cggd_df['mrr'], alternative='two-sided')
    ax3.text(0.5, 0.95, f'Mann-Whitney U test\np-value: {pval:.4f}',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. MAP Boxplot
    ax4 = axes[1, 1]
    data_for_boxplot = [
        ccggdd_df['map'].values,
        cggd_df['map'].values
    ]
    bp2 = ax4.boxplot(data_for_boxplot, labels=['CCGGDD', 'CGGD'],
                       patch_artist=True, showmeans=True)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax4.set_ylabel('MAP', fontsize=12, fontweight='bold')
    ax4.set_title('MAP Comparison (Boxplot)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add statistical test results
    stat, pval = stats.mannwhitneyu(ccggdd_df['map'], cggd_df['map'], alternative='two-sided')
    ax4.text(0.5, 0.95, f'Mann-Whitney U test\np-value: {pval:.4f}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'mrr_map_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    sns.reset_orig()


def plot_kde_comparison(ccggdd_df: pd.DataFrame, cggd_df: pd.DataFrame,
                        output_dir: Path, show_plots: bool = False):
    """
    Create KDE plots for MRR and MAP comparison.

    Parameters
    ----------
    ccggdd_df : pd.DataFrame
        CCGGDD evaluation results
    cggd_df : pd.DataFrame
        CGGD evaluation results
    output_dir : Path
        Directory to save plots
    show_plots : bool
        Whether to display plots interactively
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. MRR KDE
    ax1 = axes[0]
    sns.kdeplot(data=ccggdd_df['mrr'], label='CCGGDD_alltreat', linewidth=2,
                fill=True, alpha=0.3, ax=ax1)
    sns.kdeplot(data=cggd_df['mrr'], label='CGGD_alltreat', linewidth=2,
                fill=True, alpha=0.3, ax=ax1)

    ax1.set_xlabel('MRR', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('MRR Distribution (KDE)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add mean lines
    ax1.axvline(ccggdd_df['mrr'].mean(), color='C0', linestyle='--', linewidth=2,
                alpha=0.8, label=f"CCGGDD mean: {ccggdd_df['mrr'].mean():.4f}")
    ax1.axvline(cggd_df['mrr'].mean(), color='C1', linestyle='--', linewidth=2,
                alpha=0.8, label=f"CGGD mean: {cggd_df['mrr'].mean():.4f}")

    # 2. MAP KDE
    ax2 = axes[1]
    sns.kdeplot(data=ccggdd_df['map'], label='CCGGDD_alltreat', linewidth=2,
                fill=True, alpha=0.3, ax=ax2)
    sns.kdeplot(data=cggd_df['map'], label='CGGD_alltreat', linewidth=2,
                fill=True, alpha=0.3, ax=ax2)

    ax2.set_xlabel('MAP', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('MAP Distribution (KDE)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    # Add mean lines
    ax2.axvline(ccggdd_df['map'].mean(), color='C0', linestyle='--', linewidth=2,
                alpha=0.8, label=f"CCGGDD mean: {ccggdd_df['map'].mean():.4f}")
    ax2.axvline(cggd_df['map'].mean(), color='C1', linestyle='--', linewidth=2,
                alpha=0.8, label=f"CGGD mean: {cggd_df['map'].mean():.4f}")

    plt.tight_layout()
    output_path = output_dir / 'mrr_map_kde.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved KDE plot to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    sns.reset_orig()


def plot_scatter_mrr_vs_map(ccggdd_df: pd.DataFrame, cggd_df: pd.DataFrame,
                             output_dir: Path, show_plots: bool = False):
    """
    Create scatter plot of MRR vs MAP.

    Parameters
    ----------
    ccggdd_df : pd.DataFrame
        CCGGDD evaluation results
    cggd_df : pd.DataFrame
        CGGD evaluation results
    output_dir : Path
        Directory to save plots
    show_plots : bool
        Whether to display plots interactively
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot scatter points
    ax.scatter(ccggdd_df['mrr'], ccggdd_df['map'], alpha=0.6, s=80,
               label='CCGGDD_alltreat', edgecolors='black', linewidth=0.5)
    ax.scatter(cggd_df['mrr'], cggd_df['map'], alpha=0.6, s=80,
               label='CGGD_alltreat', edgecolors='black', linewidth=0.5)

    # Add diagonal reference line (MRR = MAP)
    max_val = max(ccggdd_df['mrr'].max(), ccggdd_df['map'].max(),
                  cggd_df['mrr'].max(), cggd_df['map'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1.5,
            label='MRR = MAP')

    ax.set_xlabel('MRR', fontsize=12, fontweight='bold')
    ax.set_ylabel('MAP', fontsize=12, fontweight='bold')
    ax.set_title('MRR vs MAP Scatter Plot', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Calculate correlation
    ccggdd_corr = ccggdd_df[['mrr', 'map']].corr().iloc[0, 1]
    cggd_corr = cggd_df[['mrr', 'map']].corr().iloc[0, 1]

    # Add correlation text
    corr_text = f'Correlation:\nCCGGDD: {ccggdd_corr:.4f}\nCGGD: {cggd_corr:.4f}'
    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = output_dir / 'mrr_vs_map_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scatter plot to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    sns.reset_orig()


def plot_rank_distribution(ccggdd_df: pd.DataFrame, cggd_df: pd.DataFrame,
                            output_dir: Path, show_plots: bool = False):
    """
    Plot distribution of ranks of first relevant item.

    Parameters
    ----------
    ccggdd_df : pd.DataFrame
        CCGGDD evaluation results
    cggd_df : pd.DataFrame
        CGGD evaluation results
    output_dir : Path
        Directory to save plots
    show_plots : bool
        Whether to display plots interactively
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Linear scale
    ax1 = axes[0]
    ax1.hist(ccggdd_df['avg_rank_first_relevant'], bins=50, alpha=0.6,
             label='CCGGDD_alltreat', edgecolor='black')
    ax1.hist(cggd_df['avg_rank_first_relevant'], bins=50, alpha=0.6,
             label='CGGD_alltreat', edgecolor='black')

    ax1.set_xlabel('Rank of First Relevant Item', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Distribution of First Relevant Item Ranks (Linear Scale)',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add median lines
    ccggdd_median = ccggdd_df['median_rank_first_relevant'].median()
    cggd_median = cggd_df['median_rank_first_relevant'].median()
    ax1.axvline(ccggdd_median, color='C0', linestyle='--', linewidth=2,
                label=f'CCGGDD median: {ccggdd_median:.0f}')
    ax1.axvline(cggd_median, color='C1', linestyle='--', linewidth=2,
                label=f'CGGD median: {cggd_median:.0f}')

    # 2. Log scale
    ax2 = axes[1]
    ax2.hist(ccggdd_df['avg_rank_first_relevant'], bins=50, alpha=0.6,
             label='CCGGDD_alltreat', edgecolor='black')
    ax2.hist(cggd_df['avg_rank_first_relevant'], bins=50, alpha=0.6,
             label='CGGD_alltreat', edgecolor='black')

    ax2.set_xlabel('Rank of First Relevant Item', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of First Relevant Item Ranks (Log Scale)',
                  fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'rank_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved rank distribution plot to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    sns.reset_orig()


def plot_coverage_comparison(ccggdd_df: pd.DataFrame, cggd_df: pd.DataFrame,
                              output_dir: Path, show_plots: bool = False):
    """
    Plot comparison of coverage and ground truth statistics.

    Parameters
    ----------
    ccggdd_df : pd.DataFrame
        CCGGDD evaluation results
    cggd_df : pd.DataFrame
        CGGD evaluation results
    output_dir : Path
        Directory to save plots
    show_plots : bool
        Whether to display plots interactively
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Ground truth edges per triple
    ax1 = axes[0]
    data_for_boxplot = [
        ccggdd_df['total_ground_truth_edges'].values,
        cggd_df['total_ground_truth_edges'].values
    ]
    bp1 = ax1.boxplot(data_for_boxplot, labels=['CCGGDD', 'CGGD'],
                       patch_artist=True, showmeans=True)
    colors = sns.color_palette("husl", 2)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Number of Ground Truth Edges', fontsize=12, fontweight='bold')
    ax1.set_title('Ground Truth Edges per Test Triple', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add mean annotations
    ccggdd_mean = ccggdd_df['total_ground_truth_edges'].mean()
    cggd_mean = cggd_df['total_ground_truth_edges'].mean()
    ax1.text(0.5, 0.95, f'CCGGDD mean: {ccggdd_mean:.2f}\nCGGD mean: {cggd_mean:.2f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Bar chart of summary statistics
    ax2 = axes[1]
    metrics = ['Mean MRR', 'Mean MAP', 'Median Rank']
    ccggdd_values = [
        ccggdd_df['mrr'].mean(),
        ccggdd_df['map'].mean(),
        ccggdd_df['median_rank_first_relevant'].median()
    ]
    cggd_values = [
        cggd_df['mrr'].mean(),
        cggd_df['map'].mean(),
        cggd_df['median_rank_first_relevant'].median()
    ]

    x = np.arange(len(metrics))
    width = 0.35

    # Normalize median rank for visualization (inverse scale)
    ccggdd_values_norm = ccggdd_values[:2] + [1.0 / ccggdd_values[2] * 1000 if ccggdd_values[2] > 0 else 0]
    cggd_values_norm = cggd_values[:2] + [1.0 / cggd_values[2] * 1000 if cggd_values[2] > 0 else 0]

    bars1 = ax2.bar(x - width/2, ccggdd_values_norm[:2] + [ccggdd_values_norm[2]],
                    width, label='CCGGDD', alpha=0.8)
    bars2 = ax2.bar(x + width/2, cggd_values_norm[:2] + [cggd_values_norm[2]],
                    width, label='CGGD', alpha=0.8)

    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Summary Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Mean MRR', 'Mean MAP', '1/Median Rank\n(x1000)'])
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.4f}',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'coverage_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved coverage comparison plot to {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    sns.reset_orig()


def generate_summary_report(ccggdd_df: pd.DataFrame, cggd_df: pd.DataFrame,
                             output_dir: Path):
    """
    Generate a text summary report of the comparison.

    Parameters
    ----------
    ccggdd_df : pd.DataFrame
        CCGGDD evaluation results
    cggd_df : pd.DataFrame
        CGGD evaluation results
    output_dir : Path
        Directory to save report
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("TracIn Evaluation Metrics Comparison Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Dataset overview
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 80)
    report_lines.append(f"CCGGDD_alltreat: {len(ccggdd_df)} triples with ground truth")
    report_lines.append(f"CGGD_alltreat: {len(cggd_df)} triples with ground truth")
    report_lines.append("")

    # MRR comparison
    report_lines.append("MEAN RECIPROCAL RANK (MRR)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {'CCGGDD':>15} {'CGGD':>15} {'Difference':>15}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Mean':<30} {ccggdd_df['mrr'].mean():>15.6f} {cggd_df['mrr'].mean():>15.6f} {ccggdd_df['mrr'].mean() - cggd_df['mrr'].mean():>15.6f}")
    report_lines.append(f"{'Median':<30} {ccggdd_df['mrr'].median():>15.6f} {cggd_df['mrr'].median():>15.6f} {ccggdd_df['mrr'].median() - cggd_df['mrr'].median():>15.6f}")
    report_lines.append(f"{'Std Dev':<30} {ccggdd_df['mrr'].std():>15.6f} {cggd_df['mrr'].std():>15.6f} {abs(ccggdd_df['mrr'].std() - cggd_df['mrr'].std()):>15.6f}")
    report_lines.append(f"{'Min':<30} {ccggdd_df['mrr'].min():>15.6f} {cggd_df['mrr'].min():>15.6f} {ccggdd_df['mrr'].min() - cggd_df['mrr'].min():>15.6f}")
    report_lines.append(f"{'Max':<30} {ccggdd_df['mrr'].max():>15.6f} {cggd_df['mrr'].max():>15.6f} {ccggdd_df['mrr'].max() - cggd_df['mrr'].max():>15.6f}")

    # Statistical test
    stat, pval = stats.mannwhitneyu(ccggdd_df['mrr'], cggd_df['mrr'], alternative='two-sided')
    report_lines.append(f"\nMann-Whitney U test: U={stat:.2f}, p-value={pval:.6f}")
    if pval < 0.05:
        report_lines.append("→ Statistically significant difference (p < 0.05)")
    else:
        report_lines.append("→ No statistically significant difference (p >= 0.05)")
    report_lines.append("")

    # MAP comparison
    report_lines.append("MEAN AVERAGE PRECISION (MAP)")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {'CCGGDD':>15} {'CGGD':>15} {'Difference':>15}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Mean':<30} {ccggdd_df['map'].mean():>15.6f} {cggd_df['map'].mean():>15.6f} {ccggdd_df['map'].mean() - cggd_df['map'].mean():>15.6f}")
    report_lines.append(f"{'Median':<30} {ccggdd_df['map'].median():>15.6f} {cggd_df['map'].median():>15.6f} {ccggdd_df['map'].median() - cggd_df['map'].median():>15.6f}")
    report_lines.append(f"{'Std Dev':<30} {ccggdd_df['map'].std():>15.6f} {cggd_df['map'].std():>15.6f} {abs(ccggdd_df['map'].std() - cggd_df['map'].std()):>15.6f}")
    report_lines.append(f"{'Min':<30} {ccggdd_df['map'].min():>15.6f} {cggd_df['map'].min():>15.6f} {ccggdd_df['map'].min() - cggd_df['map'].min():>15.6f}")
    report_lines.append(f"{'Max':<30} {ccggdd_df['map'].max():>15.6f} {cggd_df['map'].max():>15.6f} {ccggdd_df['map'].max() - cggd_df['map'].max():>15.6f}")

    # Statistical test
    stat, pval = stats.mannwhitneyu(ccggdd_df['map'], cggd_df['map'], alternative='two-sided')
    report_lines.append(f"\nMann-Whitney U test: U={stat:.2f}, p-value={pval:.6f}")
    if pval < 0.05:
        report_lines.append("→ Statistically significant difference (p < 0.05)")
    else:
        report_lines.append("→ No statistically significant difference (p >= 0.05)")
    report_lines.append("")

    # Ranking performance
    report_lines.append("RANKING PERFORMANCE")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {'CCGGDD':>15} {'CGGD':>15}")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Avg rank (first relevant)':<30} {ccggdd_df['avg_rank_first_relevant'].mean():>15.2f} {cggd_df['avg_rank_first_relevant'].mean():>15.2f}")
    report_lines.append(f"{'Median rank (first relevant)':<30} {ccggdd_df['median_rank_first_relevant'].median():>15.2f} {cggd_df['median_rank_first_relevant'].median():>15.2f}")
    report_lines.append(f"{'Avg ground truth edges':<30} {ccggdd_df['total_ground_truth_edges'].mean():>15.2f} {cggd_df['total_ground_truth_edges'].mean():>15.2f}")
    report_lines.append("")

    # Conclusion
    report_lines.append("CONCLUSION")
    report_lines.append("-" * 80)
    if ccggdd_df['mrr'].mean() > cggd_df['mrr'].mean():
        winner = "CCGGDD_alltreat"
        diff_pct = ((ccggdd_df['mrr'].mean() - cggd_df['mrr'].mean()) / cggd_df['mrr'].mean()) * 100
    else:
        winner = "CGGD_alltreat"
        diff_pct = ((cggd_df['mrr'].mean() - ccggdd_df['mrr'].mean()) / ccggdd_df['mrr'].mean()) * 100

    report_lines.append(f"{winner} shows better MRR performance ({diff_pct:+.2f}% difference)")

    if ccggdd_df['map'].mean() > cggd_df['map'].mean():
        winner = "CCGGDD_alltreat"
        diff_pct = ((ccggdd_df['map'].mean() - cggd_df['map'].mean()) / cggd_df['map'].mean()) * 100
    else:
        winner = "CGGD_alltreat"
        diff_pct = ((cggd_df['map'].mean() - ccggdd_df['map'].mean()) / ccggdd_df['map'].mean()) * 100

    report_lines.append(f"{winner} shows better MAP performance ({diff_pct:+.2f}% difference)")
    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Saved comparison report to {report_path}")

    # Also print to console
    print('\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(
        description='Visualize and compare TracIn evaluation metrics (MRR and MAP)'
    )
    parser.add_argument(
        '--ccggdd',
        type=Path,
        default=Path('/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/CCGGDD_alltreat/tracin_evaluation_results.csv'),
        help='Path to CCGGDD_alltreat evaluation results CSV'
    )
    parser.add_argument(
        '--cggd',
        type=Path,
        default=Path('/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/CGGD_alltreat/tracin_evaluation_results.csv'),
        help='Path to CGGD_alltreat evaluation results CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: analysis/tracin_comparison)'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        args.output_dir = script_dir / 'tracin_comparison'

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    ccggdd_df, cggd_df = load_evaluation_results(args.ccggdd, args.cggd)

    # Generate all plots
    logger.info("Generating comparison plots...")
    plot_mrr_map_comparison(ccggdd_df, cggd_df, args.output_dir, args.show_plots)
    plot_kde_comparison(ccggdd_df, cggd_df, args.output_dir, args.show_plots)
    plot_scatter_mrr_vs_map(ccggdd_df, cggd_df, args.output_dir, args.show_plots)
    plot_rank_distribution(ccggdd_df, cggd_df, args.output_dir, args.show_plots)
    plot_coverage_comparison(ccggdd_df, cggd_df, args.output_dir, args.show_plots)

    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(ccggdd_df, cggd_df, args.output_dir)

    logger.info("All visualizations complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
