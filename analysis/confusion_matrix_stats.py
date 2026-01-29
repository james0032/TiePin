"""
Compute confusion matrix statistics and binary classification metrics for score matrices.

For each model (ConvE, CompGCN):
- Uses score threshold to predict is_treat (True/False)
- Compares predictions against actual is_treat labels
- Computes confusion matrix, precision, recall, F1, accuracy, etc.
- Outputs metrics in JSON format
"""

import polars as pl
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class BinaryMetrics:
    """Binary classification metrics."""
    model: str
    threshold: float
    total_samples: int
    positives: int  # Actual positives (is_treat=True)
    negatives: int  # Actual negatives (is_treat=False)

    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Derived metrics
    accuracy: float
    precision: float
    recall: float  # Same as sensitivity/TPR
    specificity: float  # TNR
    f1_score: float
    balanced_accuracy: float
    mcc: float  # Matthews Correlation Coefficient

    # Additional
    positive_predictive_value: float  # Same as precision
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float


def compute_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    model_name: str,
    threshold: float
) -> BinaryMetrics:
    """Compute binary classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Confusion matrix components
    tp = int(np.sum((y_true == True) & (y_pred == True)))
    tn = int(np.sum((y_true == False) & (y_pred == False)))
    fp = int(np.sum((y_true == False) & (y_pred == True)))
    fn = int(np.sum((y_true == True) & (y_pred == False)))

    total = len(y_true)
    positives = int(np.sum(y_true == True))
    negatives = int(np.sum(y_true == False))

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Precision (PPV)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall (Sensitivity, TPR)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (TNR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Balanced Accuracy
    balanced_acc = (recall + specificity) / 2

    # NPV
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    # FPR and FNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # MCC
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return BinaryMetrics(
        model=model_name,
        threshold=threshold,
        total_samples=total,
        positives=positives,
        negatives=negatives,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        accuracy=round(accuracy, 6),
        precision=round(precision, 6),
        recall=round(recall, 6),
        specificity=round(specificity, 6),
        f1_score=round(f1, 6),
        balanced_accuracy=round(balanced_acc, 6),
        mcc=round(mcc, 6),
        positive_predictive_value=round(precision, 6),
        negative_predictive_value=round(npv, 6),
        false_positive_rate=round(fpr, 6),
        false_negative_rate=round(fnr, 6)
    )


def find_optimal_threshold(
    scores: list[float],
    y_true: list[bool],
    metric: str = "f1"
) -> tuple[float, float]:
    """Find optimal threshold that maximizes the given metric."""
    scores = np.array(scores)
    y_true = np.array(y_true)

    # Try thresholds from min to max score
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_threshold = 0.5
    best_score = 0.0

    for thresh in thresholds:
        y_pred = scores >= thresh

        tp = np.sum((y_true == True) & (y_pred == True))
        fp = np.sum((y_true == False) & (y_pred == True))
        fn = np.sum((y_true == True) & (y_pred == False))
        tn = np.sum((y_true == False) & (y_pred == False))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == "balanced_accuracy":
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (recall + specificity) / 2
        else:  # accuracy
            score = (tp + tn) / len(y_true)

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return best_threshold, best_score


def analyze_model(
    df: pl.DataFrame,
    model_name: str,
    score_col: str,
    default_threshold: float = 0.5
) -> dict:
    """Analyze a single model's predictions."""
    scores = df[score_col].to_list()
    y_true = df["is_treat"].to_list()

    # Find optimal thresholds
    opt_f1_thresh, opt_f1 = find_optimal_threshold(scores, y_true, "f1")
    opt_ba_thresh, opt_ba = find_optimal_threshold(scores, y_true, "balanced_accuracy")

    results = {
        "model": model_name,
        "score_stats": {
            "min": round(min(scores), 6),
            "max": round(max(scores), 6),
            "mean": round(np.mean(scores), 6),
            "std": round(np.std(scores), 6),
            "median": round(np.median(scores), 6)
        },
        "optimal_thresholds": {
            "f1": {"threshold": round(opt_f1_thresh, 6), "f1_score": round(opt_f1, 6)},
            "balanced_accuracy": {"threshold": round(opt_ba_thresh, 6), "balanced_accuracy": round(opt_ba, 6)}
        },
        "metrics_at_thresholds": {}
    }

    # Compute metrics at different thresholds
    thresholds = [default_threshold, opt_f1_thresh, opt_ba_thresh, 0.7, 0.8, 0.9]
    thresholds = sorted(set(thresholds))

    for thresh in thresholds:
        y_pred = [s >= thresh for s in scores]
        metrics = compute_metrics(y_true, y_pred, model_name, thresh)
        results["metrics_at_thresholds"][f"threshold_{thresh:.4f}"] = asdict(metrics)

    return results


def print_confusion_matrix(metrics: BinaryMetrics):
    """Print confusion matrix in a readable format."""
    print(f"\n{'='*60}")
    print(f"Model: {metrics.model} | Threshold: {metrics.threshold:.4f}")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Negative    Positive")
    print(f"Actual Negative   {metrics.true_negatives:>8}    {metrics.false_positives:>8}")
    print(f"Actual Positive   {metrics.false_negatives:>8}    {metrics.true_positives:>8}")
    print(f"\nSample Distribution:")
    print(f"  Total: {metrics.total_samples:,}")
    print(f"  Actual Positives (is_treat=True): {metrics.positives:,} ({100*metrics.positives/metrics.total_samples:.1f}%)")
    print(f"  Actual Negatives (is_treat=False): {metrics.negatives:,} ({100*metrics.negatives/metrics.total_samples:.1f}%)")
    print(f"\nMetrics:")
    print(f"  Accuracy:           {metrics.accuracy:.4f}")
    print(f"  Precision (PPV):    {metrics.precision:.4f}")
    print(f"  Recall (TPR):       {metrics.recall:.4f}")
    print(f"  Specificity (TNR):  {metrics.specificity:.4f}")
    print(f"  F1 Score:           {metrics.f1_score:.4f}")
    print(f"  Balanced Accuracy:  {metrics.balanced_accuracy:.4f}")
    print(f"  MCC:                {metrics.mcc:.4f}")
    print(f"  NPV:                {metrics.negative_predictive_value:.4f}")
    print(f"  FPR:                {metrics.false_positive_rate:.4f}")
    print(f"  FNR:                {metrics.false_negative_rate:.4f}")


def plot_roc_curve(
    conve_scores: list[float],
    conve_y_true: list[bool],
    compgcn_scores: list[float],
    compgcn_y_true: list[bool],
    output_path: Path
):
    """Plot ROC curves for both models with AUC. ROC is threshold-independent
    and not affected by class imbalance."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for scores, y_true, name, color in [
        (conve_scores, conve_y_true, "ConvE", "steelblue"),
        (compgcn_scores, compgcn_y_true, "CompGCN", "darkorange"),
    ]:
        scores_arr = np.array(scores)
        y_true_arr = np.array(y_true)

        # Sort by score descending to sweep thresholds
        sorted_indices = np.argsort(-scores_arr)
        sorted_labels = y_true_arr[sorted_indices]
        sorted_scores = scores_arr[sorted_indices]

        # Compute TPR and FPR at each unique threshold
        total_pos = np.sum(y_true_arr)
        total_neg = len(y_true_arr) - total_pos

        tprs = [0.0]
        fprs = [0.0]
        tp = 0
        fp = 0

        for i in range(len(sorted_labels)):
            if sorted_labels[i]:
                tp += 1
            else:
                fp += 1
            # Only record a point when score changes or at the end
            if i == len(sorted_labels) - 1 or sorted_scores[i] != sorted_scores[i + 1]:
                tprs.append(tp / total_pos if total_pos > 0 else 0.0)
                fprs.append(fp / total_neg if total_neg > 0 else 0.0)

        # Compute AUC using trapezoidal rule
        auc = np.trapz(tprs, fprs)

        ax.plot(fprs, tprs, linewidth=2, color=color,
                label=f"{name} (AUC = {auc:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"\nROC curve saved to: {output_path}")
    plt.show()


def main():
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline")

    # Load files
    print("Loading files...")
    conve_path = base_path / "scores_matrix_ConvE_sampled_test_overlap.txt"
    compgcn_path = base_path / "scores_matrix_CompGCN_sampled_test_overlap.txt"

    conve_df = pl.read_csv(conve_path)
    compgcn_df = pl.read_csv(compgcn_path)

    print(f"  ConvE rows: {len(conve_df):,}")
    print(f"  CompGCN rows: {len(compgcn_df):,}")

    # Analyze each model
    # ConvE uses 'score' column (0-1 range)
    # CompGCN uses 'sigmoid_score' column (0-1 range)

    print("\n" + "="*60)
    print("ANALYZING CONVE MODEL")
    print("="*60)
    conve_results = analyze_model(conve_df, "ConvE", "score", default_threshold=0.5)

    print("\n" + "="*60)
    print("ANALYZING COMPGCN MODEL")
    print("="*60)
    compgcn_results = analyze_model(compgcn_df, "CompGCN", "sigmoid_score", default_threshold=0.5)

    # Print detailed results for optimal F1 threshold
    print("\n\n" + "#"*60)
    print("DETAILED CONFUSION MATRIX RESULTS")
    print("#"*60)

    # ConvE at optimal F1 threshold
    opt_thresh = conve_results["optimal_thresholds"]["f1"]["threshold"]
    conve_scores = conve_df["score"].to_list()
    conve_y_true = conve_df["is_treat"].to_list()
    conve_y_pred = [s >= opt_thresh for s in conve_scores]
    conve_metrics = compute_metrics(conve_y_true, conve_y_pred, "ConvE", opt_thresh)
    print_confusion_matrix(conve_metrics)

    # CompGCN at optimal F1 threshold
    opt_thresh = compgcn_results["optimal_thresholds"]["f1"]["threshold"]
    compgcn_scores = compgcn_df["sigmoid_score"].to_list()
    compgcn_y_true = compgcn_df["is_treat"].to_list()
    compgcn_y_pred = [s >= opt_thresh for s in compgcn_scores]
    compgcn_metrics = compute_metrics(compgcn_y_true, compgcn_y_pred, "CompGCN", opt_thresh)
    print_confusion_matrix(compgcn_metrics)

    # Combine results
    all_results = {
        "ConvE": conve_results,
        "CompGCN": compgcn_results
    }

    # Save to JSON
    output_path = base_path / "binary_classification_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*60}")
    print(f"JSON metrics saved to: {output_path}")
    print(f"{'='*60}")

    # Print summary comparison
    print("\n\n" + "#"*60)
    print("MODEL COMPARISON SUMMARY (at optimal F1 threshold)")
    print("#"*60)
    print(f"\n{'Metric':<25} {'ConvE':>12} {'CompGCN':>12}")
    print("-" * 50)

    conve_opt = conve_results["metrics_at_thresholds"][f"threshold_{conve_results['optimal_thresholds']['f1']['threshold']:.4f}"]
    compgcn_opt = compgcn_results["metrics_at_thresholds"][f"threshold_{compgcn_results['optimal_thresholds']['f1']['threshold']:.4f}"]

    for metric in ["accuracy", "precision", "recall", "specificity", "f1_score", "balanced_accuracy", "mcc"]:
        print(f"{metric:<25} {conve_opt[metric]:>12.4f} {compgcn_opt[metric]:>12.4f}")

    # Plot ROC curve (handles sample imbalance)
    plot_roc_curve(
        conve_scores=conve_scores,
        conve_y_true=conve_y_true,
        compgcn_scores=compgcn_scores,
        compgcn_y_true=compgcn_y_true,
        output_path=base_path / "roc_curve.png"
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
