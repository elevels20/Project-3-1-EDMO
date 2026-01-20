import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

import cluster_prediction
import save_cluster

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR


def parse_excel_time(time_str):
    """Parses '01:30 - 02:00' into start_seconds and end_seconds."""
    try:
        start_str, end_str = time_str.split('-')
        m1, s1 = map(int, start_str.strip().split(':'))
        start_sec = (m1 * 60) + s1
        m2, s2 = map(int, end_str.strip().split(':'))
        end_sec = (m2 * 60) + s2
        return start_sec, end_sec
    except:
        return None, None


def find_matching_excel_row(json_time_sec, df_excel):
    """Find which Excel row covers the given JSON timestamp."""
    for index, row in df_excel.iterrows():
        t_range = str(row['Timestamp Range'])
        e_start, e_end = parse_excel_time(t_range)
        if e_start is not None:
            if e_start <= json_time_sec < e_end:
                return row, index + 1
    return None, None


def derive_ground_truth(row, session_type='robot'):
    """Maps communication annotation columns to cluster labels."""
    # INTENSE OVERLAP (C2)
    if (pd.notna(row.get('Competitive Turn Claiming')) or
        pd.notna(row.get('Heightened Arousal Prosody')) or
        pd.notna(row.get('Extended Monologue / Dominance'))):
        return "Intense Overlap"

    # DISENGAGED (C3)
    if pd.notna(row.get('Disalignment / Resistance')):
        return "Disengaged"

    # COLLABORATIVE INQUIRY (C4)
    if (pd.notna(row.get('High Coordination')) and
        (pd.notna(row.get('Active Engagement')) or pd.notna(row.get('Supportive Backchanneling')))):
        return "Collaborative Inquiry"

    # ACTIVE INQUIRY (C0)
    if (pd.notna(row.get('Active Engagement')) or
        pd.notna(row.get('Smooth Turn Transition'))):
        return "Active Inquiry"

    # QUIET LISTENING (C1)
    if (pd.notna(row.get('Passive Participation')) or
        pd.notna(row.get('Calming Prosody')) or
        pd.notna(row.get('Low Coordination'))):
        return "Quiet Listening"

    return "Quiet Listening"


def get_cluster_meta(session_type='robot'):
    """Returns cluster ID to label mapping."""
    if session_type == 'robot':
        return {
            0: "Active Inquiry",
            1: "Quiet Listening",
            2: "Intense Overlap",
            3: "Disengaged",
            4: "Collaborative Inquiry"
        }
    else:
        return {
            0: "Quiet Listening",
            1: "Intense Overlap",
            2: "Collaborative Inquiry",
            3: "Deep Silence"
        }


def collect_predictions_and_truth(excel_path, saved_cluster_path, test_json_files, session_type='robot'):
    """
    Collects all predictions and ground truth labels across multiple test files.

    Returns:
        all_predictions: list of predicted cluster labels
        all_truths: list of ground truth labels
        per_file_results: dict with per-file accuracy
    """
    meta = get_cluster_meta(session_type)
    saved_cluster = save_cluster.load_cluster(saved_cluster_path)

    if excel_path.endswith('.csv'):
        df_excel = pd.read_csv(excel_path)
    else:
        df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()

    all_predictions = []
    all_truths = []
    per_file_results = {}

    for json_path in test_json_files:
        membership = cluster_prediction.predict_cluster(saved_cluster, json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        valid_indices = [i for i, w in enumerate(data['audio_features']) if w.get('emotion') is not None]

        file_predictions = []
        file_truths = []

        for idx, original_idx in enumerate(valid_indices):
            if idx < len(membership):
                pred_label = meta[membership[idx]]
                json_start = data['audio_features'][original_idx]['window_start']
                matched_row, _ = find_matching_excel_row(json_start, df_excel)

                if matched_row is not None:
                    truth_label = derive_ground_truth(matched_row, session_type)
                    file_predictions.append(pred_label)
                    file_truths.append(truth_label)
                    all_predictions.append(pred_label)
                    all_truths.append(truth_label)

        # Calculate per-file accuracy
        if file_predictions:
            matches = sum(1 for p, t in zip(file_predictions, file_truths) if p == t)
            accuracy = (matches / len(file_predictions)) * 100
            per_file_results[str(json_path)] = {
                'accuracy': accuracy,
                'total': len(file_predictions),
                'correct': matches
            }

    return all_predictions, all_truths, per_file_results


def evaluate_cluster_performance(all_predictions, all_truths, session_type='robot'):
    """
    Evaluates which cluster predictions perform best.

    Returns a dict with per-cluster metrics and overall statistics.
    """
    meta = get_cluster_meta(session_type)
    cluster_labels = list(meta.values())

    # Per-cluster accuracy (when predicting this cluster, how often is it correct?)
    per_cluster_stats = defaultdict(lambda: {'correct': 0, 'total_predicted': 0, 'total_actual': 0})

    for pred, truth in zip(all_predictions, all_truths):
        per_cluster_stats[pred]['total_predicted'] += 1
        per_cluster_stats[truth]['total_actual'] += 1
        if pred == truth:
            per_cluster_stats[pred]['correct'] += 1

    # Calculate precision, recall, F1 for each cluster
    results = {}
    for cluster in cluster_labels:
        stats = per_cluster_stats[cluster]

        # Precision: correct / total_predicted (how many predictions for this cluster were right)
        precision = (stats['correct'] / stats['total_predicted'] * 100) if stats['total_predicted'] > 0 else 0

        # Recall: correct / total_actual (how many actual instances were correctly identified)
        recall = (stats['correct'] / stats['total_actual'] * 100) if stats['total_actual'] > 0 else 0

        # F1 Score
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        results[cluster] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_predicted': stats['total_predicted'],
            'total_actual': stats['total_actual'],
            'correct': stats['correct']
        }

    return results, cluster_labels


def plot_confusion_matrix(all_predictions, all_truths, cluster_labels, save_path=None):
    """Creates and plots a confusion matrix."""
    cm = confusion_matrix(all_truths, all_predictions, labels=cluster_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=cluster_labels, yticklabels=cluster_labels)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('Actual Cluster (Ground Truth)')
    plt.title('Cluster Prediction Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

    return cm


def plot_cluster_performance(results, save_path=None):
    """Creates a bar chart comparing cluster performance metrics."""
    clusters = list(results.keys())
    precision = [results[c]['precision'] for c in clusters]
    recall = [results[c]['recall'] for c in clusters]
    f1 = [results[c]['f1_score'] for c in clusters]

    x = np.arange(len(clusters))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision (%)', color='#2196F3')
    bars2 = ax.bar(x, recall, width, label='Recall (%)', color='#4CAF50')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score (%)', color='#FF9800')

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Score (%)')
    ax.set_title('Cluster Prediction Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Performance chart saved to {save_path}")
    plt.show()


def plot_prediction_distribution(all_predictions, all_truths, cluster_labels, save_path=None):
    """Shows distribution of predictions vs actual ground truth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prediction distribution
    pred_counts = pd.Series(all_predictions).value_counts()
    pred_counts = pred_counts.reindex(cluster_labels, fill_value=0)
    axes[0].bar(pred_counts.index, pred_counts.values, color='#2196F3')
    axes[0].set_title('Distribution of Predicted Clusters')
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    # Ground truth distribution
    truth_counts = pd.Series(all_truths).value_counts()
    truth_counts = truth_counts.reindex(cluster_labels, fill_value=0)
    axes[1].bar(truth_counts.index, truth_counts.values, color='#4CAF50')
    axes[1].set_title('Distribution of Actual Clusters (Ground Truth)')
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution chart saved to {save_path}")
    plt.show()


def print_detailed_report(results, per_file_results, all_predictions, all_truths):
    """Prints a comprehensive evaluation report."""
    print("\n" + "=" * 80)
    print("CLUSTER PREDICTION PERFORMANCE EVALUATION REPORT")
    print("=" * 80)

    # Overall accuracy
    overall_correct = sum(1 for p, t in zip(all_predictions, all_truths) if p == t)
    overall_accuracy = (overall_correct / len(all_predictions)) * 100 if all_predictions else 0

    print(f"\n{'OVERALL METRICS':^80}")
    print("-" * 80)
    print(f"Total Samples: {len(all_predictions)}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    # Per-file accuracy
    print(f"\n{'PER-FILE ACCURACY':^80}")
    print("-" * 80)
    for file_path, stats in per_file_results.items():
        file_name = Path(file_path).name
        print(f"  {file_name}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")

    # Per-cluster performance - sorted by F1 score
    print(f"\n{'PER-CLUSTER PERFORMANCE (Sorted by F1 Score)':^80}")
    print("-" * 80)
    print(f"{'Cluster':<25} {'Precision':>12} {'Recall':>12} {'F1 Score':>12} {'Predicted':>12} {'Actual':>10}")
    print("-" * 80)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True)

    for cluster, stats in sorted_results:
        print(f"{cluster:<25} {stats['precision']:>11.2f}% {stats['recall']:>11.2f}% "
              f"{stats['f1_score']:>11.2f}% {stats['total_predicted']:>12} {stats['total_actual']:>10}")

    # Best and worst performing clusters
    print(f"\n{'PERFORMANCE SUMMARY':^80}")
    print("-" * 80)

    best_cluster = sorted_results[0][0]
    worst_cluster = sorted_results[-1][0]

    print(f"Best Performing Cluster:  {best_cluster} (F1: {sorted_results[0][1]['f1_score']:.2f}%)")
    print(f"Worst Performing Cluster: {worst_cluster} (F1: {sorted_results[-1][1]['f1_score']:.2f}%)")

    # Most commonly confused pairs
    print(f"\n{'CONFUSION ANALYSIS':^80}")
    print("-" * 80)

    confusion_pairs = defaultdict(int)
    for pred, truth in zip(all_predictions, all_truths):
        if pred != truth:
            confusion_pairs[(truth, pred)] += 1

    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Most Common Misclassifications:")
    for (truth, pred), count in sorted_confusions:
        print(f"  {truth} -> {pred}: {count} times")

    print("\n" + "=" * 80)


def run_full_evaluation(excel_path, saved_cluster_path, test_json_files, session_type='robot',
                        save_plots=True, output_prefix='cluster_eval'):
    """
    Runs a complete evaluation of cluster prediction performance.

    Args:
        excel_path: Path to annotation Excel file
        saved_cluster_path: Path to saved cluster .pkl file
        test_json_files: List of paths to test JSON files
        session_type: 'robot' or other
        save_plots: Whether to save plots to files
        output_prefix: Prefix for saved plot files
    """
    print("Collecting predictions and ground truth...")
    all_predictions, all_truths, per_file_results = collect_predictions_and_truth(
        excel_path, saved_cluster_path, test_json_files, session_type
    )

    if not all_predictions:
        print("No predictions collected! Check your data paths.")
        return None

    print(f"Collected {len(all_predictions)} predictions across {len(test_json_files)} files.")

    # Evaluate cluster performance
    print("Evaluating cluster performance...")
    results, cluster_labels = evaluate_cluster_performance(all_predictions, all_truths, session_type)

    # Print detailed report
    print_detailed_report(results, per_file_results, all_predictions, all_truths)

    # Generate plots
    if save_plots:
        plot_confusion_matrix(all_predictions, all_truths, cluster_labels,
                            save_path=f'{output_prefix}_confusion_matrix.png')
        plot_cluster_performance(results,
                               save_path=f'{output_prefix}_performance.png')
        plot_prediction_distribution(all_predictions, all_truths, cluster_labels,
                                   save_path=f'{output_prefix}_distribution.png')
    else:
        plot_confusion_matrix(all_predictions, all_truths, cluster_labels)
        plot_cluster_performance(results)
        plot_prediction_distribution(all_predictions, all_truths, cluster_labels)

    return {
        'results': results,
        'per_file_results': per_file_results,
        'all_predictions': all_predictions,
        'all_truths': all_truths,
        'cluster_labels': cluster_labels
    }


if __name__ == "__main__":
    # Find all test JSON files
    test_data_dir = DATA_DIR / "data" / "test_data"
    test_json_files = list(test_data_dir.glob("*_features.json"))

    print(f"Found {len(test_json_files)} test files:")
    for f in test_json_files:
        print(f"  - {f.name}")

    # Run evaluation
    evaluation_results = run_full_evaluation(
        excel_path=str(DATA_DIR / "data" / "communication_annotation.xlsx"),
        saved_cluster_path=str(DATA_DIR / "full_dimensions_cluster.pkl"),
        test_json_files=test_json_files,
        session_type='robot',
        save_plots=True,
        output_prefix='cluster_eval'
    )
