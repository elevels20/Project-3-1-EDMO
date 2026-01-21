"""
Multi-label cluster evaluation.

Instead of hierarchical rule application (which assigns a single label based on priority),
this approach assigns ALL applicable labels and counts a prediction as correct if it
matches ANY of the valid labels.

"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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


def derive_ground_truth_multilabel(row):
    """
    Maps communication annotation columns to ALL applicable cluster labels.

    Returns a SET of all clusters that match the annotation, without hierarchy.
    This removes the priority bias of the hierarchical approach.
    """
    applicable_clusters = set()

    # INTENSE OVERLAP (C2) - Competitive, high arousal, extended speaking
    if (pd.notna(row.get('Competitive Turn Claiming')) or
        pd.notna(row.get('Heightened Arousal Prosody')) or
        pd.notna(row.get('Extended Monologue / Dominance'))):
        applicable_clusters.add("Intense Overlap")

    # DISENGAGED (C3) - Resistance, disalignment
    if pd.notna(row.get('Disalignment / Resistance')):
        applicable_clusters.add("Disengaged")

    # COLLABORATIVE INQUIRY (C4) - High coordination with active engagement
    if (pd.notna(row.get('High Coordination')) and
        (pd.notna(row.get('Active Engagement')) or pd.notna(row.get('Supportive Backchanneling')))):
        applicable_clusters.add("Collaborative Inquiry")

    # ACTIVE INQUIRY (C0) - Active engagement, smooth transitions
    # Note: Removed the "without High Coordination" constraint
    if (pd.notna(row.get('Active Engagement')) or
        pd.notna(row.get('Smooth Turn Transition'))):
        applicable_clusters.add("Active Inquiry")

    # QUIET LISTENING (C1) - Passive, calming, low coordination
    if (pd.notna(row.get('Passive Participation')) or
        pd.notna(row.get('Calming Prosody')) or
        pd.notna(row.get('Low Coordination'))):
        applicable_clusters.add("Quiet Listening")

    # If no annotations match, default to Quiet Listening
    if not applicable_clusters:
        applicable_clusters.add("Quiet Listening")

    return applicable_clusters


def derive_ground_truth_hierarchical(row):
    """Original hierarchical approach for comparison."""
    if (pd.notna(row.get('Competitive Turn Claiming')) or
        pd.notna(row.get('Heightened Arousal Prosody')) or
        pd.notna(row.get('Extended Monologue / Dominance'))):
        return "Intense Overlap"

    if pd.notna(row.get('Disalignment / Resistance')):
        return "Disengaged"

    if (pd.notna(row.get('High Coordination')) and
        (pd.notna(row.get('Active Engagement')) or pd.notna(row.get('Supportive Backchanneling')))):
        return "Collaborative Inquiry"

    if (pd.notna(row.get('Active Engagement')) or
        pd.notna(row.get('Smooth Turn Transition'))):
        return "Active Inquiry"

    if (pd.notna(row.get('Passive Participation')) or
        pd.notna(row.get('Calming Prosody')) or
        pd.notna(row.get('Low Coordination'))):
        return "Quiet Listening"

    return "Quiet Listening"


def get_cluster_meta():
    """Returns cluster ID to label mapping."""
    return {
        0: "Active Inquiry",
        1: "Quiet Listening",
        2: "Intense Overlap",
        3: "Disengaged",
        4: "Collaborative Inquiry"
    }


def collect_predictions_multilabel(excel_path, saved_cluster_path, test_json_files):
    """
    Collects predictions and multi-label ground truth.

    Returns:
        results: list of dicts with prediction, multi-label truth, and hierarchical truth
    """
    meta = get_cluster_meta()
    saved_cluster = save_cluster.load_cluster(saved_cluster_path)

    if excel_path.endswith('.csv'):
        df_excel = pd.read_csv(excel_path)
    else:
        df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()

    all_results = []

    for json_path in test_json_files:
        membership = cluster_prediction.predict_cluster(saved_cluster, json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        valid_indices = [i for i, w in enumerate(data['audio_features']) if w.get('emotion') is not None]

        for idx, original_idx in enumerate(valid_indices):
            if idx < len(membership):
                pred_label = meta[membership[idx]]
                json_start = data['audio_features'][original_idx]['window_start']
                matched_row, _ = find_matching_excel_row(json_start, df_excel)

                if matched_row is not None:
                    multilabel_truth = derive_ground_truth_multilabel(matched_row)
                    hierarchical_truth = derive_ground_truth_hierarchical(matched_row)

                    all_results.append({
                        'prediction': pred_label,
                        'multilabel_truth': multilabel_truth,
                        'hierarchical_truth': hierarchical_truth,
                        'file': str(json_path),
                        'window_start': json_start
                    })

    return all_results


def evaluate_multilabel(results):
    """
    Evaluates accuracy using multi-label approach.

    A prediction is correct if it matches ANY of the applicable ground truth labels.
    """
    multilabel_correct = 0
    hierarchical_correct = 0
    total = len(results)

    # Track which samples benefit from multi-label
    samples_helped = []

    for r in results:
        pred = r['prediction']

        # Multi-label: correct if prediction is in the set of valid labels
        ml_correct = pred in r['multilabel_truth']
        # Hierarchical: correct only if exact match
        h_correct = pred == r['hierarchical_truth']

        if ml_correct:
            multilabel_correct += 1
        if h_correct:
            hierarchical_correct += 1

        # Track cases where multi-label helps but hierarchical fails
        if ml_correct and not h_correct:
            samples_helped.append({
                'prediction': pred,
                'multilabel_truth': r['multilabel_truth'],
                'hierarchical_truth': r['hierarchical_truth']
            })

    return {
        'multilabel_accuracy': (multilabel_correct / total * 100) if total > 0 else 0,
        'hierarchical_accuracy': (hierarchical_correct / total * 100) if total > 0 else 0,
        'multilabel_correct': multilabel_correct,
        'hierarchical_correct': hierarchical_correct,
        'total': total,
        'samples_helped': samples_helped
    }


def analyze_label_overlap(results):
    """Analyzes how often multiple labels apply to the same sample."""
    overlap_counts = defaultdict(int)
    label_cooccurrence = defaultdict(int)

    for r in results:
        num_labels = len(r['multilabel_truth'])
        overlap_counts[num_labels] += 1

        # Track which labels co-occur
        labels = tuple(sorted(r['multilabel_truth']))
        label_cooccurrence[labels] += 1

    return overlap_counts, label_cooccurrence


def compute_per_cluster_metrics(results):
    """
    Computes per-cluster metrics (precision, recall, F1) for both multilabel and hierarchical approaches.

    Returns dict with per-cluster stats for comparison.
    """
    cluster_labels = ["Active Inquiry", "Quiet Listening", "Intense Overlap",
                      "Disengaged", "Collaborative Inquiry"]

    per_cluster = {c: {
        'ml_correct': 0, 'h_correct': 0,
        'total_predicted': 0,
        'h_actual': 0,  # Count where hierarchical_truth == cluster
        'ml_actual': 0  # Count where cluster is in multilabel_truth
    } for c in cluster_labels}

    for r in results:
        pred = r['prediction']
        per_cluster[pred]['total_predicted'] += 1

        # Count actuals for hierarchical (single label)
        h_truth = r['hierarchical_truth']
        per_cluster[h_truth]['h_actual'] += 1

        # Count actuals for multilabel (can be in multiple clusters)
        for ml_label in r['multilabel_truth']:
            per_cluster[ml_label]['ml_actual'] += 1

        # Multilabel: correct if prediction in valid set
        if pred in r['multilabel_truth']:
            per_cluster[pred]['ml_correct'] += 1

        # Hierarchical: correct if exact match
        if pred == r['hierarchical_truth']:
            per_cluster[pred]['h_correct'] += 1

    # Calculate precision, recall, F1 for both methods
    for c in cluster_labels:
        predicted = per_cluster[c]['total_predicted']

        # Hierarchical metrics
        h_actual = per_cluster[c]['h_actual']
        h_correct = per_cluster[c]['h_correct']
        per_cluster[c]['h_precision'] = (h_correct / predicted * 100) if predicted > 0 else 0
        per_cluster[c]['h_recall'] = (h_correct / h_actual * 100) if h_actual > 0 else 0
        if per_cluster[c]['h_precision'] + per_cluster[c]['h_recall'] > 0:
            per_cluster[c]['h_f1'] = 2 * per_cluster[c]['h_precision'] * per_cluster[c]['h_recall'] / \
                                     (per_cluster[c]['h_precision'] + per_cluster[c]['h_recall'])
        else:
            per_cluster[c]['h_f1'] = 0

        # Multilabel metrics
        ml_actual = per_cluster[c]['ml_actual']
        ml_correct = per_cluster[c]['ml_correct']
        per_cluster[c]['ml_precision'] = (ml_correct / predicted * 100) if predicted > 0 else 0
        per_cluster[c]['ml_recall'] = (ml_correct / ml_actual * 100) if ml_actual > 0 else 0
        if per_cluster[c]['ml_precision'] + per_cluster[c]['ml_recall'] > 0:
            per_cluster[c]['ml_f1'] = 2 * per_cluster[c]['ml_precision'] * per_cluster[c]['ml_recall'] / \
                                      (per_cluster[c]['ml_precision'] + per_cluster[c]['ml_recall'])
        else:
            per_cluster[c]['ml_f1'] = 0

        # Accuracy (for backwards compatibility)
        if predicted > 0:
            per_cluster[c]['ml_accuracy'] = per_cluster[c]['ml_correct'] / predicted * 100
            per_cluster[c]['h_accuracy'] = per_cluster[c]['h_correct'] / predicted * 100
            per_cluster[c]['improvement'] = per_cluster[c]['ml_accuracy'] - per_cluster[c]['h_accuracy']
        else:
            per_cluster[c]['ml_accuracy'] = 0
            per_cluster[c]['h_accuracy'] = 0
            per_cluster[c]['improvement'] = 0

    return per_cluster


def print_per_cluster_table(per_cluster_metrics, method='both'):
    """
    Prints formatted tables of per-cluster metrics.

    Args:
        per_cluster_metrics: Dict from compute_per_cluster_metrics
        method: 'hierarchical', 'multilabel', or 'both'
    """
    cluster_labels = ["Active Inquiry", "Quiet Listening", "Intense Overlap",
                      "Disengaged", "Collaborative Inquiry"]

    if method in ['hierarchical', 'both']:
        print(f"\n{'HIERARCHICAL - PER-CLUSTER PERFORMANCE METRICS':^90}")
        print("-" * 90)
        print(f"{'Cluster':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Predicted':<10} {'Actual':<10}")
        print("-" * 90)
        for c in cluster_labels:
            m = per_cluster_metrics[c]
            print(f"{c:<25} {m['h_precision']:>6.2f}%     {m['h_recall']:>6.2f}%     {m['h_f1']:>6.2f}%     {m['total_predicted']:<10} {m['h_actual']:<10}")
        print("-" * 90)

    if method in ['multilabel', 'both']:
        print(f"\n{'MULTI-LABEL - PER-CLUSTER PERFORMANCE METRICS':^90}")
        print("-" * 90)
        print(f"{'Cluster':<25} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Predicted':<10} {'Actual':<10}")
        print("-" * 90)
        for c in cluster_labels:
            m = per_cluster_metrics[c]
            print(f"{c:<25} {m['ml_precision']:>6.2f}%     {m['ml_recall']:>6.2f}%     {m['ml_f1']:>6.2f}%     {m['total_predicted']:<10} {m['ml_actual']:<10}")
        print("-" * 90)


def compute_per_file_metrics(results):
    """
    Computes per-file accuracy for both multilabel and hierarchical approaches.

    Returns dict with per-file stats for comparison.
    """
    from pathlib import Path

    # Group results by file
    per_file = {}

    for r in results:
        file_name = Path(r['file']).stem  # Get filename without extension
        if file_name not in per_file:
            per_file[file_name] = {
                'ml_correct': 0,
                'h_correct': 0,
                'total': 0
            }

        per_file[file_name]['total'] += 1

        # Multilabel: correct if prediction in valid set
        if r['prediction'] in r['multilabel_truth']:
            per_file[file_name]['ml_correct'] += 1

        # Hierarchical: correct if exact match
        if r['prediction'] == r['hierarchical_truth']:
            per_file[file_name]['h_correct'] += 1

    # Calculate accuracies
    for f in per_file:
        total = per_file[f]['total']
        if total > 0:
            per_file[f]['ml_accuracy'] = per_file[f]['ml_correct'] / total * 100
            per_file[f]['h_accuracy'] = per_file[f]['h_correct'] / total * 100
            per_file[f]['improvement'] = per_file[f]['ml_accuracy'] - per_file[f]['h_accuracy']
        else:
            per_file[f]['ml_accuracy'] = 0
            per_file[f]['h_accuracy'] = 0
            per_file[f]['improvement'] = 0

    return per_file


def plot_per_file_comparison(per_file_metrics, save_path=None):
    """
    Creates a grouped bar chart showing multilabel vs hierarchical accuracy per file.
    """
    files = list(per_file_metrics.keys())
    ml_acc = [per_file_metrics[f]['ml_accuracy'] for f in files]
    h_acc = [per_file_metrics[f]['h_accuracy'] for f in files]

    x = np.arange(len(files))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Multilabel bars first (left), then hierarchical (right)
    bars1 = ax.bar(x - width/2, ml_acc, width, label='Multi-label', color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, h_acc, width, label='Hierarchical', color='#9E9E9E', edgecolor='black')

    # Add value labels with correct/total
    for i, f in enumerate(files):
        m = per_file_metrics[f]
        # Multi-label bar label
        ax.annotate(f'{m["ml_accuracy"]:.1f}%\n({m["ml_correct"]}/{m["total"]})',
                   xy=(x[i] - width/2, ml_acc[i]),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
        # Hierarchical bar label
        ax.annotate(f'{m["h_accuracy"]:.1f}%\n({m["h_correct"]}/{m["total"]})',
                   xy=(x[i] + width/2, h_acc[i]),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

    # Add improvement indicators
    for i, f in enumerate(files):
        improvement = per_file_metrics[f]['improvement']
        if improvement > 0:
            y_pos = max(ml_acc[i], h_acc[i]) + 12
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(x[i], y_pos),
                       ha='center', fontsize=10, color='#4CAF50', fontweight='bold')

    ax.set_xlabel('Test Session', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-File Accuracy: Multi-label vs Hierarchical', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(files, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-file comparison saved to {save_path}")
    plt.show()

    return per_file_metrics


def print_per_file_table(per_file_metrics):
    """Prints a formatted table of per-file metrics."""
    print(f"\n{'PER-FILE ACCURACY BREAKDOWN':^80}")
    print("-" * 80)
    print(f"{'Test Session':<25} {'Hierarchical':<18} {'Multi-label':<18} {'Improvement':<12}")
    print("-" * 80)

    for f in sorted(per_file_metrics.keys()):
        m = per_file_metrics[f]
        h_str = f"{m['h_accuracy']:.1f}% ({m['h_correct']}/{m['total']})"
        ml_str = f"{m['ml_accuracy']:.1f}% ({m['ml_correct']}/{m['total']})"
        imp_str = f"+{m['improvement']:.1f}%" if m['improvement'] >= 0 else f"{m['improvement']:.1f}%"
        print(f"{f:<25} {h_str:<18} {ml_str:<18} {imp_str:<12}")

    print("-" * 80)


def plot_accuracy_comparison(eval_results, save_path=None):
    """
    Creates a bar chart comparing overall multilabel vs hierarchical accuracy.
    Shows multilabel first to emphasize the improvement.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['Multi-label', 'Hierarchical']
    accuracies = [eval_results['multilabel_accuracy'], eval_results['hierarchical_accuracy']]
    colors = ['#2196F3', '#9E9E9E']  # Blue for multilabel, gray for hierarchical

    bars = ax.bar(methods, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add improvement annotation
    improvement = eval_results['multilabel_accuracy'] - eval_results['hierarchical_accuracy']
    ax.annotate(f'+{improvement:.1f}%',
                xy=(0.5, max(accuracies) + 3),
                ha='center', fontsize=12, color='#4CAF50', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Cluster Prediction Accuracy:\nMulti-label vs Hierarchical Evaluation', fontsize=14)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add sample count info
    ax.text(0.5, -0.12, f'n = {eval_results["total"]} samples',
            transform=ax.transAxes, ha='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Accuracy comparison saved to {save_path}")
    plt.show()


def plot_per_cluster_comparison(per_cluster_metrics, save_path=None):
    """
    Creates a grouped bar chart showing multilabel vs hierarchical accuracy per cluster.
    Multilabel bars shown first (left) for each cluster.
    """
    clusters = list(per_cluster_metrics.keys())
    ml_acc = [per_cluster_metrics[c]['ml_accuracy'] for c in clusters]
    h_acc = [per_cluster_metrics[c]['h_accuracy'] for c in clusters]

    x = np.arange(len(clusters))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    # Multilabel bars first (left), then hierarchical (right)
    bars1 = ax.bar(x - width/2, ml_acc, width, label='Multi-label', color='#2196F3', edgecolor='black')
    bars2 = ax.bar(x + width/2, h_acc, width, label='Hierarchical', color='#9E9E9E', edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)

    # Add improvement indicators
    for i, c in enumerate(clusters):
        improvement = per_cluster_metrics[c]['improvement']
        if improvement > 0:
            y_pos = max(ml_acc[i], h_acc[i]) + 8
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(x[i], y_pos),
                       ha='center', fontsize=9, color='#4CAF50', fontweight='bold')

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Cluster Accuracy: Multi-label vs Hierarchical', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, rotation=30, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-cluster comparison saved to {save_path}")
    plt.show()


def plot_label_overlap_distribution(overlap_counts, total, save_path=None):
    """
    Shows distribution of how many labels apply per sample.
    Helps explain why multilabel approach improves accuracy.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    labels_count = sorted(overlap_counts.keys())
    counts = [overlap_counts[n] for n in labels_count]
    percentages = [c / total * 100 for c in counts]

    colors = ['#E3F2FD', '#2196F3', '#1565C0', '#0D47A1'][:len(labels_count)]

    bars = ax.bar([str(n) for n in labels_count], percentages, color=colors, edgecolor='black')

    for bar, pct, cnt in zip(bars, percentages, counts):
        height = bar.get_height()
        ax.annotate(f'{pct:.1f}%\n({cnt})',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Number of Valid Labels per Sample', fontsize=12)
    ax.set_ylabel('Percentage of Samples', fontsize=12)
    ax.set_title('Label Overlap Distribution\n(Samples with >1 label benefit from multi-label evaluation)', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Label overlap distribution saved to {save_path}")
    plt.show()


def plot_multilabel_confusion_matrix(results, save_path=None):
    """
    Creates a multilabel-aware confusion matrix visualization.

    Shows predictions vs. ground truth labels, with cells indicating:
    - How often each prediction occurs when each ground truth label is valid
    - Highlights valid predictions (where prediction matches a valid label)

    The matrix uses color intensity to show frequency, with annotations
    showing both counts and whether the prediction was valid for that label.
    """
    cluster_labels = ["Active Inquiry", "Quiet Listening", "Intense Overlap",
                      "Disengaged", "Collaborative Inquiry"]

    # Build matrix: rows = predictions, cols = ground truth labels that were valid
    # Cell (i,j) = count of times prediction i occurred when label j was in the valid set
    matrix = np.zeros((len(cluster_labels), len(cluster_labels)))
    valid_matrix = np.zeros((len(cluster_labels), len(cluster_labels)))  # Track valid predictions

    for r in results:
        pred_idx = cluster_labels.index(r['prediction'])
        for gt_label in r['multilabel_truth']:
            gt_idx = cluster_labels.index(gt_label)
            matrix[pred_idx, gt_idx] += 1
            # Mark as valid if prediction matches this ground truth label
            if r['prediction'] == gt_label:
                valid_matrix[pred_idx, gt_idx] += 1

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Raw counts matrix
    ax1 = axes[0]

    # Normalize by column (per ground truth label) for better visualization
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / col_sums * 100

    # Create heatmap
    im1 = ax1.imshow(matrix_normalized, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels)):
            count = int(matrix[i, j])
            pct = matrix_normalized[i, j]
            if count > 0:
                # Highlight diagonal (correct predictions) with different color
                color = 'white' if pct > 50 else 'black'
                if i == j:
                    ax1.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                                 fill=False, edgecolor='#4CAF50', linewidth=3))
                ax1.text(j, i, f'{count}\n({pct:.0f}%)', ha='center', va='center',
                        fontsize=9, color=color)

    ax1.set_xticks(range(len(cluster_labels)))
    ax1.set_yticks(range(len(cluster_labels)))
    ax1.set_xticklabels([l.replace(' ', '\n') for l in cluster_labels], fontsize=9)
    ax1.set_yticklabels(cluster_labels, fontsize=9)
    ax1.set_xlabel('Ground Truth Label (in valid set)', fontsize=11)
    ax1.set_ylabel('Predicted Cluster', fontsize=11)
    ax1.set_title('Prediction Distribution by Valid Ground Truth\n(Column-normalized %)', fontsize=12)

    plt.colorbar(im1, ax=ax1, label='% of predictions when label is valid')

    # Right plot: Valid vs Invalid predictions per cell
    ax2 = axes[1]

    # Calculate valid percentage per cell
    valid_pct_matrix = np.zeros_like(matrix)
    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels)):
            if matrix[i, j] > 0:
                valid_pct_matrix[i, j] = (valid_matrix[i, j] / matrix[i, j]) * 100

    # Custom colormap: red (invalid) -> yellow -> green (valid)
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77']
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('validity', colors)

    im2 = ax2.imshow(valid_pct_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    # Add annotations
    for i in range(len(cluster_labels)):
        for j in range(len(cluster_labels)):
            count = int(matrix[i, j])
            valid_count = int(valid_matrix[i, j])
            if count > 0:
                valid_pct = valid_pct_matrix[i, j]
                color = 'black' if 20 < valid_pct < 80 else 'white'
                # Show: valid/total
                ax2.text(j, i, f'{valid_count}/{count}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

    ax2.set_xticks(range(len(cluster_labels)))
    ax2.set_yticks(range(len(cluster_labels)))
    ax2.set_xticklabels([l.replace(' ', '\n') for l in cluster_labels], fontsize=9)
    ax2.set_yticklabels(cluster_labels, fontsize=9)
    ax2.set_xlabel('Ground Truth Label (in valid set)', fontsize=11)
    ax2.set_ylabel('Predicted Cluster', fontsize=11)
    ax2.set_title('Prediction Validity\n(Valid predictions / Total when label applies)', fontsize=12)

    cbar = plt.colorbar(im2, ax=ax2, label='% Valid Predictions')
    cbar.set_ticks([0, 50, 100])
    cbar.set_ticklabels(['0% (Invalid)', '50%', '100% (Valid)'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multilabel confusion matrix saved to {save_path}")
    plt.show()

    # Print summary statistics
    print("\nConfusion Matrix Summary:")
    print("-" * 50)
    total_predictions = len(results)
    for i, label in enumerate(cluster_labels):
        pred_count = int(matrix[i, :].sum())
        valid_count = int(valid_matrix[i, :].sum())
        if pred_count > 0:
            valid_rate = valid_count / pred_count * 100
            print(f"  {label}: {pred_count} predictions, {valid_rate:.1f}% valid")


def plot_improvement_waterfall(eval_results, per_cluster_metrics, save_path=None):
    """
    Creates a waterfall-style chart showing where improvements come from.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate improvements per cluster
    clusters = list(per_cluster_metrics.keys())
    improvements = []
    for c in clusters:
        m = per_cluster_metrics[c]
        # Number of additional correct predictions from this cluster
        additional = m['ml_correct'] - m['h_correct']
        improvements.append((c, additional))

    # Sort by improvement
    improvements.sort(key=lambda x: x[1], reverse=True)

    # Create stacked bar showing progression
    hierarchical_acc = eval_results['hierarchical_accuracy']

    # Bar 1: Hierarchical baseline
    bars = []
    bars.append(ax.bar(['Hierarchical\nBaseline'], [hierarchical_acc], color='#9E9E9E', edgecolor='black'))

    # Bar 2: Multilabel total
    multilabel_acc = eval_results['multilabel_accuracy']
    bars.append(ax.bar(['Multi-label\nTotal'], [multilabel_acc], color='#2196F3', edgecolor='black'))

    # Add value labels
    ax.annotate(f'{hierarchical_acc:.1f}%', xy=(0, hierarchical_acc), xytext=(0, 5),
                textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    ax.annotate(f'{multilabel_acc:.1f}%', xy=(1, multilabel_acc), xytext=(0, 5),
                textcoords="offset points", ha='center', fontsize=12, fontweight='bold')

    # Draw arrow showing improvement
    improvement = multilabel_acc - hierarchical_acc
    ax.annotate('', xy=(1, multilabel_acc - 2), xytext=(0, hierarchical_acc + 2),
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=2))
    ax.annotate(f'+{improvement:.1f}%\nimprovement', xy=(0.5, (hierarchical_acc + multilabel_acc) / 2),
                ha='center', fontsize=11, color='#4CAF50', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Improvement: Hierarchical → Multi-label', fontsize=14)
    ax.set_ylim(0, 100)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add breakdown text
    breakdown_text = "Improvements by cluster:\n"
    for c, imp in improvements[:3]:  # Top 3
        if imp > 0:
            breakdown_text += f"  {c}: +{imp} samples\n"

    ax.text(1.5, hierarchical_acc, breakdown_text, fontsize=9, va='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Improvement waterfall saved to {save_path}")
    plt.show()


def print_multilabel_report(eval_results, overlap_counts, label_cooccurrence):
    """Prints comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("MULTI-LABEL vs HIERARCHICAL EVALUATION COMPARISON")
    print("=" * 80)

    print(f"\n{'ACCURACY COMPARISON':^80}")
    print("-" * 80)
    print(f"Hierarchical Accuracy: {eval_results['hierarchical_accuracy']:.2f}% "
          f"({eval_results['hierarchical_correct']}/{eval_results['total']})")
    print(f"Multi-label Accuracy:  {eval_results['multilabel_accuracy']:.2f}% "
          f"({eval_results['multilabel_correct']}/{eval_results['total']})")

    improvement = eval_results['multilabel_accuracy'] - eval_results['hierarchical_accuracy']
    print(f"\nImprovement: +{improvement:.2f}% ({len(eval_results['samples_helped'])} additional correct predictions)")

    print(f"\n{'LABEL OVERLAP ANALYSIS':^80}")
    print("-" * 80)
    print("How many labels apply per sample:")
    for num_labels, count in sorted(overlap_counts.items()):
        pct = count / eval_results['total'] * 100
        print(f"  {num_labels} label(s): {count} samples ({pct:.1f}%)")

    print(f"\n{'MOST COMMON LABEL COMBINATIONS':^80}")
    print("-" * 80)
    sorted_cooccurrence = sorted(label_cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]
    for labels, count in sorted_cooccurrence:
        pct = count / eval_results['total'] * 100
        label_str = " + ".join(labels)
        print(f"  {label_str}: {count} ({pct:.1f}%)")

    print(f"\n{'SAMPLES HELPED BY MULTI-LABEL':^80}")
    print("-" * 80)

    # Group by prediction type
    helped_by_pred = defaultdict(list)
    for s in eval_results['samples_helped']:
        helped_by_pred[s['prediction']].append(s)

    for pred, samples in sorted(helped_by_pred.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n  Prediction: {pred} ({len(samples)} cases helped)")
        # Show what hierarchical label they were incorrectly compared against
        hier_labels = defaultdict(int)
        for s in samples:
            hier_labels[s['hierarchical_truth']] += 1
        for h_label, count in sorted(hier_labels.items(), key=lambda x: x[1], reverse=True):
            print(f"    - Would have been compared to '{h_label}': {count} times")

    print("\n" + "=" * 80)


def run_multilabel_evaluation(excel_path, saved_cluster_path, test_json_files,
                               save_plots=True, output_prefix='multilabel_eval'):
    """
    Runs the complete multi-label evaluation with comparison plots.

    Args:
        excel_path: Path to annotation Excel file
        saved_cluster_path: Path to saved cluster .pkl file
        test_json_files: List of paths to test JSON files
        save_plots: Whether to save plots to files
        output_prefix: Prefix for saved plot files
    """
    print("Collecting predictions with multi-label ground truth...")
    results = collect_predictions_multilabel(excel_path, saved_cluster_path, test_json_files)

    if not results:
        print("No predictions collected!")
        return None

    print(f"Collected {len(results)} predictions.")

    # Evaluate
    eval_results = evaluate_multilabel(results)

    # Analyze overlap
    overlap_counts, label_cooccurrence = analyze_label_overlap(results)

    # Compute per-cluster metrics for comparison
    per_cluster_metrics = compute_per_cluster_metrics(results)

    # Compute per-file metrics
    per_file_metrics = compute_per_file_metrics(results)

    # Print report
    print_multilabel_report(eval_results, overlap_counts, label_cooccurrence)

    # Print per-file breakdown
    print_per_file_table(per_file_metrics)

    # Print per-cluster performance metrics (precision, recall, F1)
    print_per_cluster_table(per_cluster_metrics, method='both')

    # Generate comparison plots
    print("\nGenerating comparison plots...")

    if save_plots:
        # Plot 1: Overall accuracy comparison (Multi-label vs Hierarchical)
        plot_accuracy_comparison(
            eval_results,
            save_path=f'{output_prefix}_accuracy_comparison.png'
        )

        # Plot 2: Per-cluster comparison
        plot_per_cluster_comparison(
            per_cluster_metrics,
            save_path=f'{output_prefix}_per_cluster_comparison.png'
        )

        # Plot 3: Label overlap distribution
        plot_label_overlap_distribution(
            overlap_counts,
            eval_results['total'],
            save_path=f'{output_prefix}_label_overlap.png'
        )

        # Plot 4: Improvement waterfall
        plot_improvement_waterfall(
            eval_results,
            per_cluster_metrics,
            save_path=f'{output_prefix}_improvement.png'
        )

        # Plot 5: Multilabel confusion matrix
        plot_multilabel_confusion_matrix(
            results,
            save_path=f'{output_prefix}_confusion_matrix.png'
        )

        # Plot 6: Per-file comparison
        plot_per_file_comparison(
            per_file_metrics,
            save_path=f'{output_prefix}_per_file_comparison.png'
        )
    else:
        plot_accuracy_comparison(eval_results)
        plot_per_cluster_comparison(per_cluster_metrics)
        plot_label_overlap_distribution(overlap_counts, eval_results['total'])
        plot_improvement_waterfall(eval_results, per_cluster_metrics)
        plot_multilabel_confusion_matrix(results)
        plot_per_file_comparison(per_file_metrics)

    return {
        'eval_results': eval_results,
        'overlap_counts': overlap_counts,
        'label_cooccurrence': label_cooccurrence,
        'per_cluster_metrics': per_cluster_metrics,
        'per_file_metrics': per_file_metrics,
        'raw_results': results
    }


if __name__ == "__main__":
    # Find test files
    test_data_dir = DATA_DIR / "data" / "test_data"
    test_json_files = list(test_data_dir.glob("*_features.json"))

    print(f"Found {len(test_json_files)} test files:")
    for f in test_json_files:
        print(f"  - {f.name}")

    # Run evaluation with comparison plots
    results = run_multilabel_evaluation(
        excel_path=str(DATA_DIR / "data" / "communication_annotation.xlsx"),
        saved_cluster_path=str(DATA_DIR / "full_dimensions_cluster.pkl"),
        test_json_files=test_json_files,
        save_plots=True,
        output_prefix='multilabel_eval'
    )
