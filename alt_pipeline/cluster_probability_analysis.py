import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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


def derive_ground_truth(row, session_type='robot'):
    """Maps communication annotation columns to cluster labels."""
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


def collect_predictions_with_probabilities(excel_path, saved_cluster_path, test_json_files, session_type='robot'):
    """
    Collects predictions, probabilities, and ground truth labels.

    Returns:
        data_records: list of dicts with prediction info and probabilities
    """
    meta = get_cluster_meta(session_type)
    label_to_id = {v: k for k, v in meta.items()}
    saved_cluster = save_cluster.load_cluster(saved_cluster_path)

    if excel_path.endswith('.csv'):
        df_excel = pd.read_csv(excel_path)
    else:
        df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()

    data_records = []

    for json_path in test_json_files:
        assignments, probabilities = cluster_prediction.predict_cluster_with_probabilities(saved_cluster, json_path)

        with open(json_path, 'r') as f:
            data = json.load(f)

        valid_indices = [i for i, w in enumerate(data['audio_features']) if w.get('emotion') is not None]

        for idx, original_idx in enumerate(valid_indices):
            if idx < len(assignments):
                pred_id = assignments[idx]
                pred_label = meta[pred_id]
                probs = probabilities[:, idx]  # All cluster probabilities for this sample

                json_start = data['audio_features'][original_idx]['window_start']
                matched_row, _ = find_matching_excel_row(json_start, df_excel)

                if matched_row is not None:
                    truth_label = derive_ground_truth(matched_row, session_type)
                    truth_id = label_to_id.get(truth_label, -1)

                    # Calculate metrics
                    max_prob = np.max(probs)
                    sorted_probs = np.sort(probs)[::-1]
                    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
                    entropy = -np.sum(probs * np.log(probs + 1e-10))

                    data_records.append({
                        'file': str(json_path),
                        'timestamp': json_start,
                        'pred_id': pred_id,
                        'pred_label': pred_label,
                        'truth_id': truth_id,
                        'truth_label': truth_label,
                        'is_correct': pred_label == truth_label,
                        'max_probability': max_prob,
                        'margin': margin,
                        'entropy': entropy,
                        **{f'prob_cluster_{i}': probs[i] for i in range(len(probs))}
                    })

    return pd.DataFrame(data_records), meta


def analyze_confidence_accuracy(df):
    """Analyzes relationship between prediction confidence and accuracy."""
    # Bin by confidence levels
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['0-30%', '30-50%', '50-70%', '70-90%', '90-100%']
    df['confidence_bin'] = pd.cut(df['max_probability'], bins=bins, labels=labels)

    confidence_accuracy = df.groupby('confidence_bin', observed=True).agg({
        'is_correct': ['mean', 'count']
    }).round(3)
    confidence_accuracy.columns = ['accuracy', 'count']

    return confidence_accuracy


def analyze_per_cluster_confidence(df, meta):
    """Analyzes confidence metrics per predicted cluster."""
    cluster_labels = list(meta.values())

    results = []
    for cluster in cluster_labels:
        cluster_df = df[df['pred_label'] == cluster]
        if len(cluster_df) > 0:
            results.append({
                'cluster': cluster,
                'count': len(cluster_df),
                'avg_confidence': cluster_df['max_probability'].mean(),
                'std_confidence': cluster_df['max_probability'].std(),
                'avg_margin': cluster_df['margin'].mean(),
                'avg_entropy': cluster_df['entropy'].mean(),
                'accuracy': cluster_df['is_correct'].mean() * 100,
                'correct_avg_conf': cluster_df[cluster_df['is_correct']]['max_probability'].mean() if cluster_df['is_correct'].sum() > 0 else np.nan,
                'incorrect_avg_conf': cluster_df[~cluster_df['is_correct']]['max_probability'].mean() if (~cluster_df['is_correct']).sum() > 0 else np.nan
            })

    return pd.DataFrame(results)


def analyze_misclassification_probabilities(df, meta):
    """Analyzes what probabilities look like for misclassified samples."""
    misclassified = df[~df['is_correct']].copy()
    correct = df[df['is_correct']].copy()

    analysis = {
        'correct': {
            'avg_max_prob': correct['max_probability'].mean(),
            'avg_margin': correct['margin'].mean(),
            'avg_entropy': correct['entropy'].mean(),
            'count': len(correct)
        },
        'misclassified': {
            'avg_max_prob': misclassified['max_probability'].mean(),
            'avg_margin': misclassified['margin'].mean(),
            'avg_entropy': misclassified['entropy'].mean(),
            'count': len(misclassified)
        }
    }

    # For misclassified, what was the probability of the correct cluster?
    prob_cols = [c for c in df.columns if c.startswith('prob_cluster_')]
    if len(misclassified) > 0:
        correct_probs = []
        for _, row in misclassified.iterrows():
            if row['truth_id'] >= 0 and row['truth_id'] < len(prob_cols):
                correct_probs.append(row[f'prob_cluster_{row["truth_id"]}'])
        if correct_probs:
            analysis['misclassified']['avg_true_cluster_prob'] = np.mean(correct_probs)

    return analysis


def plot_confidence_vs_accuracy(df, save_path=None):
    """Plots confidence distribution for correct vs incorrect predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Confidence distribution: correct vs incorrect
    ax = axes[0, 0]
    correct = df[df['is_correct']]['max_probability']
    incorrect = df[~df['is_correct']]['max_probability']
    ax.hist(correct, bins=20, alpha=0.7, label=f'Correct (n={len(correct)})', color='green')
    ax.hist(incorrect, bins=20, alpha=0.7, label=f'Incorrect (n={len(incorrect)})', color='red')
    ax.set_xlabel('Max Probability (Confidence)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution: Correct vs Incorrect')
    ax.legend()

    # 2. Accuracy by confidence bin
    ax = axes[0, 1]
    conf_acc = analyze_confidence_accuracy(df)
    bars = ax.bar(conf_acc.index.astype(str), conf_acc['accuracy'] * 100, color='steelblue')
    ax.set_xlabel('Confidence Bin')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Confidence Level')
    ax.set_ylim(0, 100)
    for bar, count in zip(bars, conf_acc['count']):
        ax.annotate(f'n={count}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # 3. Margin distribution
    ax = axes[1, 0]
    ax.hist(df[df['is_correct']]['margin'], bins=20, alpha=0.7, label='Correct', color='green')
    ax.hist(df[~df['is_correct']]['margin'], bins=20, alpha=0.7, label='Incorrect', color='red')
    ax.set_xlabel('Probability Margin (1st - 2nd highest)')
    ax.set_ylabel('Count')
    ax.set_title('Margin Distribution: Correct vs Incorrect')
    ax.legend()

    # 4. Entropy distribution
    ax = axes[1, 1]
    ax.hist(df[df['is_correct']]['entropy'], bins=20, alpha=0.7, label='Correct', color='green')
    ax.hist(df[~df['is_correct']]['entropy'], bins=20, alpha=0.7, label='Incorrect', color='red')
    ax.set_xlabel('Entropy (higher = more uncertain)')
    ax.set_ylabel('Count')
    ax.set_title('Entropy Distribution: Correct vs Incorrect')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confidence analysis plot saved to {save_path}")
    plt.close()


def plot_per_cluster_confidence(cluster_stats, save_path=None):
    """Plots confidence metrics per cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Average confidence with correct/incorrect breakdown
    ax = axes[0]
    x = np.arange(len(cluster_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, cluster_stats['correct_avg_conf'].fillna(0), width,
                   label='Correct Predictions', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, cluster_stats['incorrect_avg_conf'].fillna(0), width,
                   label='Incorrect Predictions', color='red', alpha=0.7)

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Average Confidence')
    ax.set_title('Average Confidence by Cluster (Correct vs Incorrect)')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_stats['cluster'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    # 2. Accuracy vs Confidence scatter
    ax = axes[1]
    scatter = ax.scatter(cluster_stats['avg_confidence'], cluster_stats['accuracy'],
                        s=cluster_stats['count'] * 3, alpha=0.7, c='steelblue')
    for i, row in cluster_stats.iterrows():
        ax.annotate(row['cluster'], (row['avg_confidence'], row['accuracy']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax.set_xlabel('Average Confidence')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Cluster Accuracy vs Confidence (bubble size = count)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Per-cluster confidence plot saved to {save_path}")
    plt.close()


def plot_probability_heatmap(df, meta, save_path=None):
    """Creates a heatmap showing average probabilities for each truth/pred combination."""
    cluster_labels = list(meta.values())
    n_clusters = len(cluster_labels)

    # Create matrix: rows = ground truth, cols = cluster probabilities
    prob_matrix = np.zeros((n_clusters, n_clusters))
    counts = np.zeros(n_clusters)

    for truth_label in cluster_labels:
        truth_id = list(meta.keys())[list(meta.values()).index(truth_label)]
        truth_df = df[df['truth_label'] == truth_label]
        if len(truth_df) > 0:
            counts[truth_id] = len(truth_df)
            for cluster_id in range(n_clusters):
                prob_col = f'prob_cluster_{cluster_id}'
                if prob_col in truth_df.columns:
                    prob_matrix[truth_id, cluster_id] = truth_df[prob_col].mean()

    plt.figure(figsize=(10, 8))
    sns.heatmap(prob_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=cluster_labels, yticklabels=cluster_labels)
    plt.xlabel('Predicted Cluster Probability')
    plt.ylabel('Ground Truth Cluster')
    plt.title('Average Cluster Probabilities by Ground Truth\n(each row shows avg probabilities for that ground truth)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability heatmap saved to {save_path}")
    plt.close()

    return prob_matrix


def plot_cluster_probability_boxplots(df, meta, save_path=None):
    """Creates boxplots of probability distributions per cluster."""
    cluster_labels = list(meta.values())
    n_clusters = len(cluster_labels)

    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]

    for idx, cluster in enumerate(cluster_labels):
        cluster_id = list(meta.keys())[list(meta.values()).index(cluster)]
        prob_col = f'prob_cluster_{cluster_id}'

        if prob_col in df.columns:
            # Split by whether this cluster was the ground truth
            is_truth = df['truth_label'] == cluster
            data = [
                df[is_truth][prob_col].dropna(),
                df[~is_truth][prob_col].dropna()
            ]
            labels = ['Is Ground Truth', 'Not Ground Truth']

            bp = axes[idx].boxplot(data, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')

            axes[idx].set_ylabel('Probability')
            axes[idx].set_title(f'{cluster}\n(n={is_truth.sum()} actual)')
            axes[idx].set_ylim(0, 1)

    plt.suptitle('Cluster Probability Distribution: Actual vs Other Samples', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Probability boxplots saved to {save_path}")
    plt.close()


def print_probability_report(df, meta, cluster_stats, misclass_analysis):
    """Prints detailed probability analysis report."""
    print("\n" + "=" * 90)
    print("CLUSTER PROBABILITY ANALYSIS REPORT")
    print("=" * 90)

    # Overall stats
    print(f"\n{'OVERALL PROBABILITY STATISTICS':^90}")
    print("-" * 90)
    print(f"Total Samples: {len(df)}")
    print(f"Average Max Probability: {df['max_probability'].mean():.4f}")
    print(f"Average Margin (1st - 2nd): {df['margin'].mean():.4f}")
    print(f"Average Entropy: {df['entropy'].mean():.4f}")

    # Correct vs incorrect
    print(f"\n{'CORRECT VS INCORRECT PREDICTIONS':^90}")
    print("-" * 90)
    print(f"{'Metric':<30} {'Correct':<20} {'Incorrect':<20}")
    print("-" * 90)
    print(f"{'Count':<30} {misclass_analysis['correct']['count']:<20} {misclass_analysis['misclassified']['count']:<20}")
    print(f"{'Avg Max Probability':<30} {misclass_analysis['correct']['avg_max_prob']:<20.4f} {misclass_analysis['misclassified']['avg_max_prob']:<20.4f}")
    print(f"{'Avg Margin':<30} {misclass_analysis['correct']['avg_margin']:<20.4f} {misclass_analysis['misclassified']['avg_margin']:<20.4f}")
    print(f"{'Avg Entropy':<30} {misclass_analysis['correct']['avg_entropy']:<20.4f} {misclass_analysis['misclassified']['avg_entropy']:<20.4f}")
    if 'avg_true_cluster_prob' in misclass_analysis['misclassified']:
        print(f"{'Avg True Cluster Prob (wrong)':<30} {'N/A':<20} {misclass_analysis['misclassified']['avg_true_cluster_prob']:<20.4f}")

    # Per-cluster confidence
    print(f"\n{'PER-CLUSTER CONFIDENCE ANALYSIS (Sorted by Accuracy)':^90}")
    print("-" * 90)
    print(f"{'Cluster':<22} {'Count':>6} {'Accuracy':>10} {'Avg Conf':>10} {'Margin':>10} {'Entropy':>10}")
    print("-" * 90)

    sorted_stats = cluster_stats.sort_values('accuracy', ascending=False)
    for _, row in sorted_stats.iterrows():
        print(f"{row['cluster']:<22} {row['count']:>6} {row['accuracy']:>9.1f}% "
              f"{row['avg_confidence']:>10.4f} {row['avg_margin']:>10.4f} {row['avg_entropy']:>10.4f}")

    # Confidence calibration
    print(f"\n{'CONFIDENCE CALIBRATION (Accuracy by Confidence Bin)':^90}")
    print("-" * 90)
    conf_acc = analyze_confidence_accuracy(df)
    print(conf_acc.to_string())

    # Key insights
    print(f"\n{'KEY INSIGHTS':^90}")
    print("-" * 90)

    # Best calibrated cluster
    best_cal = sorted_stats.iloc[0]
    worst_cal = sorted_stats.iloc[-1]
    print(f"Highest Accuracy Cluster: {best_cal['cluster']} ({best_cal['accuracy']:.1f}% acc, {best_cal['avg_confidence']:.3f} conf)")
    print(f"Lowest Accuracy Cluster:  {worst_cal['cluster']} ({worst_cal['accuracy']:.1f}% acc, {worst_cal['avg_confidence']:.3f} conf)")

    # Overconfident clusters (high confidence, low accuracy)
    overconfident = cluster_stats[(cluster_stats['avg_confidence'] > 0.5) & (cluster_stats['accuracy'] < 50)]
    if len(overconfident) > 0:
        print("\nOverconfident Clusters (high conf, low acc):")
        for _, row in overconfident.iterrows():
            print(f"  - {row['cluster']}: {row['avg_confidence']:.3f} conf, {row['accuracy']:.1f}% acc")

    # Underconfident clusters (low confidence, high accuracy)
    underconfident = cluster_stats[(cluster_stats['avg_confidence'] < 0.5) & (cluster_stats['accuracy'] > 50)]
    if len(underconfident) > 0:
        print("\nUnderconfident Clusters (low conf, high acc):")
        for _, row in underconfident.iterrows():
            print(f"  - {row['cluster']}: {row['avg_confidence']:.3f} conf, {row['accuracy']:.1f}% acc")

    print("\n" + "=" * 90)


def run_probability_analysis(excel_path, saved_cluster_path, test_json_files, session_type='robot',
                             save_plots=True, output_prefix='prob_analysis'):
    """
    Runs complete probability-based evaluation of cluster predictions.
    """
    print("Collecting predictions with probabilities...")
    df, meta = collect_predictions_with_probabilities(
        excel_path, saved_cluster_path, test_json_files, session_type
    )

    if len(df) == 0:
        print("No data collected! Check your paths.")
        return None

    print(f"Collected {len(df)} samples with probability data.")

    # Analyze
    cluster_stats = analyze_per_cluster_confidence(df, meta)
    misclass_analysis = analyze_misclassification_probabilities(df, meta)

    # Print report
    print_probability_report(df, meta, cluster_stats, misclass_analysis)

    # Generate plots
    if save_plots:
        plot_confidence_vs_accuracy(df, save_path=f'{output_prefix}_confidence.png')
        plot_per_cluster_confidence(cluster_stats, save_path=f'{output_prefix}_cluster_confidence.png')
        plot_probability_heatmap(df, meta, save_path=f'{output_prefix}_prob_heatmap.png')
        plot_cluster_probability_boxplots(df, meta, save_path=f'{output_prefix}_prob_boxplots.png')
    else:
        plot_confidence_vs_accuracy(df)
        plot_per_cluster_confidence(cluster_stats)
        plot_probability_heatmap(df, meta)
        plot_cluster_probability_boxplots(df, meta)

    return {
        'dataframe': df,
        'cluster_stats': cluster_stats,
        'misclass_analysis': misclass_analysis,
        'meta': meta
    }


if __name__ == "__main__":
    test_data_dir = DATA_DIR / "data" / "test_data"
    test_json_files = list(test_data_dir.glob("*_features.json"))

    print(f"Found {len(test_json_files)} test files:")
    for f in test_json_files:
        print(f"  - {f.name}")

    results = run_probability_analysis(
        excel_path=str(DATA_DIR / "data" / "communication_annotation.xlsx"),
        saved_cluster_path=str(DATA_DIR / "full_dimensions_cluster.pkl"),
        test_json_files=test_json_files,
        session_type='robot',
        save_plots=True,
        output_prefix='prob_analysis'
    )
