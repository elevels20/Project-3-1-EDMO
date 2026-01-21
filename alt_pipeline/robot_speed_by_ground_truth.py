"""
Robot Speed by Human-Annotated Ground Truth Labels.

Creates a bar chart showing average robot speed for each ground truth
communication strategy label from human annotations.
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

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


def derive_ground_truth(row):
    """
    Maps communication annotation columns to cluster labels.
    Uses hierarchical rules to assign a single label.
    """
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


def collect_speed_by_ground_truth(excel_path, test_json_files):
    """
    Collects robot speed data grouped by human-annotated ground truth labels.

    Returns:
        dict mapping ground truth label -> list of robot speeds
    """
    if excel_path.endswith('.csv'):
        df_excel = pd.read_csv(excel_path)
    else:
        df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()

    speed_by_label = defaultdict(list)

    for json_path in test_json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)

        robot_speeds = data.get('robot_speed_features', [])

        # Create a mapping from window_start to robot speed
        speed_lookup = {}
        for rs in robot_speeds:
            speed_lookup[rs['window_start']] = rs['avg_speed_cm_s']

        # Match each window to ground truth and get robot speed
        for window in data['audio_features']:
            if window.get('emotion') is None:
                continue

            window_start = window['window_start']
            matched_row, _ = find_matching_excel_row(window_start, df_excel)

            if matched_row is not None:
                ground_truth = derive_ground_truth(matched_row)

                # Find robot speed for this window
                if window_start in speed_lookup:
                    robot_speed = speed_lookup[window_start]
                    speed_by_label[ground_truth].append(robot_speed)

    return speed_by_label


def compute_speed_statistics(speed_by_label):
    """Compute statistics for each ground truth label."""
    stats = {}
    for label, speeds in speed_by_label.items():
        if speeds:
            stats[label] = {
                'avg_speed': np.mean(speeds),
                'std_speed': np.std(speeds),
                'min_speed': np.min(speeds),
                'max_speed': np.max(speeds),
                'count': len(speeds)
            }
        else:
            stats[label] = {
                'avg_speed': 0,
                'std_speed': 0,
                'min_speed': 0,
                'max_speed': 0,
                'count': 0
            }
    return stats


def plot_speed_by_ground_truth(stats, save_path=None):
    """
    Creates a bar chart of average robot speed by human-annotated ground truth label.

    X-Axis: Human-Annotated Labels
    Y-Axis: Average Robot Speed (cm/s)
    """
    # Define the order of labels for consistent display
    label_order = [
        "Active Inquiry",
        "Quiet Listening",
        "Intense Overlap",
        "Disengaged",
        "Collaborative Inquiry"
    ]

    # Filter to only labels that have data
    labels = [l for l in label_order if l in stats and stats[l]['count'] > 0]
    avg_speeds = [stats[l]['avg_speed'] for l in labels]
    std_speeds = [stats[l]['std_speed'] for l in labels]
    counts = [stats[l]['count'] for l in labels]

    # Create color palette
    colors = ['#2196F3', '#9C27B0', '#F44336', '#FF9800', '#4CAF50']
    color_map = {label_order[i]: colors[i] for i in range(len(label_order))}
    bar_colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, avg_speeds, yerr=std_speeds, capsize=5,
                  color=bar_colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, speed, count in zip(bars, avg_speeds, counts):
        height = bar.get_height()
        ax.annotate(f'{speed:.2f}\n(n={count})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Human-Annotated Communication Strategy', fontsize=12)
    ax.set_ylabel('Average Robot Speed (cm/s)', fontsize=12)
    ax.set_title('Average Robot Speed by Ground Truth Label', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add a note about the data
    total_samples = sum(counts)
    ax.text(0.98, 0.02, f'Total samples: {total_samples}',
            transform=ax.transAxes, ha='right', fontsize=9, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

    return stats


def print_speed_table(stats):
    """Print a formatted table of speed statistics."""
    label_order = [
        "Active Inquiry",
        "Quiet Listening",
        "Intense Overlap",
        "Disengaged",
        "Collaborative Inquiry"
    ]

    print("\n" + "=" * 85)
    print("AVERAGE ROBOT SPEED BY HUMAN-ANNOTATED GROUND TRUTH LABEL")
    print("=" * 85)
    print(f"{'Label':<25} | {'Avg Speed':<12} | {'Std Dev':<10} | {'Min':<8} | {'Max':<8} | {'Count'}")
    print("-" * 85)

    for label in label_order:
        if label in stats:
            s = stats[label]
            print(f"{label:<25} | {s['avg_speed']:>10.2f} | {s['std_speed']:>8.2f} | "
                  f"{s['min_speed']:>6.2f} | {s['max_speed']:>6.2f} | {s['count']}")

    print("-" * 85)

    # Ranking
    print("\nLabels ranked by average robot speed (highest to lowest):")
    sorted_labels = sorted(
        [(l, s) for l, s in stats.items() if s['count'] > 0],
        key=lambda x: x[1]['avg_speed'],
        reverse=True
    )
    for rank, (label, s) in enumerate(sorted_labels, 1):
        print(f"  {rank}. {label}: {s['avg_speed']:.2f} cm/s (n={s['count']})")


def run_analysis(excel_path, test_json_files, save_path=None):
    """Run the complete analysis."""
    print("Collecting robot speed data by ground truth labels...")
    speed_by_label = collect_speed_by_ground_truth(excel_path, test_json_files)

    if not speed_by_label:
        print("No data collected! Check your file paths.")
        return None

    print(f"Collected data for {len(speed_by_label)} ground truth labels.")

    stats = compute_speed_statistics(speed_by_label)
    print_speed_table(stats)
    plot_speed_by_ground_truth(stats, save_path=save_path)

    return stats


if __name__ == "__main__":
    # Find test files
    test_data_dir = DATA_DIR / "data" / "test_data"
    test_json_files = list(test_data_dir.glob("*_features.json"))

    print(f"Found {len(test_json_files)} test files:")
    for f in test_json_files:
        print(f"  - {f.name}")

    # Run analysis
    stats = run_analysis(
        excel_path=str(DATA_DIR / "data" / "communication_annotation.xlsx"),
        test_json_files=test_json_files,
        save_path='robot_speed_by_ground_truth.png'
    )
