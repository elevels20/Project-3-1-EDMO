import numpy as np
import pickle
from pathlib import Path
import json_extraction
from json_extraction import selected_features, training_files, features_labels

def load_cluster(filename):
    """Load clustered data from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_avg_robot_speed_by_cluster(pkl_path):
    """
    Extract average robot speed for each cluster from the .pkl file.

    Args:
        pkl_path: Path to the cluster .pkl file

    Returns:
        Dictionary mapping cluster ID to average robot speed
    """
    # Load cluster data
    cluster_data = load_cluster(pkl_path)
    silhouette_scores, best_score, best_k, cluster_labels, u, cntr, fpc = cluster_data

    # Load the original datapoints to get robot speed values
    datapoints = json_extraction.extract_datapoints_except_last_multiple(
        training_files,
        selected_features,
        features_labels
    )

    # Find the index of robot speed feature
    robot_speed_idx = features_labels.index("robot_speed_features_avg_speed_cm_s")

    # Build feature matrix and extract robot speed column
    X = np.array([dp.dimension_values for dp in datapoints])
    robot_speeds = X[:, robot_speed_idx]

    # Calculate average robot speed for each cluster
    cluster_speed_stats = {}

    for cluster_id in range(best_k):
        # Get indices of datapoints belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_speeds = robot_speeds[cluster_mask]

        if len(cluster_speeds) > 0:
            cluster_speed_stats[cluster_id] = {
                'avg_speed': np.mean(cluster_speeds),
                'std_speed': np.std(cluster_speeds),
                'min_speed': np.min(cluster_speeds),
                'max_speed': np.max(cluster_speeds),
                'count': len(cluster_speeds)
            }
        else:
            cluster_speed_stats[cluster_id] = {
                'avg_speed': 0,
                'std_speed': 0,
                'min_speed': 0,
                'max_speed': 0,
                'count': 0
            }

    return cluster_speed_stats, best_k

def display_results(cluster_speed_stats, best_k):
    """Display the results in a formatted table."""
    # Cluster labels mapping (from matching_prediction.py)
    cluster_names = {
        0: "Active Inquiry",
        1: "Quiet Listening",
        2: "Intense Overlap",
        3: "Disengaged",
        4: "Collaborative Inquiry"
    }

    print("\n" + "=" * 80)
    print("AVERAGE ROBOT SPEED BY CLUSTER")
    print("=" * 80)
    print(f"{'Cluster':<5} | {'Name':<22} | {'Avg Speed':<12} | {'Std Dev':<10} | {'Min':<8} | {'Max':<8} | {'Count'}")
    print("-" * 80)

    for cluster_id in range(best_k):
        stats = cluster_speed_stats[cluster_id]
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        print(f"C{cluster_id:<4} | {name:<22} | {stats['avg_speed']:>10.2f} | {stats['std_speed']:>8.2f} | {stats['min_speed']:>6.2f} | {stats['max_speed']:>6.2f} | {stats['count']}")

    print("-" * 80)

    # Show ranking by average speed
    print("\nClusters ranked by average robot speed (highest to lowest):")
    sorted_clusters = sorted(cluster_speed_stats.items(), key=lambda x: x[1]['avg_speed'], reverse=True)
    for rank, (cluster_id, stats) in enumerate(sorted_clusters, 1):
        name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        print(f"  {rank}. C{cluster_id} ({name}): {stats['avg_speed']:.2f} cm/s")

def plot_results(cluster_speed_stats, best_k):
    """Create a bar chart visualization of the results."""
    import matplotlib.pyplot as plt

    cluster_names = {
        0: "Active Inquiry",
        1: "Quiet Listening",
        2: "Intense Overlap",
        3: "Disengaged",
        4: "Collaborative Inquiry"
    }

    clusters = list(range(best_k))
    avg_speeds = [cluster_speed_stats[c]['avg_speed'] for c in clusters]
    std_speeds = [cluster_speed_stats[c]['std_speed'] for c in clusters]
    labels = [f"C{c}\n{cluster_names.get(c, '')}" for c in clusters]

    colors = plt.cm.viridis(np.linspace(0, 1, best_k))

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(labels, avg_speeds, yerr=std_speeds, capsize=5, color=colors, edgecolor='black')

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Average Robot Speed (cm/s)', fontsize=12)
    ax.set_title('Average Robot Speed by Cluster', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, speed in zip(bars, avg_speeds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{speed:.2f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('robot_speed_by_cluster.png', dpi=150)
    plt.show()
    print("\nPlot saved as 'robot_speed_by_cluster.png'")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    pkl_path = BASE_DIR / "full_dimensions_cluster.pkl"

    print(f"Loading cluster data from: {pkl_path}")

    cluster_speed_stats, best_k = get_avg_robot_speed_by_cluster(pkl_path)
    display_results(cluster_speed_stats, best_k)

    # Uncomment to generate plot
    plot_results(cluster_speed_stats, best_k)