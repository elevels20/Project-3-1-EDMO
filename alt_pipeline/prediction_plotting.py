import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cluster_prediction
import save_cluster
import json_extraction
from pathlib import Path

# Define output directory for plots
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "RESULT_TEST_DATA"
RESULTS_DIR.mkdir(exist_ok=True)

def plot_data(saved_cluster, prediction_file, type):
    # Cluster Meta Definitions
    # Communication Strategy Annotations based on cluster analysis:
    # C0: Active Inquiry -> Active Engagement, Smooth Turn Transition
    # C1: Quiet Listening -> Passive Participation, Calming Prosody, Low Coordination
    # C2: Intense Overlap -> Competitive Turn Claiming, Heightened Arousal, Extended Monologue
    # C3: Disengaged -> Disalignment/Resistance, Low Coordination
    # C4: Collaborative Inquiry -> High Coordination, Active Engagement, Supportive Backchanneling
    if type == 'robot':
        cluster_meta = {
            0: {"name": "Active Inquiry", "color": "#2ca02c",
                "annotations": ["Active Engagement", "Smooth Turn Transition"]},
            1: {"name": "Quiet Listening", "color": "#7f7f7f",
                "annotations": ["Passive Participation", "Calming Prosody", "Low Coordination"]},
            2: {"name": "Intense Overlap", "color": "#d62728",
                "annotations": ["Competitive Turn Claiming", "Heightened Arousal Prosody", "Extended Monologue"]},
            3: {"name": "Disengaged", "color": "#ff7f0e",
                "annotations": ["Disalignment/Resistance", "Low Coordination"]},
            4: {"name": "Collaborative Inquiry", "color": "#1f77b4",
                "annotations": ["High Coordination", "Active Engagement", "Supportive Backchanneling"]}
        }
    elif type == 'voice':
        cluster_meta = {
            0: {"name": "Quiet Listening", "color": "#aec7e8",
                "annotations": ["Passive Participation", "Calming Prosody"]},
            1: {"name": "Intense Overlap", "color": "#d62728",
                "annotations": ["Competitive Turn Claiming", "Heightened Arousal Prosody"]},
            2: {"name": "Collaborative Inquiry", "color": "#2ca02c",
                "annotations": ["High Coordination", "Active Engagement"]},
            3: {"name": "Deep Silence", "color": "#7f7f7f",
                "annotations": ["Passive Participation", "Low Coordination"]}
        }

    # 1. Get Cluster Predictions
    membership = cluster_prediction.predict_cluster(saved_cluster, prediction_file)
    
    # 2. Load Data from JSON
    with open(prediction_file, 'r') as f:
        data = json.load(f)
    
    audio_windows = data.get('audio_features', [])
    robot_windows = data.get('robot_speed_features', [])
    
    # 3. Align Cluster Data with Robot Speed Data
    # Filter indices where 'emotion' exists (as per your previous clustering logic)
    valid_indices = [i for i, w in enumerate(audio_windows) if w.get('emotion') is not None]
    
    df_list = []
    for idx, original_idx in enumerate(valid_indices):
        if idx < len(membership):
            cid = membership[idx]
            # Get the robot speed window at the same index
            r_window = robot_windows[original_idx]
            
            df_list.append({
                "window_num": r_window['window_index'], # Show window number
                "start_min": r_window['window_start'] / 60, # Seconds to Minutes
                "duration_min": (r_window['window_end'] - r_window['window_start']) / 60,
                "speed": r_window.get('avg_speed_cm_s', 0), # Correct key from your snippet
                "cluster": cid,
                "color": cluster_meta[cid]['color']
            })

    if not df_list:
        print("No valid data to plot.")
        return

    df = pd.DataFrame(df_list)
    k = 5 if type == 'robot' else 4

    # 4. Visualization
    fig, ax1 = plt.subplots(figsize=(16, 7))

    # A. Plot Cluster Timeline Blocks
    for _, row in df.iterrows():
        ax1.broken_barh([(row['start_min'], row['duration_min'])], 
                        (row['cluster'] - 0.4, 0.8), 
                        facecolors=row['color'], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # B. Add Window Number Label on the bar
        ax1.text(row['start_min'] + (row['duration_min']/2), row['cluster'], 
                 str(int(row['window_num'])), color='white', ha='center', va='center', 
                 fontsize=8, fontweight='bold', path_effects=None)

    # C. Plot Speed Average on Secondary Axis
    ax2 = ax1.twinx()
    ax2.plot(df['start_min'], df['speed'], color='black', marker='.', 
             linestyle='-', alpha=0.5, label='Avg Robot Speed (cm/s)')
    ax2.set_ylabel("Robot Speed (cm/s)")
    ax2.set_ylim(0, df['speed'].max() * 1.2 if not df.empty else 1)
    ax2.legend(loc='upper right')

    # Formatting
    total_duration_min = robot_windows[-1]['window_end'] / 60
    ax1.set_xlim(0, total_duration_min)
    ax1.set_yticks(range(k))
    ax1.set_yticklabels([f"C{i}: {cluster_meta[i]['name']}" for i in range(k)])
    ax1.set_xlabel("Time (Minutes)")
    ax1.set_title(f"Interaction Timeline & Speed: {data['session']} (Full Session)")
    ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'interaction_timeline_t2_{type}_updated.png')
    plt.show()

def get_cluster_meta(type):
    """Get cluster metadata based on type."""
    if type == 'robot':
        return {
            0: {"name": "Active Inquiry", "color": "#2ca02c"},
            1: {"name": "Quiet Listening", "color": "#7f7f7f"},
            2: {"name": "Intense Overlap", "color": "#d62728"},
            3: {"name": "Disengaged", "color": "#ff7f0e"},
            4: {"name": "Collaborative Inquiry", "color": "#1f77b4"}
        }
    else:  # voice
        return {
            0: {"name": "Quiet Listening", "color": "#aec7e8"},
            1: {"name": "Intense Overlap", "color": "#d62728"},
            2: {"name": "Collaborative Inquiry", "color": "#2ca02c"},
            3: {"name": "Deep Silence", "color": "#7f7f7f"}
        }


def extract_full_dataframe(saved_cluster, prediction_file, type):
    """Extract a comprehensive DataFrame with all features and cluster assignments."""
    membership = cluster_prediction.predict_cluster(saved_cluster, prediction_file)
    cluster_meta = get_cluster_meta(type)

    with open(prediction_file, 'r') as f:
        data = json.load(f)

    audio_windows = data.get('audio_features', [])
    robot_windows = data.get('robot_speed_features', [])

    valid_indices = [i for i, w in enumerate(audio_windows) if w.get('emotion') is not None]

    df_list = []
    for idx, original_idx in enumerate(valid_indices):
        if idx < len(membership):
            cid = membership[idx]
            a_window = audio_windows[original_idx]
            r_window = robot_windows[original_idx]

            # Extract emotions
            emotions = {e['label']: e['score'] for e in a_window['emotion']['emotions']}

            # Extract nonverbal features
            conv = a_window.get('nonverbal', {}).get('basic_metrics', {}).get('conversation', {})

            # Extract NLP features
            nlp = a_window.get('nlp', {})
            sentiment = nlp.get('sentiment', {})
            simple_feat = nlp.get('simple_features', {})

            df_list.append({
                "window_num": r_window['window_index'],
                "start_min": r_window['window_start'] / 60,
                "end_min": r_window['window_end'] / 60,
                "duration_min": (r_window['window_end'] - r_window['window_start']) / 60,
                "cluster": cid,
                "cluster_name": cluster_meta[cid]['name'],
                "color": cluster_meta[cid]['color'],
                # Robot features
                "speed": r_window.get('avg_speed_cm_s', 0),
                "num_detections": r_window.get('num_detections', 0),
                # Emotions
                "neutral": emotions.get('neutral', 0),
                "surprise": emotions.get('surprise', 0),
                "joy": emotions.get('joy', 0),
                "sadness": emotions.get('sadness', 0),
                "anger": emotions.get('anger', 0),
                "fear": emotions.get('fear', 0),
                "disgust": emotions.get('disgust', 0),
                # Sentiment
                "sentiment_score": sentiment.get('score', 0),
                "sentiment_label": sentiment.get('label', 'Unknown'),
                # Nonverbal
                "overlap_ratio": conv.get('overlap_ratio', 0),
                "silence_ratio": conv.get('silence_ratio', 0),
                "interruption_rate": conv.get('interruption_rate', 0),
                "total_interruptions": conv.get('total_interruptions', 0),
                "num_speakers": conv.get('num_speakers', 0),
                "total_speaking_time": conv.get('total_speaking_time', 0),
                # NLP simple features
                "collaboration_ratio": simple_feat.get('collaboration_ratio', 0),
                "hedges": simple_feat.get('hedges (uncertainty)', 0),
                "wh_questions": simple_feat.get('wh_questions (inquiry)', 0),
                "yes_no_questions": simple_feat.get('yes_no_questions (confirmation)', 0),
                "question_count": simple_feat.get('question_count', 0),
            })

    return pd.DataFrame(df_list), data['session'], cluster_meta


def plot_cluster_distribution(df, cluster_meta, session, save=True):
    """Plot cluster distribution as pie and bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate time spent in each cluster
    cluster_time = df.groupby('cluster')['duration_min'].sum()
    total_time = cluster_time.sum()

    labels = [f"C{c}: {cluster_meta[c]['name']}" for c in cluster_time.index]
    colors = [cluster_meta[c]['color'] for c in cluster_time.index]

    # Pie chart
    axes[0].pie(cluster_time, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, explode=[0.02]*len(cluster_time))
    axes[0].set_title(f'Time Distribution by Cluster\n{session}')

    # Bar chart with count
    cluster_counts = df['cluster'].value_counts().sort_index()
    bars = axes[1].bar([f"C{c}" for c in cluster_counts.index], cluster_counts.values,
                       color=[cluster_meta[c]['color'] for c in cluster_counts.index])
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Windows')
    axes[1].set_title(f'Window Count by Cluster\n{session}')

    # Add value labels on bars
    for bar, val in zip(bars, cluster_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     str(val), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'cluster_distribution_{session}.png', dpi=150)
    plt.show()


def plot_transition_matrix(df, cluster_meta, session, save=True):
    """Plot cluster transition matrix as heatmap."""
    k = len(cluster_meta)

    # Build transition matrix
    transitions = np.zeros((k, k))
    for i in range(len(df) - 1):
        from_cluster = df.iloc[i]['cluster']
        to_cluster = df.iloc[i + 1]['cluster']
        transitions[from_cluster, to_cluster] += 1

    # Normalize rows to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    trans_prob = transitions / row_sums

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = [f"C{i}" for i in range(k)]

    # Raw counts
    sns.heatmap(transitions, annot=True, fmt='.0f', cmap='Blues', ax=axes[0],
                xticklabels=labels, yticklabels=labels)
    axes[0].set_xlabel('To Cluster')
    axes[0].set_ylabel('From Cluster')
    axes[0].set_title(f'Transition Counts\n{session}')

    # Probabilities
    sns.heatmap(trans_prob, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1],
                xticklabels=labels, yticklabels=labels)
    axes[1].set_xlabel('To Cluster')
    axes[1].set_ylabel('From Cluster')
    axes[1].set_title(f'Transition Probabilities\n{session}')

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'transition_matrix_{session}.png', dpi=150)
    plt.show()


def plot_speed_by_cluster(df, cluster_meta, session, save=True):
    """Boxplot of robot speed by cluster."""
    fig, ax = plt.subplots(figsize=(10, 6))

    clusters = sorted(df['cluster'].unique())
    data_by_cluster = [df[df['cluster'] == c]['speed'].values for c in clusters]
    colors = [cluster_meta[c]['color'] for c in clusters]

    bp = ax.boxplot(data_by_cluster, patch_artist=True, labels=[f"C{c}" for c in clusters])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Cluster')
    ax.set_ylabel('Robot Speed (cm/s)')
    ax.set_title(f'Robot Speed Distribution by Cluster\n{session}')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # Add cluster names below
    ax.set_xticklabels([f"C{c}\n{cluster_meta[c]['name']}" for c in clusters], fontsize=9)

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'speed_by_cluster_{session}.png', dpi=150)
    plt.show()


def plot_emotion_trajectories(df, session, save=True):
    """Plot emotion scores over time."""
    emotions = ['neutral', 'surprise', 'joy', 'sadness', 'anger', 'fear', 'disgust']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728', '#8c564b', '#7f7f7f']

    fig, ax = plt.subplots(figsize=(14, 6))

    for emotion, color in zip(emotions, colors):
        ax.plot(df['start_min'], df[emotion], label=emotion.capitalize(),
                color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Time (Minutes)')
    ax.set_ylabel('Emotion Score')
    ax.set_title(f'Emotion Trajectories Over Time\n{session}')
    ax.legend(loc='upper right', ncol=4)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(0, df['end_min'].max())

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'emotion_trajectories_{session}.png', dpi=150)
    plt.show()


def plot_nonverbal_features(df, cluster_meta, session, save=True):
    """Plot nonverbal features (overlap, silence, interruptions) over time with cluster background."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    features = [
        ('overlap_ratio', 'Overlap Ratio', '#e74c3c'),
        ('silence_ratio', 'Silence Ratio', '#3498db'),
        ('interruption_rate', 'Interruption Rate', '#f39c12')
    ]

    for ax, (feat, label, color) in zip(axes, features):
        # Background cluster coloring
        for _, row in df.iterrows():
            ax.axvspan(row['start_min'], row['end_min'],
                      alpha=0.2, color=row['color'])

        # Plot feature line
        ax.plot(df['start_min'], df[feat], color=color, linewidth=2, marker='o', markersize=3)
        ax.fill_between(df['start_min'], 0, df[feat], alpha=0.3, color=color)
        ax.set_ylabel(label)
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    axes[-1].set_xlabel('Time (Minutes)')
    axes[0].set_title(f'Nonverbal Features Over Time (background = cluster)\n{session}')

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'nonverbal_features_{session}.png', dpi=150)
    plt.show()


def plot_cluster_duration_stats(df, cluster_meta, session, save=True):
    """Analyze and plot how long each cluster phase typically lasts."""
    # Find consecutive cluster segments
    segments = []
    current_cluster = df.iloc[0]['cluster']
    segment_start = df.iloc[0]['start_min']

    for i in range(1, len(df)):
        if df.iloc[i]['cluster'] != current_cluster:
            segment_end = df.iloc[i-1]['end_min']
            segments.append({
                'cluster': current_cluster,
                'duration': segment_end - segment_start
            })
            current_cluster = df.iloc[i]['cluster']
            segment_start = df.iloc[i]['start_min']

    # Add last segment
    segments.append({
        'cluster': current_cluster,
        'duration': df.iloc[-1]['end_min'] - segment_start
    })

    seg_df = pd.DataFrame(segments)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot of segment durations
    clusters = sorted(seg_df['cluster'].unique())
    data = [seg_df[seg_df['cluster'] == c]['duration'].values for c in clusters]
    colors = [cluster_meta[c]['color'] for c in clusters]

    bp = axes[0].boxplot(data, patch_artist=True, labels=[f"C{c}" for c in clusters])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Segment Duration (minutes)')
    axes[0].set_title(f'Duration of Consecutive Cluster Phases\n{session}')

    # Count of segments per cluster
    seg_counts = seg_df['cluster'].value_counts().sort_index()
    bars = axes[1].bar([f"C{c}" for c in seg_counts.index], seg_counts.values,
                       color=[cluster_meta[c]['color'] for c in seg_counts.index])
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Phases')
    axes[1].set_title(f'Number of Distinct Phases per Cluster\n{session}')

    for bar, val in zip(bars, seg_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     str(val), ha='center', fontweight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'cluster_duration_stats_{session}.png', dpi=150)
    plt.show()

    return seg_df


def plot_feature_correlation(df, session, save=True):
    """Plot correlation heatmap of key features."""
    feature_cols = ['speed', 'neutral', 'surprise', 'joy', 'sadness', 'anger',
                    'sentiment_score', 'overlap_ratio', 'silence_ratio',
                    'interruption_rate', 'question_count', 'collaboration_ratio']

    # Filter to columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]

    # Select only numeric columns and convert to numeric, coercing errors
    df_numeric = df[available_cols].apply(pd.to_numeric, errors='coerce')
    corr_matrix = df_numeric.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title(f'Feature Correlation Matrix\n{session}')

    plt.tight_layout()
    if save:
        plt.savefig(RESULTS_DIR / f'feature_correlation_{session}.png', dpi=150)
    plt.show()


def plot_full_dashboard(saved_cluster, prediction_file, type='robot', save=True):
    """Generate all plots for a comprehensive analysis dashboard."""
    print(f"Generating full dashboard for {prediction_file}...")

    df, session, cluster_meta = extract_full_dataframe(saved_cluster, prediction_file, type)

    if df.empty:
        print("No valid data to plot.")
        return

    print(f"\n--- Session: {session} ---")
    print(f"Total windows: {len(df)}")
    print(f"Duration: {df['end_min'].max():.1f} minutes")
    print(f"Clusters found: {sorted(df['cluster'].unique())}")

    # Generate all plots
    plot_cluster_distribution(df, cluster_meta, session, save)
    plot_transition_matrix(df, cluster_meta, session, save)
    plot_speed_by_cluster(df, cluster_meta, session, save)
    plot_emotion_trajectories(df, session, save)
    plot_nonverbal_features(df, cluster_meta, session, save)
    plot_cluster_duration_stats(df, cluster_meta, session, save)
    plot_feature_correlation(df, session, save)

    # Also plot the original timeline
    plot_data(saved_cluster, prediction_file, type)

    print(f"\nDashboard complete! All plots saved for session {session}.")
    return df


# Execution
DATA_DIR = cluster_prediction.DATA_DIR
saved_cluster = save_cluster.load_cluster(DATA_DIR / "full_dimensions_cluster.pkl")

# Run full dashboard
plot_full_dashboard(saved_cluster, DATA_DIR / "data" / "test_data" / "111455_features.json", 'robot')

# Or run individual plots:
#plot_data(saved_cluster, DATA_DIR / "data" / "test_data" / "111455_features.json", 'robot')