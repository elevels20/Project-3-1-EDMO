import json
import matplotlib.pyplot as plt
import pandas as pd
import re
import cluster_prediction
import save_cluster
import numpy as np

def plot_data(saved_cluster, prediction_file,type):
    if type == 'robot' : 
        cluster_meta = {
            0: {"name": "Moderate Activity", "color": "#1f77b4", "desc": "Steady functional talk"},
            1: {"name": "Silence/Idle", "color": "#7f7f7f", "desc": "No movement, high silence"},
            2: {"name": "Intense Engagement", "color": "#d62728", "desc": "High overlaps and emotion"},
            3: {"name": "Passive Presence A", "color": "#aec7e8", "desc": "Neutral/Quiet monitoring"},
            4: {"name": "Passive Presence B", "color": "#c5b0d5", "desc": "Low activity baseline"}
        }

    if type == 'voice' :
        cluster_meta = {
            0: {"name": "Quiet Listening", "color": "#aec7e8", "desc": "Minimal talk, steady presence"},
            1: {"name": "Intense Overlap", "color": "#d62728", "desc": "Rapid interaction, high interruptions"},
            2: {"name": "Collaborative Inquiry", "color": "#2ca02c", "desc": "Productive talk, questions, collaboration"},
            3: {"name": "Deep Silence", "color": "#7f7f7f", "desc": "Passive/Idle, no engagement"}
        }
    
    # 1. Get Membership Matrix (likely shape 7x38)
    membership = cluster_prediction.predict_cluster(saved_cluster, prediction_file)
    
    # 3. Load timestamps
    with open(prediction_file, 'r') as f:
        data = json.load(f)
    
    all_windows = data['audio_features']
    
    # We need to filter the windows to match the 38 predictions.
    valid_windows = [w for w in all_windows if w.get('emotion') is not None]
    
    df_list = []
    for i, window in enumerate(valid_windows):
        if i < len(membership):
            cid = membership[i]
            df_list.append({
                "start": window['window_start'],
                "duration": window['window_duration'],
                "cluster": cid,
                "persona": cluster_meta[cid]['name'],
                "color": cluster_meta[cid]['color']
            })

    if not df_list:
        print("No valid data to plot.")
        return

    df = pd.DataFrame(df_list)
    if type == 'robot': k = 5
    if type == 'voice': k = 4

    # 4. Visualization - Fixed Memory and Scaling
    fig, ax = plt.subplots(figsize=(15, 6)) # Wider to see full timeline
    
    for _, row in df.iterrows():
        ax.broken_barh([(row['start'], row['duration'])], (row['cluster'] - 0.4, 0.8), 
                       facecolors=row['color'], alpha=0.9, edgecolor='black', linewidth=0.5)

    # Set X-Axis to cover the FULL session duration (even if data is missing)
    total_duration = all_windows[-1]['window_end']
    ax.set_xlim(0, total_duration)

    ax.set_yticks(range(k))
    ax.set_yticklabels([f"C{i}: {cluster_meta[i]['name']}" for i in range(k)])
    ax.set_xlabel("Time (Seconds)")
    ax.set_title(f"Interaction Timeline: {data['session']} (Full Session)")
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'interaction_timeline_t2_{type}.png')
    plt.close(fig)

    print(f"Plot saved. Processed {len(df)} windows out of {len(all_windows)} total.")

DATA_DIR = cluster_prediction.DATA_DIR
saved_cluster = save_cluster.load_cluster(DATA_DIR / "full_dimensions_cluster_voice.pkl")
plot_data(saved_cluster, DATA_DIR / "data" / "test_data" / "114654_features.json",'voice')




        
        
            
            


