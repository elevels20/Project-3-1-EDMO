import pandas as pd
import json
import cluster_prediction
import save_cluster

def parse_excel_time(time_str):
    """
    Parses '01:30 - 02:00' into start_seconds (90) and end_seconds (120).
    """

    try:
        start_str, end_str = time_str.split('-')
        
        # Parse Start
        m1, s1 = map(int, start_str.strip().split(':'))
        start_sec = (m1 * 60) + s1
        
        # Parse End
        m2, s2 = map(int, end_str.strip().split(':'))
        end_sec = (m2 * 60) + s2
        
        return start_sec, end_sec
    except:
        return None, None

def find_matching_excel_row(json_time_sec, df_excel):
    """
    Scans the Excel dataframe to find which row covers the given JSON timestamp.
    """
    for index, row in df_excel.iterrows():
        t_range = str(row['Timestamp Range'])
        e_start, e_end = parse_excel_time(t_range)
        
        if e_start is not None:
            # Check if the JSON time falls inside this Excel window
            # Adding a small buffer (0.5s) for boundary edge cases
            if e_start <= json_time_sec < e_end:
                return row, index + 1
    return None, None

def derive_ground_truth(row, session_type='robot'):
    """
    Maps communication annotation columns to cluster labels.

    Cluster Mapping (based on cluster analysis):
    - C0 Active Inquiry: Active Engagement, Smooth Turn Transition (high questions, low overlap)
    - C1 Quiet Listening: Passive Participation, Calming Prosody, Low Coordination (high silence)
    - C2 Intense Overlap: Competitive Turn Claiming, Heightened Arousal, Extended Monologue (high overlap/interruptions)
    - C3 Disengaged: Disalignment/Resistance, Low Coordination (low collaboration, no questions)
    - C4 Collaborative Inquiry: High Coordination, Active Engagement, Supportive Backchanneling (high questions + collaboration)
    """
    # 1. INTENSE OVERLAP (C2) - Competitive, high arousal, extended speaking
    if (pd.notna(row.get('Competitive Turn Claiming')) or
        pd.notna(row.get('Heightened Arousal Prosody')) or
        pd.notna(row.get('Extended Monologue / Dominance'))):
        return "Intense Overlap"

    # 2. DISENGAGED (C3) - Resistance, disalignment
    if pd.notna(row.get('Disalignment / Resistance')):
        return "Disengaged"

    # 3. COLLABORATIVE INQUIRY (C4) - High coordination with active engagement
    if (pd.notna(row.get('High Coordination')) and
        (pd.notna(row.get('Active Engagement')) or pd.notna(row.get('Supportive Backchanneling')))):
        return "Collaborative Inquiry"

    # 4. ACTIVE INQUIRY (C0) - Active engagement, smooth transitions (but not high coordination)
    if (pd.notna(row.get('Active Engagement')) or
        pd.notna(row.get('Smooth Turn Transition'))):
        return "Active Inquiry"

    # 5. QUIET LISTENING (C1) - Passive, calming, low coordination
    if (pd.notna(row.get('Passive Participation')) or
        pd.notna(row.get('Calming Prosody')) or
        pd.notna(row.get('Low Coordination'))):
        return "Quiet Listening"

    # 6. DEFAULT - No annotations = Quiet Listening (largest cluster, passive)
    return "Quiet Listening"

def run_accuracy_check(excel_path, saved_cluster_path, prediction_json, session_type='robot'):
    # 1. Setup - Cluster ID to Label mapping
    # Based on cluster analysis with communication strategy annotations:
    if session_type == 'robot':
        meta = {
            0: "Active Inquiry",        # High questions, low overlap, Active Engagement
            1: "Quiet Listening",       # High silence, Passive Participation, Calming Prosody
            2: "Intense Overlap",       # High overlap/interruptions, Competitive Turn Claiming
            3: "Disengaged",            # Low collaboration, Disalignment/Resistance
            4: "Collaborative Inquiry"  # High questions + collaboration, High Coordination
        }
    else:
        meta = {
            0: "Quiet Listening",       # Passive Participation, Calming Prosody
            1: "Intense Overlap",       # Competitive Turn Claiming, Heightened Arousal
            2: "Collaborative Inquiry", # High Coordination, Active Engagement
            3: "Deep Silence"           # Passive Participation, Low Coordination
        }

    saved_cluster = save_cluster.load_cluster(saved_cluster_path)
    membership = cluster_prediction.predict_cluster(saved_cluster, prediction_json)
    
    if excel_path.endswith('.csv'):
        df_excel = pd.read_csv(excel_path)
    else:
        df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()

    with open(prediction_json, 'r') as f:
        data = json.load(f)

    valid_indices = [i for i, w in enumerate(data['audio_features']) if w.get('emotion') is not None]
    
    matches = 0
    total = 0

    print(f"--- Time-Aligned Accuracy Report: {data['session']} ---")
    print(f"{'J-Win':<6} | {'Time':<8} | {'Ex-Win':<6} | {'Prediction':<20} | {'Truth':<20} | {'Result'}")
    print("-" * 95)

    for idx, original_idx in enumerate(valid_indices):
        if idx < len(membership):
            # A. Get Prediction
            pred_label = meta[membership[idx]]
            
            # B. Get JSON Time
            json_start = data['audio_features'][original_idx]['window_start']
            
            # C. Find CORRECT Excel Row by Time (Not by ID)
            matched_row, excel_win_id = find_matching_excel_row(json_start, df_excel)
            
            if matched_row is not None:
                truth_label = derive_ground_truth(matched_row, session_type)
                
                is_match = (pred_label == truth_label)
                if is_match: matches += 1
                total += 1
                
                icon = "YES" if is_match else "NO"
                print(f"{original_idx:<6} | {json_start:<8.1f} | {excel_win_id:<6} | {pred_label:<20} | {truth_label:<20} | {icon}")
            else:
                print(f"{original_idx:<6} | {json_start:<8.1f} | {'???':<6} | {pred_label:<20} | {'(Out of Range)':<20} | WARN")

    accuracy = (matches / total) * 100 if total > 0 else 0
    print("-" * 95)
    print(f"FINAL ACCURACY: {accuracy:.2f}% ({matches}/{total} matches)")

# Run with your file paths
run_accuracy_check(
    excel_path="./data/communication_annotation.xlsx",
    saved_cluster_path="./full_dimensions_cluster.pkl",
    prediction_json="./data/test_data/114654_features.json",
    session_type='robot'
)