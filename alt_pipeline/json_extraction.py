import json

class Datapoint:
    dimension_labels: list[str]
    dimension_values: list[float]

    def __init__(self, labels, values):
        self.dimension_labels = labels
        self.dimension_values = values


selected_features = [
    "audio_features.base_window_len",                                           # 1
    "audio_features.emotion.emotions[0].score",                                 # 2
    "audio_features.emotion.emotions[1].score",                                 # 3
    "audio_features.emotion.emotions[2].score",                                 # 4
    "audio_features.emotion.emotions[3].score",                                 # 5
    "audio_features.emotion.emotions[4].score",                                 # 6
    "audio_features.emotion.emotions[5].score",                                 # 7
    "audio_features.emotion.emotions[6].score",                                 # 8
    "audio_features.nlp.sentiment.score",                                       # 9
    "audio_features.nonverbal.basic_metrics.conversation.interruption_rate",    # 10
    "audio_features.nonverbal.basic_metrics.conversation.num_speakers",         # 11
    "audio_features.nonverbal.basic_metrics.conversation.overlap_duration",     # 12
    "audio_features.nonverbal.basic_metrics.conversation.overlap_ratio",        # 13
    "audio_features.nonverbal.basic_metrics.conversation.silence_duration",     # 14
    "audio_features.nonverbal.basic_metrics.conversation.silence_ratio",       # 15
    "audio_features.nonverbal.basic_metrics.conversation.total_interruptions",  # 16
    "audio_features.nonverbal.basic_metrics.conversation.total_speaking_time",  # 17
    "audio_features.window_duration",                                           # 18
    "audio_features.window_end",                                                # 19
    "audio_features.window_index",                                              # 20
    "audio_features.window_start",                                              # 21
    "robot_speed_features.avg_speed_cm_s",                                      # 22
    "robot_speed_features.num_detections",                                      # 23
    "robot_speed_features.window_end",                                          # 24
    "robot_speed_features.window_index",                                        # 25
    "robot_speed_features.window_start"
]

features_labels = [
    "audio_features_emotion_emotions_0_score",
    "audio_features_emotion_emotions_1_score",
    "audio_features_emotion_emotions_2_score",
    "audio_features_emotion_emotions_3_score",
    "audio_features_emotion_emotions_4_score",
    "audio_features_emotion_emotions_5_score",
    "audio_features_emotion_emotions_6_score",
    "audio_features_nlp_sentiment_score",
    "audio_features_nonverbal_basic_metrics_conversation_interruption_rate",
    "audio_features_nonverbal_basic_metrics_conversation_num_speakers",
    "audio_features_nonverbal_basic_metrics_conversation_overlap_duration",
    "audio_features_nonverbal_basic_metrics_conversation_overlap_ratio",
    "audio_features_nonverbal_basic_metrics_conversation_silence_duration",
    "audio_features_nonverbal_basic_metrics_conversation_silence_ratio",
    "audio_features_nonverbal_basic_metrics_conversation_total_interruptions",
    "audio_features_nonverbal_basic_metrics_conversation_total_speaking_time",
    "audio_features_window_duration",

    "robot_speed_features_avg_speed_cm_s",
    "robot_speed_features_num_detections"
]


def get_by_path(obj, path):
    try:
        parts = path.replace("]", "").split(".")
        for p in parts:
            if "[" in p:
                if not p.endswith("["):  # Check for malformed bracket
                    key, idx_str = p.split("[", 1)
                    if idx_str:  # Check for empty index
                        obj = obj[key][int(idx_str)]
                    else:
                        return None  # Malformed path with empty brackets
                else:
                    return None  # Malformed path ending with [
            else:
                obj = obj[p]
        return obj
    except (KeyError, IndexError, TypeError, ValueError):
        return None  # return None instead of raising

def extract_datapoints_except_last(filename, feature_paths, feature_labels=None):
    # load the JSON file
    with open(filename, 'r') as f:
        data = json.load(f)
    audio_windows = {w["window_index"]: w for w in data.get("audio_features", [])}
    robot_windows = {w["window_index"]: w for w in data.get("robot_speed_features", [])}

    # Intersect window indices and remove last window
    common_indices = sorted(set(audio_windows.keys()) & set(robot_windows.keys()))
    if common_indices:
        common_indices = common_indices[:-1]  # remove last window

    datapoints = []

    for idx in common_indices:
        window_data = {
            "audio_features": audio_windows[idx],
            "robot_speed_features": robot_windows[idx]
        }

        values = []
        all_features_present = True

        for path in feature_paths:
            val = get_by_path(window_data, path)
            if isinstance(val, (int, float)):
                values.append(val)
            else:
                all_features_present = False
                break  # skip this window entirely

        if all_features_present:
            labels_to_use = feature_labels if feature_labels is not None else feature_paths
            datapoints.append(Datapoint(labels_to_use, values))

    return datapoints

def collect_numeric_paths(obj, prefix=""):
    paths = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            paths.extend(collect_numeric_paths(v, new_prefix))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            paths.extend(collect_numeric_paths(v, new_prefix))

    elif isinstance(obj, (int, float)):
        paths.append(prefix)

    return paths

def find_complete_feature_paths(data):
    audio_windows = {w["window_index"]: w for w in data.get("audio_features", [])}
    robot_windows = {w["window_index"]: w for w in data.get("robot_speed_features", [])}

    # intersect indices and ignore last window
    common_indices = sorted(set(audio_windows) & set(robot_windows))
    if common_indices:
        common_indices = common_indices[:-1]

    if not common_indices:
        return []

    # collect candidate paths from first valid window
    first_idx = common_indices[0]
    first_window = {
        "audio_features": audio_windows[first_idx],
        "robot_speed_features": robot_windows[first_idx],
    }

    candidate_paths = collect_numeric_paths(first_window)
    valid_paths = []

    for path in candidate_paths:
        valid = True
        for idx in common_indices:
            window_data = {
                "audio_features": audio_windows[idx],
                "robot_speed_features": robot_windows[idx],
            }
            val = get_by_path(window_data, path)
            if not isinstance(val, (int, float)):
                valid = False
                break
        if valid:
            valid_paths.append(path)

    return valid_paths

def extract_all_complete_features_except_last(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    feature_paths = find_complete_feature_paths(data)

    audio_windows = {w["window_index"]: w for w in data.get("audio_features", [])}
    robot_windows = {w["window_index"]: w for w in data.get("robot_speed_features", [])}

    # intersect indices and ignore last window
    common_indices = sorted(set(audio_windows) & set(robot_windows))
    if common_indices:
        common_indices = common_indices[:-1]

    datapoints = []

    for idx in common_indices:
        window_data = {
            "audio_features": audio_windows[idx],
            "robot_speed_features": robot_windows[idx],
        }

        values = [get_by_path(window_data, p) for p in feature_paths]
        datapoints.append(Datapoint(feature_paths, values))

    return datapoints

def path_to_feature_label(path: str) -> str:
    """
    Convert a JSON path like:
    audio_features.sentiment.scores[0]
    → sentiment_scores_0
    """
    label = path.replace(".", "_").replace("[", "_").replace("]", "")
    return label
