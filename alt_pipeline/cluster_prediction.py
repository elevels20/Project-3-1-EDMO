
from importlib.resources import files
import pickle
import numpy as np
import full_dimensions_clustering
import save_cluster
import json_extraction

import numpy as np
import skfuzzy as fuzz
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

def predict_cluster(clustered_data, json_path, type='robot'):
    
    if type == 'robot':
        selected_features = json_extraction.robot_selected_features
        feature_labels = json_extraction.robot_features_labels
        #Extract the new datapoints
        data = json_extraction.extract_datapoints_except_last(json_path, selected_features, feature_labels)
    else:
        selected_features = json_extraction.voice_selected_features
        feature_labels = json_extraction.voice_features_labels
        #Extract the new datapoints
        data = json_extraction.voice_extract_datapoints_except_last(json_path, selected_features, feature_labels)
   
    X_new = np.array([dp.dimension_values for dp in data])
    
    # 2. Scale the NEW data
    from sklearn.preprocessing import StandardScaler
    X_new_scaled = StandardScaler().fit_transform(X_new)

    # 3. Get centroids from pkl
    cntr = clustered_data[5]

    # 4. Predict using ONLY the new 38 points (Transposed)
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X_new_scaled.T, cntr, m=1.056, error=0.005, maxiter=1000
    )

    print(f"Correct Prediction shape: {u.shape}") # Should be (7, 38)
    
    # Return the full array of assignments
    return np.argmax(u, axis=0)


def predict_cluster_with_probabilities(clustered_data, json_path):
    """
    Same as predict_cluster but returns both assignments and full probability matrix.

    Returns:
        assignments: array of cluster assignments (argmax)
        probabilities: full membership matrix (n_clusters x n_samples)
    """
    selected_features = json_extraction.selected_features
    feature_labels = json_extraction.features_labels

    data = json_extraction.extract_datapoints_except_last(json_path, selected_features, feature_labels)
    X_new = np.array([dp.dimension_values for dp in data])

    from sklearn.preprocessing import StandardScaler
    X_new_scaled = StandardScaler().fit_transform(X_new)

    cntr = clustered_data[5]

    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X_new_scaled.T, cntr, m=1.056, error=0.005, maxiter=1000
    )

    assignments = np.argmax(u, axis=0)
    return assignments, u  # u shape: (n_clusters, n_samples)


def test_predict():
    print("Predicting clusters for test data...")
    saved_cluster = save_cluster.load_cluster(DATA_DIR / "full_dimensions_cluster_voice.pkl")
    
    assignments = predict_cluster(saved_cluster, DATA_DIR / "data" / "test_data" / "111455_features.json")

    print("Predicted cluster assignments:")
    print(assignments)
    return assignments

#test_predict()

