
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

def predict_cluster(clustered_data, json_path):
    selected_features = json_extraction.selected_features
    feature_labels = json_extraction.features_labels

    data = json_extraction.extract_datapoints_except_last(json_path, selected_features, feature_labels)
    
    print("Feature data extracted.")
    print(f"Number of datapoints extracted: {len(data)}")

    print(type(data))
    data_array = np.array(data)
    if data_array.ndim == 1:
        data_array = data_array.reshape(1, -1)
    
    print(f"Data array shape for prediction: {data_array.shape}")
    # Get centroids
    cntr = clustered_data[5]

    num_clusters, model_features = cntr.shape

    X = np.array([dp.dimension_values for dp in data])
    X_scaled = full_dimensions_clustering.StandardScaler().fit_transform(X)

    print(f"Data shape for prediction: {X.shape}")

    print(f"Pkl rows and features: {num_clusters} clusters, {model_features} model features.")

    # Predict
    u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        X_scaled.T, cntr, m=1.056, error=0.005, maxiter=1000
    )

    print(f"Prediction membership shape: {u.shape}")
    print(f"Membership values:\n{u}")
    print("Prediction complete.")
    # return np.argmax(u, axis=0)[0] 
    return np.argmax(u, axis=0)

def test_predict():
    print("Predicting clusters for test data...")
    saved_cluster = save_cluster.load_cluster(DATA_DIR / "3_cluster.pkl")
    assignments = predict_cluster(saved_cluster, DATA_DIR / "data" / "140252_features.json")

    print("Predicted cluster assignments:")
    print(assignments)
    return assignments

test_predict()

