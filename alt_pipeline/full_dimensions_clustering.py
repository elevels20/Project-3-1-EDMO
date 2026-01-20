import numpy as np
import json_extraction
from sklearn.preprocessing import StandardScaler
import dim_red_clustering_functions
from json_extraction import selected_features
from json_extraction import training_files as files
from json_extraction import features_labels as feature_labels

# load datapoints (MULTIPLE FILES)
datapoints = json_extraction.extract_datapoints_except_last_multiple(
    files,
    selected_features,
    feature_labels
)

# build feature matrix
X = np.array([dp.dimension_values for dp in datapoints])

# sanity checks
print("Feature labels:")
for lbl in feature_labels:
    print(" -", lbl)

print(f"Number of datapoints: {len(datapoints)}")
print(f"X shape: {X.shape}")

if X.size == 0:
    raise ValueError("No datapoints extracted — clustering cannot proceed.")

# scale features
X_scaled = StandardScaler().fit_transform(X)

# clustering
(
    silhouette_scores,
    best_score,
    best_k,
    cluster_labels,
    u,
    cntr,
    fpc
) = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(
    X_scaled,
    k_range=range(2, 11),
    m=2.0,
    random_state=42,
    score_method="intra_similarity"
)

# results
print("Best k:", best_k)
print("Best silhouette score:", best_score)
print("All silhouette scores:", silhouette_scores)

cluster_save = (silhouette_scores,best_score,best_k,cluster_labels,u,cntr,fpc)

import save_cluster
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
save_cluster.save_cluster(cluster_save, BASE_DIR / "full_dimensions_cluster.pkl")

