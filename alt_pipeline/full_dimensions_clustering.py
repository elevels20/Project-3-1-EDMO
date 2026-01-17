import json
import numpy as np
import json_extraction  # your module
from sklearn.preprocessing import StandardScaler
import dim_red_clustering_functions
from pathlib import Path
from alt_pipeline.json_extraction import selected_features
from alt_pipeline.json_extraction import training_files as files

# load datapoints
all_datapoints = []
file_labels = []
feature_labels, datapoints = (json_extraction.extract_datapoints_except_last(files, selected_features))

#preprocess
X = np.array([dp.dimension_values for dp in all_datapoints])
print("Feature labels:")
for lbl in feature_labels:
    print(" -", lbl)
print(f"X shape: {X.shape}")
# X: full-dimensional feature matrix
X_scaled = StandardScaler().fit_transform(X)

# performing clustering
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
    random_state=42
)

# print results
print(best_k)
print(best_score)
print(silhouette_scores)

