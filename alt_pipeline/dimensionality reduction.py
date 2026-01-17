import numpy as np
from sklearn.preprocessing import StandardScaler

import plotting_pca_clustering
import dim_red_clustering_functions
import json_extraction
from alt_pipeline.json_extraction import selected_features
from alt_pipeline.json_extraction import training_files as files
from alt_pipeline.json_extraction import features_labels as feature_labels

all_datapoints = []
file_labels = []

datapoints = (json_extraction.extract_datapoints_except_last_multiple(files, selected_features))
X = np.array([dp.dimension_values for dp in datapoints])
X_scaled = StandardScaler().fit_transform(X)
print([l for l in datapoints[0].dimension_labels if "window" in l])

for i, f in enumerate(files):
    dps = json_extraction.extract_datapoints_except_last(f, selected_features, feature_labels)
    all_datapoints.extend(dps)
    file_labels.extend([f"experiment_{i+1}"] * len(dps))  # same label for all windows of this file
print(f"Total datapoints: {len(all_datapoints)}")
X = np.array([dp.dimension_values for dp in all_datapoints])

# --- Create PCA object with 2 components ---
dim_red = dim_red_clustering_functions.create_dim_red_method("PCA", n_dimensions=2)
# --- Fit PCA ---
X_reduced = dim_red.fit(X_scaled)
# --- Apply fuzzy C-means ---
n_clusters = 3
silhouette_scores_pca ,best_score_pca, best_k_pca, cluster_labels, u, cntr, fpc = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(X_reduced)

print("score: " + str(best_score_pca))
print("best k: " + str(best_k_pca))
print("all scores pca: " + str(silhouette_scores_pca))

# here we do pls
X = np.array([dp.dimension_values[:-1] for dp in datapoints])
# Y: last feature
Y = np.array([dp.dimension_values[-1] for dp in datapoints]).reshape(-1, 1)  # column vector
# --- PLS ---
dim_red_pls = dim_red_clustering_functions.PLS(n_dimensions=2)
X_pls = dim_red_pls.fit(X, Y)  # X_pls is like PCA-reduced scores
# Fuzzy C-mean clustering on PLS
#cluster_labels_pls, u, cntr, fpc = dim_red_clustering_functions.perform_fuzzy_cmeans(X_pls, n_clusters=3)
silhouette_scores_pls ,best_score_pls, best_k_pls, cluster_labels_pls, u, cntr, fpc = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(X_pls)

#generate plots
plotting_pca_clustering.plot_clusters(X_pls, cluster_labels_pls, output_dir="output/pls_plots")
plotting_pca_clustering.plot_clusters(X_reduced, cluster_labels)
plotting_pca_clustering.plot_pca_results(X_reduced, dim_red, feature_labels, files)
features_labels_pls = feature_labels[:-1]
plotting_pca_clustering.plot_pls_results(
    X_scores=X_pls,
    Y= Y,
    dim_red=dim_red_pls,
    features_labels=features_labels_pls,
    file_labels=files,   # pass as keyword
    output_dir="output/pls_plots"
)
