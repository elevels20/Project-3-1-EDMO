# ==========================================================
# PCA / PLS dimensionality vs clustering stability experiment
# Reference = clustering without dimensionality reduction
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

import dim_red_clustering_functions
import json_extraction
from alt_pipeline.json_extraction import selected_features
from alt_pipeline.json_extraction import training_files as files


# ----------------------------------------------------------
# Data preparation
# ----------------------------------------------------------
def load_data(files, selected_features):
    datapoints = json_extraction.extract_datapoints_except_last_multiple(
        files, selected_features
    )
    X = np.array([dp.dimension_values for dp in datapoints])
    return X


def prepare_inputs(X):
    X_features = X[:, :-1]              # all features except target
    X_scaled = StandardScaler().fit_transform(X_features)

    X_pls = X_features
    Y_pls = X[:, -1].reshape(-1, 1)      # predicted feature (robot speed)

    return X_scaled, X_pls, Y_pls


# ----------------------------------------------------------
# Neighborhood preservation (Average Jaccard Index)
# ----------------------------------------------------------
def average_jaccard_index(X_orig, X_red, k):
    nn_orig = NearestNeighbors(n_neighbors=k + 1).fit(X_orig)
    nn_red = NearestNeighbors(n_neighbors=k + 1).fit(X_red)

    neigh_orig = nn_orig.kneighbors(return_distance=False)[:, 1:]
    neigh_red = nn_red.kneighbors(return_distance=False)[:, 1:]

    jaccards = []
    for i in range(X_orig.shape[0]):
        A = set(neigh_orig[i])
        B = set(neigh_red[i])
        jaccards.append(len(A & B) / len(A | B))

    return np.mean(jaccards)


# ----------------------------------------------------------
# Reference clustering (no dimensionality reduction)
# ----------------------------------------------------------
def compute_reference_clustering(X_scaled):
    (
        labels,
        u,
        cntr,
        _
    ) = dim_red_clustering_functions.perform_fuzzy_cmeans(
        X_scaled, 5
    )

    return np.asarray(labels), 5


# ----------------------------------------------------------
# Evaluation of one embedding
# ----------------------------------------------------------
def evaluate_embedding(X_reduced, X_orig, ref_labels, n_dims):
    (
        _,
        best_score,
        best_k,
        labels,
        u,
        cntr,
        _
    ) = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(
        X_reduced,
        score_method="soft_silhouette"
    )

    labels = np.asarray(labels)

    ari = adjusted_rand_score(ref_labels, labels)
    nmi = normalized_mutual_info_score(ref_labels, labels)

    return {
        "n_dimensions": n_dims,
        "best_k": best_k,
        "ARI_vs_full": ari,
        "NMI_vs_full": nmi,
        "AJI_k10": average_jaccard_index(X_orig, X_reduced, 10),
        "AJI_k20": average_jaccard_index(X_orig, X_reduced, 20),
        "AJI_k30": average_jaccard_index(X_orig, X_reduced, 30),
        "soft_silhouette": best_score,
    }


# ----------------------------------------------------------
# Run experiments
# ----------------------------------------------------------
def run_experiment(X_scaled, X_pls, Y_pls, ref_labels, dim_range=range(2, 11)):
    pca_results = []
    pls_results = []

    for n_dims in dim_range:

        # ----- PCA -----
        pca = dim_red_clustering_functions.create_dim_red_method(
            "PCA", n_dimensions=n_dims
        )
        X_pca = pca.fit(X_scaled)

        pca_results.append(
            evaluate_embedding(X_pca, X_scaled, ref_labels, n_dims)
        )

        # ----- PLS -----
        pls = dim_red_clustering_functions.PLS(n_dimensions=n_dims)
        X_pls_red = pls.fit(X_pls, Y_pls)

        pls_results.append(
            evaluate_embedding(X_pls_red, X_scaled, ref_labels, n_dims)
        )

    return pd.DataFrame(pca_results), pd.DataFrame(pls_results)


# ----------------------------------------------------------
# Plotting & saving
# ----------------------------------------------------------
def save_results(df, method_name, output_root):
    method_dir = os.path.join(output_root, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)

    df.to_csv(os.path.join(method_dir, "statistics.csv"), index=False)

    # --- ARI / NMI ---
    plt.figure(figsize=(7, 5))
    plt.plot(df["n_dimensions"], df["ARI_vs_full"], marker="o", label="ARI")
    plt.plot(df["n_dimensions"], df["NMI_vs_full"], marker="o", label="NMI")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Score")
    plt.title(f"{method_name}: clustering stability vs full space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "ari_nmi_vs_dimensions.png"))
    plt.close()

    # --- Jaccard ---
    plt.figure(figsize=(7, 5))
    plt.plot(df["n_dimensions"], df["AJI_k10"], marker="o", label="k=10")
    plt.plot(df["n_dimensions"], df["AJI_k20"], marker="o", label="k=20")
    plt.plot(df["n_dimensions"], df["AJI_k30"], marker="o", label="k=30")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Average Jaccard Index")
    plt.title(f"{method_name}: neighborhood preservation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "jaccard_vs_dimensions.png"))
    plt.close()

    # --- Optimal K ---
    plt.figure(figsize=(7, 5))
    plt.plot(df["n_dimensions"], df["best_k"], marker="o")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Optimal number of clusters")
    plt.title(f"{method_name}: dimensions vs optimal K")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "optimal_k_vs_dimensions.png"))
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    X = load_data(files, selected_features)
    X_scaled, X_pls, Y_pls = prepare_inputs(X)

    # Reference clustering
    ref_labels, ref_k = compute_reference_clustering(X_scaled)
    print(f"Reference clustering: K = {ref_k}")

    # Experiments
    df_pca, df_pls = run_experiment(
        X_scaled=X_scaled,
        X_pls=X_pls,
        Y_pls=Y_pls,
        ref_labels=ref_labels
    )

    output_root = "output"
    save_results(df_pca, "PCA", output_root)
    save_results(df_pls, "PLS", output_root)

    print("Saved results to:")
    print(" - output/pca/")
    print(" - output/pls/")


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
main()