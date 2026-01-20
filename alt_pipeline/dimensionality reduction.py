# ==========================================================
# PCA / PLS dimensionality vs fuzzy clustering experiment
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    X_scaled = StandardScaler().fit_transform(X)
    X_pls = X[:, :-1]
    Y_pls = X[:, -1].reshape(-1, 1)
    return X_scaled, X_pls, Y_pls


# ----------------------------------------------------------
# Clustering evaluation
# ----------------------------------------------------------
def evaluate_embedding(X_reduced, method_name, n_dims):
    scores_k, best_score, best_k, labels, u, cntr, fpc = \
        dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(
            X_reduced,
            score_method="soft_silhouette"
        )

    fcm_wrapper = type(
        "FCMWrapper",
        (),
        {"u": u, "centers": cntr}
    )()

    intra_sim = dim_red_clustering_functions.compute_cluster_score(
        score_method="intra_similarity",
        X=X_reduced,
        fcm_model=fcm_wrapper
    )

    return {
        "n_dimensions": n_dims,
        "best_k": best_k,
        "soft_silhouette": best_score,
        "intra_similarity": intra_sim
    }


# ----------------------------------------------------------
# Experiments
# ----------------------------------------------------------
def run_experiment(X_scaled, X_pls, Y_pls, dim_range=range(2, 11)):
    pca_results = []
    pls_results = []

    for n_dims in dim_range:

        # ----- PCA -----
        pca = dim_red_clustering_functions.create_dim_red_method(
            "PCA",
            n_dimensions=n_dims
        )
        X_pca = pca.fit(X_scaled)
        pca_results.append(evaluate_embedding(X_pca, "PCA", n_dims))

        # ----- PLS -----
        pls = dim_red_clustering_functions.PLS(n_dimensions=n_dims)
        X_pls_red = pls.fit(X_pls, Y_pls)
        pls_results.append(evaluate_embedding(X_pls_red, "PLS", n_dims))

    return (
        pd.DataFrame(pca_results),
        pd.DataFrame(pls_results)
    )


# ----------------------------------------------------------
# Plotting & Saving
# ----------------------------------------------------------
def save_plots_and_tables(df, method_name, output_root):
    method_dir = os.path.join(output_root, method_name.lower())
    os.makedirs(method_dir, exist_ok=True)

    # ---- Save CSV ----
    df.to_csv(os.path.join(method_dir, "statistics.csv"), index=False)

    # ---- Plot: Dimension vs Scores ----
    plt.figure(figsize=(7, 5))
    plt.plot(df["n_dimensions"], df["soft_silhouette"], marker="o", label="Soft silhouette")
    plt.plot(df["n_dimensions"], df["intra_similarity"], marker="o", label="Intra similarity")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Score")
    plt.title(f"{method_name}: Dimensions vs Clustering Scores")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "dimension_vs_scores.png"))
    plt.close()

    # ---- Plot: Dimension vs Optimal K ----
    plt.figure(figsize=(7, 5))
    plt.plot(df["n_dimensions"], df["best_k"], marker="o")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Optimal number of clusters")
    plt.title(f"{method_name}: Dimensions vs Optimal Clusters")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "dimension_vs_clusters.png"))
    plt.close()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    X = load_data(files, selected_features)
    X_scaled, X_pls, Y_pls = prepare_inputs(X)

    df_pca, df_pls = run_experiment(
        X_scaled=X_scaled,
        X_pls=X_pls,
        Y_pls=Y_pls
    )

    output_root = "output"
    save_plots_and_tables(df_pca, "PCA", output_root)
    save_plots_and_tables(df_pls, "PLS", output_root)

    print("Saved results to:")
    print(" - output/pca/")
    print(" - output/pls/")


# ----------------------------------------------------------
# Run
# ----------------------------------------------------------
main()
