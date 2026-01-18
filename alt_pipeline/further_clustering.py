import pickle
from sklearn.preprocessing import StandardScaler
import dim_red_clustering_functions


def perform_sub_cluster(pkl_path, X_scaled, feature_labels):
    #Loading the pkl file
    with open(pkl_path, 'rb') as f:
        loaded_data = pickle.load(f)

    main_labels = loaded_data[3]
    num_main_clusters = loaded_data[2]

    print(f"Model found {num_main_clusters} clusters")

    hierarchical_results = {}

    for id in range(num_main_clusters):
        print("-" * 20)
        print(f"Analysing cluster: {id}")

        sub_X_scaled = X_scaled[main_labels == id]
        if sub_X_scaled.shape[0] < 10:
            print("CLUSTER TOO SMALL FOR ANALYSIS")
            continue #exit and move to next cluster

        sub_X_standardised = StandardScaler().fit_transform(sub_X_scaled)

        (
            _,_,sub_k,sub_labels,_,sub_cntr,_
        ) = dim_red_clustering_functions.perform_fuzzy_cmeans_auto_k(
            sub_X_standardised,
            k_range=(2,11),
            m = 2.0,
            random_state=42
        )

        print(f"results split into {sub_k} divisions")

        # 5. Profile the Sub-Clusters (Indices 1-8 are Emotions/Sentiment)
        key_metrics = list(range(1, 9)) + [16, 17, 18] # Emotions, Speaking Time, Speed, Detections
        
        header = f"{'Behavioral Metric':<50}"
        for k in range(sub_k): header += f" | Sub-{k}"
        print(header + "\n" + "-" * len(header))

        for idx in key_metrics:
            row = f"{feature_labels[idx]:<50}"
            for k in range(sub_k):
                row += f" | {sub_cntr[k, idx]:>8.4f}"
            print(row)

        hierarchical_results[id] = {
            "sub_k": sub_k,
            "sub_labels": sub_labels,
            "sub_centroids": sub_cntr
        }

    return hierarchical_results


def test_sub_cluster():
    import full_dimensions_clustering
    X_scaled = full_dimensions_clustering.X_scaled
    feature_labels = full_dimensions_clustering.feature_labels

    perform_sub_cluster("./alt_pipeline/full_dimensions_cluster.pkl",X_scaled=X_scaled, feature_labels= feature_labels)

test_sub_cluster()
