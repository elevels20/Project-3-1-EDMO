import sys
import os
from pathlib import Path

# 1. Define the absolute path to the alt_pipeline directory
# Path(__file__).resolve() gets the absolute path of load_and_examine.py
# .parent.parent moves up to Project-3-1-EDMO, then we go into 'alt_pipeline'
BASE_DIR = Path(__file__).resolve().parent.parent
ALT_PIPELINE_PATH = str(BASE_DIR / "alt_pipeline")

# 2. Add both the root and the alt_pipeline folder to the path
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
if ALT_PIPELINE_PATH not in sys.path:
    sys.path.append(ALT_PIPELINE_PATH)

# 3. NOW perform imports
from alt_pipeline import save_cluster
from alt_pipeline import full_dimensions_clustering


feature_labels = full_dimensions_clustering.feature_labels
print("Feature labels used in clustering:")
for lbl in full_dimensions_clustering.feature_labels:
    print(" -", lbl)

loaded_cluster_full = save_cluster.load_cluster("./alt_pipeline/full_dimensions_cluster.pkl")
print("Loaded full dimensions cluster data:")

#print(loaded_cluster_full)
print("Cluster data type:", type(loaded_cluster_full))
print("Cluster data length:", len(loaded_cluster_full))
#print("cluster labels:", loaded_cluster_full[3])

print('-' * 20)
num_clusters = loaded_cluster_full[2]
print(f"Number of clusters: {num_clusters}")

cntr = loaded_cluster_full[5]
num_features = cntr.shape[1]

# Build dynamic header
header = f"{'Metric':<40} |"
for c in range(num_clusters):
    header += f" {'C' + str(c):^10} |"
print(header)
print("-" * len(header))

# Print each feature row dynamically
for i in range(num_features):
    row = f"{feature_labels[i]:<40} |"
    for c in range(num_clusters):
        row += f" {cntr[c, i]:>10.4f} |"
    print(row)

