import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from json_extraction import features_labels as feature_labels

# --- Simulated cluster data ---
# Replace these with your real clustering results
cluster_centers = np.array([
    [0.2, 0.5, 0.8, 0.1],
    [0.6, 0.3, 0.4, 0.9],
    [0.1, 0.9, 0.2, 0.5]
])  # shape = (num_clusters, num_features)

# --- Streamlit UI ---
st.title("Cluster Centers Visualization")

# Dropdown to select a feature/dimension
selected_feature = st.selectbox("Select dimension to visualize:", feature_labels)

# Find index of the selected feature
dim_index = feature_labels.index(selected_feature)

# Extract values for that dimension from each cluster
dimension_values = cluster_centers[:, dim_index]

# Plotting
fig, ax = plt.subplots()
ax.bar(range(len(dimension_values)), dimension_values)
ax.set_xticks(range(len(dimension_values)))
ax.set_xticklabels([f"Cluster {i+1}" for i in range(len(dimension_values))])
ax.set_ylabel(selected_feature)
ax.set_title(f"Cluster Centers on {selected_feature}")

st.pyplot(fig)