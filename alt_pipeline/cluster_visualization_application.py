import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from json_extraction import features_labels as feature_labels
import save_cluster

# --- Load cluster data ---
cluster = save_cluster.load_cluster("./alt_pipeline/full_dimensions_cluster_voice.pkl")
cluster_centers = cluster[5]

# --- Streamlit UI ---
st.title("Cluster Centers Visualization")

# Calculate global min and max across all dimensions for consistent scaling
global_min = cluster_centers.min()
global_max = cluster_centers.max()

# Add some padding to the y-axis limits
y_padding = (global_max - global_min) * 0.1
y_min = global_min - y_padding
y_max = global_max + y_padding

# Create a plot for each feature
num_features = len(feature_labels)
num_clusters = cluster_centers.shape[0]

# Organize plots in a grid (2 columns)
cols_per_row = 2
num_rows = (num_features + cols_per_row - 1) // cols_per_row

for idx, feature_name in enumerate(feature_labels):
    # Create new row every 2 plots
    if idx % cols_per_row == 0:
        cols = st.columns(cols_per_row)
    
    with cols[idx % cols_per_row]:
        # Extract values for this dimension from each cluster
        dimension_values = cluster_centers[:, idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(num_clusters), dimension_values)
        ax.set_xticks(range(num_clusters))
        ax.set_xticklabels([f"C{i+1}" for i in range(num_clusters)])
        ax.set_ylabel("Value")
        ax.set_xlabel("Cluster")
        ax.set_title(f"{feature_name}", fontsize=10)
        
        # Set consistent y-axis limits
        ax.set_ylim(y_min, y_max)
        
        # Add grid for easier comparison
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        st.pyplot(fig)
        plt.close(fig)

# Optional: Show summary statistics
st.subheader("Summary Statistics")
st.write(f"Global min value: {global_min:.3f}")
st.write(f"Global max value: {global_max:.3f}")
st.write(f"Number of clusters: {num_clusters}")
st.write(f"Number of dimensions: {num_features}")