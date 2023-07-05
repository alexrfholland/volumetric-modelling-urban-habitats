import pandas as pd
import numpy as np
import open3d as o3d
import random
from OctreeClass import CustomOctree

# Load data from CSV
csv_file = 'data/branchPredictions - full.csv'
data = pd.read_csv(csv_file)

# Landscape size
landscape_size = [100, 100, 100]  # Size of landscape in meters

# Get the specific tree ID
tree_id = 13

# Get tree data
tree_data = data[data['Tree.ID'] == tree_id]
offset_x, offset_y = random.uniform(0, landscape_size[0]), random.uniform(0, landscape_size[1])
tree_points = tree_data[['x', 'y', 'z']].to_numpy() + np.array([offset_x, offset_y, 0])

# Convert lists to numpy arrays for use in octree
points = np.array(tree_points)

# Create a grid of points for the ground
ground_points = np.mgrid[0:100:1, 0:100:1].reshape(2,-1).T
ground_points = np.concatenate([ground_points, np.zeros((ground_points.shape[0], 1))], axis=1)  # Set z=0 for all ground points

# Append ground points to tree points numpy array
points = np.concatenate([points, ground_points])

# Build custom octree
min_corner = [0, 0, 0]  # Minimum corner of landscape
max_corner = landscape_size  # Maximum corner of landscape
max_depth = 7
octree = CustomOctree(points, min_corner, max_corner, max_depth=max_depth)

# Create visualizer
visualize_offset = [5]
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Register key callbacks
vis.register_key_callback(ord("["), decrease_visualize_offset)
vis.register_key_callback(ord("]"), increase_visualize_offset)

# Initial update
mesh = update_visualization(vis, octree, max_depth, visualize_offset[0])
mesh.compute_vertex_normals()
