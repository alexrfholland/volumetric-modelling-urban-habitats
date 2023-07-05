from src.building_blocks import Tree
import pandas as pd
import numpy as np
import open3d as o3d


# Load data from CSV file
file_path = 'data/branchPredictions - full.csv'

data = pd.read_csv(file_path)

# Setting 'Tree.ID' as an index for faster querying
data.set_index('Tree.ID', inplace=True)

# Select the rows where 'Tree.ID' is 13
tree_data = data.loc[13]

# Extract x, y, z coordinates for Tree 13
points = tree_data[['x', 'y', 'z']].values

# Create a Tree instance for 'old - low control'
old_low_control_tree = Tree(tree_id='13', points=points, age='old', management_regime='low_control')

# Now old_low_control_tree is a Tree instance for Tree 13 with age 'old' and management_regime 'low_control'
# You can access its octree using old_low_control_tree.octree

# Optional: Visualize the octree for this tree
if old_low_control_tree.octree:
    o3d.visualization.draw_geometries([old_low_control_tree.octree])
else:
    print("Octree not built for Tree 13")
