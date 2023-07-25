#

"""
The code provided is intended for 3D volumetric modeling of urban environments using an Octree data structure. The goal is to build a detailed representation of urban spaces by grouping Octree nodes into logical sets called Blocks, such as Tree Blocks, Building Blocks, and Artificial Habitat Blocks. Each Block represents a specific object or area within the environment.


Step 1: Initialize Main Script and Load Libraries


Step 2: Load Site Data
    - Aim: Utilize `siteToBlocks.py` to import the LiDAR scan data of the urban environment. Process the data as needed for it to be inserted into the Octree structure.
    The octree expects a DataFrame with x, y, z, r, g, b, attributes [multiple of these], and blockIDs

Step 3: Initialize Custom Octree
    - Aim: Create an instance of the CustomOctree class. Initialize it with a root node covering the spatial bounds of the entire dataset. Also, initialize an empty dictionary for storing block information.

Step 4: Insert Site Data into Octree as Blocks
    - Aim: Use the site data loaded from `siteToBlocks.py` to insert points into the Octree. During insertion, assign each point to appropriate blocks such as Building Block, Ground Block, etc. based on their attributes. Store block-level information in the block dictionary initialized earlier.

Step 5: Further Processing of Blocks (Tree Blocks)
    - Aim: Add or refine blocks, with a focus on Tree Blocks. This can include further subdivision, attribute assignment, or association with Octree nodes.

Step 6: Voxelization of Octree
    - Aim: Traverse the Octree to its leaf nodes and create a point cloud from the center points of these leaf nodes. Convert this point cloud into a voxel grid for visualization purposes.

Step 7: Generate Bounding Boxes for Visualization
    - Aim: Traverse the Octree to generate bounding boxes for Octree nodes. Assign colors to each bounding box based on the node's dominant attribute and prepare these for visualization.

Step 8: Visualize Octree and Blocks
    - Aim: Utilize the OctreeVisualizer class to create a 3D visualization of the Octree. Add the voxel grid and bounding boxes to the visualization. Enable interaction with keyboard callbacks for exploration of the 3D model.
"""
# Rest of the code goes here

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
main2.py
This script is for creating a 3D volumetric model of urban environments using an Octree data structure.
"""

# === Standard Library Imports ===
import os
import sys

# === Third Party Imports ===
import open3d as o3d
import pandas as pd
import numpy as np

import concurrent.futures

import modules.convertSiteToBlocks as ConvertSites
import modules.octree as Octree
import modules.urbanforestparser as UrbanForestParser

import numba as nb
from collections import defaultdict

# File path to the LiDAR data
file_path = 'data/sites/park.parquet'

# Convert the LiDAR data to a Pandas DataFrame ready for insertion into the Octree
# Returns a dataframe with the columns being X,Y,Z,blockID,r,g,b,B,Bf,Composite,Dip (degrees),Dip direction (degrees),G,Gf,Illuminance (PCV),Nx,Ny,Nz,R,Rf,element_type,horizontality
lidar_dataframe = ConvertSites.process_lidar_data(file_path)

# Parameters for the octree
max_depth = 7  # Maximum depth of the Octree

print('from main..')
print(lidar_dataframe)
# Get the index of the 'blockId' column
blockId_index = lidar_dataframe.columns.get_loc('blockID')

# Extract all columns after 'blockId'
attribute_cols = lidar_dataframe.columns[blockId_index+1:]

# Now you can extract your points, attributes, and block IDs
lidar_points = lidar_dataframe[['X', 'Y', 'Z']].to_numpy()
lidar_block_ids = lidar_dataframe['blockID'].tolist()
lidar_attributes = lidar_dataframe[attribute_cols].to_dict('records')




print(f"Creating Octree")

# Create Octree
print(lidar_points)
#octree = Octree.CustomOctree(lidar_points, lidar_attributes, lidar_block_ids, max_depth)
#octree = Octree.CustomOctree(combined_points, combined_attributes, combined_block_ids, max_depth)
octree = Octree.CustomOctree(lidar_points, lidar_attributes, lidar_block_ids, max_depth)

print(f"Created Octree with max depth {max_depth} and extents {octree.root.min_corner} to {octree.root.max_corner}")

# Process tree block data for a specified number of trees  and add to the octree

#tree_count = 16  # Specify the number of tree blocks to process
selected_data = lidar_dataframe.loc[lidar_dataframe['element_type'].isin([2, 4]), ['X', 'Y', 'Z']]
#treeCoords = ConvertSites.select_random_ground_points(selected_data, tree_count)
print(selected_data)

treeCoords, treeAttributes = UrbanForestParser.load_urban_forest_data(octree.root.min_corner, octree.root.max_corner, selected_data)
print(f'tree Coords: {treeCoords}, treeAttributes: {treeAttributes}')
print(treeAttributes['Diameter Breast Height'])


tree_points, tree_attributes, tree_block_ids = Octree.tree_block_processing(treeCoords)

# Add new block to the octree
octree.add_block(tree_points, tree_attributes, tree_block_ids)
print("Octree updated with additional data")

octree.visualize_octree_nodes()

