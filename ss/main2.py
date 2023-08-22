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

import modules.block_inserter as block_inserter
import modules.block_info as block_info


print(f"Creating Octree")

#1 Load site data. Import the LiDAR scan data of the urban environment. Process the data as needed for it to be inserted into the Octree structure.
# File path to the LiDAR data
file_path = 'data/sites/park.parquet'

# Convert the LiDAR data to a Pandas DataFrame ready for insertion into the Octree
# Returns a dataframe with the columns being X,Y,Z,blockID,r,g,b,B,Bf,Composite,Dip (degrees),Dip direction (degrees),G,Gf,Illuminance (PCV),Nx,Ny,Nz,R,Rf,element_type,horizontality
lidar_dataframe = ConvertSites.process_lidar_data(file_path)
blockId_index = lidar_dataframe.columns.get_loc('blockID')
attribute_cols = lidar_dataframe.columns[blockId_index+1:]

lidar_points = lidar_dataframe[['X', 'Y', 'Z']].to_numpy()
lidar_block_ids = lidar_dataframe['blockID'].tolist()
lidar_attributes = lidar_dataframe[attribute_cols].to_dict('records')

# Parameters for the octree
max_depth = 7  # Maximum depth of the Octree

#1.1 Create octree. Initialize it with a root node covering the spatial bounds of the entire dataset.
# The octree expects a DataFrame with x, y, z, r, g, b, attributes [multiple of these], and blockIDs.
# Each point is represented by a tuple (x, y, z) and has a set of attributes (e.g., RGB, intensity, etc.) represented as a dictionary
# Each point also has a block_id that represents which block this point belongs to.
octree = Octree.CustomOctree(lidar_points, lidar_attributes, lidar_block_ids, max_depth)

print(f"Created Octree with max depth {max_depth} and extents {octree.root.min_corner} to {octree.root.max_corner}")

#2 Generate tree blocks to insert into octree
# Each block is a collection of points and their associated attributes. 
# Each point also has a block_id that represents which block this point belongs to.
# Blocks might be spread over multiple nodes in the octree.

#2.1 use lidar data to find ground points in octree
ground_points = lidar_dataframe.loc[lidar_dataframe['element_type'].isin([2, 4]), ['X', 'Y', 'Z']]

#2.2 use urban forest data to find location and size of trees on site and then find the nearest ground point in octree to each tree
treeCoords, treeAttributes = UrbanForestParser.load_urban_forest_data(octree.root.min_corner, octree.root.max_corner, ground_points)

print(f'treeattributes: {treeAttributes}')
#2.3 use my own geometric analysis to create tree blocks relevant to the size, locationa and type of each tree


tree_points, tree_attributes, tree_block_ids, max_tree_count = block_info.generate_tree_blocks(treeAttributes)
#tree_points, tree_attributes, tree_block_ids, max_tree_count = Octree.tree_block_processing_complex(treeAttributes)

print(f'Add blocks to octree with a maximum of {max_tree_count} blockID trees')

#3. Add tree blocks to the octree.  During insertion, assign each point to appropriate octree nodes and assign their attributes and resources.
#3.1 use tree ground points to find the nearest octree node and insert the tree points and attributes into the octree
block_inserter.add_block(octree, tree_points, tree_attributes, tree_block_ids)
print("Octree updated with additional data")

print("starting to change attributes")


#3.2 Update tree block nodes in octree with additional attributes/resources from from the lerouxdata.csv
for blockNo in range(11, max_tree_count + 1):
    print(f'changing attributes for block {blockNo}... ')
    block_inserter.distribute_changes(octree, 'isNeither', 'isBoth', .5, blockNo, 1)  # assuming the block_id is 1
    print(f'changed attributes for block {blockNo} ')

print("finished changing attributes")

#4. Visualise octree
print("visualizing octree")
octree.visualize_octree_nodes()

