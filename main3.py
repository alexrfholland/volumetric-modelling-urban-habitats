"""
main3.py
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
from modules import blocks  # new import here

print(f"Creating Octree")


#Create Octree
#1.1 Load site data. Import the LiDAR scan data of the urban environment. Process the data as needed for it to be inserted into the Octree structure.
file_path = 'data/sites/park.parquet'

# Convert the LiDAR data to a Pandas DataFrame ready for insertion into the Octree
# Returns a dataframe with the columns being X,Y,Z,blockID,r,g,b,B,Bf,Composite,Dip (degrees),Dip direction (degrees),G,Gf,Illuminance (PCV),Nx,Ny,Nz,R,Rf,element_type,horizontality
lidar_dataframe = ConvertSites.process_lidar_data(file_path)
blockId_index = lidar_dataframe.columns.get_loc('blockID')
attribute_cols = lidar_dataframe.columns[blockId_index+1:]

lidar_points = lidar_dataframe[['X', 'Y', 'Z']].to_numpy()
lidar_block_ids = lidar_dataframe['blockID'].tolist()
lidar_attributes = lidar_dataframe[attribute_cols].to_dict('records')

#1.2 Create octree. Initialize it with a root node covering the spatial bounds of the entire dataset.
# Parameters for the octree
max_depth = 7  # Maximum depth of the Octree
# The octree expects a DataFrame with x, y, z, r, g, b, attributes [multiple of these], and blockIDs.
# Each point is represented by a tuple (x, y, z) and has a set of attributes (e.g., RGB, intensity, etc.) represented as a dictionary
# Each point also has a block_id that represents which block this point belongs to.
octree = Octree.CustomOctree(lidar_points, lidar_attributes, lidar_block_ids, max_depth)

print(f"Created Octree with max depth {max_depth} and extents {octree.root.min_corner} to {octree.root.max_corner}")

#2 Update Octree with Blocks 
# Each block is a collection of points and their associated attributes. 
# Each point also has a block_id that represents which block this point belongs to.
# Blocks might be spread over multiple nodes in the octree.

print('adding blocks to octree')
ground_points = lidar_dataframe.loc[lidar_dataframe['element_type'].isin([2, 4]), ['X', 'Y', 'Z']]
treeAttributes = {}  # This should be the actual tree attributes
octree = blocks.generate_tree_blocks_and_insert_to_octree(octree, ground_points, treeAttributes)

print("blocks added to octree")

print("starting to change attributes")
#3.2 Update tree block nodes in octree with additional attributes/resources from from the lerouxdata.csv
octree = blocks.update_tree_attributes(octree)
print("finished changing attributes")

#4. Visualise octree
print("visualizing octree")
octree.visualize_octree_nodes()
