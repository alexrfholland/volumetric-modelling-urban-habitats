"""NOTE I AM GOING TO COMBINE ALL THE SCRIPTS INTO ONE LONGE ONE"""



###Main.py
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


import modules.convertSiteToBlocks as ConvertSites
import modules.octree as Octree

# File path to the LiDAR data
file_path = 'data/sites/park.parquet'

# Convert the LiDAR data to a Pandas DataFrame ready for insertion into the Octree
# Returns a dataframe with the columns being X,Y,Z,blockID,r,g,b,B,Bf,Composite,Dip (degrees),Dip direction (degrees),G,Gf,Illuminance (PCV),Nx,Ny,Nz,R,Rf,element_type,horizontality
lidar_dataframe = ConvertSites.process_lidar_data(file_path)

# Parameters for the octree
max_depth = 8  # Maximum depth of the Octree

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


# Process tree block data for a specified number of trees
tree_count = 16  # Specify the number of tree blocks to process
#selected_data = lidar_dataframe.loc[lidar_dataframe['element_type'] == 3, ['X', 'Y', 'Z']]
selected_data = lidar_dataframe.loc[lidar_dataframe['element_type'].isin([2, 4]), ['X', 'Y', 'Z']]
print('test')
print(selected_data)
#selected_data = [[0,0,0],[0,1,1]]
treeCoords = ConvertSites.select_random_ground_points(selected_data, tree_count)
tree_points, tree_attributes, tree_block_ids = Octree.tree_block_processing(treeCoords)

# Combine LiDAR and tree block data
combined_points = np.concatenate([lidar_points, tree_points])
combined_attributes = lidar_attributes + tree_attributes
combined_block_ids = lidar_block_ids + tree_block_ids

# Create Octree
print(lidar_points)
#octree = Octree.CustomOctree(lidar_points, lidar_attributes, lidar_block_ids, max_depth)
octree = Octree.CustomOctree(combined_points, combined_attributes, combined_block_ids, max_depth)
print(f"Created Octree with max depth {max_depth}")

# Visualization
vis = o3d.visualization.Visualizer()
vis.create_window()

# Update visualization
Octree.update_visualization(vis, octree, max_depth, 5, 7)

# Run visualization
vis.run()
vis.destroy_window()

####

###octree.py###

'''

CHATGPT - ALWAYS REVIEW THIS SO YOU KNOW THE AIM AND OVERALL STEPS
The code provided is intended for 3D volumetric modeling of urban environments using an Octree data structure. The goal is to build a detailed representation of urban spaces by grouping Octree nodes into logical sets called Blocks, such as Tree Blocks, Building Blocks, and Artificial Habitat Blocks. Each Block represents a specific object or area within the environment.

The key components are:
- OctreeNode class: Represents individual nodes in the Octree.
- CustomOctree class: Represents the Octree data structure with methods for creating bounding boxes and voxel grids.
- OctreeVisualizer class: Visualizes the Octree in 3D using Open3D.
- BoundingBoxToLineSet class: Converts bounding boxes into line sets for visualization.
- Blocks: Logical grouping of Octree nodes representing objects (trees, buildings, etc.)
- Node-Block Association: Nodes can be associated with one or multiple blocks.


The main steps involved are:
1. Define classes for Octree nodes and the Octree data structure.
2. Modify OctreeNode class to include an attribute for tracking associated Block identifiers.
3. Define a data structure for storing information about each Block.
4. Implement methods to associate nodes with the appropriate Blocks during insertion into the Octree.
5. Query nodes and Blocks based on specific criteria such as depth levels and Block identifiers.
6. Visualize the Octree and Blocks.



Oct tree structure: 
The purpose of this script is to construct an octree from a 3D point cloud and to provide visualizations of the octree in the form of voxelized point clouds and bounding boxes.

Key Steps:
1. Load Point Cloud Data:
    - Use pandas to load CSV data into a DataFrame.
    - Extract 3D point coordinates and attributes from the DataFrame.
    - define a Block grouping for each urban element we load data for

2. Construct a Custom Octree:
    - initialise a dictionary that has a block id and block-level attributes associated with each block
    - Initialize a root node of the octree with the minimum and maximum corners of the point cloud.
    - Recursively split the root node into eight children based on the midpoint of the parent's bounding box.
    - For each node, associate block ids with it (ie. the block id that points below to. note:  Nodes can be associated with one or multiple blocks)
    - Continue this process until a specified maximum depth is reached.
    -Upon initialization of each node, determine the dominant attribute based on the distribution of attributes within that node (the most common node attribute of leaf nodes within this node).
    -Use the determined dominant attribute to calculate and store the dominant color in each node.

3. Generate Voxel Grid from Octree's Leaf Nodes:
    - Traverse the octree to its leaf nodes.
    - Create a point cloud from the center points of these leaf nodes and assign colors based on their dominant attributes.
    - Convert the point cloud into a voxel grid using Open3D's VoxelGrid class.

4. Generate Bounding Boxes for Octree Nodes:
    - Traverse the octree, and collect nodes in levels within the minimum and maximum offset values. For each node within this offset, create a bounding box.
    - Assign a color to each bounding box based on the node's dominant attribute.
    - Save these bounding boxes for later visualization.

5. Visualize the Voxel Grid and Bounding Boxes:
    - Create a visualizer object using Open3D.
    - Add the voxel grid and bounding boxes to the visualizer.
    -Convert these bounding boxes into mesh lines, enabling customization of the width.
    - convert the voxel grid into meshes and compute mesh normals
    - Run the visualizer and enable interactions with keyboard callbacks.

Implementation Details for Proposed Block-Based Structure:

The aim is to integrate a block-based structure into the Octree. A Block refers to a logical grouping of Octree nodes that collectively represent a specific object or area within the environment. The primary types of Blocks considered are Tree Blocks, Building Blocks, and Artificial Habitat Blocks.

Key Steps:

1. Modify OctreeNode Class: Add a block_ids attribute to the OctreeNode class to keep track of Block identifiers associated with each node. 
This attribute holds a list of identifiers of the Blocks that the node belongs to.
2.Define Block Information Storage: Create a separate data structure, such as a dictionary, to store information about each Block. The information can include things like the type of Block (tree, building, artificial habitat), the minimum and maximum depths at which the Block exists, and other Block-specific attributes.
3.Implement Node-Block Association during Insertion: When inserting nodes into the Octree, associate them with the appropriate Block(s) by adding the Block identifier(s) to the node's block_ids list.
4.Implement Querying Functions: Implement functions that allow querying nodes and Blocks based on specific criteria, such as depth levels and Block identifiers.
5.Implement Visualization Methods: Implement visualization methods to visualize Blocks within the Octree. Different colors or styles could be used to distinguish different types of Blocks (e.g., trees vs. buildings).        

proposed path forward:

1. Modify OctreeNode Class: We'll start by adding a block_ids attribute to the OctreeNode class. This attribute will hold a list of identifiers of the Blocks that the node belongs to. After making this modification, we can test the OctreeNode class to ensure that it's functioning as expected.
2. Define Block Information Storage: Next, we'll create a separate data structure, such as a dictionary, to store information about each Block. We can then test this data structure by adding some dummy data and retrieving it.
3. Implement Node-Block Association during Insertion: Once we have the Block information storage in place, we can modify the node insertion process to associate nodes with the appropriate Block(s). We'll add the Block identifier(s) to the node's block_ids list during insertion. We can test this by inserting some nodes and checking if they're correctly associated with their respective Blocks.
4. Implement Querying Functions: After the node-block association is working correctly, we'll implement functions that allow querying nodes and Blocks based on specific criteria, such as depth levels and Block identifiers. We can test these functions by running some queries and checking if they return the expected results.
5. Implement Visualization Methods: Finally, we'll implement visualization methods to visualize Blocks within the Octree. We can test these methods by visualizing some Blocks and checking if they're displayed correctly.

EXTRA INFO PER STEP:

STEP 3.  associating each point with a block_id during insertion into the Octree is a good approach. 
you would iterate over the Blocks, and for each Block, 
you would iterate over its points and attributes to add them to the Octree. This way, each point is associated with its Block from the start, 
and this association is maintained as the points are inserted into the Octree.'''

"maintain the block_ids attribute as a set right from the beginning. The latter approach ensures that block_ids always contains unique elements and may be more efficient. Here's how you can modify the code to maintain the block_ids attribute as a set:"

'''
Here's the summary of the steps we've taken so far:

1.Modify the OctreeNode class to include a block_ids attribute: This attribute is a list that holds the block IDs associated with each node. It is initialized in the OctreeNode constructor and passed to the child nodes when the node splits.
2.Add block information to the DataFrame: We added a new column block_id to the DataFrame to store the block ID for each point. The block ID is created by appending the Tree.ID to the string 'Tree '.
3.Associate nodes with blocks during insertion: When inserting nodes into the Octree, they are associated with the appropriate block(s) by adding the block ID(s) to the node's block_ids list. This is done in the split method of the OctreeNode class, where the block_ids are filtered in the same way as the points and attributes and passed to the child nodes.
This approach avoids the need to create an intermediary block_info object, and should be more efficient when dealing with large datasets. However, it does require that the data is initially loaded into a DataFrame, which may not be suitable for all use cases.

###

STEP 4 and 5: implement a query function and visualise bounding boxes
#
1. Implement a query function to retrieve non-leaf nodes with bounding boxes that have only one unique block ID.
2. Implement a function to assign colors to the bounding boxes based on block ownership.
3. Visualize the octree with the colored bounding boxes.


'''
# Global dictionary to keep track of tree block counts
tree_block_count = {}


import open3d as o3d
import numpy as np
import pandas as pd
import random

try:
    from .boxlineset import BoundingBoxToLineSet  # Attempt a relative import
except ImportError:
    from boxlineset import BoundingBoxToLineSet  # Fall back to an absolute import


class OctreeNode:
    def __init__(self, min_corner, max_corner, points, attributes, block_ids, depth=0):  # Added block_ids parameter
        self.children = []  # child nodes
        self.depth = depth  # depth of this node in the tree
        self.min_corner = np.array(min_corner)  # minimum corner of bounding box
        self.max_corner = np.array(max_corner)  # maximum corner of bounding box
        self.points = points
        self.attributes = attributes
        self.block_ids = block_ids  # New line
        self.dominant_attribute, self.dominant_color = self.calculate_dominant_attribute_and_color()
        self.get_geos() #ensure bounding box and center are computed

        # Print statements for checking
        """print(f"Node at depth {self.depth} with min_corner: {self.min_corner}, max_corner: {self.max_corner}")
        print(f"Points: {self.points}")
        print(f"Attributes: {self.attributes}")
        print(f"Block IDs: {self.block_ids}")"""



    def calculate_dominant_attribute_and_color(self):
        # Logic to determine color based on node's attributes
        return ('isDeadOnly', [1, 0, 0])
        """"        counts = {"isDeadOnly": 0, "isLateralOnly": 0, "isBoth": 0, "isNeither": 0}
                for attributes in self.attributes:
                    for key in counts:
                        if attributes[key]:
                            counts[key] += 1
                            break
                max_count_attr = max(counts, key=counts.get)
                color = CustomOctree.get_color_based_on_attribute(max_count_attr)
                return max_count_attr, color """

    def get_geos(self):
        center = (self.min_corner + self.max_corner) / 2
        extent = self.max_corner - self.min_corner
        R = np.eye(3)
        color = self.dominant_color
        self.bounding_box = o3d.geometry.OrientedBoundingBox(center, R, extent)
        self.bounding_box.color = color
        self.center = center

    # The split method is responsible for dividing the current OctreeNode into 8 smaller
    # children nodes, effectively dividing the 3D space that the node represents into
    # 8 smaller cubes. This process is recursive.

    def split(self, max_depth):
        # If the current depth of the node is less than the maximum allowed depth, we can continue to split.
        if self.depth < max_depth:
            # Calculate the midpoint of the current node's bounding box.
            mid = (self.min_corner + self.max_corner) / 2
            # Store the minimum corner, midpoint, and maximum corner.
            bounds = [self.min_corner, mid, self.max_corner]

            # Iterate through the 8 potential children (2 for each dimension).
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        # Determine the minimum and maximum corners for the new child node.
                        new_min = [bounds[x][l] for x, l in zip([i, j, k], range(3))]
                        new_max = [bounds[x][l] for x, l in zip([i+1, j+1, k+1], range(3))]
                        
                        # The following code filters the points that are within the new bounds.
                        # new_min <= self.points checks if each point is greater than or equal to the new minimum corner (elementwise).
                        # self.points <= new_max checks if each point is less than or equal to the new maximum corner (elementwise).
                        # in_range will be a boolean array that will be True for points within new bounds.
                        in_range = np.all((new_min <= self.points) & (self.points <= new_max), axis=1)
                        
                        # Select the points that are within the new bounds.
                        new_points = self.points[in_range]
                        # Select the attributes corresponding to the points within the new bounds.
                        new_attributes = [self.attributes[idx] for idx, val in enumerate(in_range) if val]
                        # Select the block_ids corresponding to the points within the new bounds.
                        new_block_ids = [self.block_ids[idx] for idx, val in enumerate(in_range) if val]

                        # If there are points in the new child node, create the child node.
                        if len(new_points) > 0:
                            # Create the new child node.
                            child = OctreeNode(new_min, new_max, new_points, new_attributes, new_block_ids, self.depth + 1)
                            # Recursively call split on the new child.
                            child.split(max_depth)
                            # Append the child to the current node's children list.
                            self.children.append(child)


class CustomOctree:
    def __init__(self, points, attributes, block_ids, max_depth):  # Added block_ids parameter
        # Compute min and max corners for the bounding box
        #min_corner = np.min(points, axis=0)
        #max_corner = np.max(points, axis=0)

        min_corner, max_corner = self.fit_cube_bbox(points)

        self.root = OctreeNode(min_corner, max_corner, points, attributes, block_ids)  # Added block_ids argument
        self.root.split(max_depth)

    def fit_cube_bbox(self, points):
        min_corner = np.min(points, axis=0)
        max_corner = np.max(points, axis=0)

        length = max(max_corner - min_corner)

        max_corner += length - (max_corner - min_corner)

        return min_corner, max_corner

    @staticmethod
    def get_color_based_on_attribute(attribute):
        # Define the color mapping
        color_mapping = {"isDeadOnly": [1, 0, 0],
                         "isLateralOnly": [0, 1, 0],
                         "isBoth": [0, 0, 1],
                         "isNeither": [1, 1, 1]}
        return color_mapping[attribute]

    def get_all_bounding_boxes(self, node, boxes, min_offset_level, max_offset_level):
        if node is None:
            return
        # Check if the current depth is within the specified range
        if min_offset_level <= node.depth <= max_offset_level:
            boxes.append(node.bounding_box)

        # Recurse for children
        for child in node.children:
            self.get_all_bounding_boxes(child, boxes, min_offset_level, max_offset_level)


    def get_voxels_from_leaf_nodes(self, node, voxel_points, voxel_colors, max_depth):
        if node is None:
            return
        # If at max depth, add voxel point
        if node.depth == max_depth:
            color = node.dominant_color
            center = node.center
            voxel_points.append(center)
            voxel_colors.append(color)

        # Recurse for children
        for child in node.children:
            self.get_voxels_from_leaf_nodes(child, voxel_points, voxel_colors, max_depth)

    def getMeshesfromVoxels(self, max_depth, min_offset_level, max_offset_level):
        voxel_points = []  
        voxel_colors = []  
        bounding_boxes = []

        # Gather all the points and colors from the octree
        self.get_voxels_from_leaf_nodes(self.root, voxel_points, voxel_colors, max_depth)
        
        # Gather all the bounding boxes from the octree within the specified depth range
        self.get_all_bounding_boxes(self.root, bounding_boxes, min_offset_level, max_offset_level)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        voxel_size = np.min(self.root.max_corner - self.root.min_corner) / (2 ** max_depth)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

        return voxel_grid, bounding_boxes
    
    #def sort_nodes_by_ownership(self, node, single_block_nodes, multiple_block_nodes):
        # Base case: if node is None, return
        if node is None:
            return
        
        # If it's a non-leaf node
        if node.children:
            # If it has only one unique block ID
            if len(set(node.block_ids)) == 1:
                single_block_nodes.append(node)
            # If it has multiple block IDs
            elif len(set(node.block_ids)) > 1:
                multiple_block_nodes.append(node)
            
        # Recurse for children
        for child in node.children:
            self.sort_nodes_by_ownership(child, single_block_nodes, multiple_block_nodes)

    def sort_nodes_by_ownership(self, node, single_block_nodes, multiple_block_nodes, min_offset_level, max_offset_level):
        # Base case: if node is None, return
        if node is None:
            return
        
        # Check if the current depth is within the specified range
        if min_offset_level <= node.depth <= max_offset_level:
            # If it's a non-leaf node
            if node.children:
                # If it has only one unique block ID
                if len(set(node.block_ids)) == 1:
                    single_block_nodes.append(node)
                # If it has multiple block IDs
                elif len(set(node.block_ids)) > 1:
                    multiple_block_nodes.append(node)
        
        # Recurse for children
        for child in node.children:
            self.sort_nodes_by_ownership(child, single_block_nodes, multiple_block_nodes, min_offset_level, max_offset_level)


    @staticmethod
    def get_color_based_on_block_id(block_id):
        # Map block IDs to colors
        # Example: assigning colors based on block_id
        color_mapping = {13: [1, 0, 0],  # Red for block_id 13
                         1: [0, 1, 0]}  # Green for block_id 1
        return color_mapping.get(block_id, [1, 1, 1])  # default to white if block_id not in mapping


def update_visualization(vis, octree, max_depth, min_offset_level, max_offset_level):

    def generate_colors(unique_block_ids):
        color_mapping = {}
        for block_id in unique_block_ids:
            color_mapping[block_id] = [random.random(), random.random(), random.random()]
        return color_mapping


    voxel_grid, bounding_boxes = octree.getMeshesfromVoxels(max_depth, min_offset_level, max_offset_level)
    
    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    
    vis.clear_geometries()
    vis.add_geometry(voxel_grid)

    # Fetching nodes sorted by ownership
    single_block_nodes = []
    multiple_block_nodes = []
    octree.sort_nodes_by_ownership(octree.root, single_block_nodes, multiple_block_nodes, min_offset_level, max_offset_level)

    # Create a list of all unique block IDs
    unique_block_ids = list(set(octree.root.block_ids))

    # Generate a color for each unique block ID
    color_mapping = generate_colors(unique_block_ids)

    #Create line sets for bounding boxes with corresponding colors for single block nodes
    linesets = []

    
    for node in single_block_nodes:
        # Convert bounding box to LineSet
        lineset = BoundingBoxToLineSet([node.bounding_box], line_width=100).to_linesets()[0]['geometry']
        # Set colors of LineSet
        unique_node_block_ids = list(set(node.block_ids))
        if len(unique_node_block_ids) > 1:
            print(f"Error: More than one unique block ID found in node: {unique_node_block_ids}")
        else:
            color = color_mapping[unique_node_block_ids[0]]
            lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
            linesets.append(lineset)
    
    # Create line sets for bounding boxes with a different color for multiple block nodes
    for node in multiple_block_nodes:
        # Convert bounding box to LineSet
        lineset = BoundingBoxToLineSet([node.bounding_box], line_width=100).to_linesets()[0]['geometry']
        # Set colors of LineSet
        color = [0.5, 0.5, 0.5] # Example gray color for multiple block nodes
        lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
        linesets.append(lineset)

    # Adding linesets to the visualizer
    for lineset in linesets:
        vis.add_geometry(lineset)

    vis.get_view_control().set_lookat(octree.root.center)
    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)


def update_visualization2(vis, octree, max_depth, min_offset_level, max_offset_level):

    def generate_colors(unique_block_ids):
        color_mapping = {}
        for block_id in unique_block_ids:
            color_mapping[block_id] = [random.random(), random.random(), random.random()]
        return color_mapping


    voxel_grid, bounding_boxes = octree.getMeshesfromVoxels(max_depth, min_offset_level, max_offset_level)
    
    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    
    vis.clear_geometries()
    vis.add_geometry(voxel_grid)

    # Fetching nodes sorted by ownership
    single_block_nodes = []
    multiple_block_nodes = []
    octree.sort_nodes_by_ownership(octree.root, single_block_nodes, multiple_block_nodes, min_offset_level, max_offset_level)

    # Create a list of all unique block IDs
    unique_block_ids = list(set(octree.root.block_ids))

    # Generate a color for each unique block ID
    color_mapping = generate_colors(unique_block_ids)

    # Create line sets for bounding boxes with corresponding colors for single block nodes
    boundingboxes = []

    for node in single_block_nodes:
        box = node.bounding_box
        unique_node_block_ids = list(set(node.block_ids))
        if len(unique_node_block_ids) > 1:
            print(f"Error: More than one unique block ID found in node: {unique_node_block_ids}")
        else:
            color = color_mapping[unique_node_block_ids[0]]
            box.color = color
            boundingboxes.append(box)
    
    # Create line sets for bounding boxes with a different color for multiple block nodes
    for node in multiple_block_nodes:
        box = node.bounding_box
        box.color = [0.5, 0.5, 0.5] # Example gray color for multiple block nodes
        boundingboxes.append(box)

    for box in boundingboxes:
        vis.add_geometry(box)
    
    """
    # Create line sets for bounding boxes with corresponding colors for single block nodes
    linesets = []

    
    for node in single_block_nodes:
        # Convert bounding box to LineSet
        lineset = BoundingBoxToLineSet([node.bounding_box], line_width=100).to_linesets()[0]['geometry']
        # Set colors of LineSet
        unique_node_block_ids = list(set(node.block_ids))
        if len(unique_node_block_ids) > 1:
            print(f"Error: More than one unique block ID found in node: {unique_node_block_ids}")
        else:
            color = color_mapping[unique_node_block_ids[0]]
            lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
            linesets.append(lineset)
    
    # Create line sets for bounding boxes with a different color for multiple block nodes
    for node in multiple_block_nodes:
        # Convert bounding box to LineSet
        lineset = BoundingBoxToLineSet([node.bounding_box], line_width=100).to_linesets()[0]['geometry']
        # Set colors of LineSet
        color = [0.5, 0.5, 0.5] # Example gray color for multiple block nodes
        lineset.colors = o3d.utility.Vector3dVector([color for _ in range(len(lineset.lines))])
        linesets.append(lineset)

    # Adding linesets to the visualizer
    for lineset in linesets:
        vis.add_geometry(lineset)"""

    vis.get_view_control().set_lookat(octree.root.center)
    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)



def tree_block_processing(coordinates_list):
    """
    Load and process the tree block data.

    Args:
        coordinates_list (list): A list of tuples where each tuple contains x, y, z coordinates to translate tree blocks.

    Returns:
        Tuple[np.ndarray, dict, list]: The points, attributes, and block IDs of the processed data.
    """

    def load_and_translate_tree_block_data(dataframe, tree_id, translation):
        # Filter the data for the specific tree
        block_data = dataframe[dataframe['Tree.ID'] == tree_id].copy()

        # Apply translation
        translation_x, translation_y, translation_z = translation
        block_data['x'] += translation_x
        block_data['y'] += translation_y
        block_data['z'] += translation_z

        # Update block IDs to the format "Tree-[unique number]"
        global tree_block_count
        if tree_id in tree_block_count:
            tree_block_count[tree_id] += 1
        else:
            tree_block_count[tree_id] = 1

        block_data['BlockID'] = f'Tree-{tree_id}-{tree_block_count[tree_id]}'

        return block_data

    def get_tree_ids(count):
        return random.sample(range(1, 17), count)

    def define_attributes(combined_data):
        attributes = combined_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
        return attributes


    csv_file = 'data/branchPredictions - full.csv'

    data = pd.read_csv(csv_file)
    print(f"Loaded data with shape {data.shape}")

    # Get random tree IDs
    print(f'Coordinates list: {coordinates_list}')
    tree_count = len(coordinates_list)
    tree_ids = get_tree_ids(tree_count)
    print(f'Loading and processing {tree_ids} tree blocks...')

    # Process block data for each tree
    processed_data = [load_and_translate_tree_block_data(data, tree_id, coordinates)
                      for tree_id, coordinates in zip(tree_ids, coordinates_list)]

    # Combine the block data
    combined_data = pd.concat(processed_data)

    # Extract points, attributes, and block IDs
    points = combined_data[['x', 'y', 'z']].to_numpy()
    attributes = define_attributes(combined_data)
    block_ids = combined_data['Tree.ID'].tolist()

    return points, attributes, block_ids




def tree_block_processing2(tree_count):
    """
    Load and process the tree block data.

    Args:
        tree_count (int): The number of tree blocks to process.

    Returns:
        Tuple[np.ndarray, dict, list]: The points, attributes, and block IDs of the processed data.
    """
    # Load the point cloud data


    def load_and_translate_tree_block_data(dataframe, tree_id, translation_range=20):
        # Filter the data for the specific tree
        block_data = dataframe[dataframe['Tree.ID'] == tree_id].copy()
        
        # Apply a random translation on the horizontal plane (X, Y)
        translation_x = random.uniform(-translation_range, translation_range)
        translation_y = random.uniform(-translation_range, translation_range)
        block_data['x'] += translation_x
        block_data['y'] += translation_y
        
        # Update block IDs to the format "Tree-[unique number]"
        global tree_block_count
        if tree_id in tree_block_count:
            tree_block_count[tree_id] += 1
        else:
            tree_block_count[tree_id] = 1

        block_data['BlockID'] = f'Tree-{tree_id}-{tree_block_count[tree_id]}'
        
        return block_data

    def get_tree_ids(tree_count):
        return random.sample(range(1, 17), tree_count)
    
    def define_attributes(combined_data):
        attributes = combined_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
        return attributes
    


    csv_file = 'data/branchPredictions - full.csv'


    data = pd.read_csv(csv_file)
    print(f"Loaded data with shape {data.shape}")
    print(data.head())

    # Get random tree IDs
    tree_ids = get_tree_ids(tree_count)
    print(f'Loading and processing {tree_ids} tree blocks...')

    # Process block data for each tree
    processed_data = [load_and_translate_tree_block_data(data, i) for i in tree_ids]

    # Combine the block data
    combined_data = pd.concat(processed_data)

    print(combined_data)

    # Rename columns
    combined_data = combined_data.rename(columns={'y': 'z', 'z': 'y'})

    # Extract points, attributes, and block IDs
    points = combined_data[['x', 'y', 'z']].to_numpy()
    attributes = define_attributes(combined_data)
    block_ids = combined_data['Tree.ID'].tolist()

    return points, attributes, block_ids



def main():

    # Process tree block data for a specified number of trees
    # Example usage:
    coordinates_list = [(5, 5, 0), (-5, -5, 0), (10, -10, 0)]
    tree_block_data = tree_block_processing(coordinates_list)


    if tree_block_data is None:
        print("Error: Tree block processing failed.")
        return
    
    points, attributes, block_ids = tree_block_data


    # Create Octree
    max_depth = 7
    octree = CustomOctree(points, attributes, block_ids, max_depth)
    print(f"Created Octree with max depth {max_depth}")

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Update visualization
    update_visualization(vis, octree, max_depth, 2, 3)

    # Run visualization
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()



#### convertSitesToBlocks.py
"""
CHAT GPT PLEASE READ

We aim to construct a 3D volumetric model of urban environments using an Octree data structure. 

Initially, we perform a large-scale spatial analysis to pinpoint characteristic sites and challenges, and to unveil opportunities for interventions.

We then select representative sites from these categories for detailed study, using aerial LiDAR to capture volumetric data. 

Our platform convert these lidar scans into numerical descriptions of urban spaces using an Octree data structure. 
We aggregate Octree nodes into logical sets named Blocks, such as Tree Blocks, Building Blocks, and Artificial Habitat Blocks,
where each Block signifies a specific object or area within the environment.

For the broader urban environment, we dissect these sites into distinct 'Building Blocks,' 
which encompass the ground, buildings, trees, street furniture, and other features.
Within buildings, we conduct a nuanced analysis to identify walls and roofs that offer high potential for the addition of artificial structures, such as habitat roofs. 


Data Description:
The data is from an aerial LiDAR scan of an urban street and park, consisting of around 150,000 points in a 100x100x sample.
The data will be saved in a PLY file.
Each point has the following attributes: X, Y, Z (positions), Rf, Gf, Bf (colors in the range 0-1), Illuminance_(PCV) (shading information, grayscale, range 0-1), element_type (integers: 0 for 'tree', 1 for 'building', 2 for 'grass', 3 for 'street-furniture', 4 for 'ground'), Composite, Dip_(degrees), Dip_direction_(degrees), R, G, B, Nx, Ny, Nz.

Key Steps and Elements of the Code:

Step 1: Data Import - Use Open3D to import the PLY file.

Step 2: further_processing - Use the Dip_(degrees) information to classify points based on the horizontality: 'flat', 'angled', or 'vertical'.

Step 3: Build Query Structure - Create a function to categorize points based on combinations of attributes, for example, "element_type == 1 and horizontality == 1". This function will assign a new attribute called 'category' to each point, based on the combination of attributes specified in the query.

Step 4: Color Enhancement - Define a function that takes the original colors and enhances them using the Illuminance_(PCV) information.

Step 5: Visualization - Visualize the point cloud using Open3D, incorporating color enhancement and displaying different colors based on the 'category' attribute assigned in Step 3.

"""

import open3d as o3d
import pandas as pd
import numpy as np

try:
    from .colorMaps import Colormap  # Attempt a relative import
except ImportError:
    from colorMaps import Colormap  # Fall back to an absolute import

def enhance_colors_with_illuminance(colors, illuminance):
    return colors * illuminance[:, np.newaxis] ** 5

def assign_horizontality(dip_degrees):
    horizontality = np.empty_like(dip_degrees, dtype=int)
    horizontality[dip_degrees < 6] = 0  # flat
    horizontality[(dip_degrees >= 6) & (dip_degrees < 85)] = 1  # angled
    horizontality[dip_degrees >= 85] = 2  # vertical
    return horizontality

def create_category_mapping(data, queries):
    category = np.full(len(data), -1, dtype=int)  # default to -1
    for i, query in enumerate(queries):
        filtered_indices = data.query(query).index
        category[filtered_indices] = i + 1
    return category

def get_and_shuffle_colors(cm, queries, colormap_name):
    category_colors = cm.get_categorical_colors(len(queries) + 1, colormap_name)
    category_colors.pop(0)
    return category_colors

def assign_colors_based_on_category(data, category_colors):
    colors = np.array([category_colors[cat - 1] for cat in data['blockID']])
    return colors


def enhance_colors(colors, illuminance):
    enhanced_colors = enhance_colors_with_illuminance(colors, illuminance)
    return enhanced_colors

def convert_to_point_cloud(data, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].values)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def process_lidar_data(filepath, colormap_name='lajollaS'):
    """
    Returns a dataframe with the columns being X,Y,Z,blockID,r,g,b,B,Bf,Composite,Dip (degrees),Dip direction (degrees),G,Gf,Illuminance (PCV),Nx,Ny,Nz,R,Rf,element_type,horizontality
    with one per column. Rows are the points.
    
    Parameters:
        filepath (str): The path to the lidar data file.
        colormap_name (str, optional): The name of the colormap. Defaults to 'glasgowS'.
    
    Returns:
        pandas.DataFrame: The processed lidar data as a dataframe.
    """
        
    # Load Data
    data = pd.read_parquet(filepath)
    print(data)
    data.rename(columns={'//X': 'X'}, inplace=True)
    
    # Assign horizontality
    data['horizontality'] = assign_horizontality(data['Dip (degrees)'].values)
    
    # Queries
    queries = [
        "element_type == 0",
        "element_type == 1 and horizontality == 0",
        "element_type == 1 and horizontality == 1",
        "element_type == 1 and horizontality == 2",
        "element_type == 2",
        "element_type == 3",
        "element_type == 4"
    ]
    
    # Create category mapping
    data['blockID'] = create_category_mapping(data, queries)
    
    # Filter out uncategorized points
    data = data[data['blockID'] != -1]
    
    # Create a colormap for visualization
    cm = Colormap()
    
    # Get and shuffle colors
    category_colors = get_and_shuffle_colors(cm, queries, colormap_name)
    
    # Assign colors based on category
    colors = assign_colors_based_on_category(data, category_colors)
    
    # Enhance colors with illuminance
    enhanced_colors = enhance_colors(colors, np.array(data['Illuminance (PCV)']))
    
    # Assign the color columns directly to the data DataFrame
    data['r'] = enhanced_colors[:, 0]
    data['g'] = enhanced_colors[:, 1]
    data['b'] = enhanced_colors[:, 2]
    
    # Get the attribute columns
    attributes_columns = data.columns.difference(['X', 'Y', 'Z', 'r', 'g', 'b', 'blockID'])
    
    # Order columns
    ordered_columns = ['X', 'Y', 'Z', 'r', 'g', 'b', 'blockID'] + list(attributes_columns)
    ordered_columns = ['X', 'Y', 'Z', 'blockID', 'r', 'g', 'b'] + list(attributes_columns)

    result = data[ordered_columns]
    
    # Return the DataFrame
    return result

def select_random_ground_points(processed_data, n_points):
    """
    Randomly selects n number of points with element_type = 2 (ground) and returns their X, Y, Z coordinates.

    Parameters:
        processed_data (pandas.DataFrame): The DataFrame containing processed LiDAR data.
        n_points (int): The number of random points to select.

    Returns:
        pandas.DataFrame: A DataFrame with X, Y, Z coordinates of the randomly selected ground points.
    """
    # Selecting points where element_type = 2 (ground)
    #ground_points = processed_data[processed_data['element_type'] == 2]

    ground_points = processed_data
    print('printing ground points')
    print(ground_points)
    # Randomly select n_points from ground_points
    selected_points = ground_points.sample(n_points)

    selected_coords = selected_points[['X', 'Y', 'Z']]

    coordinates_list_as_tuples = [tuple(x) for x in selected_coords.values]


    #print(f'Selected {n_points} random points {selected_coords}')

    # Return the X, Y, Z coordinates
    return coordinates_list_as_tuples



def convertToVoxelGrid(data):
    # Extract coordinates and colors
    coordinates = data[['X', 'Y', 'Z']].values
    colors = data[['r', 'g', 'b']].values

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Convert PointCloud to VoxelGrid for visualization
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the geometry to the visualizer
    vis.add_geometry(voxel_grid)

    # Change view parameters if needed (you can adjust these values)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # This time, we won't set the front vector, but let Open3D handle it
    # Instead, we'll set the camera to look at the center of the point cloud
    bounds = np.array(pcd.get_max_bound()) - np.array(pcd.get_min_bound())
    view_control.set_lookat(np.array(pcd.get_center()))
    view_control.set_up([0, -1, 0])
    view_control.set_front([0, 0, 1])

    # Begin the visualization
    vis.run()
    vis.destroy_window()

# The main section
if __name__ == "__main__":
    # This part will only be executed if the script is run as a standalone file,
    # and not if it's imported as a module.
    
    # Filepath to the Parquet file
    filepath = 'data/sites/park.parquet'

    # Process the lidar data
    processed_data = process_lidar_data(filepath)
    
    # Convert the processed data to a VoxelGrid and visualize it
    convertToVoxelGrid(processed_data)

    ###

    ### boxlineset.py

    import open3d as o3d
import random


class BoundingBoxToLineSet:
    def __init__(self, bounding_boxes, line_width=200):
        self.bounding_boxes = bounding_boxes
        self.line_width = line_width

    def to_linesets(self):
        linesets = []
        for i, box in enumerate(self.bounding_boxes):
            lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            lineset.paint_uniform_color(box.color)

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = self.line_width

            linesets.append({"name": f"box_{i}", "geometry": lineset, "material": mat})
        return linesets


def visualize_linesets(linesets):
    geometries = []

    for line in linesets:
        geometries.append(line)

    o3d.visualization.draw(geometries)


if __name__ == "__main__":
    # Create some example bounding boxes
    boxes = [o3d.geometry.OrientedBoundingBox(center=(random.random() * 10, random.random() * 10, random.random() * 10),
                                              R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(
                                                  (random.random(), random.random(), random.random())),
                                              extent=[1, 1, 1]) for _ in range(10)]
    for box in boxes:
        box.color = [random.random() for _ in range(3)]  # Assign random colors

    # Convert bounding boxes to line sets
    converter = BoundingBoxToLineSet(boxes)
    linesets = converter.to_linesets()

    # Visualize the line sets
    visualize_linesets(linesets)


#### colourMaps.py


import json
import numpy as np
from scipy.interpolate import interp1d

#NOTE THE S VERSIONS ARE CATAGOERICAL, THE NON S VERSIONS ARE CONTINUOUS
class Colormap:
    def __init__(self, json_file_path=None):
        if json_file_path is None:
            json_file_path = 'data/colors/categorical-maps.json'  # default path
        self._colormaps = self._load_colormaps(json_file_path)

    @staticmethod
    def _load_colormaps(json_file_path):
        with open(json_file_path, 'r') as json_file:
            return json.load(json_file)

    def get_categorical_colors(self, n_colors, colormap_name=None):
        if colormap_name is None:  # if no specific colormap provided, use a default one
            colormap_name = list(self._colormaps.keys())[0]

        cm_data = np.array(self._colormaps[colormap_name])

        return cm_data[:n_colors].tolist()  # get the first n colors

    def get_continuous_colors(self, n_colors, colormap_name=None):
        if colormap_name is None:  # if no specific colormap provided, use a default one
            colormap_name = list(self._colormaps.keys())[0]

        cm_data = np.array(self._colormaps[colormap_name])

        color_func = interp1d(np.linspace(0, 1, cm_data.shape[0]), cm_data, axis=0)

        return color_func(np.linspace(0, 1, n_colors)).tolist()  # interpolate to get n colors

"""cm = Colormap()
import os
print(os.getcwd())
categorical_colors = cm.get_categorical_colors(5, 'batlow')
print(categorical_colors)

continuous_colors = cm.get_continuous_colors(5, 'batlow')
print(continuous_colors)"""


### other technical instructions
The code provided is intended for 3D volumetric modeling of urban environments using an Octree data structure. 
The goal is to build a detailed representation of urban spaces by grouping Octree nodes into logical sets called Blocks, 
such as Tree Blocks, Building Blocks, and Artificial Habitat Blocks. Each Block represents a specific object or area within the environment.

The key components are:
- OctreeNode class: Represents individual nodes in the Octree.
- CustomOctree class: Represents the Octree data structure with methods for creating bounding boxes and voxel grids.
- OctreeVisualizer class: Visualizes the Octree in 3D using Open3D.
- BoundingBoxToLineSet class: Converts bounding boxes into line sets for visualization.
- Blocks: Logical grouping of Octree nodes representing objects (trees, buildings, etc.)
- Node-Block Association: Nodes can be associated with one or multiple blocks.


The main steps involved are:
1. Define classes for Octree nodes and the Octree data structure.
2. Modify OctreeNode class to include an attribute for tracking associated Block identifiers.
3. Define a data structure for storing information about each Block.
4. Implement methods to associate nodes with the appropriate Blocks during insertion into the Octree.
5. Query nodes and Blocks based on specific criteria such as depth levels and Block identifiers.
6. Visualize the Octree and Blocks.


INFO ABOUT THE OCTREE

CHATGPT - ALWAYS REVIEW THIS SO YOU KNOW THE AIM AND OVERALL STEPS
Aim: 
The purpose of this script is to construct an octree from a 3D point cloud and to provide visualizations of the octree in the form of voxelized point clouds and bounding boxes.

Key Steps:
1. Load Point Cloud Data:
    - Use pandas to load CSV data into a DataFrame.
    - Extract 3D point coordinates and attributes from the DataFrame.

2. Construct a Custom Octree:
    - Initialize a root node of the octree with the minimum and maximum corners of the point cloud.
    - Recursively split the root node into eight children based on the midpoint of the parent's bounding box.
    - Continue this process until a specified maximum depth is reached.
    -Upon initialization of each node, determine the dominant attribute based on the distribution of attributes within that node.
    -Use the determined dominant attribute to calculate and store the dominant color in each node.

3. Generate Voxel Grid from Octree's Leaf Nodes:
    - Traverse the octree to its leaf nodes.
    - Create a point cloud from the center points of these leaf nodes and assign colors based on their dominant attributes.
    - Convert the point cloud into a voxel grid using Open3D's VoxelGrid class.

4. Generate Bounding Boxes for Octree Nodes:
    - Traverse the octree, and for each node, create a bounding box.
    - Assign a color to each bounding box based on the node's dominant attribute.
    - Save these bounding boxes for later visualization.

5. Visualize the Voxel Grid and Bounding Boxes:
    - Create a visualizer object using Open3D.
    - Add the voxel grid and bounding boxes to the visualizer.
    -Convert these bounding boxes into mesh lines, enabling customization of the width.
    - Run the visualizer and enable interactions with keyboard callbacks.
'''

[CODE]




# CSV Structure

Each tree corresponds to multiple rows in the CSV file. The current implementation uses this approach: for each unique Tree.ID in the CSV file, it takes all the rows (i.e., the branches) associated with that tree, and calculates a random offset for the x and y positions.

This offset is then added to all the branch points of that tree. This simulates the scenario where different trees are located at different positions in the landscape, while preserving the relative positions of the branches within each tree. In other words, the structure of each tree is maintained.

These tree points along with their associated attributes are then added to the points and attributes lists, respectively. The Octree is then constructed based on these points and attributes.

This means that in the Octree, each node could represent multiple branches from the same tree, or branches from different trees, depending on their positions and the depth of the Octree.

# Blocks

Urban environments are complex systems comprising natural elements such as trees and man-made structures like buildings. Modelling these environments requires a detailed representation of the spatial distribution and attributes of different elements. Octrees are hierarchical data structures ideal for volumetric modelling of 3D spaces, and can be efficiently used to model urban environments by capturing detailed information about various elements, including trees, buildings, and other structures.

## 1. Blocks

Here, a Block refers to a logical grouping of Octree nodes that collectively represent a specific object or area within the environment. There are three primary types of Blocks we consider in the urban environments:

1. Tree Blocks: These represent individual trees and are used to model arboreal habitat resources. They include information about the tree's structure and its habitat resources.
2. Building Blocks: These represent buildings and other environmental structures. They may include information such as the dimensions, materials, and other attributes of the buildings.
3. Artificial Habitat Blocks: These represent artificial structures that are specifically created to serve as habitats. This can include things like birdhouses, man-made ponds, etc.

Blocks are essentially a collection of Octree nodes that belong to a certain object or region.

### 1.2 Node Association with Blocks

Nodes in the Octree can be associated with Blocks. This association can be done by adding an additional attribute to the OctreeNode class which holds a list of identifiers of the Blocks that the node belongs to.

Higher-level nodes may belong to multiple Blocks, meaning that they are effectively shared among these Blocks. This reflects that the volume represented by that node contains elements of multiple Blocks.

As you drill down the Octree, the nodes will become more specific to individual Blocks. Eventually, at lower levels, each node might belong to only one Block, or none, if it is empty space.

### 1.3 Implementation Details

#### 1.3.1 OctreeNode Class Modification

Add a `block_ids` attribute to the `OctreeNode` class to keep track of Block identifiers associated with each node.

```python
class OctreeNode:
    def __init__(self, min_corner, max_corner, depth):
        # ... (existing code) ...
        self.block_ids = []  # This holds the list of unique identifiers of the Blocks

#### 1.3.2 Block Information Storage

Create a separate data structure, such as a dictionary, to store information about each Block. The information can include things like the type of Block (tree, building, artificial habitat), the minimum and maximum depths at which the Block exists, and other Block-specific attributes.

```python
block_info = {
    "unique_block_identifier": {
        "type": "tree",
        "min_depth": 3,
        "max_depth": 7,
        "min_corner": [float('inf'), float('inf'), float('inf')],
        "max_corner": [float('-inf'), float('-inf'), float('-inf')],
        "habitat_resources": {...},  # ... Other Block-specific attributes ...
    }
}

#### 1.3.3 Node-Block Association during Insertion

When inserting nodes into the Octree, associate them with the appropriate Block(s) by adding the Block identifier(s) to the node's block_ids list.

```python
node.block_ids.append("unique_block_identifier")

### 1.4 Querying
Implement functions that allow querying nodes and Blocks based on specific criteria, such as depth levels and Block identifiers.

### 1.5 Visualization
Implement visualization methods to visualize Blocks within the Octree. Different colors or styles could be used to distinguish different types of Blocks (e.g., trees vs. buildings

##1.x Conclusion

By implementing Blocks within an Octree structure, it is possible to create detailed volumetric models of urban environments. This structure allows for efficient representation and querying of complex spatial data, including natural elements such as trees and man-made structures. Through a flexible association of nodes with Blocks, the model can capture both the specific details of individual elements and the relationships between them in shared spaces.