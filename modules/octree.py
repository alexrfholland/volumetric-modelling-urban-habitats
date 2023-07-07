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


import open3d as o3d
import numpy as np
import pandas as pd
import random


from boxlineset import BoundingBoxToLineSet


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
        print(f"Node at depth {self.depth} with min_corner: {self.min_corner}, max_corner: {self.max_corner}")
        print(f"Points: {self.points}")
        print(f"Attributes: {self.attributes}")
        print(f"Block IDs: {self.block_ids}")



    def calculate_dominant_attribute_and_color(self):
        # Logic to determine color based on node's attributes
        counts = {"isDeadOnly": 0, "isLateralOnly": 0, "isBoth": 0, "isNeither": 0}
        for attributes in self.attributes:
            for key in counts:
                if attributes[key]:
                    counts[key] += 1
                    break
        max_count_attr = max(counts, key=counts.get)
        color = CustomOctree.get_color_based_on_attribute(max_count_attr)
        return max_count_attr, color 

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
    def __init__(self, min_corner, max_corner, points, attributes, block_ids, max_depth):  # Added block_ids parameter
        self.root = OctreeNode(min_corner, max_corner, points, attributes, block_ids)  # Added block_ids argument
        self.root.split(max_depth)

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
    
    def sort_nodes_by_ownership(self, node, single_block_nodes, multiple_block_nodes):
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

    @staticmethod
    def get_color_based_on_block_id(block_id):
        # Map block IDs to colors
        # Example: assigning colors based on block_id
        color_mapping = {13: [1, 0, 0],  # Red for block_id 13
                         1: [0, 1, 0]}  # Green for block_id 1
        return color_mapping.get(block_id, [1, 1, 1])  # default to white if block_id not in mapping

def update_visualization(vis, octree, max_depth, min_offset_level, max_offset_level):
    voxel_grid, bounding_boxes = octree.getMeshesfromVoxels(max_depth, min_offset_level, max_offset_level)
    
    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    
    vis.clear_geometries()
    vis.add_geometry(voxel_grid)

    # Fetching nodes sorted by ownership
    single_block_nodes = []
    multiple_block_nodes = []
    octree.sort_nodes_by_ownership(octree.root, single_block_nodes, multiple_block_nodes)

    # Create line sets for bounding boxes with corresponding colors for single block nodes
    linesets = []
    for node in single_block_nodes:
        # Convert bounding box to LineSet
        lineset = BoundingBoxToLineSet([node.bounding_box], line_width=100).to_linesets()[0]['geometry']
        # Set colors of LineSet
        color = CustomOctree.get_color_based_on_block_id(list(set(node.block_ids))[0])
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


def load_and_translate_block_data(dataframe, tree_id, translation_range=10):
    # Filter the data for the specific tree
    block_data = dataframe[dataframe['Tree.ID'] == tree_id].copy()
    
    # Apply a random translation on the horizontal plane (X, Y)
    translation_x = random.uniform(-translation_range, translation_range)
    translation_y = random.uniform(-translation_range, translation_range)
    block_data['x'] += translation_x
    block_data['y'] += translation_y
    
    return block_data


import os
import pandas as pd
import numpy as np

def main():
    try:
        # Load the point cloud data
        csv_file = 'data/branchPredictions - full.csv'
        if not os.path.exists(csv_file):
            print(f"Error: File not found - {csv_file}")
            return

        data = pd.read_csv(csv_file)
        print(f"Loaded data with shape {data.shape}")
        print(data.head())

        # Load and translate block data for Tree.ID == 13 and Tree.ID == 1
        # Note: Ensure that 'load_and_translate_block_data' is either a static method or
        # create an object of 'CustomOctree' with the proper parameters before calling this method.
        # Assuming it's a static method:
        block_data_13 = load_and_translate_block_data(data, 13)
        block_data_1 = load_and_translate_block_data(data, 1)

        # Combine the block data
        combined_data = pd.concat([block_data_13, block_data_1])

        # Rename columns
        combined_data = combined_data.rename(columns={'y': 'z', 'z': 'y'})

        # Extract points, attributes, and block IDs
        points = combined_data[['x', 'y', 'z']].to_numpy()
        attributes = combined_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
        block_ids = combined_data['Tree.ID'].tolist()

        # Compute min and max corners for the bounding box
        min_corner = np.min(points, axis=0)
        max_corner = np.max(points, axis=0)

        # Create Octree
        max_depth = 5
        octree = CustomOctree(min_corner, max_corner, points, attributes, block_ids, max_depth)
        print(f"Created Octree with max depth {max_depth}")

        # Visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Update visualization
        update_visualization(vis, octree, max_depth, 2, 3)

        # Run visualization
        vis.run()
        vis.destroy_window()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

