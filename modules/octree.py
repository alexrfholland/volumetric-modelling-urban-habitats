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
6. Visualize the Octree and Blocks using PyVista.



Oct tree structure: 
The purpose of this script is to construct an octree from a 3D point cloud and to provide visualizations of the octree in the form of voxelized point clouds and bounding boxes.

Key Steps:
1. Load Point Cloud Data:
    - Use pandas to load CSV data into a DataFrame.
    - Extract 3D point coordinates and attributes from the DataFrame.
    - define a Block grouping for each urban element we load data for

2. Construct a Custom Octree:
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

4. Generate Outline Cubesfor Other Nodes:
    - Traverse the octree, and collect nodes in levels within the minimum and maximum offset values. For each node within this offset, create a bounding box.
    - Find nodes that only belong to one block
    - Assign a color to eaach single block node based on the node's blockID
    
5. Visualize the Voxel Grid and Bounding Boxes:
    - Create a visualizer module called Glyphmapping that can be used to visalise nodes as a pointcloud, filled cubes, outline cubes, coloured either with RGBA or a scalar colour cmap
    - Add the voxel grid and bounding boxes to the visualizer using pyvista glyphs and other high-performance rendering techniques in PyVyista

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

"""CURRENT EXTENTION TO DO:
1.1 Extend octree class so we can add additional blocks and update the blocks (ie. recalculate the dominant attribute, single block nodes, etc)
1.1.1 write a simple test function that can add a block and update the octree
1.2 Extend octree class so we can find nearest node of a particular type
1.2.1 write a simple test function that can give a a type of node (ie. has block id = X) of then search for the nearest neighbouring node of a different type (ie. has block ID = Y)
"""
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
import random
import matplotlib.cm as cm

try:
    from . import glyphmapping  # Attempt a relative import
except ImportError:
    import glyphmapping  # Fall back to an absolute import

print("start")
from matplotlib.colors import ListedColormap


import pyvista as pv

class OctreeNode:
    def __init__(self, min_corner, max_corner, points, attributes, block_ids, depth=0):
        self.children = []
        self.depth = depth
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)
        self.points = points
        self.attributes = attributes
        self.block_ids = block_ids
        self.center, self.extent = self.get_geos()
        self.parent = None

    def calculate_dominant_attribute_and_colors(self):
        attribute_columns = ['Rf', 'Gf', 'Bf']
        
        # Convert attributes list to pandas DataFrame
        df = pd.DataFrame(self.attributes)

        # Check if 'Rf', 'Gf', 'Bf' exist in the DataFrame columns
        if all(column in df.columns for column in attribute_columns):
            # Compute mode (most common color) along each column and store result as list
            dominant_color = df[attribute_columns].mode().values[0].tolist()

            return 'isColorDominant', dominant_color, dominant_color
        else:
            return 'isDeadOnly', [1, 0, 0], [1,0,0]
    
    def get_geos(self):
        center = (self.min_corner + self.max_corner) / 2
        extent = self.max_corner - self.min_corner
        return center, extent

    def split(self, max_depth):
        if self.depth < max_depth:
            mid = (self.min_corner + self.max_corner) / 2
            bounds = [self.min_corner, mid, self.max_corner]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        new_min = [bounds[x][l] for x, l in zip([i, j, k], range(3))]
                        new_max = [bounds[x][l] for x, l in zip([i+1, j+1, k+1], range(3))]
                        in_range = np.all((new_min <= self.points) & (self.points <= new_max), axis=1)
                        new_points = self.points[in_range]
                        new_attributes = [self.attributes[idx] for idx, val in enumerate(in_range) if val]
                        new_block_ids = [self.block_ids[idx] for idx, val in enumerate(in_range) if val]
                        if len(new_points) > 0:
                            child = OctreeNode(new_min, new_max, new_points, new_attributes, new_block_ids, self.depth + 1)
                            child.split(max_depth)
                            self.children.append(child)
                            child.parent = self


    def calculate_bounds_for_point(self, point):
        # Calculate the midpoint of the current node
        mid = (self.min_corner + self.max_corner) / 2

        # The new minimum corner is the minimum of the current node's corners and the point
        new_min = np.minimum(self.min_corner, point)

        # The new maximum corner is the maximum of the current node's corners and the point
        new_max = np.maximum(self.max_corner, point)

        # Adjust the new minimum and maximum corners so that the point is within the middle of the new bounds
        new_min = np.where(point <= mid, new_min, mid)
        new_max = np.where(point > mid, new_max, mid)

        return new_min, new_max


                        

    



    
#a block is a collection of points (and their attributes) which might spread over multiple nodes in the octree

class CustomOctree:
    def __init__(self, points, attributes, block_ids, max_depth):  # Added block_ids parameter
        # Compute min and max corners for the bounding box
        #min_corner = np.min(points, axis=0)
        #max_corner = np.max(points, axis=0)

        print(f'summary stats of points, attributes and block_ids: {np.shape(points)}, {np.shape(attributes)}, {np.shape(block_ids)}')
        print(f'input data for these looks like: {points[0]}, {attributes[0]}, {block_ids[0]}')

        min_corner, max_corner = self.fit_cube_bbox(points)
        self.max_depth = max_depth

        self.root = OctreeNode(min_corner, max_corner, points, attributes, block_ids)  # Added block_ids argument
        self.root.split(max_depth)

        self.base_size = np.max(self.root.max_corner - self.root.min_corner)


    def compute_node_size(self, depth):
        return self.base_size / (2**depth)

    def fit_cube_bbox(self, points):
        min_corner = np.min(points, axis=0)
        max_corner = np.max(points, axis=0)

        length = max(max_corner - min_corner)

        max_corner += length - (max_corner - min_corner)

        return min_corner, max_corner

    
    def add_block2(self, points, attributes, block_ids):
        for point, attribute, block_id in zip(points, attributes, block_ids):
            # Find the appropriate node to insert this point into
            node, quadrant = self.find_node_for_point(point)

            # If the point is not within any existing child node, create a new one
            if node is self.root or quadrant is not None:
                min_corner, max_corner = node.calculate_bounds_for_point(point)
                child = OctreeNode(min_corner, max_corner, np.array([point]), [attribute], [block_id], node.depth + 1)
                node.children.append(child)
                child.parent = node

                # Append the block_id to the current node and all its ancestors
                node_to_update = node
                while node_to_update is not None:
                    node_to_update.block_ids.append(block_id)
                    node_to_update = node_to_update.parent
            else:
                # Append the point, attribute, and block_id to the found leaf node
                node.points = np.append(node.points, [point], axis=0)
                node.attributes.append(attribute)
                node.block_ids.append(block_id)

    def add_block(self, points, attributes, block_ids):
        def update_block_ids(node, block_id):
            while node is not None:
                node.block_ids.append(block_id)
                node = node.parent

        for point, attribute, block_id in zip(points, attributes, block_ids):
            # Find the appropriate node to insert this point into
            node, quadrant = self.find_node_for_point(point)

            # If the point is not within any existing child node, create a new one
            if node is self.root or quadrant is not None:
                min_corner, max_corner = node.calculate_bounds_for_point(point)
                child = OctreeNode(min_corner, max_corner, np.array([point]), [attribute], [block_id], node.depth + 1)
                node.children.append(child)
                child.parent = node

                # Append the block_id to the current node and all its ancestors
                update_block_ids(node, block_id)
            else:
                # Append the point, attribute, and block_id to the found leaf node
                node.points = np.append(node.points, [point], axis=0)
                node.attributes.append(attribute)
                node.block_ids.append(block_id)
                
                # Append the block_id to the found node and all its ancestors
                update_block_ids(node, block_id)





        
    def find_node_for_point(self, point):
        epsilon = 1e-9  # A small tolerance value

        # Start from the root and go down the tree
        node = self.root
        quadrant = None
        while len(node.children) > 0:
            for i, child in enumerate(node.children):
                min_corner = child.min_corner - epsilon
                max_corner = child.max_corner + epsilon
                
                if np.all(min_corner <= point) and np.all(point <= max_corner):
                    node = child
                    break
            else:
                # The point does not fit into any child -- this should not happen if the point fits into the octree bounds
                # However, in sparse octrees, it's possible that the child node was not created yet
                break  # This will stop the loop and return the current node
            quadrant = i

        return node, quadrant


        return node

    def find_node_for_point2(self, point):
        # Start from the root and go down the tree
        node = self.root
        while len(node.children) > 0:
            logging.debug(f"Checking point {point} against children of node at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}")
            for child in node.children:
                logging.debug(f"Checking point {point} against child at depth {child.depth} with bounds {child.min_corner} - {child.max_corner}")
                if np.all(child.min_corner <= point) and np.all(point <= child.max_corner):
                    node = child
                    break
            else:
                # Point does not fit into any child -- this should not happen if the point fits into the octree bounds
                logging.debug(f"Point {point} does not fit into any child of node at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}")
                return None
        logging.warning(f"Found node for point {point} at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}")
        return node
    
    def find_node_for_point3(self, point):
        epsilon = 1e-9  # A small tolerance value

        # Start from the root and go down the tree
        node = self.root
        while len(node.children) > 0:
            logging.debug(f"Checking point {point} against children of node at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}")
            for child in node.children:
                logging.debug(f"Checking point {point} against child at depth {child.depth} with bounds {child.min_corner} - {child.max_corner}")
                
                min_corner = child.min_corner - epsilon
                max_corner = child.max_corner + epsilon
                
                if np.all(min_corner <= point) and np.all(point <= max_corner):
                    node = child
                    break
            else:
                # Point does not fit into any child -- this should not happen if the point fits into the octree bounds
                logging.debug(f"Point {point} does not fit into any child of node at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}")
                return None
        logging.warning(f"Found node for point {point} at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}")
        return node



    
    @staticmethod
    def get_color_based_on_attribute(attribute):
        # Define the color mapping
        color_mapping = {"isDeadOnly": [1, 0, 0],
                         "isLateralOnly": [0, 1, 0],
                         "isBoth": [0, 0, 1],
                         "isNeither": [1, 1, 1]}
        return color_mapping[attribute]

    def get_nodes_for_visualization(self, min_offset_level, max_offset_level):
        single_block_nodes = []
        leaf_nodes = []

        # Helper method to check if parent is a single block node
        def is_parent_single_block(node):
            parent = node.parent
            while parent is not None:
                if len(set(parent.block_ids)) == 1:
                    return True
                parent = parent.parent
            return False


        # a node will be added to single_block_nodes if it's a single block node and either 
        # it's at the min_offset_level depth, or it does not have a parent that's a single block node.
        def traverse(node):
            if node is None:
                return
            logging.debug(f"For node at depth {node.depth} with bounds {node.min_corner} - {node.max_corner}, block_ids is type: {type(node.block_ids)}")

            #if node.depth == self.max_depth: # and 2 not in node.block_ids and 4:  Leaf node #and 1 not in node.block_ids #viewerviewer
            #if node.depth == self.max_depth and 2 not in node.block_ids and 3 not in node.block_ids and 4 not in node.block_ids:
            if node.depth == self.max_depth and 1 not in node.block_ids:
                leaf_nodes.append(node)
            elif min_offset_level <= node.depth <= max_offset_level:  # Within depth range
                if len(set(node.block_ids)) == 1 and (node.depth == min_offset_level or not is_parent_single_block(node)):  
                    single_block_nodes.append(node)
            # Recurse for children
            for child in node.children:
                traverse(child)


        # Kick off the traversal
        traverse(self.root)

        return {
            "single_block_nodes": [(node, self.compute_node_size(node.depth)) for node in single_block_nodes],
            "leaf_nodes": [(node, self.compute_node_size(node.depth)) for node in leaf_nodes]
        }
    

    @staticmethod
    def get_color_based_on_block_id(block_id):
        # Map block IDs to colors
        # Example: assigning colors based on block_id
        color_mapping = {13: [1, 0, 0],  # Red for block_id 13
                         1: [0, 1, 0]}  # Green for block_id 1
        return color_mapping.get(block_id, [1, 1, 1])  # default to white if block_id not in mapping
    

    def visualize_octree_nodes(self):
        # Extract nodes from octree using the correct function
        node_data = self.get_nodes_for_visualization(min_offset_level=5, max_offset_level=10)
        logging.debug(f"Found {len(node_data['single_block_nodes'])} single block nodes and {len(node_data['leaf_nodes'])} leaf nodes")


        # Process single_block_nodes: outline cubes
        single_block_nodes = [entry[0] for entry in node_data["single_block_nodes"]]
        sizes_single_block = [entry[1] for entry in node_data["single_block_nodes"]]
        positions_single_block = np.array([node.center for node in single_block_nodes])
        
        # Use blockId to create colormap
        blockIds_single_block = np.array([node.block_ids[0] for node in single_block_nodes])
        
        # Create a discrete colormap
        unique_block_ids = np.unique(blockIds_single_block)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_block_ids)))
        cmap = ListedColormap(colors)
        
        # Map each unique block ID to a unique value in range 0 to number of unique IDs
        block_id_to_index = {id: i for i, id in enumerate(unique_block_ids)}

        # Map each block ID in the list to its corresponding index
        blockId_indices = np.array([block_id_to_index[id] for id in blockIds_single_block])
        
        # Use these indices to get colors from the colormap
        colors_single_block = cmap(blockId_indices)

        color_scalar = [0] * len(blockId_indices)
        print(color_scalar)

        ##viewerviewer

        glyphmapping.add_glyphs_to_visualiser(positions_single_block, sizes_single_block, blockId_indices, solid=True, line_width=10, cmap='tab10')
        #glyphmapping.add_glyphs_to_visualiser(positions_single_block, sizes_single_block, blockId_indices, solid=False, line_width=2, cmap='tab10')

        # Specify the values to include
        include_values = [0] # replace this with your list of values
        #include_values = [11,12] # replace this with your list of values


        # Create new lists including only elements where blockId_indices is in include_values
        new_blockId_indices = [id for i, id in enumerate(blockId_indices) if id in include_values]
        new_positions_single_block = [positions_single_block[i] for i, id in enumerate(blockId_indices) if id in include_values]
        new_sizes_single_block = [sizes_single_block[i] for i, id in enumerate(blockId_indices) if id in include_values]

        # Pass these new lists to the function
        glyphmapping.add_glyphs_to_visualiser(new_positions_single_block, new_sizes_single_block, new_blockId_indices, solid=False, line_width=2, cmap='tab10')



        
        # Process leaf_nodes: solid cubes
        leaf_nodes = [entry[0] for entry in node_data["leaf_nodes"]]
        sizes_leaf = [entry[1] for entry in node_data["leaf_nodes"]]
        positions_leaf = np.array([node.center for node in leaf_nodes])
        
        # Use dominant_color for colors (array of RGB colours, values between 0-1)
        dominant_colors = [node.calculate_dominant_attribute_and_colors()[1] for node in leaf_nodes]
        colors_leaf = np.array(dominant_colors)
        
        # Add glyphs to the visualiser with the dominant colors
        glyphmapping.add_voxels_with_rgba_to_visualiser(positions_leaf, sizes_leaf, colors_leaf)
        
        # Show the glyphs using pyvista
        glyphmapping.plot()


def tree_block_processing_complex(df):
    """
    Load and process the tree block data.

    Args:
        df (DataFrame): A DataFrame contains trees data with 'X', 'Y', 'Z' and 'Tree Size' columns.

    Returns:
        Tuple[np.ndarray, dict, list]: The points, attributes, and block IDs of the processed data.
    """ 

    global tree_block_count
    tree_block_count = {'small': 10, 'medium': 11, 'large': 12}
    
    def load_and_translate_tree_block_data(dataframe, tree_id, translation, tree_size):
        print(f"Processing tree id {tree_id}, size {tree_size}")
        block_data = dataframe[dataframe['Tree.ID'] == tree_id].copy()

        # Apply translation
        translation_x, translation_y, translation_z = translation
        block_data['x'] += translation_x
        block_data['y'] += translation_y
        block_data['z'] += translation_z

        block_data['BlockID'] = tree_block_count[tree_size]

        return block_data


    def get_tree_ids(tree_size, count):
        tree_id_ranges = {'small': range(0, 5), 'medium': range(5, 10), 'large': range(10, 17)}
        print(f'count is: {count}')
        return random.choices(tree_id_ranges[tree_size], k=count)


    def define_attributes(combined_data):
        attributes = combined_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
        return attributes


    csv_file = 'data/branchPredictions - full.csv'

    data = pd.read_csv(csv_file)
    print(f"Loaded data with shape {data.shape}")

    # Process block data for each tree size
    processed_data = []
    for tree_size in ['small', 'medium', 'large']:
        tree_size_data = df[df['Tree Size'] == tree_size]
        tree_count = len(tree_size_data)
        print (f'{tree_size} trees count is {tree_count}')
        tree_ids = get_tree_ids(tree_size, tree_count)
        print(f'Loading and processing {tree_ids} {tree_size} tree blocks...')

        # Process block data for each tree
        for tree_id, row in zip(tree_ids, tree_size_data.iterrows()):
            processed_block_data = load_and_translate_tree_block_data(data, tree_id, (row[1]['X'], row[1]['Y'], row[1]['Z']), tree_size)
            processed_data.append(processed_block_data)

    # Combine the block data
    combined_data = pd.concat(processed_data)

    print(combined_data)

    # Extract points, attributes, and block IDs
    points = combined_data[['x', 'y', 'z']].to_numpy()
    attributes = define_attributes(combined_data)
    block_ids = combined_data['BlockID'].tolist()

    return points, attributes, block_ids


def tree_block_processing(coordinates_list):
    """
    Load and process the tree block data.

    Args:
        coordinates_list (list): A list of tuples where each tuple contains x, y, z coordinates to translate tree blocks.

    Returns:
        Tuple[np.ndarray, dict, list]: The points, attributes, and block IDs of the processed data.
    """ 

    global tree_block_count
    tree_block_count = {}
    
    def load_and_translate_tree_block_data(dataframe, tree_id, translation):
        # Filter the data for the specific tree
        block_data = dataframe[dataframe['Tree.ID'] == tree_id].copy()

        print(translation)

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
        print(f'count is: {count}')
        return random.choices(range(1, 17), k=count)


    def define_attributes(combined_data):
        attributes = combined_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
        return attributes


    csv_file = 'data/branchPredictions - full.csv'

    data = pd.read_csv(csv_file)
    print(f"Loaded data with shape {data.shape}")

    # Get random tree IDs
    #print(f'Coordinates list: {coordinates_list}')
    tree_count = len(coordinates_list)
    print (f'coordinates_list is {coordinates_list}')
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


# Modify main function
def main2():
    # Process tree block data for a specified number of trees
    # Example usage:
    coordinates_list = [(5, 5, 0), (-5, -5, 0), (10, -10, 0)]

    tree_block_data = tree_block_processing(coordinates_list)


    if tree_block_data is None:
        print("Error: Tree block processing failed.")
        return
    
    points, attributes, block_ids = tree_block_data


    # Create Octree
    max_depth = 8    
    octree = CustomOctree(points, attributes, block_ids, max_depth)
    print(f"Created Octree with max depth {max_depth}")

    octree.visualize_octree_nodes()

def main():
    # Load the initial data
    coordinates_list = [(20, 20, 20), (-5, -5, 0), (-20, -20, -20)]

    #coordinates_list = [(5, 5, 0), (-5, -5, 0), (10, -10, 0)]
    points, attributes, block_ids = tree_block_processing(coordinates_list)

    # Create the octree
    octree = CustomOctree(points, attributes, block_ids, max_depth=8)
    print(f"Initial Octree created with bounds of {octree.root.min_corner} - {octree.root.max_corner}")
    print(f'octree root block_ids are: {type(octree.root.block_ids)}')



    # Load additional data
    new_coordinates_list = [(2, 2, 0), (1, 1, 0), (-5, -5, 0)]
    new_points, new_attributes, new_block_ids = tree_block_processing(new_coordinates_list)

    print(f'first couple of rows of new_block_ids: {new_block_ids[:2]}')
    # Add new block to the octree
    octree.add_block(new_points, new_attributes, new_block_ids)
    print("Octree updated with additional data")
    print(f'now octree root block_ids are: {type(octree.root.block_ids)}')



    octree.visualize_octree_nodes()


def main3():
    # Create dummy data
    data = pd.DataFrame({
        'X': [20, -5, -20, 20, -5, -20, 20, -5, -20],
        'Y': [20, -5, -20, 20, -5, -20, 20, -5, -20],
        'Z': [20, 0, -20, 20, 0, -20, 20, 0, -20],
        'Tree Size': ['small', 'medium', 'large', 'small', 'medium', 'large', 'small', 'medium', 'large']
    })

    points, attributes, block_ids = tree_block_processing_complex(data)

    # Create the octree
    octree = CustomOctree(points, attributes, block_ids, max_depth=8)
    print(f"Initial Octree created with bounds of {octree.root.min_corner} - {octree.root.max_corner}")
    print(f'octree root block_ids are: {type(octree.root.block_ids)}')

    # Create new dummy data for additional blocks
    new_data = pd.DataFrame({
        'X': [2, 1, -5, 2, 1, -5, 2, 1, -5],
        'Y': [2, 1, -5, 2, 1, -5, 2, 1, -5],
        'Z': [0, 0, 0, 0, 0, 0, 0, 0, 0],
        'Tree Size': ['small', 'medium', 'large', 'small', 'medium', 'large', 'small', 'medium', 'large']
    })

    new_points, new_attributes, new_block_ids = tree_block_processing_complex(new_data)

    print(f'first couple of rows of new_block_ids: {new_block_ids[:2]}')
    # Add new block to the octree
    octree.add_block(new_points, new_attributes, new_block_ids)
    print("Octree updated with additional data")
    print(f'now octree root block_ids are: {type(octree.root.block_ids)}')

    octree.visualize_octree_nodes()

def print_summary(data, name):
    if isinstance(data, np.ndarray):
        print(f'{name} - shape: {data.shape}, dtype: {data.dtype}, size: {data.size}')
    elif isinstance(data, list):
        print(f'{name} - length: {len(data)}, element type: {type(data[0])}')




def test():
    # Load the initial data
    coordinates_list = [(20, 20, 20), (-5, -5, 0), (-20, -20, -20)]
    df = pd.DataFrame({
        'X': [coord[0] for coord in coordinates_list],
        'Y': [coord[1] for coord in coordinates_list],
        'Z': [coord[2] for coord in coordinates_list],
        'Tree Size': ['small', 'medium', 'large']
    })

    # Generate points, attributes, and block IDs with both functions
    points1, attributes1, block_ids1 = tree_block_processing(coordinates_list)
    points2, attributes2, block_ids2 = tree_block_processing_complex(df)

    # Print summaries
    print_summary(points1, 'Points from tree_block_processing')
    print_summary(points2, 'Points from tree_block_processing_complex')
    print_summary(attributes1, 'Attributes from tree_block_processing')
    print_summary(attributes2, 'Attributes from tree_block_processing_complex')
    print_summary(block_ids1, 'Block IDs from tree_block_processing')
    print_summary(block_ids2, 'Block IDs from tree_block_processing_complex')



if __name__ == "__main__":
    main3()
    #test()

