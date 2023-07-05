
# Save this as class.py

"""[PROMPT]
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