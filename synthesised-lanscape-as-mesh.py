""" 
Aim: The objective is to create a procedural landscape visualization using an Octree data structure. The Octree will efficiently organize the landscape data and allow for intuitive visualization of the terrain and its hierarchical features.

Load Data:
    Load the landscape data from a CSV file.
    This data includes numerical descriptions of trees and other attributes representing habitat resources within them.
Initialise Landscape:
    Set up the base landscape using the loaded data.
    Distribute trees randomly across the landscape.
Process Landscape:
    Organize the landscape data into an Octree structure.
    Assign attributes to each node in the Octree. This includes habitat features and any other relevant characteristics.
    Higher-level nodes inherit the most common attribute of their child nodes.
Visualize:
    Represent Octree nodes at the maximum depth as filled cubes. To do this efficiently:
    Create a point cloud from the maximum depth nodes.
    Transform this point cloud into a Voxel grid.
    Convert the Voxel grid into a mesh and apply shading for 3D visualization.
    Visualize Octree nodes at other depths as bounding boxes. Display a varying number of these non-filled boxes according to the custom Octree node depth.
    Convert these bounding boxes into mesh lines, enabling customization of the width.

 """

import open3d as o3d
import numpy as np
import pandas as pd
import random

class OctreeNode:
    def __init__(self, min_corner, max_corner, points, attributes, depth=0):
        self.children = []  # child nodes
        self.depth = depth  # depth of this node in the tree
        self.min_corner = np.array(min_corner)  # minimum corner of bounding box
        self.max_corner = np.array(max_corner)  # maximum corner of bounding box
        self.points = points
        self.attributes = attributes

    def split(self, max_depth):
        if self.depth < max_depth:
            # Calculate new bounds
            mid = (self.min_corner + self.max_corner) / 2
            bounds = [self.min_corner, mid, self.max_corner]

            # Recursively split into 8 children
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        new_min = [bounds[x][l] for x, l in zip([i, j, k], range(3))]
                        new_max = [bounds[x][l] for x, l in zip([i+1, j+1, k+1], range(3))]
                        # Filter points within new bounds
                        in_range = np.all((new_min <= self.points) & (self.points <= new_max), axis=1)
                        new_points = self.points[in_range]
                        new_attributes = [self.attributes[idx] for idx, val in enumerate(in_range) if val]
                        if len(new_points) > 0:
                            child = OctreeNode(new_min, new_max, new_points, new_attributes, self.depth + 1)
                            child.split(max_depth)
                            self.children.append(child)


class CustomOctree:
    def __init__(self, points, attributes, min_corner, max_corner, max_depth):
        self.root = OctreeNode(min_corner, max_corner, points, attributes)
        self.root.split(max_depth)

    def get_geometries(self, node, boxes, voxel_points, voxel_colors, max_depth, visualize_offset):
        if node is None:
            return

        # Determine color based on node's attributes
        color = self.get_color_based_on_attributes(node.attributes)
        center = (node.min_corner + node.max_corner) / 2
        extent = node.max_corner - node.min_corner
        R = np.eye(3)  # Identity matrix for rotation (no rotation)

        # If within visualize_offset from max_depth, add voxel point
        if max_depth - node.depth <= visualize_offset:
            voxel_points.append(center)
            voxel_colors.append(color)

            # If not at max depth, add oriented bounding box
            if node.depth < max_depth:
                box = o3d.geometry.OrientedBoundingBox(center, R, extent)
                box.color = color
                boxes.append(box)

        # Recurse for children
        for child in node.children:
            self.get_geometries(child, boxes, voxel_points, voxel_colors, max_depth, visualize_offset)


    @staticmethod
    def get_color_based_on_attributes(attributes_list):
        # Logic to determine color based on node's attributes
        counts = {"isDeadOnly": 0, "isLateralOnly": 0, "isBoth": 0, "isNeither": 0}
        for attributes in attributes_list:
            for key in counts:
                if attributes[key]:
                    counts[key] += 1
                    break
        
        # Set color based on attributes with the highest count
        max_count_attr = max(counts, key=counts.get)
        if max_count_attr == "isDeadOnly":
            return [1, 0, 0]  # Red
        if max_count_attr == "isLateralOnly":
            return [0, 1, 0]  # Green
        if max_count_attr == "isBoth":
            return [0, 0, 1]  # Blue
        if max_count_attr == "isNeither":
            return [1, 1, 0]  # Yellow
        return [1, 1, 1]  # White for the default case



# Functions
def get_octree_geometries(octree, max_depth, visualize_offset):
    voxel_points = []  # List to hold the voxel points
    voxel_colors = []  # List to hold the voxel colors
    bounding_boxes = [] # List to hold the bounding boxes
    octree.get_geometries(octree.root, bounding_boxes, voxel_points, voxel_colors, max_depth=max_depth, visualize_offset=visualize_offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(voxel_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

    # Calculate voxel size
    voxel_size = np.min(octree.root.max_corner - octree.root.min_corner) / (2 ** max_depth)

    # Create a VoxelGrid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

    # Extract filled voxels
    voxels = voxel_grid.get_voxels()

    # Initialize a triangle mesh to hold the voxel cubes
    vox_mesh = o3d.geometry.TriangleMesh()

    # Generate the voxel cubes
    for v in voxels:
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(v.color)
        cube.translate(v.grid_index, relative=False)
        vox_mesh += cube

    # Correct the position and scale of the voxel mesh
    vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
    vox_mesh.scale(voxel_size, [0, 0, 0])
    vox_mesh.translate(voxel_grid.origin, relative=True)

    return vox_mesh, bounding_boxes

def update_visualization(vis, octree, max_depth, visualize_offset):
    voxel_mesh, bounding_boxes = get_octree_geometries(octree, max_depth, visualize_offset)

    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.clear_geometries()
    vis.add_geometry(voxel_mesh)
    for box in bounding_boxes:
        vis.add_geometry(box)

    # Set rotation center to the center of the landscape
    center = np.array([50, 50, 50]) # now that landscape is a cube
    vis.get_view_control().set_lookat(center)

    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)


    
    
    # Return voxel_mesh so it can be saved later
    return voxel_mesh



def increase_visualize_offset(vis, action, mods):
    visualize_offset[0] += 1
    update_visualization(vis, octree, max_depth, visualize_offset[0])


def decrease_visualize_offset(vis, action, mods):
    visualize_offset[0] = max(0, visualize_offset[0] - 1)
    update_visualization(vis, octree, max_depth, visualize_offset[0])


def initialize_environment_from_csv(octree, csv_file):
    # Read the CSV file using pandas
    data = pd.read_csv(csv_file)

    # Extract unique tree IDs
    tree_ids = data['Tree.ID'].unique()

    # Loop through each unique tree ID
    for tree_id in tree_ids:
        # Filter data for the current tree_id
        tree_data = data[data['Tree.ID'] == tree_id]
        
        # Extract tree size from the first row of filtered data
        tree_size = tree_data.iloc[0]['Tree.size']
        
        # Randomly position the tree
        offset_x, offset_y = random.uniform(0, octree.root.max_corner[0]), random.uniform(0, octree.root.max_corner[1])
        
        # Extract points and attributes
        tree_points = tree_data[['x', 'y', 'z']].to_numpy() + np.array([offset_x, offset_y, 0])
        tree_attributes = tree_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
        
        # Insert points into octree and associate them with the current block (tree_id)
        for point, attribute in zip(tree_points, tree_attributes):
            octree.insert_point(point, attribute, tree_id)
        
        # Assign block info to the octree
        block_info = {'size': tree_size}
        octree.add_block_info(tree_id, block_info)

# Example Usage:
max_depth = 7
min_corner = [0, 0, 0]
max_corner = [100, 100, 100]
octree = CustomOctree(min_corner, max_corner, max_depth)
initialize_environment_from_csv(octree, 'data/branchPredictions - full.csv')


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

# Set initial view parameters
ctr = vis.get_view_control()
ctr.set_front([0, -1, -1.5])
ctr.set_lookat([50, 50, 5])  # center of the landscape
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.1)


# Access render options
render_option = vis.get_render_option()

# Set shading properties
render_option.point_size = 1.0  # Set point size
render_option.line_width = 100  # Set line width

"""# Function to get and transform the mesh
def get_transformed_mesh(octree, max_depth, visualize_offset):
    voxel_mesh, _ = get_octree_geometries(octree, max_depth, visualize_offset)
    
    # Calculate voxel size
    voxel_size = np.min(octree.root.max_corner - octree.root.min_corner) / (2 ** max_depth)

    # Translate relatively by half the voxel unit
    voxel_mesh.translate([0.5, 0.5, 0.5], relative=True)
    
    # Scale by the voxel size
    voxel_mesh.scale(voxel_size, [0, 0, 0])
    
    # Translate to original position
    voxel_mesh.translate(octree.root.min_corner, relative=True)
    
    # Merge close vertices
    voxel_mesh.merge_close_vertices(0.0000001)
    
    return voxel_mesh

# Get the transformed mesh
voxel_mesh = get_transformed_mesh(octree, max_depth, visualize_offset[0])

# Save the mesh to a file
output_filename = "output_mesh.ply"
o3d.io.write_triangle_mesh(output_filename, voxel_mesh)
print(f"Mesh saved as {output_filename}")

# Optionally rotate and save if required
T = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, -1, 0, 0],[0, 0, 0, 1]])
rotated_output_filename = "rotated_output_mesh.ply"
o3d.io.write_triangle_mesh(rotated_output_filename, voxel_mesh.transform(T))
print(f"Rotated mesh saved as {rotated_output_filename}")"""

# Start the visualization loop
vis.run()
vis.destroy_window()