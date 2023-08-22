import open3d as o3d
import numpy as np
import pandas as pd
import random



"""To update an Octree generation script such that each node in the Octree stores its dominant attribute and corresponding color information. The dominant attribute and color will be determined based on the majority attribute amongst the leaf nodes in that node.

Key steps:

Modify the OctreeNode class to include additional attributes: dominantAttribute and dominantColor.
Upon initialization of each node, determine the dominant attribute based on the distribution of attributes within that node.
Use the determined dominant attribute to calculate and store the dominant color in each node.
Modify the split function to ensure the new attributes are determined and stored correctly during the creation of child nodes.
Intergrate the Octree script to ensure the dominant attribute and color are being calculated and stored correctly.

"""
class OctreeNode:
    def __init__(self, min_corner, max_corner, points, attributes, depth=0):
        self.children = []  # child nodes
        self.depth = depth  # depth of this node in the tree
        self.min_corner = np.array(min_corner)  # minimum corner of bounding box
        self.max_corner = np.array(max_corner)  # maximum corner of bounding box
        self.points = points
        self.attributes = attributes
        self.dominantAttribute = self.calculate_dominant_attribute()
        self.dominantColor = self.calculate_color_based_on_attribute()

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

    def calculate_dominant_attribute(self):
        # Logic to determine dominant attribute based on node's attributes
        attribute_counts = {"isDeadOnly": 0, "isLateralOnly": 0, "isBoth": 0, "isNeither": 0}
        for attribute in self.attributes:
            for key in attribute_counts:
                if attribute[key]:
                    attribute_counts[key] += 1
                    break

        # Set dominant attribute based on highest count
        max_count_attr = max(attribute_counts, key=attribute_counts.get)
        return max_count_attr

    def calculate_color_based_on_attribute(self):
        # Set color based on dominant attribute
        if self.dominantAttribute == "isDeadOnly":
            return [1, 0, 0]  # Red
        if self.dominantAttribute == "isLateralOnly":
            return [0, 1, 0]  # Green
        if self.dominantAttribute == "isBoth":
            return [0, 0, 1]  # Blue
        if self.dominantAttribute == "isNeither":
            return [1, 1, 0]  # Yellow

        return [1, 1, 1]  # White for the default case



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

def visualize_octree(data, attributes, max_depth=4, visualize_offset=2):
    # Convert pandas DataFrame to numpy array
    points = data[['x', 'y', 'z']].values
    # Create attribute list from DataFrame
    attributes_list = attributes.to_dict('records')

    # Find min and max corners of point cloud
    min_corner = np.min(points, axis=0)
    max_corner = np.max(points, axis=0)

    # Create octree
    octree = CustomOctree(points, attributes_list, min_corner, max_corner, max_depth)

    # Retrieve geometries for visualization
    boxes = []
    voxel_points = []
    voxel_colors = []
    octree.get_geometries(octree.root, boxes, voxel_points, voxel_colors, max_depth, visualize_offset)

    # Convert voxel points and colors to point cloud for visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(voxel_points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

    # Visualize
    o3d.visualization.draw_geometries([pcd, *boxes])


# Functions
def get_octree_geometries(octree, max_depth, visualize_offset):
    voxel_points = []  # List to hold the voxel points
    voxel_colors = []  # List to hold the voxel colors
    bounding_boxes = [] # List to hold the bounding boxes
    octree.get_geometries(octree.root, bounding_boxes, voxel_points, voxel_colors, max_depth=max_depth, visualize_offset=visualize_offset)

    # Calculate voxel size
    voxel_size = np.min(octree.root.max_corner - octree.root.min_corner) / (2 ** max_depth)

    # Initialize a triangle mesh to hold the voxel cubes
    vox_mesh = o3d.geometry.TriangleMesh()

    # Generate the voxel cubes
    for i in range(len(voxel_points)):
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.paint_uniform_color(voxel_colors[i])
        cube.translate(voxel_points[i], relative=False)
        vox_mesh += cube

    # Correct the position and scale of the voxel mesh
    vox_mesh.translate([0.5, 0.5, 0.5], relative=True)
    vox_mesh.scale(voxel_size, [0, 0, 0])
    vox_mesh.translate(octree.root.min_corner, relative=True)

    return vox_mesh, bounding_boxes

# Other functions such as increase_visualize_offset, decrease_visualize_offset and update_visualization remain the same...
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


# Load data from CSV
csv_file = 'data/branchPredictions - full.csv'
data = pd.read_csv(csv_file)

# Landscape size
landscape_size = [100, 100, 100]  # Size of landscape in meters

# Get list of unique tree IDs
tree_ids = data['Tree.ID'].unique()

# Randomly distribute trees over the landscape
points = []
attributes = []
for tree_id in tree_ids:
    tree_data = data[data['Tree.ID'] == tree_id]
    offset_x, offset_y = random.uniform(0, landscape_size[0]), random.uniform(0, landscape_size[1])
    tree_points = tree_data[['x', 'y', 'z']].to_numpy() + np.array([offset_x, offset_y, 0])
    tree_attributes = tree_data[['dominantAttribute', 'dominantColor']].to_dict('records')
    points.extend(tree_points)
    attributes.extend(tree_attributes)

# Convert lists to numpy arrays for use in octree
points = np.array(points)

# Create a grid of points for the ground
ground_points = np.mgrid[0:100:1, 0:100:1].reshape(2,-1).T
ground_points = np.concatenate([ground_points, np.zeros((ground_points.shape[0], 1))], axis=1)  # Set z=0 for all ground points

# Create attributes for the ground points
ground_attributes = [{'dominantAttribute': 'ground', 'dominantColor': [0, 1, 0]} for _ in range(len(ground_points))]

# Append ground points and attributes to your points and attributes numpy array
points = np.concatenate([points, ground_points])
attributes += ground_attributes

# Build custom octree
min_corner = [0, 0, 0]  # Minimum corner of landscape
max_corner = landscape_size  # Maximum corner of landscape
max_depth = 7
octree = CustomOctree(points, attributes, min_corner, max_corner, max_depth=max_depth)


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