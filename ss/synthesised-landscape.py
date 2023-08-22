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

    return voxel_grid, bounding_boxes


def update_visualization(vis, octree, max_depth, visualize_offset):
    voxel_grid, bounding_boxes = get_octree_geometries(octree, max_depth, visualize_offset)

    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.clear_geometries()
    vis.add_geometry(voxel_grid)
    for box in bounding_boxes:
        vis.add_geometry(box)

    # Set rotation center to the center of the landscape
    center = np.array([50, 50, 50]) # now that landscape is a cube
    vis.get_view_control().set_lookat(center)

    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)


    



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
landscape_size = [100, 100]  # Size of landscape in meters




# Get list of unique tree IDs
tree_ids = data['Tree.ID'].unique()

# Randomly distribute trees over the landscape
points = []
attributes = []
for tree_id in tree_ids:
    tree_data = data[data['Tree.ID'] == tree_id]
    offset_x, offset_y = random.uniform(0, landscape_size[0]), random.uniform(0, landscape_size[1])
    tree_points = tree_data[['x', 'y', 'z']].to_numpy() + np.array([offset_x, offset_y, 0])
    tree_attributes = tree_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')
    points.extend(tree_points)
    attributes.extend(tree_attributes)

# Convert lists to numpy arrays for use in octree
points = np.array(points)

# Create a grid of points for the ground
ground_points = np.mgrid[0:100:1, 0:100:1].reshape(2,-1).T
ground_points = np.concatenate([ground_points, np.zeros((ground_points.shape[0], 1))], axis=1)  # Set z=0 for all ground points

# Create attributes for the ground points
ground_attributes = [{'isDeadOnly': False, 'isLateralOnly': False, 'isBoth': False, 'isNeither': True} for _ in range(len(ground_points))]

# Append ground points and attributes to your points and attributes numpy array
points = np.concatenate([points, ground_points])
attributes += ground_attributes



# Build custom octree
min_corner = [0, 0, 0]  # Minimum corner of landscape
# max_corner of landscape
max_corner = [100, 100, 100]

#max_corner = landscape_size + [np.max(points[:, 2])]  # Maximum corner of landscape
max_depth = 6
octree = CustomOctree(points, attributes, min_corner, max_corner, max_depth=max_depth)



# Create visualizer
visualize_offset = [5]
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Register key callbacks
vis.register_key_callback(ord("["), decrease_visualize_offset)
vis.register_key_callback(ord("]"), increase_visualize_offset)

# Initial update
update_visualization(vis, octree, max_depth, visualize_offset[0])

# Set initial view parameters
ctr = vis.get_view_control()
ctr.set_front([0, -1, -1.5])
ctr.set_lookat([50, 50, 5])  # center of the landscape
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.1)

# Start the visualization loop
vis.run()
vis.destroy_window()

