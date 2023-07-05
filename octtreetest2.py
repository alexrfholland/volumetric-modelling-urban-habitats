'''

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





import open3d as o3d
import numpy as np
import pandas as pd

from BoundingBoxToLineSet import BoundingBoxToLineSet



##this one is using draw (which works with  line widths)

class OctreeNode:
    def __init__(self, min_corner, max_corner, points, attributes, depth=0):
        self.children = []  # child nodes
        self.depth = depth  # depth of this node in the tree
        self.min_corner = np.array(min_corner)  # minimum corner of bounding box
        self.max_corner = np.array(max_corner)  # maximum corner of bounding box
        self.points = points
        self.attributes = attributes
        self.dominant_attribute, self.dominant_color = self.calculate_dominant_attribute_and_color()
        self.get_geos()  # Ensure bounding box and center are computed

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
    def __init__(self, min_corner, max_corner, points, attributes, max_depth):
        self.root = OctreeNode(min_corner, max_corner, points, attributes)
        self.root.split(max_depth)

    @staticmethod
    def get_color_based_on_attribute(attribute):
        # Define the color mapping
        color_mapping = {"isDeadOnly": [1, 0, 0],
                         "isLateralOnly": [0, 1, 0],
                         "isBoth": [0, 0, 1],
                         "isNeither": [1, 1, 1]}
        return color_mapping[attribute]

    def get_all_bounding_boxes(self, node, boxes):
        if node is None:
            return
        boxes.append(node.bounding_box)

        # Recurse for children
        for child in node.children:
            self.get_all_bounding_boxes(child, boxes)

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

    def getMeshesfromVoxels(self, max_depth):
        voxel_points = []  
        voxel_colors = []  
        bounding_boxes = []

        #gather all the points and colors from the octree
        self.get_voxels_from_leaf_nodes(self.root, voxel_points, voxel_colors, max_depth)
        #gather all the bounding boxes from the octree
        self.get_all_bounding_boxes(self.root, bounding_boxes)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        voxel_size = np.min(self.root.max_corner - self.root.min_corner) / (2 ** max_depth)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

        return voxel_grid, bounding_boxes



def update_visualization(octree, max_depth):
    voxel_grid, bounding_boxes = octree.getMeshesfromVoxels(max_depth)

    # Convert bounding boxes to line sets
    converter = BoundingBoxToLineSet(bounding_boxes, line_width=100)  # Initialize with bounding_boxes
    linesets = converter.to_linesets()

    geometries = []
    geometries.append(voxel_grid)

    # Adding linesets to the geometries list
    for lineset_info in linesets:
        geometries.append(lineset_info['geometry'])

    o3d.visualization.draw_geometries(geometries)

csv_file = 'data/branchPredictions - full.csv'
data = pd.read_csv(csv_file)
tree13_data = data[data['Tree.ID'] == 13]
tree13_data = tree13_data.rename(columns={'y': 'z', 'z': 'y'})
points = tree13_data[['x', 'y', 'z']].to_numpy()
attributes = tree13_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')

center = np.mean(points, axis=0)
max_depth = 3

min_corner = np.min(points, axis=0)
max_corner = np.max(points, axis=0)
octree = CustomOctree(min_corner, max_corner, points, attributes, max_depth=max_depth)

update_visualization(octree, max_depth)




##this one is using the visualizer
"""
class OctreeNode:
    def __init__(self, min_corner, max_corner, points, attributes, depth=0):
        self.children = []  # child nodes
        self.depth = depth  # depth of this node in the tree
        self.min_corner = np.array(min_corner)  # minimum corner of bounding box
        self.max_corner = np.array(max_corner)  # maximum corner of bounding box
        self.points = points
        self.attributes = attributes
        self.dominant_attribute, self.dominant_color = self.calculate_dominant_attribute_and_color()
        self.get_geos()  # Ensure bounding box and center are computed

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
    def __init__(self, min_corner, max_corner, points, attributes, max_depth):
        self.root = OctreeNode(min_corner, max_corner, points, attributes)
        self.root.split(max_depth)

    @staticmethod
    def get_color_based_on_attribute(attribute):
        # Define the color mapping
        color_mapping = {"isDeadOnly": [1, 0, 0],
                         "isLateralOnly": [0, 1, 0],
                         "isBoth": [0, 0, 1],
                         "isNeither": [1, 1, 1]}
        return color_mapping[attribute]

    def get_all_bounding_boxes(self, node, boxes):
        if node is None:
            return
        boxes.append(node.bounding_box)

        # Recurse for children
        for child in node.children:
            self.get_all_bounding_boxes(child, boxes)

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

    def getMeshesfromVoxels(self, max_depth):
        voxel_points = []  
        voxel_colors = []  
        bounding_boxes = []

        #gather all the points and colors from the octree
        self.get_voxels_from_leaf_nodes(self.root, voxel_points, voxel_colors, max_depth)
        #gather all the bounding boxes from the octree
        self.get_all_bounding_boxes(self.root, bounding_boxes)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(voxel_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(voxel_colors))

        voxel_size = np.min(self.root.max_corner - self.root.min_corner) / (2 ** max_depth)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)

        return voxel_grid, bounding_boxes


def update_visualization(vis, octree, max_depth):
    voxel_grid, bounding_boxes = octree.getMeshesfromVoxels(max_depth)

    view_params = vis.get_view_control().convert_to_pinhole_camera_parameters()

    vis.clear_geometries()
    vis.add_geometry(voxel_grid)

    # Convert bounding boxes to line sets before adding them to the visualizer
    converter = BoundingBoxToLineSet(bounding_boxes, line_width=100)  # Initialize with bounding_boxes
    linesets = converter.to_linesets()

    # Adding linesets to the visualizer
    for lineset_info in linesets:
        # Assuming the geometry is in 'geometry' key and material in 'material' key of lineset_info
        vis.add_geometry(lineset_info['geometry'])

    vis.get_view_control().set_lookat(octree.root.center)
    vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)

csv_file = 'data/branchPredictions - full.csv'
data = pd.read_csv(csv_file)
tree13_data = data[data['Tree.ID'] == 13]
tree13_data = tree13_data.rename(columns={'y': 'z', 'z': 'y'})
points = tree13_data[['x', 'y', 'z']].to_numpy()
attributes = tree13_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')

center = np.mean(points, axis=0)
max_depth = 5

min_corner = np.min(points, axis=0)
max_corner = np.max(points, axis=0)
octree = CustomOctree(min_corner, max_corner, points, attributes, max_depth=max_depth)


vis = o3d.visualization.Visualizer()
vis.create_window()

update_visualization(vis, octree, max_depth)

vis.run()
vis.destroy_window()
"""

