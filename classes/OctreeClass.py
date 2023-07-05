# Save this as class.py

import open3d as o3d
import numpy as np
import pandas as pd


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
    

class OctreeVisualizer:
    def __init__(self, csv_file, tree_id, max_depth):
        data = pd.read_csv(csv_file)
        tree_data = data[data['Tree.ID'] == tree_id]
        tree_data = tree_data.rename(columns={'y': 'z', 'z': 'y'})
        points = tree_data[['x', 'y', 'z']].to_numpy()
        attributes = tree_data[['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']].to_dict('records')

        min_corner = np.min(points, axis=0)
        max_corner = np.max(points, axis=0)
        self.octree = CustomOctree(min_corner, max_corner, points, attributes, max_depth=max_depth)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def update_visualization(self, max_depth, min_offset_level, max_offset_level):
        voxel_grid, bounding_boxes = self.octree.getMeshesfromVoxels(max_depth, min_offset_level, max_offset_level)

        view_params = self.vis.get_view_control().convert_to_pinhole_camera_parameters()

        self.vis.clear_geometries()
        self.vis.add_geometry(voxel_grid)

        # Convert bounding boxes to line sets before adding them to the visualizer
        converter = BoundingBoxToLineSet(bounding_boxes, line_width=100)
        linesets = converter.to_linesets()

        # Adding linesets to the visualizer
        for lineset_info in linesets:
            self.vis.add_geometry(lineset_info['geometry'])

        self.vis.get_view_control().set_lookat(self.octree.root.center)
        self.vis.get_view_control().convert_from_pinhole_camera_parameters(view_params)


    def run(self):
        self.vis.run()
        self.vis.destroy_window()


# This block ensures that the following code runs only when the script is executed directly (not imported as a module)
if __name__ == "__main__":
    csv_file = 'data/branchPredictions - full.csv'
    tree_id = 13
    max_depth = 5

    # Initialize the visualizer
    visualizer = OctreeVisualizer(csv_file, tree_id, max_depth)

    # Update the visualization (specify the range of levels of bounding boxes to display)
    visualizer.update_visualization(max_depth, 1, 2)

    # Run the visualizer
    visualizer.run() 