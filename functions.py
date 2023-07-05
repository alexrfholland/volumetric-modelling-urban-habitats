# functions.py
import numpy as np
import open3d as o3d
from classes import OctreeNode, CustomOctree

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

def increase_visualize_offset(vis, action, mods, visualize_offset):
    visualize_offset[0] += 1
    update_visualization(vis, octree, max_depth, visualize_offset[0])

def decrease_visualize_offset(vis, action, mods, visualize_offset):
    visualize_offset[0] = max(0, visualize_offset[0] - 1)
    update_visualization(vis, octree, max_depth, visualize_offset[0])
