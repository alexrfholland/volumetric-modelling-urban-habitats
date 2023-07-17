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
max_depth = 7  # Maximum depth of the Octree

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
grid, lines, voxelSize = Octree.update_visualization(vis, octree, max_depth, 4, 5)

print(voxelSize)

# Run visualization
"""vis.run()
vis.destroy_window()"""

import pyvista as pv

# Conversion function from VoxelGrid to Trianglegrid
def voxel_grid_to_mesh(voxel_grid_3d: o3d.geometry.VoxelGrid, voxel_size: float):
        voxels=voxel_grid_3d.get_voxels()
        vox_mesh=o3d.geometry.TriangleMesh()

        for v in voxels:
            cube=o3d.geometry.TriangleMesh.create_box(width=1, height=1,
            depth=1)
            cube.paint_uniform_color(v.color)
            cube.translate(v.grid_index, relative=False)
            vox_mesh+=cube

        # Scale by the voxel size
        vox_mesh.scale(voxel_size, center=(0,0,0))

        # Merge close vertices to ensure the mesh is manifold
        #vox_mesh.merge_close_vertices(0.0000001)

        # Compute vertex normals for shading
        vox_mesh.compute_vertex_normals()

        return vox_mesh

# Conversion functions
def open3d_grid_to_pyvista_polydata_with_vertex_colors(grid):
    # Fetch the vertices, faces and colors from the Open3D grid
    vertices = np.asarray(grid.vertices)
    triangles = np.asarray(grid.triangles)
    colors = np.asarray(grid.vertex_colors)

    # Convert Open3D's faces format to PyVista's faces format
    faces = np.empty((triangles.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles

    # Create the PyVista grid (PolyData)
    pv_grid = pv.PolyData(vertices, faces.flatten())

    # Add vertex colors to grid
    pv_grid.point_data["Colors"] = colors  # Assign colors to points

    return pv_grid

def open3d_lineset_to_pyvista_lineset(lineset):
    # Fetch the lines and colors from the Open3D lineset
    lines = np.asarray(lineset.lines)
    colors = np.asarray(lineset.colors)

    # Convert the points from Vector3dVector to a numpy array
    points = np.asarray(lineset.points)

    # Calculate number of lines and number of points
    n_lines = lines.shape[0]
    n_points = points.shape[0]

    # Initialize the cells array
    cells = np.empty((n_lines, 3), dtype=int)

    # First column is the number of points per line
    cells[:, 0] = 2

    # Remaining columns are indices into the points array
    cells[:, 1:] = lines

    # Initialize the colors array
    cell_colors = np.empty((n_lines, 3), dtype=float)

    # Each row is the RGB color for a line
    cell_colors = colors

    # Create the PyVista lineset (PolyData)
    pv_lineset = pv.PolyData(points, cells)

    # Add line colors to lineset
    pv_lineset.cell_data["Colors"] = cell_colors  # Assign colors to cells (lines)

    return pv_lineset


def create_random_pyvista_line():
    # Generate random start and end points for the line
    start = np.random.rand(3) * 10
    end = np.random.rand(3) * 10

    # Create a single line cell
    cells = np.array([2, 0, 1])

    # Create the points array
    points = np.vstack((start, end))

    # Create the PyVista lineset (PolyData)
    pv_lineset = pv.PolyData(points, cells)

    return pv_lineset






# Convert Open3D VoxelGrid to Open3D Trianglegrid
triangle_grid = voxel_grid_to_mesh(grid, voxelSize)
print('created mesh grid')
# Convert Open3D grid and linesets to PyVista objects
pv_grid = open3d_grid_to_pyvista_polydata_with_vertex_colors(triangle_grid)
print('converted mesh into pyvista mesh')

pv_linesets = [open3d_lineset_to_pyvista_lineset(lineset) for lineset in lines]
print('converted lineset into pyvista lineset')



# Create PyVista plotter
plotter = pv.Plotter()

# Add the mesh to the plotter
plotter.add_mesh(pv_grid, scalars="Colors", lighting=True, rgb=True)

# Create a random line
pv_line = create_random_pyvista_line()


# Add each line in the lineset to the plotter
for pv_line in pv_linesets:
    plotter.add_mesh(pv_line, scalars="Colors", lighting=True, rgb=True)

# Enable Eye Dome Lighting (EDL)
plotter.enable_eye_dome_lighting()

light2 = pv.Light(light_type='cameralight', intensity=10)
plotter.add_light(light2)

# Set camera position to have a 45 degree azimuthal angle
# You might need to adjust the position values depending on your specific use case
camera_pos = (1,1,1)
focal_point = (0,0,0)
view_up = (0,0,1)  # This is the Y direction
plotter.camera_position = [camera_pos, focal_point, view_up]
# Set near clipping distance



# Set parallel projection
#plotter.camera.parallel_projection = True

plotter.camera.position = [x * 3 for x in plotter.camera.position]

plotter.reset_camera()



# Show the plotter
plotter.show()



print("shown the tree block")