"""
CHAT GPT PLEASE READ

We aim to construct a 3D volumetric model of urban environments using an Octree data structure. 

Initially, we perform a large-scale spatial analysis to pinpoint characteristic sites and challenges, and to unveil opportunities for interventions.

We then select representative sites from these categories for detailed study, using aerial LiDAR to capture volumetric data. 

Our platform convert these lidar scans into numerical descriptions of urban spaces using an Octree data structure. 
We aggregate Octree nodes into logical sets named Blocks, such as Tree Blocks, Building Blocks, and Artificial Habitat Blocks,
where each Block signifies a specific object or area within the environment.

For the broader urban environment, we dissect these sites into distinct 'Building Blocks,' 
which encompass the ground, buildings, trees, street furniture, and other features.
Within buildings, we conduct a nuanced analysis to identify walls and roofs that offer high potential for the addition of artificial structures, such as habitat roofs. 


Data Description:
The data is from an aerial LiDAR scan of an urban street and park, consisting of around 150,000 points in a 100x100x sample.
The data will be saved in a PLY file.
Each point has the following attributes: X, Y, Z (positions), Rf, Gf, Bf (colors in the range 0-1), Illuminance_(PCV) (shading information, grayscale, range 0-1), element_type (integers: 0 for 'tree', 1 for 'building', 2 for 'grass', 3 for 'street-furniture', 4 for 'ground'), Composite, Dip_(degrees), Dip_direction_(degrees), R, G, B, Nx, Ny, Nz.

Key Steps and Elements of the Code:

Step 1: Data Import - Use Open3D to import the PLY file.

Step 2: further_processing - Use the Dip_(degrees) information to classify points based on the horizontality: 'flat', 'angled', or 'vertical'.

Step 3: Build Query Structure - Create a function to categorize points based on combinations of attributes, for example, "element_type == 1 and horizontality == 1". This function will assign a new attribute called 'category' to each point, based on the combination of attributes specified in the query.

Step 4: Color Enhancement - Define a function that takes the original colors and enhances them using the Illuminance_(PCV) information.

Step 5: Visualization - Visualize the point cloud using Open3D, incorporating color enhancement and displaying different colors based on the 'category' attribute assigned in Step 3.

"""

import open3d as o3d
import pandas as pd
import numpy as np

from colorMaps import Colormap  # replace 'colormap_module' with the actual name of your colormap python file


def enhance_colors_with_illuminance(colors, illuminance):
    return colors * illuminance[:, np.newaxis]**3

def assign_horizontality(dip_degrees):
    horizontality = np.empty_like(dip_degrees, dtype=int)
    horizontality[dip_degrees < 6] = 0  # flat
    horizontality[(dip_degrees >= 6) & (dip_degrees < 85)] = 1  # angled
    horizontality[dip_degrees >= 85] = 2  # vertical
    return horizontality

def classify_points_based_on_queries(data, queries):
    category = np.zeros(len(data), dtype=int)
    for i, query in enumerate(queries):
        filtered_indices = data.query(query).index
        category[filtered_indices] = i + 1
    return category

# Load Data
data = pd.read_parquet('data/sites/park.parquet')
data.rename(columns={'//X': 'X'}, inplace=True)

# Assign horizontality
data['horizontality'] = assign_horizontality(data['Dip (degrees)'].values)
"""
# Queries
queries = [
    "element_type == 0",
    "element_type == 1 and horizontality == 0",
    "element_type == 1 and horizontality == 1",
    "element_type == 1 and horizontality == 2",
    "element_type == 2",
    "element_type == 3",
    "element_type == 4"
]"""


# Queries
queries = [
    "element_type == 1 and horizontality == 0",
    "element_type == 1 and horizontality == 1",
    "element_type == 1 and horizontality == 2",
    "element_type == 2",
    "element_type == 3",
    "element_type == 4"
]


# Classify Points
data['category'] = classify_points_based_on_queries(data, queries)

# Create a colormap for visualization
category_colors = [
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [1, 0, 1],  # magenta
    [0, 1, 1],  # cyan
    [0.5, 0.5, 0.5]  # grey
]

# Assign colors based on category
#colors = np.array([category_colors[cat-1] for cat in data['category']])

# Create an instance of Colormap class
cm = Colormap()

# Specify the colormap name
colormap_name = 'batlowS'  # replace 'your_colormap_name' with the actual name of your colormap

import random

# Get the colors from the colormap and shuffle them
category_colors = cm.get_categorical_colors(len(queries) + 1, colormap_name)
#random.shuffle(category_colors)
category_colors.pop(0)

# Assign colors based on category
colors = np.array([category_colors[cat-1] for cat in data['category']])

illuminance = np.array(data['Illuminance (PCV)'])
enhanced_colors = enhance_colors_with_illuminance(colors, illuminance)


# Convert DataFrame to Open3D PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].values)
pcd.colors = o3d.utility.Vector3dVector(enhanced_colors)

# Convert PointCloud to VoxelGrid
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)

# Create a visualizer object
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the geometry to the visualizer
vis.add_geometry(voxel_grid)

# Change view parameters if needed (you can adjust these values)
view_control = vis.get_view_control()
view_control.set_zoom(0.8)

# Run the visualizer
vis.run()

