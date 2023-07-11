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

try:
    from .colorMaps import Colormap  # Attempt a relative import
except ImportError:
    from colorMaps import Colormap  # Fall back to an absolute import

def enhance_colors_with_illuminance(colors, illuminance):
    return colors * illuminance[:, np.newaxis] ** 5

def assign_horizontality(dip_degrees):
    horizontality = np.empty_like(dip_degrees, dtype=int)
    horizontality[dip_degrees < 6] = 0  # flat
    horizontality[(dip_degrees >= 6) & (dip_degrees < 85)] = 1  # angled
    horizontality[dip_degrees >= 85] = 2  # vertical
    return horizontality

def create_category_mapping(data, queries):
    category = np.full(len(data), -1, dtype=int)  # default to -1
    for i, query in enumerate(queries):
        filtered_indices = data.query(query).index
        category[filtered_indices] = i + 1
    return category

def get_and_shuffle_colors(cm, queries, colormap_name):
    category_colors = cm.get_categorical_colors(len(queries) + 1, colormap_name)
    category_colors.pop(0)
    return category_colors

def assign_colors_based_on_category(data, category_colors):
    colors = np.array([category_colors[cat - 1] for cat in data['blockID']])
    return colors


def enhance_colors(colors, illuminance):
    enhanced_colors = enhance_colors_with_illuminance(colors, illuminance)
    return enhanced_colors

def convert_to_point_cloud(data, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].values)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def process_lidar_data(filepath, colormap_name='lajollaS'):
    """
    Returns a dataframe with the columns being X,Y,Z,blockID,r,g,b,B,Bf,Composite,Dip (degrees),Dip direction (degrees),G,Gf,Illuminance (PCV),Nx,Ny,Nz,R,Rf,element_type,horizontality
    with one per column. Rows are the points.
    
    Parameters:
        filepath (str): The path to the lidar data file.
        colormap_name (str, optional): The name of the colormap. Defaults to 'glasgowS'.
    
    Returns:
        pandas.DataFrame: The processed lidar data as a dataframe.
    """
        
    # Load Data
    data = pd.read_parquet(filepath)
    print(data)
    data.rename(columns={'//X': 'X'}, inplace=True)
    
    # Assign horizontality
    data['horizontality'] = assign_horizontality(data['Dip (degrees)'].values)
    
    # Queries
    queries = [
        "element_type == 0",
        "element_type == 1 and horizontality == 0",
        "element_type == 1 and horizontality == 1",
        "element_type == 1 and horizontality == 2",
        "element_type == 2",
        "element_type == 3",
        "element_type == 4"
    ]
    
    # Create category mapping
    data['blockID'] = create_category_mapping(data, queries)
    
    # Filter out uncategorized points
    data = data[data['blockID'] != -1]
    
    # Create a colormap for visualization
    cm = Colormap()
    
    # Get and shuffle colors
    category_colors = get_and_shuffle_colors(cm, queries, colormap_name)
    
    # Assign colors based on category
    colors = assign_colors_based_on_category(data, category_colors)
    
    # Enhance colors with illuminance
    enhanced_colors = enhance_colors(colors, np.array(data['Illuminance (PCV)']))
    
    # Assign the color columns directly to the data DataFrame
    data['r'] = enhanced_colors[:, 0]
    data['g'] = enhanced_colors[:, 1]
    data['b'] = enhanced_colors[:, 2]
    
    # Get the attribute columns
    attributes_columns = data.columns.difference(['X', 'Y', 'Z', 'r', 'g', 'b', 'blockID'])
    
    # Order columns
    ordered_columns = ['X', 'Y', 'Z', 'r', 'g', 'b', 'blockID'] + list(attributes_columns)
    ordered_columns = ['X', 'Y', 'Z', 'blockID', 'r', 'g', 'b'] + list(attributes_columns)

    result = data[ordered_columns]
    
    # Return the DataFrame
    return result

def select_random_ground_points(processed_data, n_points):
    """
    Randomly selects n number of points with element_type = 2 (ground) and returns their X, Y, Z coordinates.

    Parameters:
        processed_data (pandas.DataFrame): The DataFrame containing processed LiDAR data.
        n_points (int): The number of random points to select.

    Returns:
        pandas.DataFrame: A DataFrame with X, Y, Z coordinates of the randomly selected ground points.
    """
    # Selecting points where element_type = 2 (ground)
    #ground_points = processed_data[processed_data['element_type'] == 2]

    ground_points = processed_data
    print('printing ground points')
    print(ground_points)
    # Randomly select n_points from ground_points
    selected_points = ground_points.sample(n_points)

    selected_coords = selected_points[['X', 'Y', 'Z']]

    coordinates_list_as_tuples = [tuple(x) for x in selected_coords.values]


    #print(f'Selected {n_points} random points {selected_coords}')

    # Return the X, Y, Z coordinates
    return coordinates_list_as_tuples



def convertToVoxelGrid(data):
    # Extract coordinates and colors
    coordinates = data[['X', 'Y', 'Z']].values
    colors = data[['r', 'g', 'b']].values

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Convert PointCloud to VoxelGrid for visualization
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the geometry to the visualizer
    vis.add_geometry(voxel_grid)

    # Change view parameters if needed (you can adjust these values)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # This time, we won't set the front vector, but let Open3D handle it
    # Instead, we'll set the camera to look at the center of the point cloud
    bounds = np.array(pcd.get_max_bound()) - np.array(pcd.get_min_bound())
    view_control.set_lookat(np.array(pcd.get_center()))
    view_control.set_up([0, -1, 0])
    view_control.set_front([0, 0, 1])

    # Begin the visualization
    vis.run()
    vis.destroy_window()

# The main section
if __name__ == "__main__":
    # This part will only be executed if the script is run as a standalone file,
    # and not if it's imported as a module.
    
    # Filepath to the Parquet file
    filepath = 'data/sites/park.parquet'

    # Process the lidar data
    processed_data = process_lidar_data(filepath)
    
    # Convert the processed data to a VoxelGrid and visualize it
    convertToVoxelGrid(processed_data)