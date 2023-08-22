import pandas as pd
import numpy as np

"""try:
    from .colorMaps import Colormap  # Attempt a relative import
except ImportError:
    from colorMaps import Colormap  # Fall back to an absolute import"""

def enhance_colors_with_illuminance(colors, illuminance):
    return colors * illuminance[:, np.newaxis] ** 5

def assign_horizontality(dip_degrees):
    horizontality = np.empty_like(dip_degrees, dtype=int)
    horizontality[dip_degrees < 6] = 0  # flat
    horizontality[(dip_degrees >= 6) & (dip_degrees < 85)] = 1  # angled
    horizontality[dip_degrees >= 85] = 2  # vertical
    return horizontality

def create_category_mapping(data, queries, populate_block_metadata):
    category = np.full(len(data), -1, dtype=int)  # default to -1
    for i, (query, label) in enumerate(queries.items()):
        filtered_indices = data.query(query).index
        category[filtered_indices] = i + 1

        populate_block_metadata(i+1, label)
    return category


"""def get_and_shuffle_colors(cm, queries, colormap_name):
    category_colors = cm.get_categorical_colors(len(queries) + 1, colormap_name)
    category_colors.pop(0)
    return category_colors"""

def get_and_shuffle_colors(queries):
    return [(0, 0, 0) for _ in range(len(queries))]

def assign_colors_based_on_category(data, category_colors):
    colors = np.array([category_colors[cat - 1] for cat in data['blockID']])
    return colors


def enhance_colors(colors, illuminance):
    enhanced_colors = enhance_colors_with_illuminance(colors, illuminance)
    return enhanced_colors

"""def convert_to_point_cloud(data, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].values)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd"""

def process_lidar_data(filepath, populate_block_metadata, colormap_name='lajollaS'):
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
        #"element_type == 0", #block id = 1, trees
        "element_type == 1 and horizontality == 0", #block id = 2, buildings, roof flat
        "element_type == 1 and horizontality == 1", #block id = 3 buildings, roof angle
        "element_type == 1 and horizontality == 2", #block id = 4, buildings, wall (vertical)
        "element_type == 2", #block id = 5, grass
        "element_type == 3", #block id = 6, street furniture
        "element_type == 4" #block id = 7, impermeable ground
    ]

        # Queries
    queries = {
        #"element_type == 0", #block id = 1, trees
        "element_type == 1 and horizontality == 0" : 'roof flat', #block id = 2, buildings, roof flat
        "element_type == 1 and horizontality == 1" : 'roof angle', #block id = 3 buildings, roof angle
        "element_type == 1 and horizontality == 2" : ' wall', #block id = 4, buildings, wall (vertical)
        "element_type == 2" : 'grass', #block id = 5, grass
        "element_type == 3" : 'street furniture', #block id = 6, street furniture
        "element_type == 4" : 'impermeable ground' #block id = 7, impermeable ground
        }


    catagories = create_category_mapping(data, queries, populate_block_metadata)
    print(f'catagories are {catagories}')
    # Create category mapping
    data['blockID'] = catagories
    
    # Filter out uncategorized points
    data = data[data['blockID'] != -1]
    
    """# Create a colormap for visualization
    cm = Colormap()
    
    # Get and shuffle colors
    category_colors = get_and_shuffle_colors(cm, queries, colormap_name)"""

    category_colors = get_and_shuffle_colors(queries)

    
    # Assign colors based on category
    colors = assign_colors_based_on_category(data, category_colors)
    
    # Enhance colors with illuminance
    enhanced_colors = enhance_colors(colors, np.array(data['Illuminance (PCV)']))

    
    # Assign the color columns directly to the data DataFrame
    data['r'] = enhanced_colors[:, 0]
    data['g'] = enhanced_colors[:, 1]
    data['b'] = enhanced_colors[:, 2]
    
    # Get the attribute columns
    attributes_columns = data.columns.difference(['X', 'Y', 'Z', 'r', 'g', 'b', 'blockID','Rf','Bf','Gf'])
    
    # Order columns
    ordered_columns = ['X', 'Y', 'Z', 'r', 'g', 'b', 'blockID','Rf','Bf','Gf'] + list(attributes_columns)

    result = data[ordered_columns]
    
    print(result)
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


# The main section
if __name__ == "__main__":
    # This part will only be executed if the script is run as a standalone file,
    # and not if it's imported as a module.
    
    # Filepath to the Parquet file
    filepath = 'data/sites/park.parquet'

    # Process the lidar data
    processed_data = process_lidar_data(filepath)
    