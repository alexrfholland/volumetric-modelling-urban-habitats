import open3d as o3d
import pandas as pd
import numpy as np
from .colorMaps import Colormap

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
    colors = np.array([category_colors[cat - 1] for cat in data['category']])
    return colors

def enhance_colors(colors, illuminance):
    enhanced_colors = enhance_colors_with_illuminance(colors, illuminance)
    return enhanced_colors

def convert_to_point_cloud(data, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].values)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def process_lidar_data(filepath, colormap_name='glasgowS'):
    # Load Data
    data = pd.read_parquet(filepath)
    data.rename(columns={'//X': 'X'}, inplace=True)
    
    # Assign horizontality
    data['horizontality'] = assign_horizontality(data['Dip (degrees)'].values)
    
    # Queries
    queries = [
        "element_type == 1 and horizontality == 0",
        "element_type == 1 and horizontality == 1",
        "element_type == 1 and horizontality == 2",
        "element_type == 2",
        "element_type == 3",
        "element_type == 4"
    ]
    
    # Create category mapping
    data['category'] = create_category_mapping(data, queries)
    
    # Filter out uncategorized points
    data = data[data['category'] != -1]
    
    # Create a colormap for visualization
    cm = Colormap()
    
    # Get and shuffle colors
    category_colors = get_and_shuffle_colors(cm, queries, colormap_name)
    
    # Assign colors based on category
    colors = assign_colors_based_on_category(data, category_colors)
    
    # Enhance colors with illuminance
    illuminance = np.array(data['Illuminance (PCV)'])
    enhanced_colors = enhance_colors(colors, illuminance)
    
    # Convert to point cloud
    pcd = convert_to_point_cloud(data, enhanced_colors)

    # Return the processed point cloud
    return pcd

if __name__ == "__main__":
    # This part will only be executed if the script is run as a standalone file,
    # and not if it's imported as a module.
    
    filepath = 'data/sites/park.parquet'
    pcd = process_lidar_data(filepath)

    # Convert PointCloud to VoxelGrid for visualization
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1.0)

    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the geometry to the visualizer
    vis.add_geometry(voxel_grid)

    # Change view parameters if needed (you can adjust these values)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([-0.5, -0.5, -0.5])
    view_control.set_lookat([2, 2, 2])
    view_control.set_up([0, 0, 1])

    # Begin the visualization
    o3d.visualization.draw_geometries([voxel_grid])
