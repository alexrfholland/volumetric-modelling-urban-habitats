import pandas as pd
import numpy as np
import pyvista as pv

def reduce_point_cloud_to_voxel_grid(data, voxel_size):
    """
    Reduces the point cloud to a voxel grid and returns a DataFrame with the 
    attributes of the first point that falls into each voxel. The coordinates
    are adjusted to be the centers of the voxels.
    """
    # Calculate voxel indices for each point
    voxel_indices_df = (data[['X', 'Y', 'Z']] / voxel_size).astype(int)
    
    # Add these voxel indices back to the original dataframe for grouping
    data['Voxel_X'] = voxel_indices_df['X']
    data['Voxel_Y'] = voxel_indices_df['Y']
    data['Voxel_Z'] = voxel_indices_df['Z']

    # Group by voxel indices and take the first point in each group
    reduced_data = data.groupby(['Voxel_X', 'Voxel_Y', 'Voxel_Z']).first().reset_index()

    # Adjust the coordinates to be the voxel centers instead of voxel corners
    reduced_data['X'] = (reduced_data['Voxel_X'] + 0.5) * voxel_size
    reduced_data['Y'] = (reduced_data['Voxel_Y'] + 0.5) * voxel_size
    reduced_data['Z'] = (reduced_data['Voxel_Z'] + 0.5) * voxel_size

    # Remove voxel indices columns
    reduced_data = reduced_data.drop(['Voxel_X', 'Voxel_Y', 'Voxel_Z'], axis=1)
    print(reduced_data)

    return reduced_data

def rgb_to_pyvista_rgb(rgb_colors):
    """Convert RGB colors from [0, 255] to [0, 1] range."""
    return rgb_colors * 255

def visualize_point_cloud_with_pyvista(data):
    print(data)
    # Convert DataFrame to PyVista Mesh
    points = data[['X', 'Y', 'Z']].values
    # Convert color values to range [0, 1]
    rgba = data[['Rf', 'Gf', 'Bf']].values
    # Add an opacity channel (fully opaque)
    rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1))], axis=1)
    
    plotter = pv.Plotter()
    plotter.add_points(points, scalars=rgba, point_size=30, render_points_as_spheres=True, rgba=True)
    plotter.enable_eye_dome_lighting()
    plotter.show()

if __name__ == "__main__":
    data = pd.read_parquet('data/sites/park.parquet')
    data = data.rename(columns={'//X': 'X'})
    voxel_size = .5
    reduced_data = reduce_point_cloud_to_voxel_grid(data, voxel_size)
    visualize_point_cloud_with_pyvista(reduced_data)
