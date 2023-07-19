import pandas as pd
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

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

    return reduced_data

from matplotlib.colors import ListedColormap

# This code visualizes a voxel grid using PyVista by directly assigning RGBA colors to each voxel.
# The key steps are as follows:
#
# 1. Obtain the RGB color values for each voxel from the data. These are assumed to be floating-point numbers
#    in the range [0, 1]. If your data has RGB values in the range [0, 255], you should divide by 255 to 
#    convert to the [0, 1] range.
#
# 2. Add an alpha channel to these RGB colors to create RGBA colors. The alpha channel controls opacity,
#    with 0 being fully transparent and 1 being fully opaque. We set alpha to 1 for all voxels.
#
# 3. Convert the voxel coordinates and colors into a PyVista PolyData structure. The coordinates become the
#    points in this structure, and the colors are stored in the point_data attribute.

# 4. Use the PyVista add_mesh function to visualize the voxel grid. The colors argument is set to the name of
#    the array in the PolyData structure that contains the colors (in this case, "colors"). The rgba argument
#    is set to True to indicate that the colors should be interpreted as RGBA colors.

def visualize_voxel_grid_with_pyvista(data, voxel_size):
    # Convert DataFrame to PyVista PolyData
    ugrid = pv.PolyData(data[['X', 'Y', 'Z']].values)
    # Prepare the colors (RGB + alpha channel, range [0, 1])
    colors = data[['Rf', 'Gf', 'Bf']].values # RGB
    alpha = np.ones((colors.shape[0], 1))  # Alpha channel (full opacity)
    colors = np.hstack((colors, alpha))
    # Assign colors to points in the PolyData structure
    ugrid.point_data["colors"] = colors
    # Create the box glyph and shift it to the center
    glyph = pv.Box().scale(voxel_size / 2.0, voxel_size / 2.0, voxel_size / 2.0)
    plotter = pv.Plotter()
    plotter.add_mesh(ugrid.glyph(geom=glyph, scale=False), rgba=True, scalars="colors")
    plotter.enable_eye_dome_lighting()
    plotter.show()


def visualize_point_cloud_with_pyvista(data):
    # Convert DataFrame to PyVista Mesh
    points = data[['X', 'Y', 'Z']].values
    # Convert color values to range [0, 1]
    rgba = data[['Rf', 'Gf', 'Bf']].values
    # Add an opacity channel (fully opaque)
    rgba = np.concatenate([rgba, np.ones((rgba.shape[0], 1))], axis=1)
    
    plotter = pv.Plotter()
    plotter.add_points(points, scalars=rgba, point_size=20, render_points_as_spheres=True, rgba=True)
    plotter.enable_eye_dome_lighting()

    plotter.show()

if __name__ == "__main__":
    data = pd.read_parquet('data/sites/park.parquet')
    data = data.rename(columns={'//X': 'X'})
    voxel_size = 1
    reduced_data = reduce_point_cloud_to_voxel_grid(data, voxel_size)
    visualize_voxel_grid_with_pyvista(reduced_data, voxel_size)
    visualize_point_cloud_with_pyvista(reduced_data)
