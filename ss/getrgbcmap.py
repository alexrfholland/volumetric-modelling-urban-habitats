"""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate random points and colors
np.random.seed(42)  # Set a seed for reproducibility
num_points = 1000
points = np.random.rand(num_points, 2)  # Generate random 2D points between 0 and 1
colors = np.random.rand(num_points, 3)  # Generate random RGB colors for each point

# Create a colormap based on the colors
cmap = ListedColormap(colors)

# Plot the points with directly assigned colors
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c=colors)
plt.title('Points with Directly Assigned Colors')
plt.show()

# Plot the points using the custom created colormap
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], c=np.arange(num_points), cmap=cmap)
plt.title('Points with Custom Colormap')
plt.colorbar(label='Scalar Value')
plt.show()
"""

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

# Load your data and preprocess it
data = pd.read_parquet('data/sites/park.parquet')
data = data.rename(columns={'//X': 'X'})
voxel_size = .5
reduced_data = reduce_point_cloud_to_voxel_grid(data, voxel_size)

# Collect the colors
colors = reduced_data[['Rf', 'Gf', 'Bf']].values

# Convert the RGB colors to unique string representations
color_strings = [''.join(map(str, color)) for color in colors]

# Convert the string representations to unique integer indices
le = LabelEncoder()
color_indices = le.fit_transform(color_strings)

# Create a colormap from the unique colors
cmap = ListedColormap(colors)

num_colors = len(cmap.colors)
print(f"The colormap contains {num_colors} unique colors.")


# Convert DataFrame to PyVista UnstructuredGrid
ugrid = pv.PolyData(reduced_data[['X', 'Y', 'Z']].values)
ugrid["color_indices"] = color_indices

# Create the plotter and add the glyphs
plotter = pv.Plotter()
# Create the box glyph and shift it to the center
# Create the box glyph and shift it to the center
glyph = pv.Box().scale(voxel_size / 2.0, voxel_size / 2.0, voxel_size / 2.0)
plotter.add_mesh(ugrid.glyph(geom=glyph, scale=False), cmap=cmap, scalars="color_indices")
plotter.enable_eye_dome_lighting()


# Show the plot
plotter.show()