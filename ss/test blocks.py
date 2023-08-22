import numpy as np
import pyvista as pv

def visualize_voxel_grid_with_pyvista(positions, sizes, colors, base_voxel_size=1.0):
    """
    Visualizes a voxel grid with PyVista.
    
    Args:
    - positions (list): A list of voxel positions, where each position is a list or tuple with 3 coordinates (x, y, z).
    - sizes (list): A list of voxel sizes which scales the voxel by that amount.
    - colors (list): A list of voxel colors, where each color is a list or tuple with 3 RGB values in the range [0, 1].
    - base_voxel_size (float): The base size of each voxel before scaling. Default is 1.0.
    
    Returns:
    - None
    """
    
    # Convert lists to NumPy arrays
    positions_np = np.array(positions)
    sizes_np = np.array(sizes)
    colors_np = np.array(colors)
    
    # Create a PolyData structure from the positions
    ugrid = pv.PolyData(positions_np)
    
    # Assign sizes and colors to points in the PolyData structure
    ugrid.point_data["sizes"] = sizes_np
    ugrid.point_data["colors"] = np.hstack((colors_np, np.ones((colors_np.shape[0], 1))))  # RGBA colors
    
    # Create the box glyph for each voxel and scale it
    glyph = pv.Box().scale(base_voxel_size / 2.0, base_voxel_size / 2.0, base_voxel_size / 2.0)
    
    # Visualize the voxel grid
    plotter = pv.Plotter()
    plotter.add_mesh(ugrid.glyph(geom=glyph, scale="sizes"), rgba=True, scalars="colors")
    plotter.enable_eye_dome_lighting()
    plotter.show()

# Sample usage with 10,000 voxels
num_voxels = 10000
positions = np.random.rand(num_voxels, 3) * 50  # Random positions in a space of 50x50x50
sizes = np.random.rand(num_voxels) * 5  # Random sizes between 0 and 5
colors = np.random.rand(num_voxels, 3)  # Random RGB colors

visualize_voxel_grid_with_pyvista(positions.tolist(), sizes.tolist(), colors.tolist())
