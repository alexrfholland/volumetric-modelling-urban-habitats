import pyvista as pv
import numpy as np

"""def draw_cube(center, length=1):
    #Draws a wireframe cube at the specified center point.
    half_length = length / 2
    corners = [
        center + np.array([-half_length, -half_length, -half_length]),
        center + np.array([half_length, -half_length, -half_length]),
        center + np.array([half_length, half_length, -half_length]),
        center + np.array([-half_length, half_length, -half_length]),
        center + np.array([-half_length, -half_length, half_length]),
        center + np.array([half_length, -half_length, half_length]),
        center + np.array([half_length, half_length, half_length]),
        center + np.array([-half_length, half_length, half_length]),
    ]

    lines = []
    for i in range(4):
        lines.append([corners[i], corners[(i+1)%4]])
        lines.append([corners[i+4], corners[(i+1)%4+4]])
        lines.append([corners[i], corners[i+4]])

    cube = pv.PolyData()
    for line in lines:
        cube += pv.Line(line[0], line[1])

    return cube

# Create the plotter
plotter = pv.Plotter()

# Define the colors we will use for each bucket
bucket_colors = np.array([
    [1, 0, 0],  # Red
    [0, 1, 0],  # Green
    [0, 0, 1],  # Blue
    [1, 1, 0],  # Yellow
    [1, 0, 1],  # Magenta
    [0, 1, 1],  # Cyan
    [0.5, 0.5, 0.5],  # Gray
    [1, 0.5, 0],  # Orange
    [0.5, 0, 1],  # Purple
    [0, 0.5, 0.5],  # Teal
])

# Create 10 buckets
for i in range(10):
    print(f'bucket {i}')
    # Create 1000 random points
    points = np.random.rand(100, 3) * 10

    # Create a cube at each point and combine them into a MultiBlock
    bucket = pv.MultiBlock([draw_cube(point) for point in points])

    # Add the cubes in this bucket to the plotter with the bucket's color
    plotter.add_mesh(bucket, color=bucket_colors[i], line_width=2)

# Adjust the camera's position and view to include all objects in the scene
plotter.reset_camera()

# Show the plot
plotter.show()
"""

import pyvista as pv
import numpy as np

# Define voxel centers
centers = np.mgrid[0:10, 0:10, 0:10].reshape(3, -1).T

# Assign random sizes to each cube
sizes = np.random.random(centers.shape[0])

# Create a unit cube to use as the glyph
cube = pv.Cube()

# Create a PolyData with the voxel centers as points
points = pv.PolyData(centers)

# Add the voxel sizes as a point array (this will be used to scale the glyphs)
points["sizes"] = sizes

# Assign random scalars to each cube
scalars = np.random.random(centers.shape[0])
points["scalars"] = scalars

# Use the cube as a glyph and scale it by the voxel sizes
glyphs = points.glyph(scale="sizes", factor=1.0, geom=cube)

# Plot the glyphs, using the "scalars" array to define the colors
p = pv.Plotter()
p.add_mesh(glyphs, scalars="scalars", cmap="rainbow")

# Enable eye dome lighting
p.enable_eye_dome_lighting()

p.show()
