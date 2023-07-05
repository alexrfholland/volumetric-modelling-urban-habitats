import open3d as o3d

# Load the mesh
mesh = o3d.io.read_triangle_mesh("output_mesh.ply")

# Compute the normals for shading
mesh.compute_vertex_normals()

# Create a visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the mesh to the visualizer
vis.add_geometry(mesh)

# Start the visualization
vis.run()

# Destroy the visualizer window after visualization ends
vis.destroy_window()
