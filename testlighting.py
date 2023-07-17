import pyvista as pv
import numpy as np

def create_dummy_line():
    # Create two points for the line
    point1 = [np.random.randint(10), np.random.randint(10), np.random.randint(10)]
    point2 = [np.random.randint(10), np.random.randint(10), np.random.randint(10)]

    # Print the points to console
    print(f"Creating line from {point1} to {point2}")

    # Create a line cell
    lines = np.array([[2, 0, 1]])

    # Create the points array
    points = np.array([point1, point2])

    # Create a PolyData object with the points and lines
    line = pv.PolyData(points, lines)

    return line

# Create the dummy line
dummy_line = create_dummy_line()

# Create PyVista plotter
plotter = pv.Plotter()

# Add the dummy line to the plotter
plotter.add_mesh(dummy_line, color="red", line_width=5)

# Show the plotter
plotter.show()
