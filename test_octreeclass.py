# Example script to use OctreeVisualizer
from classes import OctreeVisualizer

csv_file = 'data/branchPredictions - full.csv'
tree_id = 13
max_depth = 5

# Initialize the visualizer
visualizer = OctreeVisualizer(csv_file, tree_id, max_depth)

# Update the visualization (specify the range of levels of bounding boxes to display)
visualizer.update_visualization(max_depth, 0, max_depth - 1)

# Run the visualizer
visualizer.run()
