###make a test script that creates a couple of bounding boxes, and converts them into line widths and then visualises them 
#using the approach described in line_widths.py provided by open3d located in the examples folder (examples/python/visualization/line_widths.py)
import open3d as o3d
import numpy as np
import random


# Define colors for different properties
property_colors = {"hollow": [1.0, 0.0, 0.0],  # Red
                   "leaf": [0.0, 1.0, 0.0],  # Green
                   "road": [0.0, 0.0, 1.0],  # Blue
                   "building": [1.0, 1.0, 0.0],  # Yellow
                   "pole": [0.0, 1.0, 1.0]}  # Cyan

def allocate_lines(boxes, properties, line_width=200):
    linesets = {}
    for i, (box, prop) in enumerate(zip(boxes, properties)):
        if prop not in linesets:
            linesets[prop] = []
        lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
        lineset.paint_uniform_color(property_colors[prop])

        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = line_width

        linesets[prop].append({"name": f"{prop}_{i}", "geometry": lineset, "material": mat})
    return linesets


def visualize_linesets(linesets):
    geometries = []

    for prop_lines in linesets.values():
        for line in prop_lines:
            geometries.append(line)

    o3d.visualization.draw(geometries)

def main():
    boxes = [o3d.geometry.OrientedBoundingBox(center=(random.random()*10, random.random()*10, random.random()*10),
                                              R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((random.random(), random.random(), random.random())),
                                              extent=[1, 1, 1]) for _ in range(30)]
    properties = random.choices(["hollow", "leaf", "road", "building", "pole"], k=30)

    # Allocate lines into linesets based on their properties
    linesets = allocate_lines(boxes, properties)

    # Visualize the linesets
    visualize_linesets(linesets)

if __name__ == "__main__":
    main()


