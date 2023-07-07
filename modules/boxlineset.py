import open3d as o3d
import random


class BoundingBoxToLineSet:
    def __init__(self, bounding_boxes, line_width=200):
        self.bounding_boxes = bounding_boxes
        self.line_width = line_width

    def to_linesets(self):
        linesets = []
        for i, box in enumerate(self.bounding_boxes):
            lineset = o3d.geometry.LineSet.create_from_oriented_bounding_box(box)
            lineset.paint_uniform_color(box.color)

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = self.line_width

            linesets.append({"name": f"box_{i}", "geometry": lineset, "material": mat})
        return linesets


def visualize_linesets(linesets):
    geometries = []

    for line in linesets:
        geometries.append(line)

    o3d.visualization.draw(geometries)


if __name__ == "__main__":
    # Create some example bounding boxes
    boxes = [o3d.geometry.OrientedBoundingBox(center=(random.random() * 10, random.random() * 10, random.random() * 10),
                                              R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(
                                                  (random.random(), random.random(), random.random())),
                                              extent=[1, 1, 1]) for _ in range(10)]
    for box in boxes:
        box.color = [random.random() for _ in range(3)]  # Assign random colors

    # Convert bounding boxes to line sets
    converter = BoundingBoxToLineSet(boxes)
    linesets = converter.to_linesets()

    # Visualize the line sets
    visualize_linesets(linesets)
