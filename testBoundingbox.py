import open3d as o3d
import numpy as np

def main():
    # Define an OrientedBoundingBox
    oriented_bbox = o3d.geometry.OrientedBoundingBox(center=(0, 0, 0),
                                              R=o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz((0, 0, 0)),
                                              extent=[1, 1, 1])
    
    # Define an AxisAlignedBoundingBox
    axis_aligned_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -1), max_bound=(1, 1, 1))
    
    # Convert bounding boxes to LineSets
    lineset_oriented_bbox = o3d.geometry.LineSet.create_from_oriented_bounding_box(oriented_bbox)
    lineset_axis_aligned_bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(axis_aligned_bbox)
    
    # Visualize the LineSets
    o3d.visualization.draw_geometries([lineset_oriented_bbox, lineset_axis_aligned_bbox])

if __name__ == "__main__":
    main()
