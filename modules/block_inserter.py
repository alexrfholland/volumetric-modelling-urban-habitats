from __future__ import annotations
from typing import List, Tuple, Any

import numpy as np

#se Python's "forward references". Forward references are string literals that specify the fully qualified name of a type.


def add_block(octree: 'Octree', child: 'OctreeNode', points: np.ndarray, attributes: List[dict], block_ids: List[int], max_depth: int) -> None:
    def update_block_ids(node, block_id):
        while node is not None:
            node.block_ids.append(block_id)
            node = node.parent

    for point, attribute, block_id in zip(points, attributes, block_ids):
        # Find the appropriate node to insert this point into
        node, quadrant = octree.find_node_for_point(point)

        # If the point is not within any existing child node, create a new one
        if node is octree.root or quadrant is not None:
            min_corner, max_corner = node.calculate_bounds_for_point(point)
            child = OctreeNode(min_corner, max_corner, np.array([point]), [attribute], [block_id], node.depth + 1)
            node.children.append(child)
            child.parent = node
            # Append the block_id to the current node and all its ancestors
            update_block_ids(node, block_id)

            child.split(max_depth + 1)

        else:
            # Append the point, attribute, and block_id to the found leaf node
            node.points = np.append(node.points, [point], axis=0)
            node.attributes.append(attribute)
            node.block_ids.append(block_id)
                
            # Append the block_id to the found node and all its ancestors
            update_block_ids(node, block_id)

            node.split(max_depth + 1)
            
def generate_dummy_data(num_points):
    points = np.random.rand(num_points, 3) * 20 - 10  # random points in a 20x20x20 cube centered at origin
    attributes = [{'Rf': np.random.rand(), 'Bf': np.random.rand(), 'Gf': np.random.rand(), 
                   'B': np.random.randint(256), 'Composite': np.random.rand(), 'Dip (degrees)': np.random.rand()*180, 
                   'Dip direction (degrees)': np.random.rand()*360, 'G': np.random.randint(256), 
                   'Illuminance (PCV)': np.random.rand(), 'Nx': np.random.rand(), 'Ny': np.random.rand(), 
                   'Nz': np.random.rand(), 'R': np.random.randint(256), 'element_type': np.random.randint(10), 
                   'horizontality': np.random.randint(2)} for _ in range(num_points)]
    block_ids = np.random.randint(1, 5, num_points)  # block ids between 1 and 5

    return points, attributes, block_ids
