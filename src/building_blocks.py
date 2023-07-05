import open3d as o3d
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class BuildingBlock:
    def __init__(self, name: str, properties: Dict[str, Any]):
        self.name = name
        self.properties = properties


class Tree(BuildingBlock):
    def __init__(self, tree_id: str, points: np.ndarray, age: str = None, management_regime: str = None, properties: Dict[str, Any] = None):
        super().__init__("Tree", properties or {})
        self.tree_id = tree_id
        self.age = age
        self.management_regime = management_regime
        self.octree = self._build_octree(points)

    def _build_octree(self, points: np.ndarray, max_depth: int = 6) -> o3d.geometry.Octree:
        """
        Builds an octree from the given points
        """
        # Convert numpy array to Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Build an octree
        octree = o3d.geometry.Octree(max_depth=max_depth)
        octree.convert_from_point_cloud(point_cloud)
        
        return octree


class ArtificialStructure(BuildingBlock):
    def __init__(self, structure_type: str, properties: Dict[str, Any]):
        super().__init__("ArtificialStructure", properties)
        self.structure_type = structure_type
