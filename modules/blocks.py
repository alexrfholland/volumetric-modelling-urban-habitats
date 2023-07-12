import pandas as pd

from typing import List, Dict, Any, Optional


# read the csv files
df_tree_mapping = pd.read_csv('./data/treemapping.csv')
df_branch_predictions = pd.read_csv('./data/branchPredictions - full.csv')
df_attributes = pd.read_csv('./data/lerouxdata.csv')  # assuming you have this DataFrame



class Block:
    def __init__(self, pointcloud: Optional[Any] = None, name: str = '', conditions: Optional[List[Any]] = None, attributes: Optional[Dict[str, Any]] = None):
        self.pointcloud = pointcloud
        self.name = name
        self.conditions = conditions if conditions else []
        self.attributes = attributes if attributes else {}

    def __str__(self) -> str:
        return f"Block(name={self.name}, conditions={self.conditions}, attributes={self.attributes})"


class TreeBlock(Block):
    def __init__(self, control_level: str = '', tree_id_value: int = 0, size: str = '', otherData: Optional[pd.DataFrame] = None, **kwargs):
        super().__init__(**kwargs)
        self.control_level = control_level
        self.tree_id_value = tree_id_value
        self.size = size
        self.otherData = otherData if otherData is not None else pd.DataFrame()

    def __str__(self) -> str:
        return f"TreeBlock(control_level={self.control_level}, tree_id_value={self.tree_id_value}, size={self.size}, {super().__str__()})"

# assume df_attributes, df_tree_mapping, and df_branch_predictions are already defined and are of type pd.DataFrame

# Step 1: Create tree blocks based on tree mapping file
tree_blocks: List[TreeBlock] = []
for i, row in df_tree_mapping.iterrows():
    for col in df_tree_mapping.columns:
        if col != 'control level':
            otherData = df_branch_predictions[df_branch_predictions['Tree.ID'] == row[col]]
            tree_block = TreeBlock(
                control_level=row['control level'],
                tree_id_value=row[col],
                size=col,
                name=f"{row['control level']}_{col}",
                conditions=[],
                attributes={},
                otherData=otherData
            )
            tree_blocks.append(tree_block)

def control_level_size_df(df_attributes, control_level, size):
    """
    Listed ittributes are:
    Diameter at breast height (DBH cm)
    Height (m)
    Canopy width (m)
    Number of epiphytes
    Number of hollows
    Number of fallen logs (> 10 cm DBH 10 m radius of tree)
    % of peeling bark cover on trunk/limbs
    % of dead branches in canopy
    % of litter cover (10 m radius of tree)
    """
    control_level_low_val = f"{control_level} low"
    control_level_high_val = f"{control_level} high"

    size_filter = df_attributes['Size'] == size

    df_filtered = df_attributes[size_filter][['Attribute', control_level_low_val, control_level_high_val]]

    return df_filtered

# Step 4: Add attributes from the lerouxdata.csv
for tree_block in tree_blocks:
    attr_df = control_level_size_df(df_attributes, tree_block.control_level, tree_block.size)
    
    low_dict: Dict[str, float] = attr_df.set_index('Attribute')[f'{tree_block.control_level} low'].to_dict()
    high_dict: Dict[str, float] = attr_df.set_index('Attribute')[f'{tree_block.control_level} high'].to_dict()
    
    tree_block.attributes = {
        'low': low_dict,
        'high': high_dict
    }
print(tree_blocks[1].attributes['low']['Number of fallen logs (> 10 cm DBH 10 m radius of tree)'])

# Convert TreeBlock objects to a list of dictionaries
tree_blocks_dict = [
    {
        'control_level': block.control_level,
        'size': block.size,
        'tree_block': block
    }
    for block in tree_blocks
]

# Convert list of dictionaries to DataFrame
tree_blocks_df = pd.DataFrame(tree_blocks_dict)

# Pivot DataFrame
pivot_df = tree_blocks_df.pivot(index='control_level', columns='size', values='tree_block')

#use open3d to
#convert to 0.5m voxel grids with attributes. The coordinates are the x,y,z columns. The atrributes are the columns: "isDeadOnly","isLateralOnly","isBoth","isNeither".
#to populate the attributes, for each voxel:
# create a dictionary with the keys being the four types of attributes (Branch.length, isDeadOnly, isLateralOnly, isBoth, isNeither)
#1 find all branches that are within the bounds of the voxel
#2 for each branch, check if it is dead only, lateral only, both, or neither
#3 add the length of the branch to the corresponding attribute
#4 once all branches have been allocated to each voxel, find the length of all branches of each type per voxel
#5 have a dominant attribute colour that is the attribute with the highest length
#6 assign colour to the voxel based on the dominant attribute


import open3d as o3d
import numpy as np
from typing import Tuple

def add_branch_to_voxel(voxel: Dict[str, Any], branch: pd.Series):
    if branch['Branch.type'] == 'dead' and branch['Branch.angle'] > 20:
        category = 'isDeadOnly'
    elif branch['Branch.type'] != 'dead' and branch['Branch.angle'] <= 20:
        category = 'isLateralOnly'
    elif branch['Branch.type'] == 'dead' and branch['Branch.angle'] <= 20:
        category = 'isBoth'
    else:
        category = 'isNeither'
    
    voxel[category] += branch['Branch.length']

            
def compute_dominant_attribute(voxel: Dict[str, Any]) -> str:
    categories = ['isDeadOnly', 'isLateralOnly', 'isBoth', 'isNeither']
    lengths = [voxel[category] for category in categories]
    dominant_index = np.argmax(lengths)
    print(voxel)
    return categories[dominant_index]

def assign_color_to_dominant_attribute(dominant_attribute: str) -> Tuple[float, float, float]:
    colors = {
        'isDeadOnly': (1, 0, 0),     # red
        'isLateralOnly': (0, 1, 0),  # green
        'isBoth': (0, 0, 1),         # blue
        'isNeither': (1, 1, 1)       # white
    }
    return colors[dominant_attribute]

def create_voxel_grid(tree_block: TreeBlock, voxel_size: float = 0.5) -> o3d.geometry.PointCloud:
    # Define the voxel grid
    voxel_grid: Dict[str, Dict[str, Any]] = {}

    for _, branch in tree_block.otherData.iterrows():
        # Determine voxel for this branch
        voxel_x = np.floor(branch['x'] / voxel_size) * voxel_size
        voxel_y = np.floor(branch['y'] / voxel_size) * voxel_size
        voxel_z = np.floor(branch['z'] / voxel_size) * voxel_size
        voxel_key = f"{voxel_x}_{voxel_y}_{voxel_z}"

        # Initialize voxel if not already in grid
        if voxel_key not in voxel_grid:
            voxel_grid[voxel_key] = {
                'x': voxel_x,
                'z': voxel_y,
                'y': voxel_z,
                'isDeadOnly': 0,
                'isLateralOnly': 0,
                'isBoth': 0,
                'isNeither': 0
            }

        # Add branch to voxel
        add_branch_to_voxel(voxel_grid[voxel_key], branch)

    # Compute dominant attribute and color for each voxel
    for voxel in voxel_grid.values():
        dominant_attribute = compute_dominant_attribute(voxel)
        voxel['color'] = assign_color_to_dominant_attribute(dominant_attribute)

    # Convert voxel grid to point cloud for visualization
    point_cloud = o3d.geometry.PointCloud()
    for voxel in voxel_grid.values():
        point = np.array([voxel['x'], voxel['y'], voxel['z']])
        color = voxel['color']
        point_cloud.points.append(point)
        point_cloud.colors.append(color)

    return point_cloud

"""# Apply create_voxel_grid to each TreeBlock in tree_blocks
for tree_block in tree_blocks:
    point_cloud = create_voxel_grid(tree_block)
    o3d.visualization.draw_geometries([point_cloud])
"""

def assign_color_to_dominant_attribute(dominant_attribute: str) -> Tuple[float, float, float]:
    colors = {
        'isDeadOnly': (1, 0, 0),     # Red: RGB value of (1, 0, 0)
        'isLateralOnly': (0, 1, 0),  # Green: RGB value of (0, 1, 0)
        'isBoth': (0, 0, 1),         # Blue: RGB value of (0, 0, 1)
        'isNeither': (.25,.25,.25)       # White: RGB value of (1, 1, 1)
    }
    return colors[dominant_attribute]


def create_voxel_grid_from_point_cloud(tree_block: TreeBlock, voxel_size: float = 0.5) -> o3d.geometry.VoxelGrid:
    # Convert tree_block to a point cloud
    point_cloud = create_voxel_grid(tree_block, voxel_size)

    # Create a voxel grid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size)

    return voxel_grid

# Apply create_voxel_grid_from_point_cloud to each TreeBlock in tree_blocks
for tree_block in tree_blocks:
    voxel_grid = create_voxel_grid_from_point_cloud(tree_block)
    o3d.visualization.draw_geometries([voxel_grid])
