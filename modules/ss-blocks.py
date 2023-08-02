import pandas as pd

from typing import List, Dict, Tuple, Any, Optional
from collections import Counter

import os


# read the csv files
df_tree_mapping = pd.read_csv('./data/treemapping.csv')
df_branch_predictions = pd.read_csv('./data/branchPredictions - full.csv')
df_attributes = pd.read_csv('./data/lerouxdata.csv')  # assuming you have this DataFrame


try:
    from .colorMaps import Colormap  # Attempt a relative import
except ImportError:
    from colorMaps import Colormap  # Fall back to an absolute import


class Block:
    def __init__(self, pointcloud: Optional[Any] = None, name: str = '', conditions: Optional[List[Any]] = None, attributes: Optional[Dict[str, Any]] = None):
        self.pointcloud = pointcloud
        self.name = name
        self.conditions = conditions if conditions else []
        self.attributes = attributes if attributes else {}

    """def __str__(self) -> str:
        return f"Block(name={self.name}, conditions={self.conditions}, attributes={self.attributes})"""


class TreeBlock(Block):
    def __init__(self, control_level: str = '', tree_id_value: int = 0, size: str = '', otherData: Optional[pd.DataFrame] = None, **kwargs):
        super().__init__(**kwargs)
        self.control_level = control_level
        self.tree_id_value = tree_id_value
        self.size = size
        self.otherData = otherData if otherData is not None else pd.DataFrame()

    def __str__(self) -> str:
        #return f"TreeBlock(control_level={self.control_level}, tree_id_value={self.tree_id_value}, size={self.size}, {super().__str__()})"
        return f"TreeBlock, control: {self.control_level}, size: {self.size}, scanned tree no: {self.tree_id_value}{super().__str__()})"

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
    
    low_dict: Dict[str, float] = {k: float(v) for k, v in attr_df.set_index('Attribute')[f'{tree_block.control_level} low'].to_dict().items()}
    high_dict: Dict[str, float] = {k: float(v) for k, v in attr_df.set_index('Attribute')[f'{tree_block.control_level} high'].to_dict().items()}
    
    tree_block.attributes = {
        'low': low_dict,
        'high': high_dict
    }
print(tree_blocks[1].attributes['low']['Number of fallen logs (> 10 cm DBH 10 m radius of tree)'])


#print(tree_blocks[1].attributes['low']['Number of fallen logs (> 10 cm DBH 10 m radius of tree)'])

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
pivot_df = tree_blocks_df.pivot(index='size', columns='control_level', values='tree_block')

df_attributes = pd.read_csv('./data/lerouxdata.csv')  # assuming you have this DataFrame

import open3d as o3d
import numpy as np
import random

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
    return categories[dominant_index]


def assign_color_to_dominant_attribute(dominant_attribute: str) -> Tuple[float, float, float]:
    colors = {
        'isDeadOnly': (1, 0, 0),     # Red: RGB value of (1, 0, 0)
        'isLateralOnly': (0, 1, 0),  # Green: RGB value of (0, 1, 0)
        'isBoth': (0, 0, 1),         # Blue: RGB value of (0, 0, 1)
        'isNeither': (.25, .25, .25), # Gray: RGB value of (.25, .25, .25)
        'hollows': (1, 1, 0),        # Yellow: RGB value of (1, 1, 0)
        'epiphytes': (0, 1, 1),      # Cyan: RGB value of (0, 1, 1)
        'peeling_bark': (1, 0, 1)    # Magenta: RGB value of (1, 0, 1)
    }
    return colors.get(dominant_attribute, (1, 1, 1))  # Default color to white (1, 1, 1) if key not found


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
                'isNeither': 0,
                'synthesized_type': None
            }

        # Add branch to voxel
        add_branch_to_voxel(voxel_grid[voxel_key], branch)

    # Compute dominant attribute and color for each voxel
    for voxel in voxel_grid.values():
        dominant_attribute = compute_dominant_attribute(voxel)
        voxel['color'] = assign_color_to_dominant_attribute(dominant_attribute)

    return voxel_grid


def select_voxels_for_synthesis(voxel_grid: Dict[str, Dict[str, Any]], n_voxels: int) -> List[Dict[str, Any]]:
    # Group voxels by attribute
    voxels_by_attribute = {
        'isNeither': [],
        'isLateralOnly': [],
        'isDeadOnly': [],
        'isBoth': []
    }

    for voxel in voxel_grid.values():
        dominant_attribute = compute_dominant_attribute(voxel)
        voxels_by_attribute[dominant_attribute].append(voxel)

    # Sequentially draw from each group
    selected_voxels = []
    attribute_priority = ['isNeither', 'isLateralOnly', 'isDeadOnly', 'isBoth']

    for attribute in attribute_priority:
        while len(selected_voxels) < n_voxels and voxels_by_attribute[attribute]:
            # Randomly select a voxel from the current attribute group
            i = random.randint(0, len(voxels_by_attribute[attribute]) - 1)
            selected_voxel = voxels_by_attribute[attribute].pop(i)
            selected_voxels.append(selected_voxel)

    return selected_voxels




def set_attributes_for_tree_block(tree_block, df_attributes):
    """Set the attribute values for a tree block based on its control level and size."""
    attr_df = control_level_size_df(df_attributes, tree_block.control_level, tree_block.size)
    low_dict: Dict[str, float] = attr_df.set_index('Attribute')[f'{tree_block.control_level} low'].to_dict()
    high_dict: Dict[str, float] = attr_df.set_index('Attribute')[f'{tree_block.control_level} high'].to_dict()

    tree_block.attributes = {
        'low': low_dict,
        'high': high_dict
    }


def synthesize_voxels(tree_block, attributes_to_voxel):
    """For each attribute in attributes_to_voxel, synthesize voxels based on the attribute's random value between its low and high limits."""
    for attribute, synthesized_type in attributes_to_voxel:
        low = float(tree_block.attributes['low'][attribute])
        high = float(tree_block.attributes['high'][attribute])
        value = random.uniform(low, high)

        if attribute == '% of peeling bark cover on trunk/limbs':
            number_of_voxels = round(len(tree_block.voxel_grid) * (value / 100))
        else:
            number_of_voxels = round(value)

        selected_voxels = select_voxels_for_synthesis(tree_block.voxel_grid, number_of_voxels)

        # Assign the synthesized type to the selected voxels
        for voxel in selected_voxels:
            voxel['synthesized_type'] = synthesized_type

        # After the synthesized type has been set, decide the voxel_type and color
        for voxel in tree_block.voxel_grid.values():
            voxel['voxel_type'] = voxel['synthesized_type'] if voxel['synthesized_type'] else compute_dominant_attribute(voxel)

        print(f"Number of voxels to convert to {synthesized_type}: {number_of_voxels}")

def adjust_dead_voxel_count(tree_block: TreeBlock, attr_df: pd.DataFrame):
    total_voxel_count = len(tree_block.voxel_grid)

    dead_branch_percent_low = float(tree_block.attributes['low']['% of dead branches in canopy'])
    dead_branch_percent_high = float(tree_block.attributes['high']['% of dead branches in canopy'])
    dead_branch_percent = random.uniform(dead_branch_percent_low, dead_branch_percent_high)

    print(f'expected % dead branches in {tree_block} is {dead_branch_percent}')

    desired_dead_voxel_count = round(total_voxel_count * (dead_branch_percent / 100))

    current_dead_voxel_count = sum(voxel['voxel_type'] in ['isDeadOnly', 'isBoth'] for voxel in tree_block.voxel_grid.values())

    print(f"current % dead branches: {(current_dead_voxel_count/total_voxel_count)*100}")
    print(f"Total voxels: {total_voxel_count}")
    print(f"Desired dead voxels: {desired_dead_voxel_count}")
    print(f"Current dead voxels: {current_dead_voxel_count}")



    # If the current count is less than desired, convert neither and lateral voxels to dead and both
    if current_dead_voxel_count < desired_dead_voxel_count:
        # First go with isNone type
        for voxel in tree_block.voxel_grid.values():
            if voxel['voxel_type'] == 'isNeither' and current_dead_voxel_count < desired_dead_voxel_count:
                #print(f"Converting a voxel from 'isNeither' to 'isDeadOnly'")
                voxel['voxel_type'] = 'isDeadOnly'
                current_dead_voxel_count += 1

        # Then go to isLateral type
        for voxel in tree_block.voxel_grid.values():
            if voxel['voxel_type'] == 'isLateralOnly' and current_dead_voxel_count < desired_dead_voxel_count:
                #print(f"Converting a voxel from 'isLateralOnly' to 'isBoth'")
                voxel['voxel_type'] = 'isBoth'
                current_dead_voxel_count += 1

    # If the current count is higher than desired, convert dead voxels to neither
    elif current_dead_voxel_count > desired_dead_voxel_count:
        # First start with isBoth type
        for voxel in tree_block.voxel_grid.values():
            if voxel['voxel_type'] == 'isBoth' and current_dead_voxel_count > desired_dead_voxel_count:
                #print(f"Converting a voxel from 'isBoth' to 'isLateralOnly'")
                voxel['voxel_type'] = 'isLateralOnly'
                current_dead_voxel_count -= 1

        # Then go to isDead type
        for voxel in tree_block.voxel_grid.values():
            if voxel['voxel_type'] == 'isDeadOnly' and current_dead_voxel_count > desired_dead_voxel_count:
                #print(f"Converting a voxel from 'isDeadOnly' to 'isNeither'")
                voxel['voxel_type'] = 'isNeither'
                current_dead_voxel_count -= 1

    print(f"Adjusted dead voxels to: {current_dead_voxel_count}")


def assign_voxel_colors(tree_block: TreeBlock):
    # Create a colormap instance (using your default json file)
    cmap = Colormap()

    # Define a dictionary to map voxel types to colors
    voxel_type_to_color = {
        'isNeither': cmap.get_categorical_colors(1, 'MyColorMap')[0],  # replace 'MyColorMap' with the actual colormap name
        'isLateralOnly': cmap.get_categorical_colors(1, 'MyColorMap')[0],
        'isDeadOnly': cmap.get_categorical_colors(1, 'MyColorMap')[0],
        'isBoth': cmap.get_categorical_colors(1, 'MyColorMap')[0],
    }

    # Assign colors to voxels
    for voxel in tree_block.voxel_grid.values():
        voxel_type = voxel['voxel_type']
        voxel['color'] = voxel_type_to_color[voxel_type]
def add_ground_cover_voxels(tree_block, ground_cover_attributes, voxel_size=0.5):
    def get_center_of_bounding_box(point_cloud):
        """Calculates and returns the center of the bounding box of the given point cloud."""
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        center = (min_coords + max_coords) / 2
        return center
    
    def round_to_increment(value, increment):
        return round(value / increment) * increment

    def add_log_voxel(log_length, x, y, z, direction, synthesized_type):
        for i in range(log_length):
            if direction == 'x':
                log_voxel = (x + i * voxel_size, y, z)
            else:  # direction == 'z'
                log_voxel = (x, y, z + i * voxel_size)

            if log_voxel not in tree_block.voxel_grid:
                new_voxel = {
                    'x': log_voxel[0],
                    'y': log_voxel[1],
                    'z': log_voxel[2],
                    'synthesized_type': synthesized_type,
                    'voxel_type': synthesized_type,
                }
                tree_block.voxel_grid[log_voxel] = new_voxel
    
    tree_block_center = (0,0,0)

    for attribute, synthesized_type in ground_cover_attributes:
        low = float(tree_block.attributes['low'][attribute])
        high = float(tree_block.attributes['high'][attribute])
        value = random.uniform(low, high)

        # If the attribute is litter_cover, it's a percentage
        if attribute == '% of litter cover (10 m radius of tree)':
            total_voxels = (10/voxel_size)**2  # Total voxels in the 10m x 10m area
            value = (value / 100) * total_voxels
            print(f"For attribute {attribute}, {value} of ground should be covered)")
        else:
            print(f"For attribute {attribute}, {value} voxels should be present)")


        number_of_voxels = round(value)
        print(number_of_voxels)
        print(f'this is approximately {number_of_voxels} voxels.')


        created_voxels = 0

        # Iterate over the number of voxels
        for _ in range(number_of_voxels):
            # Get a random point within a 10m radius, clamping the coordinates to the nearest voxel increment
            x = round_to_increment(random.uniform(tree_block_center[0]-10, tree_block_center[0]+10), voxel_size)
            y = 0
            z = round_to_increment(random.uniform(tree_block_center[1]-10, tree_block_center[1]+10), voxel_size)

            if (x, y, z) not in tree_block.voxel_grid:
                # Create a new voxel if it doesn't exist already
                new_voxel = {
                    'x': x,
                    'y': y,
                    'z': z,
                    'synthesized_type': synthesized_type,
                    'voxel_type': synthesized_type,
                }

                tree_block.voxel_grid[(x, y, z)] = new_voxel

                created_voxels += 1

                # If this is a log voxel, add additional log voxels in the y or z direction
                if synthesized_type == 'fallen_logs':
                    log_length = random.randint(3, 6)
                    print(log_length)
                    direction = random.choice(['x', 'z'])
                    add_log_voxel(log_length, x, y, z, direction, synthesized_type)

        print(f"Successfully created {created_voxels} voxels for attribute {attribute}.")


def assign_color_to_dominant_attribute(dominant_attribute: str) -> Tuple[float, float, float]:
    """colors = {
        'isDeadOnly': (1, 0, 0),     # Red: RGB value of (1, 0, 0) ""pretty high intensity colour"
        'isLateralOnly': (0, 1, 0),  # Green: RGB value of (0, 1, 0) "low intensity colour - a medium grey"
        'isBoth': (0, 0, 1),         # Blue: RGB value of (0, 0, 1) "a very high intensity colour"
        'isNeither': (.25, .25, .25), # Gray: RGB value of (.25, .25, .25) "lowest colour - a lighter grey"
        'hollows': (1, 1, 0),        # Yellow: RGB value of (1, 1, 0) "the highest intensity colour"
        'epiphytes': (0, 1, 1),      # Cyan: RGB value of (0, 1, 1) "a very high intensity colour"
        'peeling_bark': (1, 0, 1)    # Magenta: RGB value of (1, 0, 1) "moderate intesnity coloyr - a dark grey"
    }"""

    """    colors = {
        'isNeither': (0.8, 0.8, 0.8),  # Light gray
        'isLateralOnly': (0.6, 0.6, 0.6),  # Medium gray

        # Picking color from 'inferno' colormap for diversity and not grey
        'peeling_bark': (0.795666, 0.220803, 0.339161),  # Moderate intensity

        # More intense colors from 'viridis' colormap
        'isDeadOnly': (0.152566, 0.392007, 0.682171),  # Moderately intense
        'isBoth': (0.122312, 0.633153, 0.530398),  # Highly intense

        # From 'plasma' colormap
        'epiphytes': (0.827018, 0.184201, 0.422179),  # Moderately intense color

        # Most intense color from 'viridis' colormap
        'hollows': (0.280046, 0.004866, 0.329415),  # Highly intense color
    }"""
    colors = {
        'isNeither': (0.8, 0.8, 0.8),  # Light gray
        'isLateralOnly': (0.2, 0.2, 0.2),  # Darker gray

        # Colors from 'viridis' colormap
        'peeling_bark': (0.267004, 0.004874, 0.329415),  # Moderate intensity, deep purple
        'isDeadOnly': (0.20803, 0.718701, 0.472873),  # Light intensity, bright green
        'isBoth': (0.993248, 0.906157, 0.143936),  # High intensity, bright yellow

        # Vivid colors
        'hollows': (1, 0, 1),  # Vivid pink
        'epiphytes': (0, 1, 1),  # Vivid cyan

        #ground
        'fallen_logs' :  (0.8, 0.4, 0), #Darker Orange
        'litter_cover' : (0.7, 0.7, 0.7),  # Lighter gray
    }


    return colors.get(dominant_attribute, (1, 1, 1))  # Default color to white (1, 1, 1) if key not found



def convert_voxel_grid_to_point_cloud(tree_block, point_size):
    def voxel_grid_to_mesh(voxel_grid_3d: o3d.geometry.VoxelGrid, voxel_size: float):
        voxels=voxel_grid_3d.get_voxels()
        vox_mesh=o3d.geometry.TriangleMesh()

        scaling = 0.8

        for v in voxels:
            cube=o3d.geometry.TriangleMesh.create_box(width=1 * scaling, height=1  * scaling,
            depth=1  * scaling)
            cube.paint_uniform_color(v.color)
            cube.translate(v.grid_index, relative=False)
            vox_mesh+=cube

        # Scale by the voxel size
        vox_mesh.scale(voxel_size, center=(0,0,0))

        # Merge close vertices to ensure the mesh is manifold
        #vox_mesh.merge_close_vertices(0.0000001)

        # Compute vertex normals for shading
        vox_mesh.compute_vertex_normals()

        return vox_mesh


    def voxel_grid_to_point_cloud(voxel_grid: Dict[str, Dict[str, Any]], point_size: float = 0.01) -> o3d.geometry.PointCloud:
        points = []
        colors = []
        for voxel in voxel_grid.values():
            voxel['color'] = assign_color_to_dominant_attribute(voxel['voxel_type'])
            #points.append([voxel['x'], voxel['y'], voxel['z']])
            points.append([voxel['x'], voxel['z'], voxel['y']])

            colors.append(voxel['color'])
        

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.scale(1/point_size, center=pcd.get_center())

        return pcd

    def point_cloud_to_voxel_grid(pcd: o3d.geometry.PointCloud, voxel_size: float = 0.5) -> o3d.geometry.VoxelGrid:
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
        return voxel_grid

    """Convert the voxel grid of a tree block to a point cloud."""
    # Generate a point cloud and visualize it
    pcd = voxel_grid_to_point_cloud(tree_block.voxel_grid, point_size)
    
    # Convert point cloud to voxel grid
    voxel_grid_3d = point_cloud_to_voxel_grid(pcd, .5)

    voxel_mesh = voxel_grid_to_mesh(voxel_grid_3d, voxel_size)

    # Return the 3D voxel grid
    return voxel_mesh


# Attributes to convert to voxels
attributes_to_voxel = [
    ('Number of hollows', 'hollows'),
    ('Number of epiphytes', 'epiphytes'),
    ('% of peeling bark cover on trunk/limbs', 'peeling_bark')
]

# Define your desired point size
point_size = 1
voxel_size = .25

print(pivot_df)
processedBlocks = tree_blocks
#processedBlocks = [pivot_df.loc['large', 'minimal'], pivot_df.loc['large', 'maximum']]

def finalise_colors(tree_block):
    def voxel_grid_to_point_cloud(voxel_grid: Dict[str, Dict[str, Any]]):
        points = []
        colors = []
        for voxel in voxel_grid.values():
            voxel['color'] = assign_color_to_dominant_attribute(voxel['voxel_type'])
            points.append([voxel['x'], voxel['y'], voxel['z']])
            colors.append(voxel['color'])

        return points, colors

    """Finalise colors of a voxel grid of a tree block and return the point cloud, points and colors."""
    # Generate a point cloud
    points, colors = voxel_grid_to_point_cloud(tree_block)

    # Save points and colors to the tree block object
    tree_block.points = points
    tree_block.colors = colors



import pyvista as pv

def open3d_mesh_to_pyvista_polydata_with_vertex_colors(mesh):
    # Fetch the vertices, faces and colors from the Open3D mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)

    # Convert Open3D's faces format to PyVista's faces format
    faces = np.empty((triangles.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles

    # Create the PyVista mesh (PolyData)
    pv_mesh = pv.PolyData(vertices, faces.flatten())

    # Add vertex colors to mesh
    pv_mesh.point_data["Colors"] = colors  # Assign colors to points

    return pv_mesh

def capture_views(pv_mesh, name, views=(0, 90, 180, 270), screenshot_res=(1500,1500), camera_dist=3):
    
    # Center of the mesh
    center = pv_mesh.center

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=True, window_size=screenshot_res)

    # Add the mesh to the plotter
    plotter.add_mesh(pv_mesh, scalars="Colors", lighting=True, rgb=True)

    # Set up lighting
    light = pv.Light(light_type='cameralight', intensity=2)
    light.specular = 0.5  # Reduced specular reflection
    plotter.add_light(light)
    plotter.enable_eye_dome_lighting()

    # Capture screenshots from different views
    for view in views:
        # Set the view
        plotter.view_vector((-np.sin(np.deg2rad(view)), -np.cos(np.deg2rad(view)), 0.3), viewup=(0, 0, 1)) # adjust the third component of the view vector to change the tilt

        # Zoom to maintain a constant distance
        plotter.camera.zoom(camera_dist)
        # Capture screenshot
        plotter.screenshot(f'{name}_{view}_degree.png')

    plotter.close()





def capture_views2(pv_mesh, name, views=(0, 90, 180, 270), screenshot_res=(1500,1500), camera_dist=3):
    def rotate_position(position, azimuth, center=(0,0,0)):
        # Convert azimuth to radians
        azimuth_rad = np.deg2rad(azimuth)
        
        # Calculate distances from center
        dx = position[0] - center[0]
        dy = position[1] - center[1]
        
        # Calculate rotated position
        xp = dx*np.cos(azimuth_rad) - dy*np.sin(azimuth_rad) + center[0]
        yp = dx*np.sin(azimuth_rad) + dy*np.cos(azimuth_rad) + center[1]
        
        return (xp, yp, position[2])  # Keep the same z value# Define the views (angles) for rotation# Create a PyVista plotter
    
    plotter = pv.Plotter(off_screen=True, window_size=screenshot_res)

    # Add the mesh to the plotter
    plotter.add_mesh(pv_mesh, scalars="Colors", lighting=True, rgb=True)

    # Set up lighting
    light = pv.Light(light_type='cameralight', intensity=2)
    plotter.add_light(light)
    plotter.enable_eye_dome_lighting()

    # Set camera position to have a 45 degree azimuthal angle
    camera_pos = (1,1,.5)
    focal_point = (0,0,0)
    view_up = (0,0,1)
    plotter.camera_position = [camera_pos, focal_point, view_up]

    plotter.camera.parallel_projection = True
    plotter.camera.position = [x * 3 for x in plotter.camera.position]
    plotter.camera.clipping_range = (0.1, 1000)  # Adjust as necessary for your scene

    # Capture screenshots from different views
    for view in views:
        # Rotate camera position
        new_camera_pos = rotate_position(camera_pos, view)
        plotter.camera_position = [new_camera_pos, focal_point, view_up]

        # After setting the camera position
        plotter.reset_camera()

        # Capture screenshot
        plotter.screenshot(f'{name}_{view}_degree.png')



# Apply create_voxel_grid and additional processing to each TreeBlock in tree_blocks
for tree_block in processedBlocks:
    # Create voxel grid for the tree block
    tree_block.voxel_grid = create_voxel_grid(tree_block)

    # Set the attribute values for the tree block
    set_attributes_for_tree_block(tree_block, df_attributes)

    # Synthesize voxels for the tree block
    synthesize_voxels(tree_block, attributes_to_voxel)

    # Adjust dead voxel count for the tree block
    expected_dead_voxels = adjust_dead_voxel_count(tree_block, df_attributes)

    # Add ground cover
    ground_cover_attributes = [
    ('Number of fallen logs (> 10 cm DBH 10 m radius of tree)', 'fallen_logs'),
    ('% of litter cover (10 m radius of tree)', 'litter_cover'),
    ]
    add_ground_cover_voxels(tree_block, ground_cover_attributes, voxel_size)

    #asign colours
    #finalise_colors(tree_block)


    # Convert the voxel grid of the tree block to a point cloud
    voxel_grid_3d = convert_voxel_grid_to_point_cloud(tree_block, point_size)

    
    #visualize_treeblock_with_voxelization(tree_block,0.5)

    print("converted the tree block into an open3d mesh")
    


    # Visualize voxel grid
    # Convert to PyVista mesh
    pv_mesh = open3d_mesh_to_pyvista_polydata_with_vertex_colors(voxel_grid_3d)

    print("converted the tree block into a pyvista mesh")

    capture_views(pv_mesh, f'{tree_block.control_level}_{tree_block.size}')

    """ # Create PyVista plotter
    plotter = pv.Plotter()
    plotter.window_size = (1500, 1500)


    # Add the mesh to the plotter
    plotter.add_mesh(pv_mesh, scalars="Colors", lighting=True, rgb=True)
    print(pv_mesh)

    light = pv.Light(light_type='scenelight', intensity=1)
    #plotter.add_light(light)

    # Add a camera light
    light2 = pv.Light(light_type='cameralight', intensity=2)
    plotter.add_light(light2)


    # Enable Eye Dome Lighting (EDL)
    plotter.enable_eye_dome_lighting()

    # Set camera position to have a 45 degree azimuthal angle
    # You might need to adjust the position values depending on your specific use case
    #camera_pos = (1,1,1)
    camera_pos = (1,1,.5)

    focal_point = (0,0,0)
    
    view_up = (0,0,1)  # This is the Y direction
    plotter.camera_position = [camera_pos, focal_point, view_up]
    # Set near clipping distance

    plotter.show(auto_close=False)  # Prevent plotter from closing after show


    # Set parallel projection
    plotter.camera.parallel_projection = True

    plotter.camera.position = [x * 3 for x in plotter.camera.position]

    plotter.reset_camera()

    name = f'{tree_block.control_level}_{tree_block.size}'

    def capture_views(plotter, file_name):
        # Define four camera positions
        camera_positions = [
            ((100, 100, 100), (0, 0, 0), (1,1,.5)),
            ((-100, 100, 100), (0, 0, 0), (1,1,.5)),
            ((-100, -100, 100), (0, 0, 0), (1,1,.5)),
            ((100, -100, 100), (0, 0, 0), (1,1,.5))
        ]
        
        for i, camera_position in enumerate(camera_positions):
            # Set the camera position
            plotter.camera_position = camera_position

            # Set screenshot file name
            screenshot_name = f"{file_name}_{i}.png"
            
            # Capture screenshot
            plotter.screenshot(screenshot_name)


    capture_views(plotter, name)
    # Close the plotter
    plotter.close()"""

# Usage:
# capture_views(plotter, mesh, 'your_file_name', resolution=(800, 600))




    print("shown the tree block")

   # print(f"\nTree Block: {tree_block}")
   # print(f"Total number of voxels: {total_voxels}")
   # print(f"Number of dead voxels (isDead and isBoth): {dead_voxels}")
   # print(f"Expected number of dead voxels (based on % dead in canopy): {expected_dead_voxels}")
    #print(f"Number of voxels changed: {tree_block.voxels_changed}")



def print_voxel_stats(tree_blocks):
    print(tree_block)
    for tree_block in tree_blocks:
        voxel_types = [voxel['voxel_type'] for voxel in tree_block.voxel_grid.values()]
        voxel_counter = Counter(voxel_types)
        print(f"\nTree Block: {tree_block}")
        print("Voxel count by type:")
        for voxel_type, count in voxel_counter.items():
            print(f"    {voxel_type}: {count}")


# Now you can call this function after creating and processing all tree_blocks:
#print_voxel_stats(tree_blocks)

print("done")

"""
Step 1: Read the CSV Files

In this step, we're reading CSV files that contain important data regarding tree mapping, branch predictions, and attributes into pandas DataFrames. This raw data will be used in the upcoming steps for data processing and visualization.

Step 2: Define the Block and TreeBlock 

The Block and TreeBlock classes represent the primary structures for processing and visualizing tree data. The Block class includes essential properties such as a point cloud, name, and various conditions, while the TreeBlock class extends the Block class, adding specific attributes for tree blocks like control level, tree ID value, size, and more.

Step 3: Define Utility Functions

To support the voxel grid creation and attribute synthesis, various utility functions are defined. These include adding branch information to a voxel, computing the voxel's dominant attribute, assigning a color based on this dominant attribute, creating a voxel grid from a TreeBlock, and selecting voxels for synthesis based on certain conditions.

Step 4: Main Code Execution

This step includes several sub-steps that make up the main part of the program.

Step 4.1: Re-reading the CSV Files

CSV files are re-read here to ensure their content is accessible for further processing. This step might not be necessary if the data from Step 1 is already accessible here.

Step 4.2: Create TreeBlock Objects

Using the data from the CSV files and other relevant details, TreeBlock objects are created. Each TreeBlock represents a unique tree and encapsulates all necessary data for voxel grid creation and attribute synthesis.

Step 4.3: Attribute Synthesis (Branch-based)

In this step, for each TreeBlock, we perform attribute synthesis using the branch data contained in tree_block.otherData. Here, the branch types are counted for each voxel (isDead, isLateral, isBoth, isNone), and the dominant attribute for each voxel is determined based on the longest cumulative branch length.

Step 4.4: Attribute Synthesis (Synthesized Attributes)

For each TreeBlock, additional synthesized attributes are obtained from tree_block.attributes and added to the voxel grid. These attributes are typically statistical properties derived from the data.

Step 4.5: Voxel Selection and Conversion

Based on attribute priority, voxels are selected for conversion starting with the 'isNeither' type. If additional voxels are needed, 'isLateralOnly' type voxels are considered next, followed by 'isDeadOnly' and 'isBoth' types. The selected voxels are then converted by assigning the synthesized attribute types to them.

Step 4.6: Voxel Grid Creation and Visualization

For each TreeBlock, a voxel grid is created from the branch data and the synthesized attribute information. Then, the voxel grid is converted into a point cloud using the Open3D library for visualization purposes.

Step 4.7: Visualize the Voxel Grid

Lastly, the created voxel grids are visualized using the capabilities of the Open3D library, providing an interactive way to explore the synthesized tree block data.

The entire process takes raw data about trees, processes it, synthesizes new attribute data, incorporates this data into a voxel grid, and then visualizes the result. 
"""