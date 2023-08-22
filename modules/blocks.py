
#step 1. identify tree coordinates, size and control level

#step 2. generate point clouds and attributes that represent tree blocks

#step 3. add tree blocks to the octree.

#step 4. update tree block nodes in octree with additional attributes based on the statistical surveying

#step 4.1 convert existing leaf nodes to a different reource type (peeling bark, dead branches, epipyhtes, hollows)
#step 4.1 generate new leaf nodes to define ground resources (leaf litter and logs)

from . import block_inserter
from . import urbanforestparser as UrbanForestParser
from . import block_info
import logging

# Global block metadata dictionary
block_metadata = {}

def populate_block_metadata(block_id, _type, location=None, size=None, control=None):
    global block_metadata

    # Populate the block metadata dictionary
    block_metadata[block_id] = {
        'location': location,
        'size': size,
        'control': control,
        'type' : _type
    }

    print(f'Populated block metadata for block {block_id} with location {location}, size {size}, control {control}, and type {_type}')

def log_stats(block_id, amount, column, resource_type=None):
    global block_metadata
    print(block_metadata[block_id])
    
    # Get the DataFrame for the specific block ID
    resource_log_df = block_metadata[block_id]['resource log']

    if resource_type is None:
        resource_log_df[column] = amount
    
    else:
        # Find the row index corresponding to the resource named in resource_type_to
        row_index = resource_log_df[resource_log_df['Attribute'] == resource_type].index[0]

        # Update the 'total nodes' value for the identified row
        resource_log_df.loc[row_index, column] = amount


def generate_tree_blocks_and_insert_to_octree(octree, ground_points, treeAttributes):
    """
    This function takes the octree object, ground points, and tree attributes as input,
    performs the tree block generation and inserts them into the octree.

    Parameters:
    octree (CustomOctree): The octree object.
    ground_points (np.ndarray): The ground points from LiDAR data.
    treeAttributes (dict): The attributes of the trees.

    Returns:
    octree (CustomOctree): The updated octree object.
    max_tree_count (int): The maximum tree count.
    """

    # Step 1: Use urban forest data to find location and size of trees on site 
    # and then find the nearest ground point in octree to each tree
    treeCoords, treeAttributes = UrbanForestParser.load_urban_forest_data(
        octree.root.min_corner, octree.root.max_corner, ground_points)

    # Step 2: Generate tree blocks relevant to the size, location, and type of each tree
    tree_points, tree_attributes, tree_block_ids, sizeList, blockList = block_info.generate_tree_blocks(treeAttributes, populate_block_metadata)

    # Step 3: Add tree blocks to the octree
    block_inserter.add_block(octree, tree_points, tree_attributes, tree_block_ids)

    print(f'Add blocks with IDs {blockList} to octree')

    return octree

def update_tree_attributes(octree):

    """
    This function updates the tree block nodes in the octree with additional attributes.

    Parameters:
    octree (CustomOctree): The octree object.
    max_tree_count (int): The maximum tree count.

    Returns:
    octree (CustomOctree): The updated octree object.
    """
    global block_metadata

    # Extract all block ids where type is 'tree'
    tree_block_ids = [block_id for block_id, metadata in block_metadata.items() if metadata['type'] == 'tree']

    # For each block in the octree
    for id in tree_block_ids:
        # Calculate the number of nodes to update
        size = block_metadata[id]['size']

        if size == 'medium':
            size = 'large'
            
        control = block_metadata[id]['control']
        resourceStats = block_info.calculate_leaf_nodes(size, control)

        block_metadata[id]['resource log'] = resourceStats

        # Distribute the changes in the octree
        block_inserter.distribute_changes_across_resources(octree, resourceStats, id, 1, block_metadata, log_stats)

        print(f'log stats for block {id} of size {size} and control {control} are \n{block_metadata[id]["resource log"]}')

        print(f'Updated block {id} with additional attributes')

        print(f'changing angles for tree block {id}')
        #block_inserter.changeAngles(octree, id)

    return octree

def confirm_tree_attributes(octree):
    # Get all leaf nodes
    leaves = octree.get_leaves(octree.root)

    # Get list of ids of trees
    tree_ids = [key for key, value in block_metadata.items() if value['type'] == 'tree']

    # Partition leaves by block_ids using tree_ids
    partitioned_leaves = {block_id: [] for block_id in tree_ids}
    for leaf in leaves:
        for block_id in leaf.block_ids:
            if block_id in partitioned_leaves:
                partitioned_leaves[block_id].append(leaf)

    for id in tree_ids:
        resourcelog = block_metadata[id]['resource log']

        # Add a new column for Found Totals, initialize with zeros
        resourcelog['Found Totals'] = 0

        for index, row in resourcelog.iterrows():
            resource_type = row['Attribute']

            # Count the number of leaf nodes that have the resource_type in their resource_types and have the specific id
            count = sum(resource_type in leaf.resource_types for leaf in partitioned_leaves[id])

            # Add that value to the column of the row of the resource_type
            resourcelog.loc[index, 'Found Totals'] = count

        print(f'FINAL log stats for block {id} of size {block_metadata[id]["size"]} and control {block_metadata[id]["control"]} are \n{block_metadata[id]["resource log"]}')

