
#step 1. identify tree coordinates, size and control level

#step 2. generate point clouds and attributes that represent tree blocks

#step 3. add tree blocks to the octree.

#step 4. update tree block nodes in octree with additional attributes based on the statistical surveying

#step 4.1 convert existing leaf nodes to a different reource type (peeling bark, dead branches, epipyhtes, hollows)
#step 4.1 generate new leaf nodes to define ground resources (leaf litter and logs)

from . import block_inserter
from . import urbanforestparser as UrbanForestParser
from . import block_info


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
    tree_points, tree_attributes, tree_block_ids, sizeList, blockList = block_info.generate_tree_blocks(treeAttributes)

    # Step 3: Add tree blocks to the octree
    block_inserter.add_block(octree, tree_points, tree_attributes, tree_block_ids)
    
    return octree, sizeList, blockList

def update_tree_attributes(octree, sizeList, blockList):


    """
    This function updates the tree block nodes in the octree with additional attributes.

    Parameters:
    octree (CustomOctree): The octree object.
    max_tree_count (int): The maximum tree count.

    Returns:
    octree (CustomOctree): The updated octree object.
    """

    print(f'range of blocklist is {range(len(blockList))}')
    # For each block in the octree
    for i in range(len(blockList)):
        # Calculate the number of nodes to update
        #resourceStats = block_info.calculate_leaf_nodes(sizeList[i], 'minimal')
        resourceStats = block_info.calculate_leaf_nodes(sizeList[i], 'maximum')
        print(f'{i}: stats for block {blockList[i]} of size {sizeList[i]} are {resourceStats}')

        # Distribute the changes in the octree
        #block_inserter.distribute_changes(octree, blockList[i], 1, resourceStats)
        block_inserter.distribute_changes_across_resources(octree, resourceStats, blockList[i], 1)

    return octree
