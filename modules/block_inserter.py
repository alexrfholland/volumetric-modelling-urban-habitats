from __future__ import annotations
from typing import List, Tuple, Any
import numpy as np
import time

#see Python's "forward references". Forward references are string literals that specify the fully qualified name of a type.

import random
import warnings
from tqdm import tqdm  # Importing the tqdm module for the progress bar


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start:.2f} seconds to run.")
        return result
    return wrapper

# Helper function to check if a point lies on the edge or outside the Octree bounds
def is_outside_or_on_edge(point: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> bool:
    return np.any(point <= min_corner) or np.any(point >= max_corner)
    
def add_block(octree: 'CustomOctree', points: np.ndarray, attributes: List[dict], block_ids: List[int]) -> None:
    """
    Add a block of points to the Octree.

    Each block is a collection of points and their associated attributes. Blocks might be spread over multiple nodes in the octree.
    Each point is represented by a tuple (x, y, z) and has a set of attributes (e.g., RGB, intensity, etc.) represented as a dictionary.
    Each point also has a block_id that represents which block this point belongs to.

    This function also checks if a point lies on the edge of the Octree bounds. If a point lies on the edge or outside, the point is ignored.

    Args:
        octree ('Octree'): The Octree object to which the points are added.
        points (np.ndarray): A numpy array containing the 3D coordinates of the points.
        attributes (List[dict]): A list of dictionaries, each representing the attributes of a corresponding point.
        block_ids (List[int]): A list of block IDs, each corresponding to a point.

    Returns:
        None

    """



    for point, attribute, block_id in zip(points, attributes, block_ids):
        # Check if the point lies on the edge or outside the Octree bounds
        if is_outside_or_on_edge(point, octree.root.min_corner, octree.root.max_corner):
            continue

        # Find the appropriate node to insert this point into
        node, quadrant = octree.find_node_for_point(point)



        # If the point is not within any existing child node, create a new one
        if node is octree.root or quadrant is not None:
            min_corner, max_corner = node.calculate_bounds_for_point(point)
            child = octree.create_child_node(min_corner, max_corner, np.array([point]), [attribute], [block_id], [attribute['type']], node.depth + 1)
            node.children.append(child)
            child.parent = node

            # Append the block_id  and resource types to the current node and all its ancestors
            octree.update_block_ids_and_resource_types(node, block_id, attribute['type'])
            
            if attribute['type'] == 'fallen logs':
                print(f'fallen logs added to node {node} with block id {block_id} to child with resource type {child.resource_types}')

    

            child.split(octree.max_depth + 1)

        else:
            # Append the point, attribute, resourcetype, and block_id to the found leaf node
            node.points = np.append(node.points, [point], axis=0)
            node.attributes.append(attribute)
            
            
            # Append the block_id and resource types to the found node and all its ancestors
            
            #old way 
            # TODO: make generic function ('ie. new way' work that swaps the blockID and nodetype. currently assumes blockID and resourve type
            # lists for ancestral nodes are empty and does not remove the types first)
            node.block_ids.append(block_id)
            node.resource_types.append(attribute['type'])
            octree.update_block_ids_and_resource_types(node, block_id, attribute['type'])

            #new way - currently not working 
            #octree.update_ancestral_nodes(node, block_id, attribute['type'])

            if attribute['type'] == 'fallen logs':
                print(f'fallen logs added to node {node} with block id {block_id} and resource type {node.resource_types}')

            node.split(octree.max_depth + 1)


@timing_decorator
def distribute_changes(octree: 'CustomOctree', resource_type_from: str, resource_type_to: str, amount: int, block_id: int, temperature: float, zones=None, zone_min_offset_level = 2) -> None:
    """
    Distribute changes across the Octree, changing a certain amount of resources from one type to another type.

    The function works as follows:
    1. Identify the zones of the specified block ID without specifying the resource type.
    2. Calculate the frequency of the resource type to change from in each zone by counting its occurrences in the zone's resource_type list. Then, sort these zones by this frequency.
    3. Begin distributing changes in these zones, starting from the one with the highest frequency of the resource type to change from. Find leaf nodes with the resource type to change from, and change their resource type to the desired type. Repeat until the specified amount of changes is done or there's a switch to another zone based on the temperature parameter.
    4. Check if all changes are done.

    Args:
        octree (CustomOctree): The octree object to distribute changes across.
        resource_type_from (str): The type of resources to change.
        resource_type_to (str): The type to change the resources to.
        amount (int): The number of resources to change.
        block_id (int): The block ID to restrict the changes to.
        temperature (float): The temperature parameter controlling randomness. 
        For temperature=1, all changes start from the zone with the highest frequency of the resource type to change from. 
        As temperature decreases, the likelihood of changing zones before this allocation is completed increases.

    Returns:
        None

    TODO: fix temperature parameter
    """
    print(f'Searching for nodes with block id {block_id} and resource type {resource_type_from}')
    
    print("Step 1: Identifying zones of the block")

    # if zones are not provided, compute them
    if zones is None:
        zones = octree.get_leaf_and_block_nodes(min_offset_level=zone_min_offset_level, max_offset_level=10, block_id=block_id)['single_block_nodes']    

    if not zones:
        warnings.warn(f"WARNING: No zones with block ID {block_id} and resource type {resource_type_from} were found. Moving on to the next block.")
        return

    print("Step 2: Calculating the frequency of the resource type to change from in each zone")
    zone_frequencies = {}
    for zone in zones:
        resource_from_frequency = zone[0].resource_types.count(resource_type_from)
        resource_to_frequency = zone[0].resource_types.count(resource_type_to)
        zone_frequencies[zone] = {resource_type_from : resource_from_frequency, resource_type_to : resource_to_frequency}
        #print(f"Zone has a frequency of {resource_from_frequency} for resource type {resource_type_from} and a frequency of {resource_to_frequency} for resource type {resource_type_to}.")

    sorted_zones = sorted(zone_frequencies.items(), key=lambda x: (x[1][resource_type_to], x[1][resource_type_from]), reverse=True)
    print(f"Zones sorted by frequency of the resource type {resource_type_to} and then {resource_type_from}")

    print("Step 3: Distributing changes...")
    changes_done = 0
    for zone, _ in sorted_zones:
        leaf_nodes = octree.get_leaves(zone[0], block_id=block_id, resource_type=resource_type_from)
        for leaf in leaf_nodes:
            #old way 
            leaf.resource_types[0] = resource_type_to
            octree.change_block_ids_and_resource_types(leaf, block_id_from = None, block_id_to=None, resource_type_from = resource_type_from, resource_type_to = resource_type_to)
            
            #new way 
            # TODO: make generic function that works (ie. also swaps the attributes in the block inserter)
            #octree.update_ancestral_nodes(leaf, resource_type_to)
           
            changes_done += 1
            
            if changes_done >= amount:
                break
        if changes_done >= amount:
                break

    print("Step 4: Checking if all changes are done...")
    if changes_done < amount:
        warnings.warn(f"WARNING for BLOCK ID {block_id}. {changes_done} nodes converted to {resource_type_to}. Not all planned changes were done. {amount - changes_done} changes are still needed.")
    else:
        print(f"{changes_done} nodes converted to {resource_type_to}. All planned changes were successfully done for Block ID {block_id}")

    return changes_done

def assignDead(resourceLog, resource_type_from, octree, block_id, temperature, logStats):    
    resource_type_to = 'dead branches'
    tableValue = resourceLog.loc[resourceLog['Attribute'] == resource_type_to, 'Leaf Nodes (Random)'].values[0]
    total_leaf_nodes = octree.root.block_ids.count(block_id)
    amount = round(total_leaf_nodes * tableValue)
    print(f'for block {block_id} there should be {amount} leaf nodes that are {resource_type_to}, or {tableValue} * {total_leaf_nodes} leaf nodes')
    distribute_changes(octree, resource_type_from, resource_type_to, amount, block_id, temperature)

    logStats(block_id, total_leaf_nodes, 'Total Nodes')
    logStats(block_id, amount, 'Planned Changes', resource_type_to)

def assignPeelingBark(resourceLog, resource_type_from, octree, block_id, temperature, logStats):
    resource_type_to = 'peeling bark'    
    tableValue = resourceLog.loc[resourceLog['Attribute'] == resource_type_to, 'Leaf Nodes (Random)'].values[0]
    total_leaf_nodes = octree.root.block_ids.count(block_id)
    amount = round(total_leaf_nodes * tableValue)
    print(f'for block {block_id} there should be {amount} leaf nodes that are {resource_type_to}, or {tableValue} * {total_leaf_nodes} leaf nodes')
    distribute_changes(octree, resource_type_from, resource_type_to, amount, block_id, temperature, zone_min_offset_level = 7)

    logStats(block_id, amount, 'Planned Changes', resource_type_to)


def assignEpiphytesAndHollows(resourceLog, resource_type_from, octree, block_id, logStats):
    for resource_type_to in ['hollows', 'epiphytes']:
        tableValue = resourceLog.loc[resourceLog['Attribute'] == resource_type_to, 'Leaf Nodes (Random)'].values[0]
        total_leaf_nodes = octree.root.block_ids.count(block_id)
        amount = round(tableValue)
        if(amount > 0):
            # Use the values directly from 'Leaf Nodes (Random)'
            distribute_to_higher_elevation_nodes(octree, resource_type_from, resource_type_to, amount, block_id)
        logStats(block_id, amount, 'Planned Changes', resource_type_to)


def assignFallenLogs(block_metadata, resourceLog, octree, block_id, logStats, block_id_with_desired_type):
    print(f'grass and concrete has block id {block_id_with_desired_type}')
    resource_type_to = 'fallen logs'

    # find number of fallen logs to insert into tree block
    # chose a random start point at the same z coordinate as TreePos and between 0 and 10 meters away
    # get a random angle and a random length value between a min and max length (ie. 0.5 to 3m)
    # draw a line between them and create a list of points that are 0.2m along the line
    # use the function octree.find_leaf_node(self, node: 'OctreeNode', block_id: int, node_types: List[str]) -> 'OctreeNode' to get the closest ground node to each point (use block id of 2 as it is ground)
    # check if the distance between the treePos and the ground node is less than 10m. if not, discard the point.
    # if it is, change the z location of the point to the z cordinate of the node + 0.2.
    # create a corrosponding list of dictionaries called 'attributes', one per point, and add the value 'fallen log' with the key 'type'.
    #create a list of blockids, one per point, all repeating the block_id value
    #call add_block(octree: 'CustomOctree', points: np.ndarray, attributes: List[dict], block_ids: List[int]) -> None to insert the points into the octree

    print(f'block metadata is {block_metadata}')

    treePos = block_metadata[block_id]['location']
    tableValue = resourceLog.loc[resourceLog['Attribute'] == resource_type_to, 'Leaf Nodes (Random)'].values[0]
    amount = round(tableValue)  # Number of fallen logs to insert


    for _ in range(amount):
        # Get random start point, angle, and length
        start_point = treePos[0] + random.uniform(0, 10), treePos[1] + random.uniform(0, 10), treePos[2]
        angle = random.uniform(0, 2 * np.pi)
        length = random.uniform(0.5, 3)

        # Generate points along the line
        points = [(start_point[0] + length * np.cos(angle) * t, start_point[1] + length * np.sin(angle) * t, start_point[2]) for t in np.arange(0, 1, 0.2)]

        # Get closest ground nodes and check distance
        new_points = []
        attributes = []
        block_ids = []
        for point in points:
            node, distance = octree.find_closest_leaf_node(treePos, block_id_with_desired_type, return_distance = True) #Find closest grass ground node                    
            if node is not None:
                if distance < 10:
                    new_points.append((point[0], point[1], node.max_corner[2] + octree.compute_node_size(node.depth)/2))  # Change z-coordinate
                    # Create corresponding attributes and block IDs
                    attributes.append({'type': resource_type_to})
                    block_ids.append(block_id)
                else:
                    print(f'{resource_type_to} point {point} is too far from the ground node at {node.center} with distance {distance} and will not be added to the octree')
            else:
                print(f'no node could be found for {resource_type_to} point')
        
        # Insert points into octree
        add_block(octree, np.array(new_points), attributes, block_ids)

        print(f'added {len(new_points)} fallen logs to block {block_id} at positions {new_points}')

    logStats(block_id, amount, 'Planned Changes', resource_type_to)    

    if amount == 0:
        print(f'{amount} {resource_type_to} to add for tree {block_id}, skipping')
    else:
        print(f"adding {amount} {resource_type_to} to {block_metadata[block_id]['size']}, {block_metadata[block_id]['control']} tree {block_id} at position {treePos}")

def assignLeafLitter(block_metadata, resourceLog, octree, block_id, logStats, block_id_with_desired_type):
    resource_type_to = 'leaf litter'
    tableValue = resourceLog.loc[resourceLog['Attribute'] == resource_type_to, 'Leaf Nodes (Random)'].values[0]

    treePos = block_metadata[block_id]['location']
    radius = 10  # 10m radius
    percentage_leaf_litter = tableValue  # % leaf litter

    # Calculate how many points there would be in a 1m x 1m grid in that area
    area = np.pi * (radius**2)
    grid_points = area  # 1m x 1m grid

    # Find the number of leaf litter points
    amount = round(grid_points * percentage_leaf_litter)
    

    print(f"Adding {amount} {resource_type_to} to {block_metadata[block_id]['size']}, {block_metadata[block_id]['control']} tree {block_id} at position {treePos}")

    # Randomly create those points at the cell locations
    points = []
    for _ in range(amount):
        x = treePos[0] + random.uniform(-radius, radius)
        y = treePos[1] + random.uniform(-radius, radius)
        node, distance = octree.find_closest_leaf_node((x, y, treePos[2]), block_id_with_desired_type, return_distance = True)
        
        if node is not None:
            if distance < radius:
                z = node.max_corner[2] + octree.compute_node_size(node.depth)/2
                points.append((x, y, z))
            else:
                print(f'{resource_type_to} point ({x}, {y}, {z}) is too far from the ground node at {node.center} with distance {distance} and will not be added to the octree')
        else:
            print(f'no node could be found for {resource_type_to} point')

    # Create a list of attributes that are key-value pairs with the key being 'type' and the value being 'leaf litter'
    attributes = [{'type': resource_type_to} for _ in range(len(points))]
    
    # Create a list of block IDs, which is just the block ID value repeated for each point
    block_ids = [block_id for _ in range(len(points))]

    # With these 3 lists (list of points, list of attributes, list of block IDs), call add_block
    add_block(octree, np.array(points), attributes, block_ids)
    print(f'Added {len(points)} {resource_type_to} to block {block_id} at positions {points}')

    logStats(block_id, amount, 'Planned Changes', resource_type_to)    


    if amount == 0:
        print(f'{amount} {resource_type_to} to add for tree {block_id}, skipping')
    else:
        print(f"adding {amount} {resource_type_to} to {block_metadata[block_id]['size']}, {block_metadata[block_id]['control']} tree {block_id} at position {treePos}")

    return -1



def distribute_changes_across_resources(octree: 'CustomOctree', df: pd.DataFrame, block_id: int, temperature: float, block_metadata: dict, logStats) -> None:
    """
    Distribute changes across the Octree, iterating through each resource in a DataFrame.

    For each resource, this function calculates the amount to be distributed and then calls 
    the distribute_changes function to handle the actual distribution.

    Args:
        octree (CustomOctree): The octree object to distribute changes across.
        df (pd.DataFrame): A DataFrame containing the changes to be distributed.
        block_id (int): The block ID to restrict the changes to.
        temperature (float): The temperature parameter controlling randomness. 
        blockmetadata (dict)
        logStats (function)

    Returns:
        None
    """
    resource_type_from = 'branch'
    resourceLog = block_metadata[block_id]['resource log'] 

    assignDead(resourceLog, resource_type_from, octree, block_id, temperature, logStats)
    assignPeelingBark(resourceLog, resource_type_from, octree, block_id, temperature, logStats)
    assignEpiphytesAndHollows(resourceLog, resource_type_from, octree, block_id, logStats)

    #calculate ground stuff

    # Get the block id with the desired type
    desired_type = 'grass'
    block_id_with_desired_type = []
    for block_id_in_list, metadata in block_metadata.items():
        if metadata['type'] == desired_type:
            block_id_with_desired_type.append(block_id_in_list)
            break

    # Get the block id with the desired type
    desired_type = 'impermeable ground'
    for block_id_in_list, metadata in block_metadata.items():
        if metadata['type'] == desired_type:
            block_id_with_desired_type.append(block_id_in_list)
            break

    assignFallenLogs(block_metadata, resourceLog, octree, block_id, logStats, block_id_with_desired_type)
    assignLeafLitter(block_metadata, resourceLog, octree, block_id, logStats, block_id_with_desired_type)
        
def distribute_changes_across_resources2(octree: 'CustomOctree', df: pd.DataFrame, block_id: int, temperature: float, block_metadata: dict, logStats) -> None:
    """
    Distribute changes across the Octree, iterating through each resource in a DataFrame.

    For each resource, this function calculates the amount to be distributed and then calls 
    the distribute_changes function to handle the actual distribution.

    Args:
        octree (CustomOctree): The octree object to distribute changes across.
        df (pd.DataFrame): A DataFrame containing the changes to be distributed.
        block_id (int): The block ID to restrict the changes to.
        temperature (float): The temperature parameter controlling randomness. 
        blockmetadata (dict)
        logStats (function)

    Returns:
        None
    """
    resource_type_from = 'branch'

    amount = -1
    verifiedChanges = -1
    # Iterate through each resource in the DataFrame

    for index, row in df.iterrows():
        resource_type_to = row['Attribute']
        tableValue = row['Leaf Nodes (Random)']

        # Check if the resource is 'dead branches' or 'peeling bark'
        if resource_type_to in ['dead branches', '2peeling bark']:
            # Calculate the total number of leaf nodes
            total_leaf_nodes = octree.root.block_ids.count(block_id)
            
            # Calculate the number of leaf nodes to convert
            amount = round(total_leaf_nodes * tableValue)
            print(f'for block {block_id} there should be {amount} leaf nodes that are {resource_type_to}, or {tableValue} * {total_leaf_nodes} leaf nodes')
            logStats(block_id, total_leaf_nodes, 'Total Nodes')

            if resource_type_to == 'peeling bark': #distribute across much finer zones
                distribute_changes(octree, resource_type_from, resource_type_to, amount, block_id, temperature, zone_min_offset_level = 7)
            else:
                distribute_changes(octree, resource_type_from, resource_type_to, amount, block_id, temperature)
        elif resource_type_to in ['hollows', 'epiphytes']:
            if(amount > 0):
                # Use the values directly from 'Leaf Nodes (Random)'
                amount = row['Leaf Nodes (Random)']
                distribute_to_higher_elevation_nodes(octree, resource_type_from, resource_type_to, amount, block_id)

        # Get the block id with the desired type
        desired_type = 'grass'
        block_id_with_desired_type = []
        for block_id_in_list, metadata in block_metadata.items():
            if metadata['type'] == desired_type:
                block_id_with_desired_type.append(block_id_in_list)
                break

                # Get the block id with the desired type
        desired_type = 'impermeable ground'
        for block_id_in_list, metadata in block_metadata.items():
            if metadata['type'] == desired_type:
                block_id_with_desired_type.append(block_id_in_list)
                break

        if resource_type_to == 'fallen logs':
            print(f'{desired_type} has block id {block_id_with_desired_type}')
            
            # find number of fallen logs to insert into tree block
            # chose a random start point at the same z coordinate as TreePos and between 0 and 10 meters away
            # get a random angle and a random length value between a min and max length (ie. 0.5 to 3m)
            # draw a line between them and create a list of points that are 0.2m along the line
            # use the function octree.find_leaf_node(self, node: 'OctreeNode', block_id: int, node_types: List[str]) -> 'OctreeNode' to get the closest ground node to each point (use block id of 2 as it is ground)
            # check if the distance between the treePos and the ground node is less than 10m. if not, discard the point.
            # if it is, change the z location of the point to the z cordinate of the node + 0.2.
            # create a corrosponding list of dictionaries called 'attributes', one per point, and add the value 'fallen log' with the key 'type'.
            #create a list of blockids, one per point, all repeating the block_id value
            #call add_block(octree: 'CustomOctree', points: np.ndarray, attributes: List[dict], block_ids: List[int]) -> None to insert the points into the octree

            print(f'block metadata is {block_metadata}')
            treePos = block_metadata[block_id]['location']
            amount = round(tableValue)  # Number of fallen logs to insert

            #break out of if statement if num_logs == 0
            if amount == 0:
                print(f'{amount} to add for tree {block_id}, skipping')
                continue

            print(f"adding {amount} {resource_type_to} to {block_metadata[block_id]['size']}, {block_metadata[block_id]['control']} tree {block_id} at position {treePos}")

            for _ in range(amount):
                # Get random start point, angle, and length
                start_point = treePos[0] + random.uniform(0, 10), treePos[1] + random.uniform(0, 10), treePos[2]
                angle = random.uniform(0, 2 * np.pi)
                length = random.uniform(0.5, 3)

                # Generate points along the line
                points = [(start_point[0] + length * np.cos(angle) * t, start_point[1] + length * np.sin(angle) * t, start_point[2]) for t in np.arange(0, 1, 0.2)]

                # Get closest ground nodes and check distance
                new_points = []
                attributes = []
                block_ids = []
                for point in points:
                    node, distance = octree.find_closest_leaf_node(treePos, block_id_with_desired_type, return_distance = True) #Find closest grass ground node                    
                    if node is not None:
                        if distance < 10:
                            new_points.append((point[0], point[1], node.max_corner[2] + octree.compute_node_size(node.depth)/2))  # Change z-coordinate
                            # Create corresponding attributes and block IDs
                            attributes.append({'type': resource_type_to})
                            block_ids.append(block_id)
                        else:
                            print(f'{resource_type_to} point {point} is too far from the ground node at {node.center} with distance {distance} and will not be added to the octree')
                    else:
                        print(f'no node could be found for {resource_type_to} point')
                           

                # Insert points into octree
                add_block(octree, np.array(new_points), attributes, block_ids)
                print(f'added {len(new_points)} fallen logs to block {block_id} at positions {new_points}')

            
        elif resource_type_to == 'leaf litter':
            treePos = block_metadata[block_id]['location']
            radius = 10  # 10m radius
            percentage_leaf_litter = tableValue  # % leaf litter


            # Calculate how many points there would be in a 1m x 1m grid in that area
            area = np.pi * (radius**2)
            grid_points = area  # 1m x 1m grid

            # Find the number of leaf litter points
            amount = round(grid_points * percentage_leaf_litter)
            
            if amount == 0:
                print(f'{amount} to add for tree {block_id}, skipping')
                continue

            print(f"Adding {amount} {resource_type_to} to {block_metadata[block_id]['size']}, {block_metadata[block_id]['control']} tree {block_id} at position {treePos}")

            # Randomly create those points at the cell locations
            points = []
            for _ in range(amount):
                x = treePos[0] + random.uniform(-radius, radius)
                y = treePos[1] + random.uniform(-radius, radius)
                node, distance = octree.find_closest_leaf_node((x, y, treePos[2]), block_id_with_desired_type, return_distance = True)
                
                if node is not None:
                    if distance < radius:
                        z = node.max_corner[2] + octree.compute_node_size(node.depth)/2
                        points.append((x, y, z))
                    else:
                        print(f'{resource_type_to} point ({x}, {y}, {z}) is too far from the ground node at {node.center} with distance {distance} and will not be added to the octree')
                else:
                    print(f'no node could be found for {resource_type_to} point')

            # Create a list of attributes that are key-value pairs with the key being 'type' and the value being 'leaf litter'
            attributes = [{'type': resource_type_to} for _ in range(len(points))]
            
            # Create a list of block IDs, which is just the block ID value repeated for each point
            block_ids = [block_id for _ in range(len(points))]

            # With these 3 lists (list of points, list of attributes, list of block IDs), call add_block
            add_block(octree, np.array(points), attributes, block_ids)
            print(f'Added {len(points)} {resource_type_to} to block {block_id} at positions {points}')

        logStats(block_id, amount, 'Planned Changes', resource_type_to)


def distribute_to_higher_elevation_nodes(octree: 'CustomOctree', resource_type_from: str, resource_type_to: str, amount: int, block_id: int) -> None:
    """
    Distribute changes to random higher elevation leaf nodes in the Octree, changing a certain amount of resources from one type to another type.

    Args:
        octree (CustomOctree): The octree object to distribute changes across.
        resource_type_from (str): The type of resources to change.
        resource_type_to (str): The type to change the resources to.
        amount (int): The number of resources to change.
        block_id (int): The block ID to restrict the changes to.

    Returns:
        None
    """
    # Get all leaf nodes in the block with the specified resource type
    leaf_nodes = octree.get_leaves(octree.root, block_id=block_id, resource_type=resource_type_from)

    if not leaf_nodes:
        print(f"No leaf nodes with block ID {block_id} and resource type {resource_type_from} were found. Moving on to the next resource.")
        return

    # Sort the leaf nodes by elevation in descending order
    sorted_leaf_nodes = sorted(leaf_nodes, key=lambda node: node.points[0][2], reverse=True)
    print(f'sorted leaf nodes are {sorted_leaf_nodes}')


    # Calculate the index that represents the top 10% of nodes
    top_10_percent_index = len(sorted_leaf_nodes) * 0.1

    top_10_percent_index = int(top_10_percent_index)
    print(f'top 10% index is {top_10_percent_index}')


    if len(sorted_leaf_nodes) <= amount:
        selected_nodes = sorted_leaf_nodes
    else:
        # Select the top 10% of nodes
        top_10_percent_nodes = sorted_leaf_nodes[:top_10_percent_index]

        if len(top_10_percent_nodes) <= amount:
            selected_nodes = top_10_percent_nodes
        else:
            # Randomly select 'amount' nodes from the top 10%
            selected_nodes = random.sample(top_10_percent_nodes, int(amount))

    # Change the resource type of the selected nodes
    for node in selected_nodes:
        node.resource_types[0] = resource_type_to
        octree.change_block_ids_and_resource_types(
            node,
            block_id_from=None,
            block_id_to=None,
            resource_type_from=resource_type_from,
            resource_type_to=resource_type_to
        )

def changeAngles(octree: 'CustomOctree', block_id: int) -> None:
     # Get all leaf nodes in the block with the specified resource type
    leaf_nodes = octree.get_leaves(octree.root, block_id=block_id, resource_type=['dead branches', 'branch'])
    print(f'found {len(leaf_nodes)} dead branch and branch leaf nodes for block {block_id}')
    sorted_nodes = sorted(leaf_nodes, key=lambda node: node.attributes[0]['Branch.angle'])
    lateralCount = 0
    perchCount = 0
    for node in sorted_nodes:
            if node.attributes[0]['Branch.angle'] <= 20:
                if node.resource_types[0] == 'dead branches':
                    node.resource_types[0] = 'perch branch'
                    perchCount = perchCount + 1
                    octree.change_block_ids_and_resource_types(node, resource_type_from = 'dead branches', resource_type_to = 'perch branch')
                else:
                    node.resource_types[0] = 'lateral branch'
                    octree.change_block_ids_and_resource_types(node, resource_type_from = 'branches', resource_type_to = 'lateral branch')
                    lateralCount = lateralCount + 1

    

    print(f'tree {block_id}: changed {lateralCount} branches to lateral branches and {perchCount} branches to perch branches')
            


