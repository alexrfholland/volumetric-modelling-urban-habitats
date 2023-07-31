from __future__ import annotations
from typing import List, Tuple, Any
import numpy as np
import time
try:
    from . import block_inserter
except ImportError:
    import block_inserter

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

    TODO: Optimize edge check by first checking if the block center lies near the Octree edge.
    """

    # Helper function to check if a point lies on the edge or outside the Octree bounds
    def is_outside_or_on_edge(point: np.ndarray, min_corner: np.ndarray, max_corner: np.ndarray) -> bool:
        return np.any(point <= min_corner) or np.any(point >= max_corner)

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
            octree.update_ancestral_nodes(node, block_id, attribute['type'])

            node.split(octree.max_depth + 1)


@timing_decorator
def distribute_changes(octree: 'CustomOctree', resource_type_from: str, resource_type_to: str, amount: int, block_id: int, temperature: float) -> None:
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
    zones = octree.get_leaf_and_block_nodes(min_offset_level=2, max_offset_level=10, block_id=block_id)['single_block_nodes']

    if not zones:
        warnings.warn(f"WARNING: No zones with block ID {block_id} and resource type {resource_type_from} were found. Moving on to the next block.")
        return

    print("Step 2: Calculating the frequency of the resource type to change from in each zone")
    zone_frequencies = {}
    for zone in zones:
        resource_from_frequency = zone[0].resource_types.count(resource_type_from)
        resource_to_frequency = zone[0].resource_types.count(resource_type_to)
        zone_frequencies[zone] = {resource_type_from : resource_from_frequency, resource_type_to : resource_to_frequency}
        print(f"Zone has a frequency of {resource_from_frequency} for resource type {resource_type_from} and a frequency of {resource_to_frequency} for resource type {resource_type_to}.")

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

    print("Step 4: Checking if all changes are done...")
    if changes_done < amount:
        warnings.warn(f"WARNING for BLOCK ID {block_id}. {changes_done} nodes converted to {resource_type_to}. Not all planned changes were done. {amount - changes_done} changes are still needed.")
    else:
        print(f"{changes_done} nodes converted to {resource_type_to}. All planned changes were successfully done for Block ID {block_id}")


def distribute_changesLAST(octree: 'CustomOctree', resource_type_from: str, resource_type_to: str, amount: int, block_id: int, temperature: float) -> None:
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
    """
    
    print(f'searching for nodes with block id {block_id} and resource type {resource_type_from}')
    
    print("Step 1: Identifying zones of the block")
    zones = octree.get_leaf_and_block_nodes(min_offset_level=2, max_offset_level=10, block_id=block_id)['single_block_nodes']
    print(zones)

    print("Step 2: Calculating the frequency of the resource type to change from in each zone")
    zone_frequencies = {}
    for zone in zones:
        resource_from_frequency = zone[0].resource_types.count(resource_type_from)
        resource_to_frequency = zone[0].resource_types.count(resource_type_to)
        zone_frequencies[zone] = {resource_type_from : resource_from_frequency, resource_type_to : resource_to_frequency}
        print(f"Zone has a frequency of {resource_from_frequency} for resource type {resource_type_from} and a frequency of {resource_to_frequency} for resource type {resource_type_to}.")

    # The lambda function is used to define a custom sorting order for the sorted function. 
    # In this case, the lambda function takes in x, which is a tuple that contains two elements: a zone and a dictionary.
    # The dictionary, x[1], maps resource types to their frequencies in the given zone.
    # This lambda function returns a tuple (x[1][resource_type_to], x[1][resource_type_from]), which is used to sort the zones.
    # The sorted function first sorts the zones by the frequency of 'resource_type_to' in descending order (because reverse=True).
    # If two zones have the same frequency of 'resource_type_to', then they are further sorted by the frequency of 'resource_type_from' in descending order.
    sorted_zones = sorted(zone_frequencies.items(), key=lambda x: (x[1][resource_type_to], x[1][resource_type_from]), reverse=True)
    print(f"Zones sorted by frequency of the resource type {resource_type_to} and then sorted by {resource_from_frequency}")

    print("Step 3: Distributing changes...")
    changes_done = 0
    for zone, _ in sorted_zones:
        leaf_nodes = octree.get_leaves(zone[0], block_id=block_id, resource_type=resource_type_from)
        #print(f'number of lead nodes is {len(leaf_nodes)} of type {resource_type_from}')
        for leaf in leaf_nodes:
            #print(f'leaf children are {leaf.children}')
            leaf.resource_types[0] = resource_type_to
            #print(f'leaf resource types are now {leaf.resource_types}')
            changes_done += 1
            if changes_done >= amount: # or random.random() < 1 - temperature:
                break

    print("Step 4: Checking if all changes are done...")
    if changes_done < amount:
        warnings.warn(f"WARNING for BLOCK ID {block_id}. {changes_done} nodes converted to {resource_type_to}. Not all planned changes were done. {amount - changes_done} changes are still needed.")
    else:
        print(f"{changes_done} nodes converted to {resource_type_to}. All planned changes were successfully done for Block ID {block_id}")


