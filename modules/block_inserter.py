from __future__ import annotations
from typing import List, Tuple, Any
import numpy as np

try:
    from . import block_inserter
except ImportError:
    import block_inserter

#see Python's "forward references". Forward references are string literals that specify the fully qualified name of a type.

import random
import warnings
from tqdm import tqdm  # Importing the tqdm module for the progress bar

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

    TODO: 
        Optimize edge check by first checking if the block center lies near the Octree edge.
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
            node.block_ids.append(block_id)
            node.resource_types.append(attribute['type'])

            # Append the block_id and resource types to the found node and all its ancestors
            octree.update_block_ids_and_resource_types(node, block_id, attribute['type'])

            node.split(octree.max_depth + 1)



def add_block2(octree: 'CustomOctree', points: np.ndarray, attributes: List[dict], block_ids: List[int]) -> None:
    """
    Add a block of points to the Octree.

    Each block is a collection of points and their associated attributes. Blocks might be spread over multiple nodes in the octree.
    Each point is represented by a tuple (x, y, z) and has a set of attributes (e.g., RGB, intensity, etc.) represented as a dictionary.
    Each point also has a block_id that represents which block this point belongs to.

    Args:
        octree ('Octree'): The Octree object to which the points are added.
        points (np.ndarray): A numpy array containing the 3D coordinates of the points.
        attributes (List[dict]): A list of dictionaries, each representing the attributes of a corresponding point.
        block_ids (List[int]): A list of block IDs, each corresponding to a point.

    Returns:
        None
    """
    for point, attribute, block_id in zip(points, attributes, block_ids):
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
            node.block_ids.append(block_id)
            node.resource_types.append(attribute['type'])

            # Append the block_id and resource types to the found node and all its ancestors
            octree.update_block_ids_and_resource_types(node, block_id, attribute['type'])

            node.split(octree.max_depth + 1)

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
    """
    print("Step 1: Identifying zones of the block")
    zones = octree.get_leaf_and_block_nodes(min_offset_level=2, max_offset_level=10, block_id=block_id)['single_block_nodes']

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
        print(f'number of lead nodes is {len(leaf_nodes)} of type {resource_type_from}')
        for leaf in leaf_nodes:
            print(f'leaf children are {leaf.children}')
            leaf.resource_types[0] = resource_type_to
            print(f'leaf resource types are now {leaf.resource_types}')
            changes_done += 1
            if changes_done >= amount: # or random.random() < 1 - temperature:
                break

    print("Step 4: Checking if all changes are done...")
    if changes_done < amount:
        warnings.warn(f"{changes_done} nodes converted to {resource_type_to}. Not all planned changes were done. {amount - changes_done} changes are still needed.")
    else:
        print("All planned changes were successfully done.")





def distribute_changesB(octree: 'CustomOctree', change_type_to: str, resource_type: str, amount: int, block_id: int, temperature: float) -> None:
    """
    Distribute changes across the Octree, changing a certain amount of resources of a certain type to another type.

    Args:
        change_type_to (str): The type to change the resources to.
        resource_type (str): The type of resources to change.
        amount (int): The number of resources to change.
        block_id (int): The block ID to restrict the changes to.
        temperature (float): The temperature parameter controlling randomness (0 = totally random, 1 = totally focused).

    Returns:
        None
    """
    # 1. Find the local root of the block
    local_root = octree.find_lowest_common_ancestor_of_block(octree.root, block_id)

    if not local_root:
        print(f"No local root found for block_id {block_id} in octree node with block ids of {octree.root.block_ids}.")
        return


    # 2. Get the zones (children of the local root)
    zones = local_root.children

    print(f"Found {len(zones)} zones for block_id {block_id}.")

    # 3. Calculate the number of changes per zone
    total_leaves = sum([len(octree.get_leaves(zone, block_id, resource_type)) for zone in zones])
    changes_per_zone = [len(octree.get_leaves(zone, block_id, resource_type)) / total_leaves for zone in zones]

    zone_changes = dict()
    changes_needed = amount

    for zone, changes_ratio in zip(zones, changes_per_zone):
        if temperature == 0:  # Totally random distribution
            changes = changes_needed / len(zones)
        elif temperature == 1:  # Totally focused distribution
            changes = changes_needed * changes_ratio
        else:  # Weighted distribution
            changes = changes_needed * ((1 - temperature) / len(zones) + temperature * changes_ratio)
        zone_changes[zone] = changes

    print(f"Planned changes for {len(zone_changes)} zones.")

    # 4. Make the changes in each zone
    changes_done = 0
    for zone, changes in zone_changes.items():
        # Get the specific number of leaves we want
        leaves = octree.get_leaves(zone, block_id, resource_type, int(changes))

        if not leaves:
            warnings.warn(f"No leaves found with block_id {block_id} and resource_type {resource_type} in zone {zone.block_ids}.")
            continue

        # Apply the changes to each leaf
        for leaf in leaves:
            try:
                if leaf.resource_types[0] != resource_type:
                    raise ValueError(f"Unexpected resource_type found in leaf {leaf.block_ids}. Expected {resource_type}, found {leaf.resource_types[0]}.")
                leaf.resource_types[0] = change_type_to
                changes_done += 1
            except IndexError:
                warnings.warn(f"No resource_types found in leaf {leaf.block_ids}.")
                continue
            except ValueError as e:
                print(str(e))
                changes_needed += 1
                continue

    print(f"Changes done: {changes_done}. Changes planned: {amount}.")
    if changes_done < amount:
        warnings.warn(f"Changes done: {changes_done}. Not all planned changes were done. {amount - changes_done} changes are still needed.")


        
#without the resource type code
def add_block2(octree: 'CustomOctree', points: np.ndarray, attributes: List[dict], block_ids: List[int]) -> None:
    """
    Add a block of points to the Octree.

    Each block is a collection of points and their associated attributes. Blocks might be spread over multiple nodes in the octree.
    Each point is represented by a tuple (x, y, z) and has a set of attributes (e.g., RGB, intensity, etc.) represented as a dictionary.
    Each point also has a block_id that represents which block this point belongs to.

    Args:
        octree ('Octree'): The Octree object to which the points are added.
        points (np.ndarray): A numpy array containing the 3D coordinates of the points.
        attributes (List[dict]): A list of dictionaries, each representing the attributes of a corresponding point.
        block_ids (List[int]): A list of block IDs, each corresponding to a point.

    Returns:
        None
    """

    for point, attribute, block_id in zip(points, attributes, block_ids):
        # Find the appropriate node to insert this point into
        node, quadrant = octree.find_node_for_point(point)

        # If the point is not within any existing child node, create a new one
        if node is octree.root or quadrant is not None:
            min_corner, max_corner = node.calculate_bounds_for_point(point)
            child = octree.create_child_node(min_corner, max_corner, np.array([point]), [attribute], [block_id], node.depth + 1)
            node.children.append(child)
            child.parent = node
            # Append the block_id to the current node and all its ancestors
            octree.update_block_ids(node, block_id)

            child.split(octree.max_depth + 1)

        else:
            # Append the point, attribute, and block_id to the found leaf node
            node.points = np.append(node.points, [point], axis=0)
            node.attributes.append(attribute)
            node.block_ids.append(block_id)
                
            # Append the block_id to the found node and all its ancestors
            octree.update_block_ids(node, block_id)

            node.split(octree.max_depth + 1)
    
import random
import warnings
from typing import List

def change_attributes_old(octree: 'Octree', change_type_to: str, node_types_to_convert: List[str], amount: int, block_id: int, temperature: float) -> None:
    """
    Change the attributes of certain nodes in the Octree.

    Find leaf nodes of a certain block ID that match one of the specified node types to convert,
    and change their type to a new type. The selection between nodes can be random or by 'growing' 
    based on the temperature parameter.

    Args:
        octree (Octree): The Octree to change the nodes in.
        change_type_to (str): The type to change the nodes to.
        node_types_to_convert (List[str]): The types of nodes to convert.
        amount (int): The number of nodes to convert.
        block_id (int): The block ID to restrict the search to.
        temperature (float): Controls the randomness of the node selection. At 0, nodes are entirely selected at random. 
                            At 1, nodes are entirely selected by 'growing'. 

    Returns:
        None
    """
    # Get all the leaf nodes of the block with the specified ID
    leaf_nodes = octree.get_leaves(octree.root, block_id)

    if not leaf_nodes:
        warnings.warn(f"No leaf nodes found with block_id {block_id}.")
        return

    print(f"Found {len(leaf_nodes)} leaf nodes with block_id {block_id}.")

    # Filter the leaf nodes that match one of the node types to convert
    nodes_to_convert = []
    #attribute_name = 'Branch.type'
    attribute_name = 'type'
    for node in leaf_nodes:
        try:
            node_type = node.attributes[0][attribute_name]  # Attempt to access 'Branch.type' key in attributes
            if node_type in node_types_to_convert:
                nodes_to_convert.append(node)
        except KeyError:
            # If 'type' key is not found in the attributes, issue a warning
            warnings.warn(f"The node with attributes {node.attributes[0]} does not have a {attribute_name} key.")

    if not nodes_to_convert:
        warnings.warn(f"No nodes found with types {node_types_to_convert} in block_id {block_id}.")
        return

    print(f"Found {len(nodes_to_convert)} nodes with types {node_types_to_convert}.")

    # Change the type of the nodes to convert and their siblings
    changed_nodes_count = 0
    for node in nodes_to_convert:
        if random.random() > temperature:
            # With probability (1 - temperature), start growing from a new randomly chosen node
            node = random.choice(nodes_to_convert)
            
        siblings = octree.get_siblings(node)
        for sibling in siblings:
            if sibling.attributes[0][attribute_name] in node_types_to_convert:
                sibling.attributes[0]['type'] = change_type_to
                changed_nodes_count += 1
                if changed_nodes_count >= amount:
                    break
        if changed_nodes_count >= amount:
            break

    # Add a warning if fewer nodes were changed than requested
    if changed_nodes_count < amount:
        warnings.warn(f"Only {changed_nodes_count} nodes were available to convert, less than requested {amount}.")

    print(f"Converted {changed_nodes_count} nodes to type {change_type_to}. Change of attributes completed.")



###

from tqdm import tqdm  # Importing the tqdm module for the progress bar

def change_attributes_current(octree: 'Octree', change_type_to: str, node_types_to_convert: List[str], amount: int, block_id: int, temperature: float) -> None:
    """
    Change the attributes of certain nodes in the Octree.

    Args:
        octree (Octree): The Octree to change the nodes in.
        change_type_to (str): The type to change the nodes to.
        node_types_to_convert (List[str]): The types of nodes to convert.
        amount (int): The number of nodes to convert.
        block_id (int): The block ID to restrict the search to.
        temperature (float): The temperature parameter controlling randomness (0 = totally random, 1 = totally growing)

    Returns:
        None
    """
    
    print('getting leaf nodes...')
    leaf_nodes = octree.get_leaves(octree.root, block_id)
    if not leaf_nodes:
        warnings.warn(f"No leaf nodes found with block_id {block_id}.")
        return

    print(f"Found {len(leaf_nodes)} leaf nodes with block_id {block_id}.")

    nodes_to_convert = []
    checked_nodes = set()
    attribute_name = 'type'
    for node in leaf_nodes:
        try:
            node_type = node.attributes[0][attribute_name] 
            siblings = octree.get_siblings(node)
            if node_type in node_types_to_convert and siblings:  # Only add nodes that have siblings
                nodes_to_convert.append(node)
                node.attributes[0]['grow_depth'] = 0  # Ensure every node in nodes_to_convert has 'grow_depth' attribute
        except KeyError:
            warnings.warn(f"The node with attributes {node.attributes[0]} does not have a {attribute_name} key.")

    if not nodes_to_convert:
        warnings.warn(f"No nodes found with types {node_types_to_convert} in block_id {block_id}.")
        return

    print(f"Found {len(nodes_to_convert)} nodes with types {node_types_to_convert}.")

    origin_nodes_count = 0
    changed_nodes_count = 0
    node = random.choice(nodes_to_convert)  # Choose an initial origin node

    with tqdm(total=amount, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
        while changed_nodes_count < amount:
            siblings = [s for s in octree.get_leaves(node, block_id, change_type_to) if s not in checked_nodes]
            if not siblings:
                # If no siblings left, move up to the parent node and then down to its leaf nodes
                node = node.parent
                if 'grow_depth' not in node.attributes[0]:
                    node.attributes[0]['grow_depth'] = 0  # If the parent node doesn't have 'grow_depth', initialize it
                node.attributes[0]['grow_depth'] += 1
                siblings = octree.get_leaves(node, block_id, change_type_to)
            elif random.random() > temperature:
                # With probability (1 - temperature), start growing from a new randomly chosen node
                node = random.choice(nodes_to_convert)
                node.attributes[0]['grow_depth'] = 0
                origin_nodes_count += 1
                print(f"Chose new origin node. Total origin nodes so far: {origin_nodes_count}.")
                siblings = octree.get_leaves(node, block_id, change_type_to)

            for sibling in siblings:
                try:
                    if sibling.attributes[0][attribute_name] in node_types_to_convert and sibling not in checked_nodes:
                        sibling.attributes[0]['type'] = change_type_to
                        sibling.attributes[0]['grow_depth'] = node.attributes[0]['grow_depth'] + 1  # Increment grow depth from origin node
                        checked_nodes.add(sibling)
                        changed_nodes_count += 1
                        pbar.update(1)
                        if changed_nodes_count >= amount:
                            break
                except KeyError:
                    print(f"Could not find {attribute_name} in node attributes: {sibling.attributes[0]}, {sibling.block_ids}")
            if changed_nodes_count >= amount:
                break
            if changed_nodes_count >= len(nodes_to_convert):
                print("All possible nodes have been converted, stopping...")
                break
            node.attributes[0]['grow_depth'] += 1
      # Increment grow depth for next iteration

     # Increment grow depth for next iteration

        """while changed_nodes_count < amount:
            siblings = [s for s in octree.get_siblings(node) if s not in checked_nodes]
            if not siblings:
                # If no siblings left, move up to the parent node
                node = node.parent
                if 'grow_depth' not in node.attributes[0]:
                    node.attributes[0]['grow_depth'] = 0  # If the parent node doesn't have 'grow_depth', initialize it
                node.attributes[0]['grow_depth'] += 1
            elif random.random() > temperature:
                # With probability (1 - temperature), start growing from a new randomly chosen node
                node = random.choice(nodes_to_convert)
                node.attributes[0]['grow_depth'] = 0
                origin_nodes_count += 1
                print(f"Chose new origin node. Total origin nodes so far: {origin_nodes_count}.")

            siblings = [s for s in octree.get_siblings(node) if s not in checked_nodes]
            for sibling in siblings:
                try:
                    if sibling.attributes[0][attribute_name] in node_types_to_convert and sibling not in checked_nodes:
                        sibling.attributes[0]['type'] = change_type_to
                        sibling.attributes[0]['grow_depth'] = node.attributes[0]['grow_depth'] + 1  # Increment grow depth from origin node
                        checked_nodes.add(sibling)
                        changed_nodes_count += 1
                        pbar.update(1)
                        if changed_nodes_count >= amount:
                            break
                except KeyError:
                    print(f"Could not find {attribute_name} in node attributes: {sibling.attributes[0]}, {sibling.block_ids}")
            if changed_nodes_count >= amount:
                break
            if changed_nodes_count >= len(nodes_to_convert):
                print("All possible nodes have been converted, stopping...")
                break
            node.attributes[0]['grow_depth'] += 1  # Increment grow depth for next iteration"""

    print(f"Converted {changed_nodes_count} nodes to type {change_type_to}. Origin nodes count: {origin_nodes_count}. Change of attributes completed.")



def change_attributes(octree: 'Octree', change_type_to: str, node_types_to_convert: List[str], amount: int, block_id: int, temperature: float) -> None:
    """
    Change the attributes of certain nodes in the Octree.

    Args:
        octree (Octree): The Octree to change the nodes in.
        change_type_to (str): The type to change the nodes to.
        node_types_to_convert (List[str]): The types of nodes to convert.
        amount (int): The number of nodes to convert.
        block_id (int): The block ID to restrict the search to.
        temperature (float): The temperature parameter controlling randomness (0 = totally random, 1 = totally growing)

    Returns:
        None
    """
    node = octree.find_leaf_node(octree.root, block_id, node_types_to_convert)

    if node is None:
        warnings.warn(f"No leaf node found with block_id {block_id} and types {node_types_to_convert}.")
        return

    nodes_to_convert = set()  # use a set for faster lookup and removal
    nodes_to_convert.add(node)

    origin_nodes_count = 0
    changed_nodes_count = 0
    node.attributes[0]['grow_depth'] = 0

    while changed_nodes_count < amount and nodes_to_convert:
        if random.random() > temperature:
            # With probability (1 - temperature), start growing from a new randomly chosen node
            node = octree.find_leaf_node(octree.root, block_id, node_types_to_convert)
            if node is not None:
                nodes_to_convert.add(node)
                node.attributes[0]['grow_depth'] = 0
                origin_nodes_count += 1

        siblings = octree.get_siblings(node)

        for sibling in siblings:
            if sibling in nodes_to_convert and sibling.attributes[0]['type'] in node_types_to_convert:
                sibling.attributes[0]['type'] = change_type_to
                sibling.attributes[0]['grow_depth'] = node.attributes[0]['grow_depth'] + 1  # Increment grow depth from origin node
                nodes_to_convert.remove(sibling)  # remove converted node from nodes_to_convert
                changed_nodes_count += 1
                if changed_nodes_count >= amount:
                    break
        node.attributes[0]['grow_depth'] += 1  # Increment grow depth for next iteration

    print(f"Converted {changed_nodes_count} nodes to type {change_type_to}. Origin nodes count: {origin_nodes_count}. Change of attributes completed.")



def change_attributesB(octree: 'Octree', change_type_to: str, node_types_to_convert: List[str], amount: int, block_id: int, temperature: float) -> None:
    """
    Change the attributes of certain nodes in the Octree.

    Args:
        octree (Octree): The Octree to change the nodes in.
        change_type_to (str): The type to change the nodes to.
        node_types_to_convert (List[str]): The types of nodes to convert.
        amount (int): The number of nodes to convert.
        block_id (int): The block ID to restrict the search to.
        temperature (float): The temperature parameter controlling randomness (0 = totally random, 1 = totally growing)

    Returns:
        None
    """
    leaf_nodes = octree.get_leaves(octree.root, block_id)

    if not leaf_nodes:
        warnings.warn(f"No leaf nodes found with block_id {block_id}.")
        return

    print(f"Found {len(leaf_nodes)} leaf nodes with block_id {block_id}.")

    nodes_to_convert = []
    attribute_name = 'type'
    for node in leaf_nodes:
        try:
            node_type = node.attributes[0][attribute_name] 
            if node_type in node_types_to_convert:
                nodes_to_convert.append(node)
        except KeyError:
            warnings.warn(f"The node with attributes {node.attributes[0]} does not have a {attribute_name} key.")

    if not nodes_to_convert:
        warnings.warn(f"No nodes found with types {node_types_to_convert} in block_id {block_id}.")
        return

    print(f"Found {len(nodes_to_convert)} nodes with types {node_types_to_convert}.")

    origin_nodes_count = 0
    changed_nodes_count = 0
    node = random.choice(nodes_to_convert)  # Choose an initial origin node
    node.attributes[0]['grow_depth'] = 0
    while changed_nodes_count < amount:
        if random.random() > temperature:
            # With probability (1 - temperature), start growing from a new randomly chosen node
            node = random.choice(nodes_to_convert)
            node.attributes[0]['grow_depth'] = 0
            origin_nodes_count += 1
        siblings = octree.get_siblings(node)
        for sibling in siblings:
            if sibling.attributes[0][attribute_name] in node_types_to_convert:
                sibling.attributes[0]['type'] = change_type_to
                sibling.attributes[0]['grow_depth'] = node.attributes[0]['grow_depth'] + 1  # Increment grow depth from origin node
                changed_nodes_count += 1
                if changed_nodes_count >= amount:
                    break
        node.attributes[0]['grow_depth'] += 1  # Increment grow depth for next iteration
    print(f"Converted {changed_nodes_count} nodes to type {change_type_to}. Origin nodes count: {origin_nodes_count}. Change of attributes completed.")

    
    
def change_attributes2(octree: 'Octree', change_type_to: str, node_types_to_convert: List[str], amount: int, block_id: int) -> None:
    """
    Change the attributes of certain nodes in the Octree.

    Find leaf nodes of a certain block ID that match one of the specified node types to convert,
    and change their type to a new type.

    Args:
        octree (Octree): The Octree to change the nodes in.
        change_type_to (str): The type to change the nodes to.
        node_types_to_convert (List[str]): The types of nodes to convert.
        amount (int): The number of nodes to convert.
        block_id (int): The block ID to restrict the search to.

    Returns:
        None
    """
    # Get all the leaf nodes of the block with the specified ID
    leaf_nodes = octree.get_leaves(octree.root, block_id)

    if not leaf_nodes:
        warnings.warn(f"No leaf nodes found with block_id {block_id}.")
        return

    print(f"Found {len(leaf_nodes)} leaf nodes with block_id {block_id}.")

    # Filter the leaf nodes that match one of the node types to convert
    nodes_to_convert = []
    attribute_name = 'Branch.type'
    for node in leaf_nodes:
        try:
            node_type = node.attributes[0][attribute_name]  # Attempt to access 'Branch.type' key in attributes
            if node_type in node_types_to_convert:
                nodes_to_convert.append(node)
        except KeyError:
            # If 'type' key is not found in the attributes, issue a warning
            warnings.warn(f"The node with attributes {node.attributes[0]} does not have a {attribute_name} key.")

    if not nodes_to_convert:
        warnings.warn(f"No nodes found with types {node_types_to_convert} in block_id {block_id}.")
        return

    print(f"Found {len(nodes_to_convert)} nodes with types {node_types_to_convert}.")

    # Randomly select a subset of the nodes to convert if the amount is less than the total number of nodes to convert
    if len(nodes_to_convert) > amount:
        nodes_to_convert = random.sample(nodes_to_convert, amount)
    elif len(nodes_to_convert) < amount:
        warnings.warn(f"Only {len(nodes_to_convert)} nodes are available to convert, less than requested {amount}.")

    print(f"Converting {len(nodes_to_convert)} nodes to type {change_type_to}.")

    # Change the type of the nodes to convert
    for node in nodes_to_convert:
        node.attributes[0]['type'] = change_type_to

    print("Change of attributes completed.")
