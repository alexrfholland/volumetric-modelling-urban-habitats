import pandas as pd
import numpy as np
import random


def generate_tree_blocks(df):
    """
    This function generates a tree block based on the input DataFrame.

    Args:
        df (DataFrame): A DataFrame contains trees data with 'X', 'Y', 'Z' and 'Tree Size' columns.

    Returns:
        tuple: The points, attributes, and block_ids of the generated tree block.
    """ 
    def load_and_translate_tree_block_data(dataframe, tree_id, translation, tree_size, tree_block_count):
        print(f"Processing tree id {tree_id}, size {tree_size}")
        block_data = dataframe[dataframe['Tree.ID'] == tree_id].copy()

        # Apply translation
        translation_x, translation_y, translation_z = translation
        block_data['x'] += translation_x
        block_data['y'] += translation_y
        block_data['z'] += translation_z

        # Add control level
        block_data['control'] = 'low'

        tree_block_count = tree_block_count + 1
        print(f'tree_block_count is: {tree_block_count}')

        #block_data['BlockID'] = tree_block_count[tree_size]
        block_data['BlockID'] = tree_block_count

        return block_data, tree_block_count

    def get_tree_ids(tree_size, count):
        tree_id_ranges = {'small': range(0, 5), 'medium': range(10, 17), 'large': range(10, 17)}

        print(f'count is: {count}')
        return random.choices(tree_id_ranges[tree_size], k=count)

    def define_attributes(combined_data):
        #extract the data from these columns
        attributes = combined_data[['Branch.type', 'Branch.angle', 'Tree.size', 'type', 'control']].to_dict('records')
        return attributes

    csv_file = 'data/edited_branchPredictions - adjusted.csv'

    data = pd.read_csv(csv_file)
    print(f"Loaded data with shape {data.shape}")

    # Process block data for each tree size
    tree_block_count = 10
    processed_data = []
    sizeList = []
    blockList =[]

    for tree_size in ['small', 'medium', 'large']:
        tree_size_data = df[df['Tree Size'] == tree_size]
        tree_count = len(tree_size_data)
        print (f'{tree_size} trees count is {tree_count}')
        tree_ids = get_tree_ids(tree_size, tree_count)
        print(f'Loading and processing {tree_ids} {tree_size} tree blocks...')

        # Process block data for each tree
        for tree_id, row in zip(tree_ids, tree_size_data.iterrows()):
            processed_block_data, tree_block_count = load_and_translate_tree_block_data(data, tree_id, (row[1]['X'], row[1]['Y'], row[1]['Z']), tree_size, tree_block_count)
            processed_data.append(processed_block_data)
            blockList.append(tree_block_count)
            sizeList.append(tree_size)
            print(f'processed_block_data for tree with block_id {tree_block_count} and size {tree_size}')

    # Combine the block data
    combined_data = pd.concat(processed_data)

    print(combined_data)

    # Extract points, attributes, and block IDs
    points = combined_data[['x', 'y', 'z']].to_numpy()
    attributes = define_attributes(combined_data)
    block_ids = combined_data['BlockID'].tolist()

    print(f'sizeList is: {sizeList}, blockList is: {blockList}')

    return points, attributes, block_ids, sizeList, blockList

import pandas as pd
import numpy as np

def calculate_leaf_nodes(tree_size: str, control_level: str) -> pd.DataFrame:
    """
    This function takes the number of leaf nodes, tree size, and control level as input,
    calculates how many leaf nodes are needed for each attribute and returns the results as a DataFrame.
    
    Parameters:
    leaf_count (int): The total number of leaf nodes in a block.
    tree_size (str): The size of the tree ('small', 'medium', 'large').
    control_level (str): The control level ('minimal', 'moderate', 'maximum').

    Returns:
    pd.DataFrame: DataFrame containing attributes, control level, and corresponding leaf nodes.
    """
    
    # Load the data
    df = pd.read_csv('data/lerouxdata-long.csv')

    # Create a dictionary to map old attribute names to new ones
    attribute_dict = {'% of peeling bark cover on trunk/limbs': 'peeling bark',
                      '% of dead branches in canopy': 'dead branches',
                      'Number of hollows': 'hollows',
                      'Number of epiphytes' : 'epiphytes',
                      '% of litter cover (10 m radius of tree)': 'leaf litter',
                      'Number of fallen logs (> 10 cm DBH 10 m radius of tree)': 'fallen logs'}
    
    # Replace attribute names in DataFrame
    df['Attribute'] = df['Attribute'].replace(attribute_dict)

    # Filter data based on tree size
    df = df[df['Tree Size'] == tree_size]

    # Normalize the control_level
    control = control_level.capitalize() + ' Control'

    # Create a new DataFrame to store results
    results = pd.DataFrame()

    # Get unique attributes
    unique_attributes = df['Attribute'].unique()

    # Loop over unique attributes and calculate leaf nodes for each attribute
    for attribute in unique_attributes:
        # Filter the data for the given attribute and control_level
        df_attribute = df[(df['Attribute'] == attribute)]

        print(f'df attribute is {df_attribute}')

        # Fetch the low and high estimates
        low_estimate = df_attribute[df_attribute['Estimate Type'] == 'Low'][control].values[0]
        high_estimate = df_attribute[df_attribute['Estimate Type'] == 'High'][control].values[0]

        # If attribute is a percentage, calculate number of leaf nodes based on leaf count
        if attribute in ['peeling bark', 'dead branches', 'leaf litter']:
            leaf_nodes_low = low_estimate / 100
            leaf_nodes_high = high_estimate / 100
        else:  # For absolute attributes, leaf nodes is same as attribute value
            leaf_nodes_low = low_estimate
            leaf_nodes_high = high_estimate
        
        # Generate a random number of leaf nodes within the calculated range
        leaf_nodes_random = np.random.uniform(leaf_nodes_low, leaf_nodes_high)

        if attribute in ['hollows', 'epiphytes']:
            leaf_nodes_random = round(leaf_nodes_random)
            print(f'leaf_nodes_random is {leaf_nodes_random}')
                
        results = pd.concat([results, pd.DataFrame({'Attribute': [attribute], 'Control Level': [control], 
                                                    'Leaf Nodes (Low)': [leaf_nodes_low], 
                                                    'Leaf Nodes (High)': [leaf_nodes_high], 
                                                    'Leaf Nodes (Random)': [leaf_nodes_random]})], ignore_index=True)
    
    results = results[['Attribute', 'Control Level', 'Leaf Nodes (Low)', 'Leaf Nodes (High)', 'Leaf Nodes (Random)']]
    return results

# Usage
leaf_nodes_df = calculate_leaf_nodes('large', 'minimal')
print(leaf_nodes_df)