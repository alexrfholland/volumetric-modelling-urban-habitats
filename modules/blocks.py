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

print(pivot_df.loc['moderate', 'medium'])