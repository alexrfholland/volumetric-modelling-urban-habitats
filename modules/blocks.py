import pandas as pd

df_attributes = pd.read_csv('./data/lerouxdata.csv')

def control_level_size_df(df_attributes, control_level, size):
    control_level_low = f"{control_level} low"
    control_level_high = f"{control_level} high"

    size_filter = df_attributes['Size'] == size

    df_filtered = df_attributes[size_filter][['Attribute', control_level_low, control_level_high]]

    return df_filtered

# Example usage:
control_level = "maximum"
size = "large"
df_attributes = pd.read_csv('./data/lerouxdata.csv')  # assuming you have this DataFrame

print(control_level_size_df(df_attributes, control_level, size))





class Block:
    def __init__(self, pointcloud=None, name='', conditions=None, attributes=None):
        self.pointcloud = pointcloud
        self.name = name
        self.conditions = conditions if conditions else []
        self.attributes = attributes if attributes else {}

    def __str__(self):
        return f"Block(name={self.name}, conditions={self.conditions}, attributes={self.attributes})"

class TreeBlock(Block):
    def __init__(self, control_level='', tree_id_value=0, size='', otherData=None, **kwargs):
        super().__init__(**kwargs)
        self.control_level = control_level
        self.tree_id_value = tree_id_value
        self.size = size
        self.otherData = otherData if otherData is not None else pd.DataFrame()

    def __str__(self):
        return f"TreeBlock(control_level={self.control_level}, tree_id_value={self.tree_id_value}, size={self.size}, {super().__str__()})"

# read the csv files
df_tree_mapping = pd.read_csv('./data/treemapping.csv')
df_branch_predictions = pd.read_csv('./data/branchPredictions - full.csv')

# create tree blocks
tree_blocks = []
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

# print created tree blocks
for tree_block in tree_blocks:
    print(tree_block)
    #print(tree_block.otherData)
