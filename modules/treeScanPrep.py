import pandas as pd
import numpy as np

# Load the CSV data into a DataFrame
df = pd.read_csv('data/branchPredictions - full.csv')

# conditions to categorize branches
conditions = [
    (df['Branch.type'] == 'dead') & (df['Branch.angle'] > 20),
    (df['Branch.type'] != 'dead') & (df['Branch.angle'] <= 20),
    (df['Branch.type'] == 'dead') & (df['Branch.angle'] <= 20),
]

# corresponding categories
categories = ['isDeadOnly', 'isLateralOnly', 'isBoth']

# Define a new column 'type' based on existing columns
df['type'] = np.select(conditions, categories, default='isNeither')

# Drop unnecessary columns
df = df.drop(['isNeither', 'isLateralOnly', 'isDeadOnly', 'isBoth'], axis=1)

# Save the modified DataFrame back to a CSV file
df.to_csv('data/edited_branchPredictions - adjusted.csv', index=False)
