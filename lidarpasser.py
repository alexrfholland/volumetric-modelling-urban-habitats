
import pandas as pd
import open3d as o3d
import numpy as np

# Read the CSV file
csv_file = 'data/sites/park.csv'  # Replace with the correct file path
df = pd.read_csv(csv_file)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(df.iloc[:, :3].values)  # Assuming the first 3 columns contain x, y, and z coordinates

points_array = np.asarray(pcd.points)
x_min, x_max = np.min(points_array[:, 0]), np.max(points_array[:, 0])
y_min, y_max = np.min(points_array[:, 1]), np.max(points_array[:, 1])
midpoint = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, 0.0])  # Assuming z = 0.0 for translation

pcd.translate(-midpoint)

translated_points = np.asarray(pcd.points)
translated_attributes = df.iloc[:, 3:].values  # Assuming the attributes start from the fourth column

translated_df = pd.DataFrame(np.concatenate((translated_points, translated_attributes), axis=1), columns=df.columns)
translated_df = translated_df.astype(df.dtypes)

print(translated_df)

# Convert to Parquet and save to the desired location
parquet_file = 'data/sites/park.parquet'  # Replace with your desired file path
translated_df.to_parquet(parquet_file)

print("Conversion to Parquet completed!")