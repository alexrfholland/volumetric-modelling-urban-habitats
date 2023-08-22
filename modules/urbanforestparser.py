import pandas as pd
import numpy as np
from scipy.spatial import KDTree
#import matplotlib.pyplot as plt



def load_urban_forest_data(min_corner, max_corner, ground_points_list):

    ground_points = convert_to_np_format(ground_points_list)

    parkshift = [-318660.15415893, -5814580.27203155, -0]

    # Load the data
    df = pd.read_csv('data/trees-park-site.csv')
    
    # Apply the translation to the easting and northing columns
    df['Easting'] = df['Easting'] + parkshift[0]
    df['Northing'] = df['Northing'] + parkshift[1]

    # Filter the data by bounding box
    mask = (
        (df['Easting'] >= min_corner[0]) & 
        (df['Easting'] <= max_corner[0]) & 
        (df['Northing'] >= min_corner[1]) & 
        (df['Northing'] <= max_corner[1])
    )
    df = df[mask]

    # Convert points to numpy array and add z-coordinate
    points = df[['Easting', 'Northing']].values
    points = np.concatenate([points, np.zeros((points.shape[0], 1))], axis=1)
    points = find_z_points(points, ground_points)
    
    # Split the points into separate x, y, and z arrays
    x, y, z = points[:,0], points[:,1], points[:,2]
    
    # Update the trees DataFrame to include the new x, y, z coordinates
    trees = df
    trees['X'] = x
    trees['Y'] = y
    trees['Z'] = z  # Assuming you want the Z-coordinate under column 'Z'

    # Remove rows where 'Diameter Breast Height' is NaN
    before_rows = trees.shape[0]
    trees = trees.dropna(subset=['Diameter Breast Height'])
    after_rows = trees.shape[0]

    # Print the number of rows removed
    print(f'Removed {before_rows - after_rows} rows with NaN in the "Diameter Breast Height" column.')

    # Add a new column 'Tree Size' based on 'Diameter Breast Height'
    trees['Tree Size'] = trees['Diameter Breast Height'].apply(lambda x: 'small' if x < 50 else ('medium' if 50 <= x < 80 else 'large'))

    # Print the number of each tree size
    tree_size_counts = trees['Tree Size'].value_counts()
    print(tree_size_counts)

    print(trees)

    return points, trees




def find_z_points(points, ground_points):
    ground_tree = KDTree(ground_points[:,:2])
    dist, idx = ground_tree.query(points[:,:2])
    points[:,2] = ground_points[idx, 2]
    return points


def convert_to_np_format(ground_points):
    return np.array(ground_points)

"""
def main():
    
    # Test bounding box
    min_corner = [-100, -100]
    max_corner = [100, 100]

    ground_points_list = [(-21.07499694, 2.50499726, 6.1621579), (-38.15499877, 0.12499238, 15.74215782)]
    
    points, trees = load_urban_forest_data(min_corner, max_corner, ground_points_list)
    
    # Plot the points
    plt.scatter(points[:,0], points[:,1])
    plt.xlim(min_corner[0], max_corner[0])
    plt.ylim(min_corner[1], max_corner[1])
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Urban Forest Data')
    plt.show()


if __name__ == "__main__":
    main()"""