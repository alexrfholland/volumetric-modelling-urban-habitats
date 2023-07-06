
import pandas as pd

# Read the CSV file
csv_file = 'data/sites/park.csv'  # Replace with the correct file path
df = pd.read_csv(csv_file)

print(df)

# Convert to Parquet and save to the desired location
parquet_file = 'data/sites/park.parquet'  # Replace with your desired file path
df.to_parquet(parquet_file)

print("Conversion to Parquet completed!")
