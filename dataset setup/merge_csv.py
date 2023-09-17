import pandas as pd

# Define the paths to your two CSV file
csv_file1 = "C:/Users/Ayberk/Dev/coords_bigger.csv"
csv_file2 = 'succesful_coords.csv'

# Read both CSV files into pandas DataFrames
df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

# Merge the two DataFrames based on a common column (e.g., 'common_column')
# You can specify the type of merge (inner, outer, left, or right) depending on your needs
merged_df = pd.merge(df1, df2,how='inner')

# Save the merged DataFrame to a new CSV file
merged_csv_file = 'merged_file.csv'
merged_df.to_csv(merged_csv_file, index=False)

print(f"Merged data saved to {merged_csv_file}")