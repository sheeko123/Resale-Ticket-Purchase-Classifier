import pandas as pd

# Read the CSV file
df = pd.read_csv('p17processed_data.csv')

# Print column names
print("Available columns in the CSV file:")
print(df.columns.tolist())

# Define columns to check for duplicates (using timestamp instead of listing_time)
duplicate_cols = ['event_name', 'zone', 'section', 'row', 'timestamp', 'price', 'quantity']

# Find duplicates
duplicates = df[df.duplicated(subset=duplicate_cols, keep=False)]

# Convert 'quantity' to integer to avoid sorting issues
duplicates['quantity'] = duplicates['quantity'].astype(int)

# Print results
print('\nNumber of duplicate listings:', len(duplicates))
print('\nDuplicate listings:')
# Only use unique columns for display and sorting
print(duplicates[duplicate_cols].sort_values(by=duplicate_cols).head(10)) 