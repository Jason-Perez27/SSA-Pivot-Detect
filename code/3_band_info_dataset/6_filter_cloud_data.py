import pandas as pd
import ast  

# Load the CSV file into a DataFrame
csv_file_path = '/Users/jasonperez/decoded_qa_bands.csv'
df = pd.read_csv(csv_file_path)

# Convert string representations of dictionaries in 'Pixel_QA' and 'Cloud_QA' to actual dictionaries in python
df['Pixel_QA'] = df['Pixel_QA'].apply(ast.literal_eval)
df['Cloud_QA'] = df['Cloud_QA'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Define a function to filter rows based on cloud conditions
def filter_cloud_conditions(row):
    # Check 'Pixel_QA' for cloud conditions
    pixel_qa = row['Pixel_QA']
    if pixel_qa['Cloud'] != 'No' or pixel_qa['Cloud_Shadow'] != 'No':
        return False

    # Check 'Cloud_QA' for Landsat 4-5 specific conditions
    cloud_qa = row['Cloud_QA']
    if isinstance(cloud_qa, dict):  
        if cloud_qa['Cloud'] != 'Not present' or cloud_qa['Cloud_Shadow'] != 'Not present':
            return False

    return True

# Apply the filter function
df_filtered = df[df.apply(filter_cloud_conditions, axis=1)]

# Save the filtered DataFrame back to CSV
df_filtered.to_csv('/Users/jasonperez/filtered_no_clouds.csv', index=False)

print("Filtered CSV file has been saved.")