import pandas as pd

# Load the CSV file
csv_file = '/Users/jasonperez/filtered_no_clouds.csv'
data = pd.read_csv(csv_file)

# Define a function to convert Label to Label ID
def label_to_id(label):
    if label == "Active CP":
        return "1"
    elif label == "Inactive CP":
        return "0"
    elif label == "No CP":
        return "0"
    else:
        return None

# Apply the function to create a new 'Label ID' column
data['Label ID'] = data['Label'].apply(label_to_id)

# Ensure 'Label ID' is in the eighth column position
column_order = data.columns.tolist()
column_order.insert(7, column_order.pop(column_order.index('Label ID')))
data = data[column_order]

# Save the modified DataFrame back to CSV
data.to_csv('/Users/jasonperez/filtered_no_clouds_label_id.csv', index=False)

print("CSV file has been processed and saved with the new 'Label ID' column.")