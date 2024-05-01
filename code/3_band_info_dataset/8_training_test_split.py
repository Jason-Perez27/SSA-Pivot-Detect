import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit

# Load the CSV data
csv_file = '/Users/jasonperez/filtered_no_clouds.csv'
data = pd.read_csv(csv_file)


# Initialize GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

# Split the data ensuring that the same 'TIF ID' stays in the same split
train_idx, test_idx = next(gss.split(data, groups=data['TIF ID']))

# Create training and testing datasets
train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

# Save the training and test data to separate CSV files on the desktop
desktop_path = os.path.expanduser("~/Desktop")
train_csv_file = os.path.join(desktop_path, 'bands_info_training.csv')
test_csv_file = os.path.join(desktop_path, 'bands_info_test.csv')

train_data.to_csv(train_csv_file, index=False)
test_data.to_csv(test_csv_file, index=False)

print("Data splitting and saving completed.")
print(f"Training CSV file saved on the desktop: {train_csv_file}")
print(f"Test CSV file saved on the desktop: {test_csv_file}")