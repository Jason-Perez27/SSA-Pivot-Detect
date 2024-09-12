import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = '/home/waves/data/SSA-Pivot-Detect/data/3_script_data/5Landsat_total_data.csv'
df = pd.read_csv(csv_file_path)

qa_columns = ['QA_PIXEL']
for col in qa_columns:
    df[col] = df[col].fillna(0).astype(int) 

columns_to_drop = ['Surface_Reflectance_Aerosol', 'Atmospheric_Opacity', 'Radiometric_Saturation_QA']
df.drop(columns=columns_to_drop, inplace=True)

def decode_pixel_qa(value):
    decoded = {
        'Dilated_Cloud': 'Yes' if (value & (1 << 1)) != 0 else 'No',
        'Cirrus': 'Yes' if (value & (1 << 2)) != 0 else 'No',
        'Cloud': 'Yes' if (value & (1 << 3)) != 0 else 'No',
        'Cloud_Shadow': 'Yes' if (value & (1 << 4)) != 0 else 'No',
        'Snow': 'Yes' if (value & (1 << 5)) != 0 else 'No',
        'Clear': 'Yes' if (value & (1 << 6)) != 0 else 'No',
        'Water': 'Yes' if (value & (1 << 7)) != 0 else 'No',
    }
    return decoded

def decode_qa_bands(row):
    landsat_version = row['Landsat']
    if landsat_version in [7, 8, 9, 4, 5]:
        row.update(decode_pixel_qa(row['QA_PIXEL']))
    return row

# Apply the decoding functions to each row in the DataFrame
df = df.apply(decode_qa_bands, axis=1)

# Save the updated DataFrame back to CSV
df.to_csv('/home/waves/data/SSA-Pivot-Detect/data/3_script_data/decoded_qa_bands.csv', index=False)