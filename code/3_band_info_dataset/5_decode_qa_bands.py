import pandas as pd

# Load the CSV file into a DataFrame
csv_file_path = '/home/waves/data/SSA-Pivot-Detect/data/3_script_data/5Landsat_total_data.csv'
df = pd.read_csv(csv_file_path)

qa_columns = ['Cloud_QA', 'Pixel_QA']
for col in qa_columns:
    df[col] = df[col].fillna(0).astype(int) 

columns_to_drop = ['Surface_Reflectance_Aerosol', 'Atmospheric_Opacity', 'Radiometric_Saturation_QA']
df.drop(columns=columns_to_drop, inplace=True)

def decode_cloud_qa(value):
    decoded = {
        'Cloud': 'Present' if (value & (1 << 1)) != 0 else 'Not present',
        'Cloud_Shadow': 'Present' if (value & (1 << 2)) != 0 else 'Not present',
        'Adjacent_to_Cloud': 'Present' if (value & (1 << 3)) != 0 else 'Not present',
    }
    return decoded


def decode_pixel_qa(value):
    decoded = {
        'Cloud_Shadow': 'Yes' if (value & (1 << 3)) != 0 else 'No',
        'Cloud': 'Yes' if (value & (1 << 5)) != 0 else 'No',
        'Confidence_Cloud': 'High' if (value & (1 << 6)) != 0 else ('Medium' if (value & (1 << 7)) != 0 else 'Low'),
    }
    return decoded



def decode_qa_bands(row):
    landsat_version = row['Landsat']
    if landsat_version in [7, 8, 9]:
        row['Pixel_QA'] = decode_pixel_qa(row['Pixel_QA'])
    elif landsat_version in [4, 5]:
        row['Cloud_QA'] = decode_cloud_qa(row['Cloud_QA'])
        row['Pixel_QA'] = decode_pixel_qa(row['Pixel_QA'])
    return row

# Apply the decoding functions to each row in the DataFrame
df = df.apply(decode_qa_bands, axis=1)

# Save the updated DataFrame back to CSV
df.to_csv('/home/waves/data/SSA-Pivot-Detect/data/3_script_data/decoded_qa_bands.csv', index=False)