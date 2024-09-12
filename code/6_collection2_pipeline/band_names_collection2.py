import pandas as pd

# Path to the CSV file
csv_file_path = '/home/waves/data/SSA-Pivot-Detect/data/3_script_data/5Landsat_total_data.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Dictionary mapping Landsat versions to their band names for Collection 2
band_names = {
    "Landsat_4_5": {
        "SR_B1": "Blue",
        "SR_B2": "Green",
        "SR_B3": "Red",
        "SR_B4": "Near_Infrared",
        "SR_B5": "SWIR1",
        "ST_B6": "Thermal",
        "SR_B7": "SWIR2",
        "QA_PIXEL": "Pixel_QA"
    },
    "Landsat_7_8_9": {
        "SR_B1": "Coastal_Aerosol",
        "SR_B2": "Blue",
        "SR_B3": "Green",
        "SR_B4": "Red",
        "SR_B5": "Near_Infrared",
        "SR_B6": "SWIR1",
        "SR_B7": "SWIR2",
        "ST_B10": "TIRS1",
        "ST_B11": "TIRS2",
        "QA_PIXEL": "Pixel_QA",
        "QA_RADSAT": "Radiometric_Saturation_QA"
    }
}

# Function to rename columns for the entire DataFrame based on Landsat version
def rename_columns_based_on_landsat(df):
    # Determine the Landsat version and apply appropriate column names
    for version_key, names_map in band_names.items():
        # Check if the Landsat version is in the DataFrame
        if any(version_key.split('_')[-1] in s for s in df['Landsat'].unique()):
            # Map old column names to new names based on the dictionary
            rename_dict = {old: new for old, new in names_map.items() if old in df.columns}
            df.rename(columns=rename_dict, inplace=True)

    # Drop any remaining old band columns that are not needed
    old_cols = set([col for sublist in band_names.values() for col in sublist])
    cols_to_drop = [col for col in old_cols if col in df.columns and col not in rename_dict.values()]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return df

# Apply the renaming function to the DataFrame
df_renamed = rename_columns_based_on_landsat(df)

# Save the updated DataFrame back to CSV
df_renamed.to_csv(csv_file_path, index=False)

print("CSV file has been updated with new band names for Landsat Collection 2.")