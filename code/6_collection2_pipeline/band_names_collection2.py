import pandas as pd

# Path to the Excel file
csv_file_path = 'c:/Users/jdper/Desktop/collection_2_dataset_unsplit.xlsx'
# Load the Excel file into a DataFrame
df = pd.read_excel(csv_file_path)

# Dictionary mapping Landsat versions to their band names
band_names = {
    "Landsat_4_5_7": {
        "B1": "BLUE",
        "B2": "GREEN",
        "B3": "RED",
        "B4": "NIR",
        "B5": "SWIR1",
        "B6": "SWIR2",
        "B7": "ATMOS_OPACITY",
        "B8": "CLOUD_QA",
        "B9": "TIR",
        "B10": "ATMOS_TRANSMITTANCE",
        "B11": "CLOUD_DISTANCE",
        "B12": "DOWNWARD_RADIANCE",
        "B13": "EMISSITIVITY",
        "B14": "EMISSITIVITY_SD",
        "B15": "QA",
        "B16": "UPWARD_RADIANCE",
        "B17": "UPWARD_RADIANCE_SD",
        "B18": "PIXEL_QA",
        "B19": "RADIOMETRIC_SATURATION_QA"
    },
    "Landsat_8_9": {
        "B1": "COASTAL_AEROSOL",
        "B2": "BLUE",
        "B3": "GREEN",
        "B4": "RED",
        "B5": "NIR",
        "B6": "SWIR1",
        "B7": "SWIR2",
        "B8": "AEROSOL_QA",
        "B9": "TIR",
        "B10": "ATMOS_TRANSMITTANCE",
        "B11": "CLOUD_DISTANCE",
        "B12": "DOWNWARD_RADIANCE",
        "B13": "EMISSITIVITY",
        "B14": "EMISSITIVITY_SD",
        "B15": "QA",
        "B16": "UPWARD_RADIANCE",
        "B17": "UPWARD_RADIANCE_SD",
        "B18": "PIXEL_QA",
        "B19": "RADIOMETRIC_SATURATION_QA"
    }
}

def rename_columns_based_on_landsat(df):
    # Create a new DataFrame to store renamed columns
    df_renamed = df.copy()

    # Apply renaming based on Landsat version
    for landsat_version, names_map in band_names.items():
        if landsat_version == "Landsat_4_5_7":
            version_rows = df['Landsat'].isin(['Landsat4', 'Landsat5', 'Landsat7'])
        elif landsat_version == "Landsat_8_9":
            version_rows = df['Landsat'].isin(['Landsat8', 'Landsat9'])
        else:
            continue

        # Map old column names to new names based on the dictionary
        for old, new in names_map.items():
            if old in df.columns:
                # If the new column already exists, append the data
                if new in df_renamed.columns:
                    df_renamed.loc[version_rows, new] += df.loc[version_rows, old]
                else:
                    df_renamed.loc[version_rows, new] = df.loc[version_rows, old]

    # Drop any remaining old band columns that are not needed
    old_cols = set([col for sublist in band_names.values() for col in sublist])
    cols_to_drop = [col for col in df.columns if col in old_cols]
    df_renamed.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    return df_renamed

df_renamed = rename_columns_based_on_landsat(df)
df_renamed.to_excel('c:/Users/jdper/Desktop/collection_2_dataset_unsplit_renamed.xlsx', index=False)