import os
import rasterio

# Directory containing the TIF files
tif_directory = '/Users/jasonperez/Downloads/Landsat7'

# Output file path
output_file_path = '/Users/jasonperez/Desktop/7band_names.txt'

# Open the output file in write mode
with open(output_file_path, 'w') as output_file:
    # Iterate over the TIF files
    for tif_name in os.listdir(tif_directory):
        if tif_name.endswith('.tif'):
            tif_path = os.path.join(tif_directory, tif_name)
            output_file.write(f"Band names for {tif_name}:\n")
            with rasterio.open(tif_path) as src:
                for i, name in zip(src.indexes, src.descriptions):
                    output_file.write(f"Band {i}: {name}\n")