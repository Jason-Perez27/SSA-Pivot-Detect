import rasterio
import csv
import os

# Define paths
tif_folder = '/home/waves/data/SSA-Pivot-Detect/data/2_script_data/SSA_TIF_REQUEST'
csv_file_path = '/home/waves/data/SSA-Pivot-Detect/data/3_script_data/5LS_dataset.csv'
csv_columns = ['Pixel ID', 'TIF ID', 'TIF Name', 'Landsat', 'Year', 'Month', 'X Value', 'Y Value', 'Longitude', 'Latitude', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'sr_aerosol', 'sr_atmos_opacity', 'sr_cloud_qa', 'pixel_qa', 'radsat_qa']

# Prepare CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_columns)

    pixel_id = 1  # Initialize pixel ID counter

    # Process each TIF file
    for tif_filename in os.listdir(tif_folder):
        tif_file_path = os.path.join(tif_folder, tif_filename)
        with rasterio.open(tif_file_path) as src:
            _, _, year, month, landsat = tif_filename.split('_')
            month = month.split('.')[0]

            # Iterate over each pixel in the image
            for y in range(src.height):
                for x in range(src.width):
                    # Transform pixel coordinates (x, y) to geographic coordinates (lon, lat)
                    lon, lat = src.xy(y, x)

                    band_data = []
                    for band_name in csv_columns[11:]:  
                        if band_name in src.descriptions:
                            band_index = src.descriptions.index(band_name) + 1
                            band = src.read(band_index, window=((y, y+1), (x, x+1)))
                            if band.size > 0:
                                band_data.append(band[0, 0])
                            else:
                                band_data.append('N/A')

                    # Write data to CSV, including the Pixel ID, TIF ID, year, month, and pixel data
                    tif_id = int(tif_filename.split('_')[0])  # Assuming TIF ID is part of the filename
                    csv_row = [pixel_id, tif_id, tif_filename, landsat, year, month, x, y, lon, lat] + band_data
                    csv_writer.writerow(csv_row)
                    pixel_id += 1 