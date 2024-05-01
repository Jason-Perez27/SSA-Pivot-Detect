import json
import rasterio
import csv
import os

# Step 1: Read JSON File
json_file_path = '/Users/jasonperez/Desktop/tif_labels.json'
with open(json_file_path, 'r') as json_file:
    annotations_data = json.load(json_file)

# Step 2: Create a mapping of image filenames to annotation data
image_to_annotation = {}
tif_name_to_id = {}  # Add this line
current_tif_id = 1  # Add this line
for annotation in annotations_data:
    filename = annotation['file_upload']
    # Remove unwanted prefix and replace "_RGB.jpg" with ".tif"
    clean_filename = filename.split('-', 1)[1].replace('_RGB.jpg', '.tif')
    if clean_filename not in tif_name_to_id:  # Add this line
        tif_name_to_id[clean_filename] = current_tif_id  # Add this line
        current_tif_id += 1  # Add this line
    image_to_annotation[clean_filename] = annotation

# Step 3: Read the band names from the text file
band_names = {}
with open('/Users/jasonperez/Desktop/band_names.txt', 'r') as band_names_file:
    current_tif_name = None
    for line in band_names_file:
        if line.startswith('Band names for '):
            current_tif_name = line.split('Band names for ')[1].strip().split('.')[0]
            band_names[current_tif_name] = []
        elif line.startswith('Band '):
            band_names[current_tif_name].append(line.split(': ')[1].strip())

# Step 4: Prepare CSV file
csv_file_path = '/Users/jasonperez/Desktop/5LS_dataset.csv'
# Add 'Month', 'Year', and 'TIF ID' to the csv_columns list
csv_columns = ['TIF ID', 'TIF Name', 'Landsat', 'Year', 'Month', 'X Value', 'Y Value', 'Label', 'X-Coord', 'Y-Coord','B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'sr_aerosol', 'sr_atmos_opacity', 'sr_cloud_qa', 'pixel_qa', 'radsat_qa']

# Step 5: Write data to CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_columns)

    tif_folder = '/Users/jasonperez/Desktop/SSA_TIF_REQUEST'
    geojson_folder = '/Users/jasonperez/Desktop/SSA_TIF_GEOJSON'
    for clean_filename, annotation in image_to_annotation.items():
        tif_file_path = f'{tif_folder}/{clean_filename}'
        with rasterio.open(tif_file_path) as src:
            band_data = ['N/A'] * len(csv_columns[9:])  

            _, _, year, month, landsat = clean_filename.split('_')
            month = month.split('.')[0]  
            # Get the GeoJSON filename
            geojson_filename = f'{clean_filename.split(".tif")[0]}.geojson'
            geojson_file_path = os.path.join(geojson_folder, geojson_filename)

            # Open the GeoJSON file
            with open(geojson_file_path, 'r') as geojson_file:
                geojson_data = json.load(geojson_file)

            bottom_left = geojson_data['features'][0]['geometry']['coordinates'][0][0]
            top_right = geojson_data['features'][0]['geometry']['coordinates'][0][2]

            # Extract annotation data 
            drafts = annotation.get('drafts', [])
            if drafts:
                for result in drafts[0]['result']:
                    if result['type'] == 'keypointlabels':
                        x_percent = result['value']['x']
                        y_percent = result['value']['y']
                        label = result['value']['keypointlabels'][0]

                        # Convert percentage to pixel coordinates
                        x_pixel = int(x_percent * src.width / 100)
                        y_pixel = int(y_percent * src.height / 100)
                        x = bottom_left[0] + (top_right[0] - bottom_left[0]) * (x_percent / 100)
                        y = bottom_left[1] + (top_right[1] - bottom_left[1]) * (y_percent / 100)
                        # Extract all bands from the TIFF image that are available
                        for i, band_name in enumerate(csv_columns[9:24]):
                            if band_name in src.descriptions:
                                band_index = src.descriptions.index(band_name) + 1
                                band = src.read(band_index, window=((y_pixel, y_pixel+1), (x_pixel, x_pixel+1)))
                                if band.size > 0:
                                    band_data[i] = band[0, 0]  

                        # Write data to CSV, including the TIF ID, year, month, and label
                        tif_id = tif_name_to_id[clean_filename]  
                        csv_row = [tif_id, clean_filename, landsat, year, month, x_percent, y_percent, label, x, y] + band_data  
                        csv_writer.writerow(csv_row)