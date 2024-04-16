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
tif_name_to_id = {}  
current_tif_id = 1  
for annotation in annotations_data:
    filename = annotation['file_upload']
    # Remove unwanted prefix and replace "_RGB.jpg" with ".tif"
    clean_filename = filename.split('-', 1)[1].replace('_RGB.jpg', '.tif')
    if clean_filename not in tif_name_to_id:
        tif_name_to_id[clean_filename] = current_tif_id
        current_tif_id += 1
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
csv_file_path = '/Users/jasonperez/Desktop/5_landsat_dataset.csv'
# Add 'Month', 'Year', 'TIF ID', 'Latitude', and 'Longitude' to the csv_columns list
csv_columns = ['TIF ID', 'TIF Name', 'Year', 'Month', 'X Value', 'Y Value', 'Label', 'Latitude', 'Longitude'] + list(band_names.keys())
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_columns)

    tif_folder = '/Users/jasonperez/Desktop/SSA_TIF_REQUEST'
    geojson_folder = '/Users/jasonperez/Desktop/SSA_TIF_GEOJSON'
    for clean_filename, annotation in image_to_annotation.items():
        tif_file_path = f'{tif_folder}/{clean_filename}'
        with rasterio.open(tif_file_path) as src:
            # Get the GeoJSON filename
            geojson_filename = f'{clean_filename.split(".tif")[0]}.geojson'
            geojson_file_path = os.path.join(geojson_folder, geojson_filename)

            # Open the GeoJSON file
            with open(geojson_file_path, 'r') as geojson_file:
                geojson_data = json.load(geojson_file)

            # Get the latitude and longitude of the point
            x_pixel = annotation['x_pixel']
            y_pixel = annotation['y_pixel']
            x_percent = x_pixel / src.width
            y_percent = y_pixel / src.height
            x = geojson_data['features'][0]['geometry']['coordinates'][0][0] + (geojson_data['features'][0]['geometry']['coordinates'][0][2] - geojson_data['features'][0]['geometry']['coordinates'][0][0]) * x_percent
            y = geojson_data['features'][0]['geometry']['coordinates'][0][1] + (geojson_data['features'][0]['geometry']['coordinates'][0][3] - geojson_data['features'][0]['geometry']['coordinates'][0][1]) * y_percent

            # Extract the label from the annotation data
            label = annotation['label']

            # Extract all bands from the TIFF image that are available
            band_data = ['N/A'] * len(band_names)
            for i, band_name in enumerate(band_names.keys()):
                if band_name in src.descriptions:
                    band_index = src.descriptions.index(band_name) + 1
                    band = src.read(band_index, window=((y_pixel, y_pixel+1), (x_pixel, x_pixel+1)))
                    band_data[i] = band[0,0] if band.size > 0 else 'N/A'

            # Extract the Year and Month from the filename
            _, _, year, month = clean_filename.split('_')
            month = month.split('.')[0]  # Remove the file extension


            # Write the CSV row to the CSV file
            csv_row = [tif_name_to_id[clean_filename], clean_filename, year, month, x_percent, y_percent, label, x, y] + band_data
            csv_writer.writerow(csv_row)