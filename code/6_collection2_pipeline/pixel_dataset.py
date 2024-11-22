import json
import rasterio
import pandas as pd
import os

# Step 1: Read JSON File
json_file_path = 'c:/Users/jdper/Desktop/tif_labels.json'
with open(json_file_path, 'r') as json_file:
    annotations_data = json.load(json_file)

# Step 2: Create a mapping of image filenames to annotation data
image_to_annotation = {}
tif_name_to_id = {}
current_tif_id = 1

# Create a mapping based on pivot_id, year, and month from the JSON file format
for annotation in annotations_data:
    filename = annotation['file_upload']
    # Extract pivot_id, year, and month from the JSON filename format
    clean_filename = filename.split('-', 1)[1].replace('_RGB.jpg', '')
    pivot_id, year, month = clean_filename.split('_')[1:]  # Skip the 'CenterPivot' part
    key = (pivot_id, year, month)
    
    if key not in tif_name_to_id:
        tif_name_to_id[key] = current_tif_id
        current_tif_id += 1
    image_to_annotation[key] = annotation

# Step 3: Read the band names from the text file
band_names = {}
with open('c:/Users/jdper/Desktop/band_names_collection2.txt', 'r') as band_names_file:
    current_tif_name = None
    for line in band_names_file:
        if line.startswith('Band names for '):
            current_tif_name = line.split('Band names for ')[1].strip().split('.')[0]
            band_names[current_tif_name] = []
        elif line.startswith('Band '):
            band_names[current_tif_name].append(line.split(': ')[1].strip())

# Step 4: Prepare DataFrame
data = []
columns = ['TIF ID', 'TIF Name', 'Landsat', 'Year', 'Month', 'Day', 'X Value', 'Y Value', 'Label', 'X-Coord', 'Y-Coord',
           'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19']

tif_folder = 'c:/Users/jdper/Desktop/landsat_collection_2_request'
geojson_folder = 'c:/Users/jdper/Desktop/collection_2_geojson'
for tif_filename in os.listdir(tif_folder):
    if not tif_filename.endswith('.tif'):
        continue

    # Extract pivot_id, landsat_name, year, month, and day from the new filename format
    parts = tif_filename.split('_')
    pivot_id = parts[1]
    landsat_name = parts[2]
    year, month, day = parts[3].split('-')
    day = day.split('.')[0]
    key = (pivot_id, year, month)

    if key not in image_to_annotation:
        continue

    annotation = image_to_annotation[key]
    tif_file_path = f'{tif_folder}/{tif_filename}'
    with rasterio.open(tif_file_path) as src:
        band_data = ['N/A'] * 19  # Ensure you have 19 bands to read

        # Get the GeoJSON filename
        geojson_filename = f'{tif_filename.split(".tif")[0]}.geojson'
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

                    # Extract the first 19 bands from the TIFF image
                    for i in range(19):  # Assuming you want the first 19 bands
                        band_index = i + 1  # Band indices in rasterio are 1-based
                        band = src.read(band_index, window=((y_pixel, y_pixel+1), (x_pixel, x_pixel+1)))
                        if band.size > 0:
                            band_data[i] = band[0, 0]

                    # Append data to list
                    tif_id = tif_name_to_id[key]  
                    row = [tif_id, tif_filename, landsat_name, year, month, day, x_percent, y_percent, label, x, y] + band_data  
                    data.append(row)
# Step 5: Write data to Excel file
df = pd.DataFrame(data, columns=columns)
excel_file_path = 'c:/Users/jdper/Desktop/collection_2_dataset_unsplit.xlsx'
df.to_excel(excel_file_path, index=False)