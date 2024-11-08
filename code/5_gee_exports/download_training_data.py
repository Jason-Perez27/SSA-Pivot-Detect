
'''
Previously, in 2_labeling_data/1_gee_request.py, we downloaded (and later labeled) collection 1 data from google earth engine. 
To achieve this, we subset our CP data ('/home/waves/data/SSA-Pivot-Detect/data/1_script_data/filtered_center_pivots.shp') 
and downloaded a random image for each landsat (4, 5, 7, 8, 9). 
This image was chosen by specifying a random month and year within that landsat's years in service, 
and choosing the first image that was in that year/month and had information within the CP's buffer area and satisfying some cloud cover criteria.

We want to update our model to use collection 2 data, so here we
(1) retrieve the pivot id, month, year, and landsat of each image we orginally downloaded
(2) request this same image to be chosen and downloaded, just using the exporter we created to download the collection 2 data. 
These new images should be identical to the first ones we downloaded, 
but they are from collection 2, scaled according to GEE's recommendations, and have a mask for quality control. 
'''

import os
import logging
from data_exporter import LandsatDataExporter

'''
First, we reteive the information about the pivot IDs, month, year, and landsat we originally downloaded. 
'''

# Define the folder where the original TIF files are stored
tif_folder = '/home/waves/data/SSA-Pivot-Detect/data/2_script_data/SSA_TIF_REQUEST'

# Initialize empty lists to store the parsed information
pivot_ids = []
landsats = []
years = []
months = []

# Function to parse filename and retrieve pivot_id, landsat, year, and month
def parse_filename(filename):
    parts = filename.split('_')
    if len(parts) >= 5:
        pivot_id = parts[1]
        year = parts[2]
        month = parts[3]
        landsat = parts[4].split('.')[0]
        return pivot_id, landsat, year, month
    return None, None, None, None

# Iterate through each file in the folder
for file in os.listdir(tif_folder):
    if file.endswith('.tif'):
        # Parse the filename to get the required details
        pivot_id, landsat, year, month = parse_filename(file)
        
        # Check that parsing was successful before adding to the lists
        if pivot_id and landsat and year and month:
            pivot_ids.append(pivot_id)
            landsats.append(landsat)
            years.append(year)
            months.append(month)

'''
Next, we initialize the exporter class and download the data. 
'''

# Setup logging
log_name='collection_2_training_data.log'

# The CPs we chose for training
shapefile_path = '/home/waves/data/SSA-Pivot-Detect/data/1_script_data/filtered_center_pivots.shp'

# The name of the google drive folder to save the data into
drive_folder = 'collection2_landsat_data'

# GEE parameters
service_account='gee-export-service@center-pivot-collection2.iam.gserviceaccount.com'
service_account_key_path = '/home/waves/data/SSA-Pivot-Detect/center-pivots-collection2-1408cb0e7b9f.json'

# Initialize 
exporter = LandsatDataExporter(shapefile_path, drive_folder, service_account, service_account_key_path)

# Download these specific images, with a buffer. 
exporter.download(log_name='collection_2_training_data.log', pivot_ids=pivot_ids, months=months, years=years, landsats=landsats, buffer=True, max_cloud_cover=10)