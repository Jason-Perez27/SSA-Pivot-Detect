'''
This script uses the LandsatDataExporter to download the full Landsat time series for each center pivot in the shapefile. 
'''

import os
from data_exporter import LandsatDataExporter

# Setup logging
log_name='collection_2_timeseries.log'

# The CPs we chose for training
shapefile_path = '/home/waves/data/SSA-Pivot-Detect/data/combined_CPIS/combined_CPIS.shp'

# The name of the google drive folder to save the data into
drive_folder = 'timeseries_c2'

# GEE parameters
service_account='gee-export-service@center-pivot-collection2.iam.gserviceaccount.com'
service_account_key_path = '/home/waves/data/SSA-Pivot-Detect/center-pivots-collection2-1408cb0e7b9f.json'

# Initialize 
exporter = LandsatDataExporter(shapefile_path, drive_folder, service_account, service_account_key_path)

# Download the full time series for all center pivots
exporter.download(log_name='c2_time_series.log', buffer=False, max_cloud_cover=100, completed_pivot_file='completed_pivots_timeseries.txt')
