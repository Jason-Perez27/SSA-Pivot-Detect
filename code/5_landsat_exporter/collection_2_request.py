import ee
import geopandas as gpd
import os
import logging
import time
import re
from datetime import datetime

# Setup logging
logging.basicConfig(filename='exporter.log', level=logging.INFO, format='%(asctime)s %(message)s')

class LandsatDataExporter:
    def __init__(self, shapefile_path, tif_folder, service_account_key_path, max_tasks=50):
        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(
            service_account='gee-export-service@center-pivot-collection2.iam.gserviceaccount.com',
            private_key_file=service_account_key_path
        )
        ee.Initialize(credentials)
        
        self.shapefile_path = os.path.expanduser(shapefile_path)
        self.center_pivot_gdf = gpd.read_file(self.shapefile_path)
        self.tif_folder = tif_folder
        self.max_tasks = max_tasks
        self.landsat_collections = {
            'Landsat4': 'LANDSAT/LT04/C02/T1_L2',
            'Landsat5': 'LANDSAT/LT05/C02/T1_L2',
            'Landsat7': 'LANDSAT/LE07/C02/T1_L2',
            'Landsat8': 'LANDSAT/LC08/C02/T1_L2',
            'Landsat9': 'LANDSAT/LC09/C02/T1_L2'
        }

    def clean_description(self, desc):
        cleaned_desc = re.sub(r'[^a-zA-Z0-9.,:;_-]', '_', desc)
        return cleaned_desc[:100]

    def parse_filename(self, filename):
        parts = filename.split('_')
        if len(parts) >= 5:
            pivot_id = parts[1]
            year = parts[2]
            month = parts[3]
            landsat = parts[4].split('.')[0]
            date = f"{year}-{month}-01"
            return pivot_id, date, landsat, year, month
        return None, None, None, None, None

    def export_landsat_data(self):
        for file in os.listdir(self.tif_folder):
            if file.endswith('.tif'):
                pivot_id, date, landsat = self.parse_filename(file)
                if pivot_id and date and landsat:
                    row = self.center_pivot_gdf[self.center_pivot_gdf['ID'] == pivot_id].iloc[0]
                    if self.check_active_tasks() < self.max_tasks:
                        self.export_pivot(row, date, landsat)
                    else:
                        logging.info("Max tasks reached, waiting...")
                        time.sleep(600)

    def export_pivot(self, row, date, landsat, year, month):
        geom = row['geometry']
        ee_geom = ee.Geometry(geom.__geo_interface__)
        collection_path = self.landsat_collections.get(landsat)
        
        if not collection_path:
            logging.warning(f"No Landsat collection found for {landsat}")
            return

        start_date = ee.Date(date)
        end_date = start_date.advance(1, 'month')

        collection = ee.ImageCollection(collection_path)\
                      .filterDate(start_date, end_date)\
                      .filterBounds(ee_geom)\
                      .filter(ee.Filter.lt('CLOUD_COVER', 10))

        if collection.size().getInfo() > 0:
            image = collection.first()
            
            # Get the exact date from the image
            image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            year = image_date.split('-')[0]
            month = image_date.split('-')[1]
            day = image_date.split('-')[2]
        
            scaled = self.apply_scale_factors(image)
            masked = self.qa_mask(scaled)
            pivot_id = row['ID']
            
            # Include day in the description
            description = f'CenterPivot_{pivot_id}_{year}_{month}_{day}_{landsat}_collection2'
            
            export_options = {
                'image': masked.toFloat(),
                'description': self.clean_description(description),
                'folder': 'collection2_landsat_data',
                'scale': 30,
                'region': ee_geom,
                'maxPixels': 10000000000000
            }
            task = ee.batch.Export.image.toDrive(**export_options)
            task.start()
            logging.info(f'Started export task for pivot {pivot_id} for {landsat} on {image_date}')
        else:
            logging.info(f'No image found for pivot {row["ID"]} on {date}')

    def apply_scale_factors(self, image):
        optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        thermal_band = image.select('ST_B10').multiply(0.00341802).add(149.0)
        return image.addBands(optical_bands, None, True).addBands(thermal_band, None, True)

    def qa_mask(self, image):
        return image.updateMask(image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0))

    def check_active_tasks(self):
        tasks = ee.batch.Task.list()
        running_tasks = [task for task in tasks if task.state in ['RUNNING', 'READY']]
        return len(running_tasks)


# Usage
shapefile_path = '/home/waves/data/SSA-Pivot-Detect/data/World_CPIS_2021/combined_CPIS.shp'
tif_folder = '/home/waves/data/SSA-Pivot-Detect/data/Data_collection_2_request'
service_account_key_path = '/home/waves/data/SSA-Pivot-Detect/center-pivots-collection2-1408cb0e7b9f.json'
exporter = LandsatDataExporter(shapefile_path, tif_folder, service_account_key_path)
exporter.export_landsat_data()