import ee
import geopandas as gpd
import os
import logging
import time
import re
import json

# Setup logging
logging.basicConfig(filename='exporter.log', level=logging.INFO, format='%(asctime)s %(message)s')

class LandsatDataExporter:
    def __init__(self, shapefile_path, service_account_key_path, max_tasks=50, progress_file='export_progress.json'):
        # Initialize with service account
        credentials = ee.ServiceAccountCredentials(
            'gee-export-service@center-pivot-collection2.iam.gserviceaccount.com',
            service_account_key_path
        )
        ee.Initialize(credentials)
        
        self.shapefile_path = os.path.expanduser(shapefile_path)
        self.center_pivot_gdf = gpd.read_file(self.shapefile_path)
        self.max_tasks = max_tasks
        self.progress_file = progress_file
        self.progress = self.load_progress()
        self.landsat_collections = {
            'LANDSAT/LT04/C02/T1_L2': (1982, 1993, 'Landsat 4'),
            'LANDSAT/LT05/C02/T1_L2': (1984, 2012, 'Landsat 5'),
            'LANDSAT/LC08/C02/T1_L2': (2013, 2021, 'Landsat 8'),
            'LANDSAT/LE07/C02/T1_L2': (1999, 2022, 'Landsat 7'),
            'LANDSAT/LC09/C02/T1_L2': (2021, 2023, 'Landsat 9')
        }
    def load_progress(self):
        if not os.path.exists(self.progress_file):
            # Create an empty JSON file if it doesn't exist
            with open(self.progress_file, 'w') as f:
                json.dump({}, f)
            return {}
        
        with open(self.progress_file, 'r') as f:
            return json.load(f)

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)

    def clean_description(self, desc):
        cleaned_desc = re.sub(r'[^a-zA-Z0-9.,:;_-]', '_', desc)
        return cleaned_desc[:100]

    def export_landsat_data(self):
        for index, row in self.center_pivot_gdf.iterrows():
            pivot_id = row['ID']
            if pivot_id not in self.progress:
                self.progress[pivot_id] = {}
            
            for collection_path, (start_year, end_year, landsat_name) in self.landsat_collections.items():
                if landsat_name not in self.progress[pivot_id]:
                    if self.check_active_tasks() < self.max_tasks:
                        try:
                            self.export_pivot(row, collection_path, start_year, end_year, landsat_name)
                            self.progress[pivot_id][landsat_name] = 'completed'
                            self.save_progress()
                        except Exception as e:
                            logging.error(f'Failed to process pivot {pivot_id} for {landsat_name}: {str(e)}')
                    else:
                        logging.info("Max tasks reached, waiting...")
                        time.sleep(600)

    def export_pivot(self, row, collection_path, start_year, end_year, landsat_name):
        geom = row['geometry']
        ee_geom = ee.Geometry(geom.__geo_interface__)
        collection = ee.ImageCollection(collection_path)\
                      .filterDate(f'{start_year}-01-01', f'{end_year}-12-31')\
                      .filterBounds(ee_geom)\
                      .filter(ee.Filter.lt('CLOUD_COVER', 10))
        
        image_count = collection.size().getInfo()
        
        if image_count > 0:
            images = collection.toList(image_count)
            for i in range(image_count):
                image = ee.Image(images.get(i))
                image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                scaled = self.apply_scale_factors(image)
                masked = self.qa_mask(scaled)
                pivot_id = row['ID']
                description = f'CenterPivot_{pivot_id}_{landsat_name}_{image_date}'
                export_options = {
                    'image': masked.toFloat(),
                    'description': self.clean_description(description),
                    'folder': 'time_series_landsat_data',
                    'scale': 30,
                    'region': ee_geom,
                    'maxPixels': 10000000000000
                }
                task = ee.batch.Export.image.toDrive(**export_options)
                task.start()
                logging.info(f'Started export task for pivot {pivot_id} for {landsat_name}, image date: {image_date}')
        else:
            logging.info(f'No images found for pivot {row["ID"]} in {landsat_name} collection')

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

    def resume_export(self):
        logging.info("Resuming export from last saved progress")
        self.export_landsat_data()

# Usage
shapefile_path = '/home/waves/data/SSA-Pivot-Detect/data/World_CPIS_2021/combined_CPIS.shp'
service_account_key_path = '/home/waves/data/SSA-Pivot-Detect/center-pivots-collection2-1408cb0e7b9f.json'
exporter = LandsatDataExporter(shapefile_path, service_account_key_path)
exporter.export_landsat_data()