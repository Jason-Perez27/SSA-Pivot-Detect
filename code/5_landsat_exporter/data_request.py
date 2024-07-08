import ee
import geopandas as gpd
import os
import random
import re
import time
import logging

# Initialize the Earth Engine API.
ee.Initialize()

# Setup logging
logging.basicConfig(filename='exporter.log', level=logging.INFO, format='%(asctime)s %(message)s')

class LandsatDataExporter:
    def __init__(self, shapefile_path, max_tasks=50):
        self.shapefile_path = os.path.expanduser(shapefile_path)
        self.center_pivot_gdf = gpd.read_file(self.shapefile_path)
        self.max_tasks = max_tasks
        self.landsat_collections = {
            'LANDSAT/LT04/C01/T1_L2': (1982, 1993, 'Landsat 4'),
            'LANDSAT/LT05/C01/T1_L2': (1984, 2012, 'Landsat 5'),
            'LANDSAT/LC08/C01/T1_L2': (2013, 2021, 'Landsat 8'),
            'LANDSAT/LE07/C01/T1_L2': (1999, 2022, 'Landsat 7'),
            'LANDSAT/LC09/C01/T1_L2': (2021, 2023, 'Landsat 9')
        }

    def clean_description(self, desc):
        cleaned_desc = re.sub(r'[^a-zA-Z0-9.,:;_-]', '_', desc)
        return cleaned_desc[:100]

    def divide_pivots(self):
        num_pivots = len(self.center_pivot_gdf)
        pivots_per_group = max(10, num_pivots // 100)  # Adjust the number of groups based on your needs
        self.pivot_groups = [self.center_pivot_gdf.iloc[i:i+pivots_per_group] for i in range(0, num_pivots, pivots_per_group)]
        if num_pivots % 100 != 0:
            self.pivot_groups.append(self.center_pivot_gdf.iloc[-(num_pivots % 100):])

    def export_landsat_data(self):
        self.divide_pivots()
        for collection_path, (start_year, end_year, landsat_name) in self.landsat_collections.items():
            collection = ee.ImageCollection(collection_path)
            for pivot_group in self.pivot_groups:
                self.process_group(pivot_group, collection, start_year, end_year, landsat_name)

    def process_group(self, pivot_group, collection, start_year, end_year, landsat_name):
        for index, row in pivot_group.iterrows():
            if self.check_active_tasks() < self.max_tasks:
                try:
                    self.export_pivot(row, collection, start_year, end_year, landsat_name)
                except Exception as e:
                    logging.error(f'Failed to process pivot {row["ID"]}: {str(e)}')
                    continue
            else:
                logging.info("Max tasks reached, waiting...")
                time.sleep(600)  # Wait for 10 minutes before checking again
                self.process_group(pivot_group, collection, start_year, end_year, landsat_name)
                break

    def export_pivot(self, row, collection, start_year, end_year, landsat_name):
        geom = row['geometry']
        month = random.randint(1, 13)
        year = random.randint(start_year, end_year)
        buffer_distance = 1000 / 100000
        ee_geom = ee.Geometry(geom.__geo_interface__)
        filtered = collection.filterDate(f'{year}-{month:02d}-01', f'{year}-{month:02d}-28')\
                             .filterBounds(ee_geom)\
                             .filter(ee.Filter.lt('CLOUD_COVER', 10))
        pivot_id = row['ID']
        clipped_image = ee.Image(filtered.first()).toFloat()
        description = f'CenterPivot_{pivot_id}_{year}_{month:02d}_{landsat_name}'
        export_options = {
            'image': clipped_image,
            'description': self.clean_description(description),
            'folder': 'TOTAL_TIF_REQUEST',
            'scale': 30,
            'region': ee_geom,
            'maxPixels': 10000000000000
        }
        task = ee.batch.Export.image.toDrive(**export_options)
        task.start()
        logging.info(f'Started export task for pivot {pivot_id} for {landsat_name} in {year}-{month:02d}')

    def check_active_tasks(self):
        tasks = ee.batch.Task.list()
        running_tasks = [task for task in tasks if task.state in ['RUNNING', 'READY']]
        return len(running_tasks)

# Usage
shapefile_path = '/home/waves/data/SSA-Pivot-Detect/data/combined_CPIS/combined_CPIS.shp'
exporter = LandsatDataExporter(shapefile_path)
exporter.export_landsat_data()