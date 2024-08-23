import ee
import geopandas as gpd
import os
import logging
import time
# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize(project='center-pivots')

# Setup logging
logging.basicConfig(filename='exporter.log', level=logging.INFO, format='%(asctime)s %(message)s')

class LandsatDataExporter:
    def __init__(self, shapefile_path, max_tasks=50):
        self.shapefile_path = os.path.expanduser(shapefile_path)
        self.center_pivot_gdf = gpd.read_file(self.shapefile_path)
        self.max_tasks = max_tasks
        self.landsat_collections = {
            'LANDSAT/LT04/C02/T1_L2': (1982, 1993, 'Landsat 4'),
            'LANDSAT/LT05/C02/T1_L2': (1984, 2012, 'Landsat 5'),
            'LANDSAT/LC08/C02/T1_L2': (2013, 2021, 'Landsat 8'),
            'LANDSAT/LE07/C02/T1_L2': (1999, 2022, 'Landsat 7'),
            'LANDSAT/LC09/C02/T1_L2': (2021, 2023, 'Landsat 9')
        }

    def clean_description(self, desc):
        cleaned_desc = re.sub(r'[^a-zA-Z0-9.,:;_-]', '_', desc)
        return cleaned_desc[:100]

    def export_landsat_data(self):
        for index, row in self.center_pivot_gdf.iterrows():
            for collection_path, (start_year, end_year, landsat_name) in self.landsat_collections.items():
                if self.check_active_tasks() < self.max_tasks:
                    try:
                        self.export_pivot(row, collection_path, start_year, end_year, landsat_name)
                    except Exception as e:
                        logging.error(f'Failed to process pivot {row["ID"]}: {str(e)}')
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
        scaled = collection.map(self.apply_scale_factors)
        masked = scaled.map(self.qa_mask)
        pivot_id = row['ID']
        description = f'CenterPivot_{pivot_id}_{landsat_name}_TimeSeries'
        export_options = {
            'image': masked.median().toFloat(),
            'description': self.clean_description(description),
            'folder': 'TOTAL_TIF_REQUEST',
            'scale': 30,
            'region': ee_geom,
            'maxPixels': 10000000000000
        }
        task = ee.batch.Export.image.toDrive(**export_options)
        task.start()
        logging.info(f'Started export task for pivot {pivot_id} for {landsat_name} time series')

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
shapefile_path = '/home/waves/data/SSA-Pivot-Detect/data/combined_CPIS/combined_CPIS.shp'
exporter = LandsatDataExporter(shapefile_path)
exporter.export_landsat_data()