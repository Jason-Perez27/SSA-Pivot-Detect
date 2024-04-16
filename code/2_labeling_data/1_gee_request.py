import ee
import geopandas as gpd
import os
import random
import re
from osgeo import gdal

# Initialize the Earth Engine API.
ee.Initialize()

class LandsatDataExporter:
    def __init__(self, shapefile_path):
        # Base constructor class, reads in shapefiles and consists of the different Landsat collections
        self.shapefile_path = os.path.expanduser(shapefile_path)
        self.center_pivot_gdf = gpd.read_file(self.shapefile_path)
        self.landsat_collections = {
            'LANDSAT/LT04/C01/T1_L2': (1982, 1993, 'Landsat 4'),
            'LANDSAT/LT05/C01/T1_L2': (1984, 2012, 'Landsat 5'),
            'LANDSAT/LC08/C01/T1_L2': (2013, 2021, 'Landsat 8'),
            'LANDSAT/LE07/C01/T1_L2': (1999, 2022, 'Landsat 7'),
            'LANDSAT/LC09/C01/T1_L2': (2021, 2022, 'Landsat 9')
        }

    def clean_description(self, desc):
        cleaned_desc = re.sub(r'[^a-zA-Z0-9.,:;_-]', '_', desc)
        return cleaned_desc[:100]

    def divide_pivots(self):
        # Divide Pivots into Groups to be assigned to a Landsat Collection
        num_pivots = len(self.center_pivot_gdf)
        pivots_per_group = num_pivots // 5
        self.pivot_groups = [self.center_pivot_gdf.iloc[i:i+pivots_per_group] for i in range(0, num_pivots, pivots_per_group)]
        if num_pivots % 5 != 0:
            self.pivot_groups[-1] = self.center_pivot_gdf.iloc[-(num_pivots % 5):]

    def export_landsat_data(self):
        # Determines each Center Pivot's Landsat Version and assigns the correct collection
        self.divide_pivots()
        for collection_path, (start_year, end_year, landsat_name) in self.landsat_collections.items():
            collection = ee.ImageCollection(collection_path)
            for pivot_group in self.pivot_groups:
                self.process_group(pivot_group, collection_path, collection, start_year, end_year, landsat_name)

    def process_group(self, pivot_group, collection_path, collection, start_year, end_year, landsat_name):
        for index, row in pivot_group.iterrows():
            geom = row['geometry']
            month = random.randint(1, 12)
            year = random.randint(start_year, end_year)
            buffer_distance = 1000 / 100000
            geom_buffer = geom.buffer(buffer_distance)
            geom_buffer_json = geom_buffer.__geo_interface__
            ee_geom_buffer = ee.Geometry(geom_buffer_json)
            filtered = collection.filterDate(f'{year}-{month:02d}-01', f'{year}-{month:02d}-28')\
                                 .filterBounds(ee_geom_buffer)\
                                 .filter(ee.Filter.lt('CLOUD_COVER', 10))
            pivot_id = row['Id']
            clipped_image = ee.Image(filtered.first()).toFloat()
            description = f'CenterPivot_{pivot_id}_{year}_{month:02d}'
            export_options = {
                'image': clipped_image,
                'description': self.clean_description(description),
                'folder': 'SSA_TIF_REQUEST',
                'scale': 30,
                'region': ee_geom_buffer,
                'maxPixels': 10000000000000
            }
            # Add Landsat version metadata
            landsat_version = self.landsat_collections[collection_path][2]
            metadata = {'landsat_version': landsat_version}
            metadata_xml = '<GDALMetadata>{}</GDALMetadata>'.format(''.join(['<Item name="{0}">{1}</Item>'.format(k, v) for k, v in metadata.items()]))
            export_options['metadata'] = metadata_xml

            # Export image to Drive
            task = ee.batch.Export.image.toDrive(**export_options)
            task.start()

# Start Landsat Request Exporter
shapefile_path = 'data/World_CPIS_2021/filtered_center_pivots.shp'
exporter = LandsatDataExporter(shapefile_path)
exporter.export_landsat_data()
