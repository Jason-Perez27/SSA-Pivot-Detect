import ee
import geopandas as gpd
import os
import logging
import time
import random
import re
# Initialize the Earth Engine API.
ee.Initialize()

class LandsatDataExporter:
    def __init__(self, shapefile_path, drive_folder, tif_folder=None, max_tasks=500):
        self.shapefile_path = shapefile_path
        self.drive_folder = drive_folder
        self.center_pivot_gdf = gpd.read_file(shapefile_path)
        self.max_tasks = max_tasks
        self.tif_folder = tif_folder
        self.landsat_collections = {
            'LANDSAT/LT04/C02/T1_L2': (1982, 1993, 'Landsat4'),
            'LANDSAT/LT05/C02/T1_L2': (1984, 2012, 'Landsat5'),
            'LANDSAT/LC08/C02/T1_L2': (2013, 2021, 'Landsat8'),
            'LANDSAT/LE07/C02/T1_L2': (1999, 2022, 'Landsat7'),
            'LANDSAT/LC09/C02/T1_L2': (2021, 2023, 'Landsat9')
        }

    def parse_tif_filenames(self, tif_folder):
        if not self.tif_folder:
            return None, None, None, None
        # Regular expression to match the filename pattern
        pattern = r'CenterPivot_(\d+)_(\d{4})_(\d{1,2})_(Landsat\d+)'
        
        # Lists to store the extracted information
        pivot_ids = []
        years = []
        months = []
        landsats = []
        
        # Iterate over all files in the folder
        for filename in os.listdir(tif_folder):
            match = re.match(pattern, filename)
            if match:
                pivot_id, year, month, landsat = match.groups()
                pivot_ids.append(int(pivot_id))
                years.append(int(year))
                months.append(int(month))
                landsats.append(landsat)
        
        return pivot_ids, years, months, landsats
    
    def download(self, log_name, pivot_ids=None, months=None, years=None, landsats=None, buffer=True, max_cloud_cover=10, completed_pivot_file=None):
        """
        max_cloud_cover: int, default 10. The maximum cloud cover percentage allowed for the Landsat images.
        If trying to download the full time series should be set to 100 since no images should be missed. 
        """
        if pivot_ids is None and self.tif_folder:
            pivot_ids, years, months, landsats = self.parse_tif_filenames(self.tif_folder)
        # Assert that if months, years, or landsats is specified, all three are specified and they are the same length as pivot_ids. 
        # Also assert that if any of these are specified, completed_pivot_file is not specified
        if months is not None or years is not None or landsats is not None:
            assert months is not None and years is not None and landsats is not None, 'If months, years, or landsats are specified, all three must be specified'
            assert len(months) == len(pivot_ids) and len(years) == len(pivot_ids) and len(landsats) == len(pivot_ids), 'If months, years, or landsats are specified, they must be the same length as pivot_ids'
            assert completed_pivot_file is None, 'If months, years, or landsats are specified, completed_pivot_file must be None else some pivots will be skipped over'
    
        # Setup logging
        logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s %(message)s')

        alternative_collections = {
            'Landsat4': ['Landsat5'],
            'Landsat5': ['Landsat4'],
            'Landsat7': ['Landsat8', 'Landsat9'],
            'Landsat8': ['Landsat7', 'Landsat9'],
            'Landsat9': ['Landsat7', 'Landsat8']
        }

        # Set default for pivot_ids to all Ids in the GeoDataFrame
        if pivot_ids is None:
            pivot_ids = self.center_pivot_gdf['ID'].tolist()
            random.shuffle(pivot_ids) # shuffle the pivots so that we have a random sample even if we haven't finished downloading all pivots
        
        # remove the completed pivots from the list of pivots to download if a file with completed pivots is provided
        if completed_pivot_file:
            # create the file if necessary
            if not os.path.exists(completed_pivot_file):
                with open(completed_pivot_file, 'w') as f:
                    pass
            with open(completed_pivot_file, 'r') as f:
                completed_pivots = f.read().splitlines()
            pivot_ids = [pivot_id for pivot_id in pivot_ids if pivot_id not in completed_pivots]

        # Go through all the pivots and download the images
        for index, pivot_id in enumerate(pivot_ids): 

            # Retrieve the center pivot geometry and buffer it if necessary:
            row = self.center_pivot_gdf[self.center_pivot_gdf['ID'] == pivot_id]
            if row.empty:
                logging.warning(f'Pivot ID {pivot_id} not found in the shapefile.')
                continue
            row = row.iloc[0]
            geom = row['geometry']
            if buffer:
                buffer_distance = 1000 / 100000
                geom = geom.buffer(buffer_distance)
            geom_json = geom.__geo_interface__
            ee_geom = ee.Geometry(geom_json)

            images_found = False  # Flag to track if images are found

            # Go through all the different landsat collections and either download all images or only the specific image with that month and year
            for collection_path, (start_year, end_year, landsat_name) in self.landsat_collections.items():
                
                # Skip this landsat collection if we're looking for a specific landsat/month/year combination
                if landsats is not None and landsats[index] != landsat_name: 
                    continue

                # Get the start and end date for the image collection
                if months is not None and years is not None: # make the start and end date correspond to a specific month and year
                    month = months[index]
                    year = years[index]
                    start_date = ee.Date(f'{year}-{month:02d}-01')
                    end_date = ee.Date(f'{year}-{month:02d}-28') # Misses the last few days
                else: # make the start and end date correspond to all years the landsat collection is available
                    start_date = ee.Date(f'{start_year}-01-01')
                    end_date = ee.Date(f'{end_year}-12-31')

                # Filter the image collection by date, bounds, and cloud cover
                collection = ee.ImageCollection(collection_path)\
                    .filterDate(start_date, end_date)\
                    .filterBounds(ee_geom)\
                    .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
                
                image_count = collection.size().getInfo()

                # Check if there are any images in the filtered collection
                if image_count == 0:
                    logging.info(f'No images found for pivot {pivot_id} in {landsat_name} collection between {start_date} and {end_date}')
                    
                    # Check alternative collections if no images found
                    for alt_landsat in alternative_collections.get(landsat_name, []):
                        alt_collection_path = next((path for path, (_, _, name) in self.landsat_collections.items() if name == alt_landsat), None)
                        if alt_collection_path:
                            alt_collection = ee.ImageCollection(alt_collection_path)\
                                .filterDate(start_date, end_date)\
                                .filterBounds(ee_geom)\
                                .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
                            
                            if alt_collection.size().getInfo() > 0:
                                logging.info(f'Found images for pivot {pivot_id} in alternative {alt_landsat} collection between {start_date} and {end_date}')
                                collection = alt_collection
                                landsat_name = alt_landsat
                                image_count = alt_collection.size().getInfo()  # Update image_count
                                images_found = True
                                break
                else:
                    images_found = True

                if images_found:
                    # Proceed with downloading images if the collection is not empty
                    logging.info(f'Starting download for {image_count} images for pivot {pivot_id} in {landsat_name} collection between {start_date.format("YYYY-MM-dd").getInfo()} and {end_date.format("YYYY-MM-dd").getInfo()}')

                    # Download the images to Google Drive
                    images = collection.toList(image_count)
                    for i in range(image_count):
                        image = ee.Image(images.get(i))
                        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()

                        # Mask and scale the image
                        scaled = self.apply_scale_factors(image)
                        masked = self.qa_mask(scaled)

                        # If buffer = False, the image should also be masked by the original geometry
                        if not buffer:
                            masked = masked.clip(ee_geom)

                        description = f'CP_{pivot_id}_{landsat_name}_{image_date}'
                        export_options = {
                            'image': masked.toFloat(),
                            'description': description,
                            'folder': self.drive_folder,
                            'scale': 30,
                            'region': ee_geom,
                            'maxPixels': 10000000000000
                        }
                        task = ee.batch.Export.image.toDrive(**export_options)
                        
                        while self.check_active_tasks() >= self.max_tasks:
                            logging.info("Max tasks reached, waiting...")
                            time.sleep(60) # wait one minute before checking again
                        
                        try:
                            task.start()
                            logging.info(f'Successfully started export task for pivot {pivot_id} for {landsat_name}, image date: {image_date}')
                        except Exception as e:
                            logging.error(f'Failed to start export task for pivot {pivot_id}: {e}')
            
            if not images_found:
                logging.info(f'No images found for pivot {pivot_id} in any collection.')
                # Handle the case where no images are found in any collection

            logging.info(f'Finished downloading images for pivot {pivot_id}')
            
            if completed_pivot_file:
                with open(completed_pivot_file, 'a') as f:
                    f.write(f'{pivot_id}\n')
                

    def apply_scale_factors(self, image):
            # Apply scale factors to the optical bands
            optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
            
            # Check for the thermal band and apply the correct scale factor
            if 'ST_B10' in image.bandNames().getInfo():
                thermal_band = image.select('ST_B10').multiply(0.00341802).add(149.0)
            elif 'ST_B6' in image.bandNames().getInfo():
                thermal_band = image.select('ST_B6').multiply(0.00341802).add(149.0)
            else:
                thermal_band = None

            # Add scaled bands back to the image
            image = image.addBands(optical_bands, None, True)
            if thermal_band:
                image = image.addBands(thermal_band, None, True)
            
            return image

    def qa_mask(self, image): 
            # see  https://developers.google.com/earth-engine/landsat_c1_to_c2#colab-python
            # Mask out pixels where the QA_PIXEL bits indicate poor quality
            return image.updateMask(image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0))
        
    def check_active_tasks(self):
            tasks = ee.batch.Task.list()
            running_tasks = [task for task in tasks if task.state in ['RUNNING', 'READY']]
            return len(running_tasks)

# Usage
shapefile_path = 'c:/Users/jdper/Desktop/combined_CPIS/combined_CPIS.shp'
drive_folder = 'landsat_collection_2_request'
tif_folder = 'c:/Users/jdper/Desktop/Data_collection_2_request'
exporter = LandsatDataExporter(shapefile_path, drive_folder, tif_folder)
exporter.download(log_name='collection_2_noauth.log')