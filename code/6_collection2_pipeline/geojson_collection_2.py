import os
import rasterio
import json
from pyproj import Transformer

# Directory containing the TIF files
tif_directory = 'c:/Users/jdper/Desktop/landsat_collection_2_request'

# Output file path
geojson_directory = 'c:/Users/jdper/Desktop/collection_2_geojson' 


os.makedirs(geojson_directory, exist_ok=True)

# Iterate over the TIF files
for tif_name in os.listdir(tif_directory):
    if tif_name.endswith('.tif'):
        tif_path = os.path.join(tif_directory, tif_name)

        # Split the TIF filename into its name and extension parts
        tif_name_no_ext, _ = os.path.splitext(tif_name)

        # Construct the GeoJSON filename with the .geojson extension
        geojson_filename = f'{tif_name_no_ext}.geojson'
        geojson_file_path = os.path.join(geojson_directory, geojson_filename)

        # Open the TIF file with rasterio
        with rasterio.open(tif_path) as src:
            # Get the CRS of the TIF file
            crs = src.crs
            print(f"CRS of {tif_name}: {crs}")

            # Get the UTM zone from the CRS
            utm_zone = crs.to_string().split(":")[-1]

            # Create a transformer to convert from the UTM coordinate system to EPSG 4326
            transformer = Transformer.from_crs(f"EPSG:{utm_zone}", "EPSG:4326")

            # Get the bounding box coordinates
            minx, miny, maxx, maxy = src.bounds

            # Convert the bounding box coordinates from the UTM coordinate system to EPSG 4326
            minx_4326, miny_4326 = transformer.transform(minx, miny)
            maxx_4326, maxy_4326 = transformer.transform(maxx, maxy)

            # Create a GeoJSON dictionary
            geojson_data = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [minx_4326, miny_4326],
                                    [minx_4326, maxy_4326],
                                    [maxx_4326, maxy_4326],
                                    [maxx_4326, miny_4326],
                                    [minx_4326, miny_4326]
                                ]
                            ]
                        }
                    }
                ]
            }

            # Write the GeoJSON dictionary to the GeoJSON file
            with open(geojson_file_path, 'w') as geojson_file:
                json.dump(geojson_data, geojson_file, indent=2)

        print(f"GeoJSON file saved at: {geojson_file_path}")