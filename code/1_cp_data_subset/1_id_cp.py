import geopandas as gpd
import os

# Read the shapefile
shapefile_path = 'data/World_CPIS_2021/World_CPIS_2021.shp'
gdf = gpd.read_file(shapefile_path)

# Add an ID column
gdf['Id'] = range(1, len(gdf) + 1)

# Save the updated shapefile
output_shapefile_path = "data/World_CPIS_2021/id_cp_shapefile.shp"
gdf.to_file(output_shapefile_path)

print("IDs assigned and saved to the shapefile.")
