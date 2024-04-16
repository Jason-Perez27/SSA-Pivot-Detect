import os
import geopandas as gpd

# Input paths
shapefile_path = 'data/World_CPIS_2021/id_cp_shapefile.shp'
txt_file_path = os.path.join('data/cp_ids.txt')

# Output file path
output_shapefile_path = os.path.join('data/World_CPIS_2021/filtered_center_pivots.shp')

# Read the IDs from the text file
with open(txt_file_path, 'r') as f:
    sampled_cp_ids = [int(line.strip()) for line in f]

# Read the original shapefile
original_cp_gdf = gpd.read_file(shapefile_path)

# Filter the original shapefile based on the IDs from the text file
filtered_cp_gdf = original_cp_gdf[original_cp_gdf['Id'].isin(sampled_cp_ids)]

# Write the filtered center pivots to a new shapefile
filtered_cp_gdf.to_file(output_shapefile_path)

print(f"Filtered center pivots saved to {output_shapefile_path}")
