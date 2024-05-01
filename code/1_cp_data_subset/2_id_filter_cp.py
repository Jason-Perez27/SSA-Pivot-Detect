import os
import geopandas as gpd
import random

# Set random seed for reproducibility
random.seed(42)

# Input paths
shapefile_path = os.path.join('data/World_CPIS_2021/id_cp_shapefile.shp')
ssa_geojson_path = 'data/map.geojson'

# Output file path
output_txt_file_path = os.path.join('data/cp_ids.txt')

# Read data
original_cp_gdf = gpd.read_file(shapefile_path)
ssa_bbox_gdf = gpd.read_file(ssa_geojson_path)

# Calculate centroids for original shapefile
original_cp_gdf['centroid'] = original_cp_gdf['geometry'].centroid

# Spatial join to get center pivots in Sub-Saharan Africa
center_pivots_in_ssa = gpd.sjoin(original_cp_gdf, ssa_bbox_gdf, op='intersects')

# Divide Sub-Saharan Africa into grid cells
grid_size = 1  
center_pivots_in_ssa['grid'] = center_pivots_in_ssa.apply(
    lambda row: (
        int(row.centroid.x / grid_size),
        int(row.centroid.y / grid_size)
    ),
    axis=1
)

# Initialize an empty list to store the IDs of the sampled center pivots
sampled_cp_ids = []

# maximum number of CPs to admit from the same grid cell
threshold = 4

# Stratified sampling
for grid_cell in center_pivots_in_ssa['grid'].unique():
    # Extract center pivots in the current grid cell
    grid_cell_pivots = center_pivots_in_ssa[center_pivots_in_ssa['grid'] == grid_cell]

    # Define the number of CPs to sample
    sample_size = min(threshold, len(grid_cell_pivots))

    # Sample from the current grid cell
    if sample_size > 0:
        grid_cell_sample = grid_cell_pivots.sample(n=sample_size, random_state=42)
        sampled_cp_ids.extend(grid_cell_sample['Id'])

# Write the list of IDs of sampled center pivots to a text file
with open(output_txt_file_path, 'w') as f:
    for cp_id in sampled_cp_ids:
        f.write(str(cp_id) + '\n')

print(f"IDs of sampled center pivots written to {output_txt_file_path}")

