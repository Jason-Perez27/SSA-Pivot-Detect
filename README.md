# SSA-Pivot-Detect
Repository for detecting activity status in center pivots across Sub-Saharan Africa. Leveraging ML to understand agricultural practices

Authors: Jason Perez, Anna Boser, Kelly Caylor

## Shapefile containing Center Pivots across the World

You can obtain the shp file containing the world's center pivots [here](https://github.com/DetectCPIS/global_cpis_shp)

## Code Description and Packages

| File Name and Location | Necessary Packages in Environment
|---------------|----------------|
| `code/cp_data_subset/id_cp.py` | `pip install geopandas os`
| `code/cp_data_subset/id_filter_cp.py`   | `pip install geopandas os random`
| `code/cp_data_subset/ee_landsat_data.py` | `pip install earthengine-api geopandas os random numpy re`
| `code/cp_data_subset/ee_landsat_data.py` | `pip install geopandas os`

| File Name and Location | Purpose of Code
|---------------|----------------|
| `code/cp_data_subset/id_cp.py` | Assigns ID column and Unique ID's to Center Pivots
| `code/cp_data_subset/id_filter_cp.py`   | Filters CP's to include a representative example of SSA, saves ID's in a test file
| `code/cp_data_subset/data_subset.py` | Uses Google Earth Engine to request Landsat data of Center Pivots, will request data from Landsat 4, 5, 8 and 9
| `code/cp_data_subset/data_subset.py` | Uses SHP file and txt file to filter the shape file to include CP's who have ID's listed in the text file

## Obtaining a subset of training/test data

1. To get our subsets of training and test data we first need to download the shp file containing all center pivots across the world. We also need to use [geojson](geojson.io) to create a JSON file with the geometric shape of Sub-Saharan Africa. This can be done by using geojson's line feature which will allow you to trace your shape and export the coordinates (Lattitude and Longitude) into a JSON file.

2. We then need to assign ID's to all the center pivots in our global CP shp file so that we can later filter/subset these center pivots to fit a representative dataset of center pivots across Sub-Saharan Africa. The script `id_cp.py` reads in our global shape file and creates a new column for "ID", which will create a unique ID for each center pivot in the shp file.

3. Our next step is to filter our shape file to only include CP's in Sub-Saharan Africa and make our samples representative of all of SSA. The script `id_filter_cp.py` reads in our GeoJSON map to only include CP's located in SSA. Then by using a stratisfying method, our GeoJSON map is split into multiple grid cells. The script then randomly samples a maximum of 4 center pivots per grid and saves their Center Pivot ID into a text file.

4. Since we now have our text file containing our desired center pivots and our shapefile containing CP ID's we then use `data_subset.py` which allows us to filter our shapefile to only include center pivots who's ID's are listed in the text file. 

