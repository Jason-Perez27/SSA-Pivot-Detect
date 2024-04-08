# SSA-Pivot-Detect
Repository for detecting activity status in center pivots across Sub-Saharan Africa. Leveraging ML to understand agricultural practices

Authors: Jason Perez, Anna Boser, Kelly Caylor

## Setup
The code in this directory is written in Python, and we use anaconda and a requirements.txt to manage our packages. Download anaconda and run the following commands in your terminal to set up your environment: 

```{bash}
conda create -n cp_pipeline python=3.9 -y
conda activate cp_pipeline
pip install -r requirements.txt
```

## Shapefile containing Center Pivots across the World

You can obtain the shp file containing the world's center pivots [here](https://github.com/DetectCPIS/global_cpis_shp). 

To extract the Center Pivots identified in 2021, download all files in [this folder](https://github.com/DetectCPIS/global_cpis_shp/tree/main/World_CPIS_2021) and run the following code in the terminal: 

```{bash}
cd ~/Downloads 
zip -s 0 World_CPIS_2021.zip --out World_CPIS_2021_together.zip
unzip World_CPIS_2021_together.zip
```

## Inventory of the data folder
Because many of the data used in this project are too large to upload to GitHub, the data folder is in the .gitignore file and therefore none of its contents are uploaded. However, here we include a description of all datasets and their location within the data folder. These data can also be found in [this google drive folder](https://drive.google.com/drive/folders/1tpn4sNm4YDX0psiPLOqaD3Kvbd8m87Re?usp=drive_link). 

- map.goejson: A geojson of Sub-Saharan Africa, used to subset only CPIS in SSA in `code/cp_data_subset/2_id_filter_cp.py`
- World_CPIS_2021
    - The original World CPIS data from [this folder](https://github.com/DetectCPIS/global_cpis_shp/tree/main/World_CPIS_2021)
    - A modified version of this dataset with IDs, created in `code/cp_data_subset/1_id_cp.py`
    - Only the CPIS to be used for training and testing (whose IDs appear in `data/cp_ids.txt`). Created in `code/cp_data_subset/3_data_subset.py`
- cp_ids.txt
    - A list of CP IDs to be used in training and testing of the algorithm. Determined in `code/cp_data_subset/2_id_filter_cp.py`. 

## Code Description

| File Name and Location | Purpose of Code
|---------------|----------------|
| `code/cp_data_subset/1_id_cp.py` | Assigns ID column and Unique ID's to Center Pivots
| `code/cp_data_subset/2_id_filter_cp.py`   | Filters CP's to include a representative example of SSA, saves ID's in a test file
| `code/cp_data_subset/3_data_subset.py` | Uses SHP file and txt file to filter the shape file to include CP's who have ID's listed in the text file
| `code/labeling_data/1_gee_req.py` | Uses Google Earth Engine to request Landsat data of Center Pivots, will request data from Landsat 4, 5, 7, 8 and 9
| `code/labeling_data/2_cp_image_creator` | Converts TIF images from Google Earth Engine into 4 images. RGB, NDVI, LST, and Combined.
## Obtaining a subset of training/test data

1. To get our subsets of training and test data we first need to download the shp file containing all center pivots across the world. We also need to use [geojson](geojson.io) to create a JSON file with the geometric shape of Sub-Saharan Africa. This can be done by using geojson's line feature which will allow you to trace your shape and export the coordinates (Lattitude and Longitude) into a JSON file.

2. We then need to assign ID's to all the center pivots in our global CP shp file so that we can later filter/subset these center pivots to fit a representative dataset of center pivots across Sub-Saharan Africa. The script `1_id_cp.py` reads in our global shape file and creates a new column for "ID", which will create a unique ID for each center pivot in the shp file.

3. Our next step is to filter our shape file to only include CP's in Sub-Saharan Africa and make our samples representative of all of SSA. The script `2_id_filter_cp.py` reads in our GeoJSON map to only include CP's located in SSA. Then by using a stratisfying method, our GeoJSON map is split into multiple grid cells. The script then randomly samples a maximum of 4 center pivots per grid and saves their Center Pivot ID into a text file.

4. Since we now have our text file containing our desired center pivots and our shapefile containing CP ID's we then use `3_data_subset.py` which allows us to filter our shapefile to only include center pivots who's ID's are listed in the text file. 

## Using Google Earth Engine to Request Landsat Data and Creating Images for Labeling

1. Now that we have a shp file with our subsetted center pivots across SSA we will use `1_gee_req.py` to request Landsat data for the center pivots in our shp file. This script utilizes object-oriented programming to create a "LandsatExporter" class that can be called to request data. This script divides our center pivots into groups of 5, then calls the "LandsatExporter" for each group. This script requires Google Earth Engine API, as well as access to Google Drive. The script will return TIF images to the identified Google Drive folder and be named by the ID, year, and month.

2. Before uploading the images to LabelStudio, we must convert the TIFs into JPEG images. Using the script `2_cp_image_creator.py` we will define our folder of requested TIF images as our input, then this script will run recursively over each TIF image and create a total of 4 images per TIF **SAVED WHERE IN WHAT FORMAT?**. Each image constitutes a different representation of the many bands in a Landsat image which helps in the manual annotation. The four images created are:
    - Red, Green, and Blue (RGB). This is a regual visual representation of the image, which is what is actually loaded into LabelStudio for labeling. **correct??**
    - Normalized Difference Vegetation Index (NDVI). This image will allow us to see if there is vegetation on a given center pivot. 
    - Thermal infrared (TIR). Cooler temperatures signify that irrigation is likely.
    - The final image is a combination of the RGB, NDVI, and LST images side by side, which is a helpful reference while labeling. **correct??**
