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

3. Our next step is to filter our shape file to only include CP's in Sub-Saharan Africa and make our samples representative of all of SSA. The script `2_id_filter_cp.py` reads in our GeoJSON map to only include CP's located in SSA. Our GeoJSON of SSA is then split into grids, and thresholds determine the size of the grids and how many center pivots are sampled from each grid. Using a grid sizes of 1 or (1% of the TIF) and a maximum threshold of 4 center pivots per grids 1341 center pivots were returned. The ID's of these center pivots are then saved into a text file which will be used to subset our shapefile to include a balanced representation of Sub Saharan Africa.

4. Since we now have our text file containing our desired center pivots and our shapefile containing CP ID's we then use `3_data_subset.py` which allows us to filter our shapefile to only include center pivots who's ID's are listed in the text file. 

## Using Google Earth Engine to Request Landsat Data and Creating Images for Labeling

1. Now that we have a shp file with our subsetted center pivots across SSA we will use `1_gee_req.py` to request Landsat data for the center pivots in our shp file. This script utilizes object-oriented programming to create a "LandsatExporter" class that can be called to request data. This script divides our center pivots into groups of 5, then calls the "LandsatExporter" for each group. This script requires Google Earth Engine API, as well as access to Google Drive. The script will return TIF images to the identified Google Drive folder and be named by the ID, year, month, and landsat.

2. Before uploading the images to LabelStudio, we must convert the TIFs into JPEG images. Using the script `2_cp_image_creator.py` we will define our folder of requested TIF images as our input, then this script will run recursively over each TIF image and create a total of 4 images per TIF. The JPEG images created from each TIF will be saved with their original filename, with the image type added before the extension. All images will be saved into a folder "LandSat_CP_Images". Each image constitutes a different representation of the many bands in a Landsat image which helps in the manual annotation. The four images created are:
    - Red, Green, and Blue (RGB). This is a regual visual representation of the image, which is what is actually loaded into LabelStudio for labeling. 
    - Normalized Difference Vegetation Index (NDVI). This image will allow us to see if there is vegetation on a given center pivot. 
    - Thermal infrared (TIR). Cooler temperatures signify that irrigation is likely.
    - The final image is a combination of the RGB, NDVI, and LST images side by side, which is a helpful reference while labeling, allowing us to compare characteristics from all 3 images. 

3. Now that we have RGB Jpeg images for each TIF we upload our images to Label Studio. I labeled using 4 different keypoint labels: "Active CP", "Inactive CP", "No CP", "Cloud". I labeled using multiple keypoint labels per JPEG image. "Active CP" labels represent center pivots that had clear signs of irrigation or vegetation. This was determined through the side-by-side images produced in `2_cp_image_creator.py` which plot the National Vegetation Index and Land Surface Temperature. "No CP" labels represent center pivots that were no longer in usage, or had signs of abandonment due to a lack of NDVI and LST. "No CP" labels represent surrounding areas of center pivots and were gathered to obtain additonal information on the surrounding areas of center pivots. "Cloud" labels represent clouds found in the TIF images. These are obtained to provide additional information on images with large cloud coverage.

4. Once finished labeling you can submit your annotations to LabelStudio and export all your annotations into a JSON file. This JSON file includes the X and Y percentage of each label, allowing us to locate where in the TIF image the label is. It also includes information such as which file the annotation is from, allowing us to use our previous TIF files to extract band data on our keypoint labels.

## Extracting Band Data and Creating the 5 Landsat Dataset

1. Now that we have a JSON file containing these annotations we must extract band data and write it into a csv. However, since we grabbed data from Landsats 4, 5, 7, 8, and 9, not all TIFS have the same band information. Landsat 4 and 5 have some bands that differentiate from the TIFS from Landsats 7, 8, and 9 so we must account for this in our dataset. The script `1_band_names.py` reads in your folder containing all the TIF Images and recieves the metadata description for each band. It writes these band names into a txt file containing band names for each TIF image.

2. Given that our JSON file from LabelStudio provides us with the X and Y percentage location of each keypoint label we can use this information to find out the latitude and longitude coordinates. `2_tif_gjsons.py` is a script that also reads in your TIF folder and produces a geojson containing the bounding box coordinates of the TIF. This series of geojsons will be saved to a folder named "SSA_TIF_GEOJSON" and 1 geojson will be produced for each TIF. Using the percentages from our JSON file and our bounding box coordinates we can calculate the exact position of each label.

3. To create our complete dataset we must now use the files returned from `1_band_names.py` and `2_tif_gjsons.py` to construct our CSV file. The script `3_bands_qa_unsplit_dataset.py' first reads in our JSON file containing the annotations, creates a mapping of the filename in the annotation JSON file to the TIF filenames