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

- World_CPIS_2021
    - The original World CPIS data from [this folder](https://github.com/DetectCPIS/global_cpis_shp/tree/main/World_CPIS_2021)
    - A modified version of this dataset with IDs, created in `code/cp_data_subset/1_id_cp.py`
    - Only the CPIS to be used for training and testing (whose IDs appear in `data/cp_ids.txt`). Created in `code/cp_data_subset/3_data_subset.py`
- 1_script_data: Includes files contained from scripts within `1_cp_data_subset`
  - id_cp_shapefile.shp: Same file as `World_CPIS_2021` but IDs assigned to each center pivot. Used in `code/cp_data_subset/1_id_cp.py`
  - cp_ids.txt: A list of CP IDs to be used in the training and testing of the algorithm. Determined in `code/cp_data_subset/2_id_filter_cp.py`.
  - map.geojson: GeoJSON file containing a geometric shape of Africa and its coordinates. Used to subset `World_CPIS_2021` and for images.
  - filtered_center_pivots.shp: Shapefile containing center pivots only within Africa. Filtered using the previous files in 
    `code/cp_data_subset/3_data_subset.py`
- 2_script_data: Includes files contained from scripts within `2_labeling_data`
  - SSA_TIF_REQUEST: A folder containing all the TIF data obtained from Google Earth Engine in `1_gee_request.py`. This data is used to extract 
    Landsat band info and create images that are used for labeling.
  - Landsat_CP_images: A folder containing all of the images obtained from `2_cp_image_creator.py`. There is a JPEG for each type of image 
    obtained (RGB, NDVI, LST) as well as a combined image containing all 3 which is in our labeling process.
- 3_script_data: Includes files contained from scripts within `3_band_info_dataset`
  - band_names.txt: A text file containing the band names for each TIF Image, used for reorganizing column names in our dataset to be in 
    accordance with the band name rather than numerical index. Obtained from `1_band_names.py`.
  - SSA_TIF_GEOJSONS: A folder containing all the bounding box coordinates obtained from `2_tif_gjsons.py`. These GeoJSONS are used to provide 
    the general location of the center pivot, to then use our labeling data to pinpoint the exact lat/lon coordinates.
  - tif_labels.json: JSON file containing our keypoint annotation data obtained from LabelStudio. Used from `3_bands_qa_unsplit.py` to extract 
    data from the center pivots within our TIF images.
  - 5LS_dataset.csv: Dataset containing Landsat band info, coordinates, and other data. Obtained by using our TIF images and annotation data in
    `3_bands_qa_unsplit.py` and will later be prepared and modified before being used to train the classification model.
  - 5Landsat_total_dataset.csv: Modified version of the previous script that is obtained from `4_dataset_columns`. This modified version 
    reorganizes the data frame column names by their band name to avoid identical band indexes from providing different data.
  - decoded_qa_bands.csv: Modified version of the previous dataset. It is obtained from `5_decode_qa_bands.py` and has altered QA columns that 
    have been decoded to include a dictionary of information about the TIF image quality.
  - filtered_no_clouds.csv: Modified version of the previous dataset which has been filtered to only include images that were not affected by 
    clouds or cloud shadows. Obtained from `6_filter_cloud_data.py`
  - bands_info_training.csv: Dataset that will be used to train the classification model. Is a random 70% split of `filtered_no_clouds.csv` but 
    ensures that annotations from the same TIF are all included in the dataset. Obtained in `8_training_test_split.py`
  - bands_info_test.csv: Dataset that will be used to test the classification model. Is a random 30% split of `filtered_no_clouds.csv` but 
    ensures that annotations from the same TIF are all included in the dataset. This dataset will not be introduced to the model until after 
    training to test the performance. Obtained in `8_training_test_split.py`
- 4_script_images: Includes images/maps contained from scripts within `4_cross_validation_model`
  - cv_bar_graph.png: PNG image of a bar graph showing the performance scores of the cross-validation.
  - confusion_matrix.png: PNG image of a confusion matrix showing the models' correct and incorrect predictions
  - regional_map.png: PNG image containing a map of Africa, showing performance by region via F-1 scores.
  - new_ssa_map.png: PNG image containing a map of Africa that has the location of the center pivots plotted, showing correct/incorrect 
    predictions for each center pivot.
  - feature_performance.png: PNG image containing a bar graph conveying feature importance for our classification model, allowing us to see what 
    features played a bigger role in helping predict activity status.
  - four_season_map.png: PNG image containing 4 separate maps of Africa, each of the maps showcases the performance by region for each season of 
    the year.

 

## Code Description

| File Name and Location | Purpose of Code
|---------------|----------------|
| `code/1_cp_data_subset/1_id_cp.py` | Assigns ID column and Unique ID's to Center Pivots
| `code/1_cp_data_subset/2_id_filter_cp.py`   | Filters CP's to include a representative example of SSA, saves ID's in a test file
| `code/1_cp_data_subset/3_data_subset.py` | Uses SHP file and txt file to filter the shape file to include CP's who have ID's listed in the text file
| `code/2_labeling_data/1_gee_req.py` | Uses Google Earth Engine to request Landsat data of Center Pivots, will request data from Landsat 4, 5, 7, 8 and 9
| `code/2_labeling_data/2_cp_image_creator` | Converts TIF images from Google Earth Engine into 4 images. RGB, NDVI, LST, and Combined.
| `code/3_band_info_dataset/1_band_names.py` | Creates a text file including the band names for each CP TIF image
| `code/3_band_info_dataset/2_tif_gjsons.py` | Creates a GeoJSON bounding box in (lat, lon) for each CP TIF image
| `code/3_band_info_dataset/3_bands_qa_unsplit_dataset.py` | Reads in files/annotations and extracts bands/info for our complete unsplit dataset
| `code/3_band_info_dataset/4_dataset_columns.py` | Changes band info column names from a numerical identifier to their actual meaning
| `code/3_band_info_dataset/5_decode_qa_bands.py` | Decodes quality control bands to binary bits and evaluates if a TIF image includes clouds or cloud shadows
| `code/3_band_info_dataset/6_filter_cloud_data.py` | Uses the decoded binary to assess which images contain clouds, and filter them out of the dataset
| `code/3_band_info_dataset/7_label_id.py` | Assigns numerical binary label ID and merges our "Inactive" and "No CP" classes (Class 0)
| `code/3_band_info_dataset/8_training_test_split.py` | Creates 70/30 training/test datasets and ensures CPs with the same TIF ID are part of the same dataset
| `code/4_cross_validation_model/1_cross_valid_model.py` | random forest classification model that uses 5-fold cross-validation returning a classification report of the cross-validation
| `code/4_cross_validation_model/2_cv_bar_graph.py` | Performs the cross validation like `1_cross_valid_model.py`, using the cross-validation scores to return a bar graph of the "precision", "recall", and "F1-score" values
| `code/4_cross_validation_model/3_cv_ssa_map.py` | Using cross-validation scores this script returns a map of Africa consisting of the CP locations and model performance
| `code/4_cross_validation_model/4_cv_conf_matrix.py` | Using cross-validation scores this script returns a 2x2 confusion matrix
| `code/4_cross_validation_model/5_feature_plot.py` | Creates a horizontal bar graph conveying the importance of each feature provided to the model

## Obtaining a subset of training/test data

1. To get our subsets of training and test data we first need to download the shp file containing all center pivots across the world. We also need to use [geojson](geojson.io) to create a JSON file with the geometric shape of Sub-Saharan Africa. This can be done by using Geojson's line feature which will allow you to trace your shape and export the coordinates (Lattitude and Longitude) into a JSON file.

2. We then need to assign IDs to all the center pivots in our global CP shp file so that we can later filter/subset these center pivots to fit a representative dataset of center pivots across Sub-Saharan Africa. The script `1_id_cp.py` reads in our global shape file and creates a new column for "ID", which will create a unique ID for each center pivot in the shp file.

3. Our next step is to filter our shape file to only include CP's in Sub-Saharan Africa and make our samples representative of all of SSA. The script `2_id_filter_cp.py` reads in our GeoJSON map to only include CP's located in SSA. Our GeoJSON of SSA is then split into grids, and thresholds determine the size of the grids and how many center pivots are sampled from each grid. Using a grid size of 1 or (1% of the TIF) and a maximum threshold of 4 center pivots per grid 1341 center pivots were returned. The IDs of these center pivots are then saved into a text file which will be used to subset our shapefile to include a balanced representation of Sub-Saharan Africa.

4. Since we now have our text file containing our desired center pivots and our shapefile containing the CP IDs we then use `3_data_subset.py` which allows us to filter our shapefile to only include center pivots whose IDs are listed in the text file. 

## Using Google Earth Engine to Request Landsat Data and Creating Images for Labeling

1. Now that we have a shp file with our subsetted center pivots across SSA we will use `1_gee_req.py` to request Landsat data for the center pivots in our shp file. This script utilizes object-oriented programming to create a "LandsatExporter" class that can be called to request data. This script divides our center pivots into groups of 5, then calls the "LandsatExporter" for each group. This script requires Google Earth Engine API, as well as access to Google Drive. The script will return TIF images to the identified Google Drive folder and be named by the ID, year, month, and Landsat.

2. Before uploading the images to LabelStudio, we must convert the TIFs into JPEG images. Using the script `2_cp_image_creator.py` we will define our folder of requested TIF images as our input, then this script will run recursively over each TIF image and create a total of 4 images per TIF. The JPEG images created from each TIF will be saved with their original filename, with the image type added before the extension. All images will be saved into a folder "LandSat_CP_Images". Each image constitutes a different representation of the many bands in a Landsat image which helps in the manual annotation. The four images created are:
    - Red, Green, and Blue (RGB). This is a regular visual representation of the image, which is what is actually loaded into LabelStudio for labeling. 
    - Normalized Difference Vegetation Index (NDVI). This image will allow us to see if there is vegetation on a given center pivot. 
    - Thermal infrared (TIR). Cooler temperatures signify that irrigation is likely.
    - The final image is a combination of the RGB, NDVI, and LST images side by side, which is a helpful reference while labeling, allowing us to compare characteristics from all 3 images.


3. Now that we have RGB Jpeg images for each TIF we upload our pictures to Label Studio. I labeled using 4 different keypoint labels: "Active CP", "Inactive CP", "No CP", and "No CP Irrigated". Despite using four labels, we will use a binary classifier within our model by dividing these 4 labels into two classes. The first class will represent irrigated agriculture and include "Active CP" and "No CP Irrigated", our second class will represent non-irrigated land or agriculture and include "No CP" and "Inactive CP". When training our model we form these classes because we want our model to be able to distinguish irrigated and non-irrigated land; however, we include these subclasses to perform deeper validation checks for how the model will perform. I labeled using multiple keypoint labels per JPEG image. "Active CP" labels represent center pivots that had clear signs of irrigation or vegetation. This was determined through the side-by-side images produced in `2_cp_image_creator.py` which plot the National Vegetation Index and Land Surface Temperature. "Inactive CP" labels represent center pivots that were no longer in usage or had signs of abandonment due to a lack of NDVI and LST. "No CP" labels represent surrounding areas of center pivots and were gathered to obtain additional information on the surrounding areas of center pivots, making sure not to choose irrigated/vegetated areas. "No CP Irrigated" labels represent irrigated land areas such as fields or other farmland that are not center pivots.

## Example of labeling process
# Label Key: Green-Active CP, Red-Inactive CP, Brown-No CP, Blue-No CP Irrigated

Key Factors in Label Decision Making:
  -LST Image: Is the land surface temperature cool compared to the surrounding area? (Indicates water beneath the center pivot)
  -NDVI Image: Are there indications of vegetation? (Center Pivots are used to irrigate crops)
  -RGB Image: Identify the location within the image and determine if the Center Pivot look active, or abandoned? (Visual observations are 
   helpful but the prior two are the key determiners)
# Combined Image Used for Labeling Analysis
<img width="928" alt="Screen Shot 2024-05-31 at 4 07 39 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/33b59d82-970b-4f99-a6cc-3ba1971d8c09"> 

# Labels Created Via Analysis Process
<img width="518" alt="Screen Shot 2024-07-02 at 9 49 09 AM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/4a004ccd-2941-411f-a7d8-03326e6d1339">

The image on the top is our combined JPEG image obtained from `2_cp_image_creator.py`. In order the images are RGB, Infrared (not used), NDVI, and LST. The bottom image represents some of the keypoint labels I made for this image based on our JPEG images. In this annotation, I created 4 Active labels. These labels were listed as Active because we can see visually that the center pivot is a darker brown than the surrounding areas and the LST image shows that they have a much cooler temperature as well. Even though there is only slight vegetation on these pivots, the other pieces of information still allow us to label these active considering there are stages where center pivots are active but don't have crops. The Inactive labels were determined as a result of the lack of vegetation and LST. When a center pivot is abandoned it is no longer being watered, thus causing land to become dry and similar to the surrounding area. For our No CP labels, it is important to avoid random areas of land that also contain a cooler LST than the surrounding areas. These labels are included to present data on the surrounding areas of the center pivot to showcase the difference between an irrigated center pivot and the surrounding areas. Finally, for our No CP Irrigated labels, areas that included irrigated farmlands were chosen and verified using our LST image. 

4. Once finished labeling you can submit your annotations to LabelStudio and export all your annotations into a JSON file. This JSON file includes the X and Y percentage of each label, allowing us to locate where in the TIF image the label is. It also includes information such as which file the annotation is from, allowing us to use our previous TIF files to extract band data on our keypoint labels.

## Extracting Band Data and Creating the Landsat Dataset

1. Now that we have a JSON file containing these annotations we must extract band data and write it into a csv. However, since we grabbed data from Landsats 4, 5, 7, 8, and 9, not all TIFS have the same band information. Landsat 4 and 5 have some bands that differentiate from the TIFS from Landsats 7, 8, and 9 so we must account for this in our dataset. The script `1_band_names.py` reads in your folder containing all the TIF Images and receives the metadata description for each band. It writes these band names into a txt file containing band names for each TIF image.

2. Given that our JSON file from LabelStudio provides us with the X and Y percentage location of each keypoint label we can use this information to find out the latitude and longitude coordinates. `2_tif_gjsons.py` is a script that also reads in your TIF folder and produces a geojson containing the bounding box coordinates of the TIF. This series of geojsons will be saved to a folder named "SSA_TIF_GEOJSON" and 1 geojson will be produced for each TIF. Using the percentages from our JSON file and our bounding box coordinates we can calculate the exact position of each label.

3. To create our complete dataset we must now use the files returned from `1_band_names.py` and `2_tif_gjsons.py` to construct our CSV file. The script `3_bands_qa_unsplit_dataset.py` first reads in our JSON file containing the annotations, creates a mapping of the filename in the annotation JSON file to the TIF filenames, and constructs a CSV file containing the extracted band information, TIF name, TIF ID, Label, and the coordinates in (lat, lon). It also reads in our text file from `1_band_names.py` to correctly identify the correct bands to extract from our TIF image, making sure to only fill columns that correlate to the TIF's landsat band structure. The script uses the JSON data to assess the location within the TIF we will extract the band data from, as well as the label the annotation was assigned in LabelStudio.

4. After extracting our band information it is important to consider that since we obtained data from 4 different landsats that bands could share the same name, but hold different meaning (ex. Band 1 in Landsats 4&5 correspond to Blue, while in Landsats 8&9 it corresponds to coastal aerosol). In order to correctly organize our CSV file we use `4_dataset_columns.py`. This script holds dictionaries that contain a mapping from a Landsat's band number to the information it represents. It reads in our previous CSV file and iterates over every row, using it's "Landsat" column to identify which dictionary each annotation should access. It then reconstructs our CSV file to organize our band columns by their name, rather than band number. 

5. When training our model we want to avoid annotations that have data hindered by clouds or cloud shadows. Using the quality control bands obtained from our TIF images we use the script `5_decode_qa_bands.py` to convert our quality control bands to bits, and then using a dictionaries created from [USGS Website](https://hub.arcgis.com/documents/cc558a8c29fa491595762224665c399d/explore) uses the bits to analyze cloud and cloud shadow presence within the annotation. After decoding the bits it alters the values quality control columns to include a dictionary conveying whether there is Cloud Presence or Cloud Shadows altering the data. We then use `6_filter_cloud_data.py` to read in our updated CSV file and filter out any annotations with hindered data.

6. Now that we've filtered our data to only include high quality annotations we need to prepare the CSV file to be read in to our training model. Since ML models cannot take strings as input data we need to adjust our CSV to include a numeric "Label ID" to encode our labels. `7_label_id.py` reads in our CSV file and accesses the "Label" column. If the label value is "Active CP" it is assigned a value 1, while "No CP" and "Inactive CP" are assigned 0. We use this binary classification because we already have knowledge of the location of all CP's in the world, therefore by training our classification model to identify "Active CP" vs "No CP/Inactive CP", we can later apply this model strictly to location of the known CP locations to determine their activity status. Since we made sure not to annotate irrigated/vegetated areas of land for our "No CP" label, this won't cause confusion between labels as irrigation and vegetation were key factors in determining our "Active CP" label. After merging our "No CP" and "Inactive CP" labels via our "Label ID" column we can now split our data into our training/test datasets. 

7. Using `8_training_test_split.py`, we read our dataset and apply SKLearn's "Group Shuffle Split" to initialize the size of our split, and the random seed used to determine what data will be stored in each file. In order to avoid annotations from the same TIF appearing in both datasets, we ensure that annotations with the same "TIF ID" appear in the same dataset. This allows us to prevent a biased model that was trained on annotations of center pivots that appear in the test dataset. This script divides our dataset into 70% for our training subset and 30% for our test subset. This split was chosen because while training our model//setting parameter we will incorporate 5-fold group cross-validation, allowing us to validate our model using just our training dataset. The test dataset will not be used until we complete our classification model, allowing us to determine its final performance.

## Training Our Random Forest Classification Model

1. Using the 70% training dataset gathered from `8_training_test_split.py` we will begin training our model using 5-fold group cross-validation. This approach of training our model splits our training dataset into 5 groups, then runs a recursive training process using 4 of the groups, while testing itself on the remaining group. This allows us to see which type of classifier, features, parameters, and algorithms will yield the best results. During my trial process, I trained my model using 3 different classifiers: Random Forest, CatBoost, and Gradient Boost; I found that random forest consistently yielded better predictions than the other 2. Now that we've chosen to use the random forest classification model, we need to do some data manipulation and case handling to properly provide a valid dataset for our model. Within our model, we create an "X" data frame and a "Y" data frame, the "X" data frame stores all the "feature values" from our dataset, and the "Y" data frame stores the encoded labels. Essentially, we are training our model to use the data stored in "X" to predict the correct encoded label stored in "Y" (target).

2. We begin by omitting columns from our data frame that contain strings (invalid) and the quality control bands we used to determine whether the TIF images contained clouds/cloud shadows. After excluding these columns we need to alter some of the features from our dataset to account for differences in bands via Landsat or to adjust them to a categorical variable. Due to Landsats 4 and 5 not containing TIRS 1 or 2 bands, these columns are empty in the dataset, in order to avoid Nan values we fill these cells with 0. We then assign "Landsat", "Month", and "Year" to be a category type, semantically indicating that these features can take on a finite amount of values and conveying that CPs with the values of the same feature hold a categorical relevance to one another. We then encode our labels (Binary: Active CP or Inactive/No CP) using our "Label ID" column assign "X" to our dataset excluding the labels, and assign our encoded labels to "Y". After assigning our "X" and "Y" dataframes we initialize our randomforest classifier and begin training our model to look for additional features/parameters to boost performance.

3. In the script `1_cross_valid_model.py` we began improving our model by adding additional features that were calculated/gathered from data within our training dataset. In order to determine what features could be included to improve the performance of the model, the script includes functions that calculate the correlation between features and the encoded labels. It also returns which features held the most importance during the cross-validation process, allowing us to analyze which features should be added. As a result of the labeling criteria for active CP's heavily focusing on land surface temperature and vegetation, our next step was to use the Landsat band info to calculate LST, GNDVI, and EVI. Including these features provides the model with insight as to whether there is irrigation or vegetation. Due to our model performing a binary classification, we used a confidence threshold of 0.5. After performing data preprocessing and altering our model to include additional features, we once again performed our 5-fold cross-validation to assess how our model is performing.

**Updated `1_cross_valid_model.py` Classification Report**

<img width="640" alt="Screen Shot 2024-05-16 at 4 57 13 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/3a23299e-d8da-47ea-85e0-4c2081e874cb">


4. The script `2_cv_bar_graph.py` utilizes the same methodology for classification as `1_cross_valid_model.py`; however, the purpose of this script is to create a bar graph plot of our cross-validation scores. This script performs 5-fold group cross-validation on our training set, then uses "precision", "recall", and "F1-score" to create a bar graph. Precision represents the model's ability to correctly identify the assigned label (accuracy metric for specific labels). Recall score is a metric that represents the percentage of the relevant cases the model got correct (percentage of true positives correctly identified out of all true positives in the dataset). The f-1 score is a metric that conveys a mean average between the precision and recall scores.

![cv_bar_graph](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/64ba1730-2714-4678-af0c-ec377db8ee53)


5. Similar to the previous script, `3_cv_ssa_map.py` uses our model's cross-validation and creates a visual based on its scores. This script first reads in our GeoJSON file containing the bounding box of SSA. Since our GeoJSON map is saved in EPSG 4326, we can visualize performance by location/region. This script returns a map of SSA with the locations of the CP labels. Green dots represent that the model correctly identified the label, while red means that the model failed to identify the correct label.

![new_ssa_map](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/b0835f27-a8a2-4f4c-8326-41806795838d)

6. Now that we received our scores we want additional knowledge on where the model was incorrectly identifying labels. To do this we use our cross-validation scores in the script `4_cv_conf_matrix.py` to create a 2x2 confusion matrix. In this image, the Y-axis identifies the true labels, while the X-axis represents the predicted labels provided by the model. The top left box displays the correctly identified "Inactive/No CP" labels and the bottom right conveys the correctly identified "Active CP" labels. The bottom left demonstrates CPs that were predicted to be inactive but were actually active, and the top right demonstrates CPs the model predicted were active but were inactive or had no CP.

<img width="1034" alt="Screen Shot 2024-05-16 at 12 43 59 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/c157ae28-1846-4283-a6e9-ec0ffa60ddb3">

7. When evaluating our model and trying to improve its performance, identifying the features that most benefit the model can reveal trends or types of data that benefit the model more than other features. In order to get a better understanding of how my model was predicting labels the script `5_feature_importance.py` creates a horizontal bar graph to display each feature considered in the model, as well as how important it was in determining the probability of an active center pivot.

<img width="875" alt="Screen Shot 2024-05-22 at 4 47 31 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/c6db7e4a-f086-4515-82c9-3bd9d368483b">


8. The script `6_region_perf.py` uses the stratification method used in `2_id_filter_cp.py` to divide our GeoJSON map of Africa and uses the F-1 scores calculated during the cross-validation to show performance within the grids. The script uses grid cell size 2 for visible segmentations of the map.

<img width="817" alt="Screen Shot 2024-05-22 at 4 38 29 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/12ffe825-f332-4cd0-ae7d-91d8eba79263">


9. The script `7_seasonal_perf.py` analyzes the performance of the model by season. Giving the accuracy, precision, recall, and f-1 scores for each season.  
<img width="194" alt="Screen Shot 2024-05-22 at 4 37 23 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/e11ba711-1779-4f37-8efe-50abc859e1cd"><img width="194" alt="Screen Shot 2024-05-22 at 4 37 05 PM" src="https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/bfe3178c-c06d-4f7c-a551-a102543534a7">

10. Using the performance scores from `7_seasonal_perf.py`, to account for seasons varying across different parts of Africa, the script `8_seasonal_map.py` creates an image containing four different maps of Africa. There is one map corresponding to each season of the year and uses their F-1 scores to show performance by region using grid cells.

![four_seaon_map](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/18989475-b3b4-47b0-828c-5f1292d03bd2)

11. The script `9_dataset_info.py` creates 3 different images to provide insight for our dataset. The first image includes our number of observations per Landsat, allowing us to ensure there is a large enough quantity per Landsat. The second image shows the number of observations per label subclass we have, including "Active CP", "Inactive CP", "No CP", and "No CP Irrigated", and our third image shows us the number of observations per month.
![observation_landsat](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/8b43ec2e-56b8-4e8d-8d67-036eb702daa6)
![observations_label](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/e053330c-3d2f-4c4f-be35-f7dafef77a8b)
![observations_month](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/412aaa9a-771e-4599-a7ed-3f85df4b1cfb)

12. Our script `10_combined_matrix.py` produces an image conveying the model's predictions for each of our subclass labels. Since we already know that subclasses "No CP' and "Inactive CP" are representations of "non-irrigated land" and the subclasses "Active CP" and "No CP Irrigated" represent "irrigated land", we can use this image to analyze the number of correct and incorrect predictions per subclass label.
![combined_matrix](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/a2bff999-1e6f-4ac4-b859-78b937ed2cc2)

13. The script `11_label_graph.py` produces 3 separate bar graph images to show our model's performance regarding subclass label accuracy, f-1 score by different Landsat, and f-1 score  by month.
# Accuracy Score by Subclass Label
![label_graph](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/f327c5a4-f1d9-4d16-839f-8c4563733803)

# F-1 Score per Landsat Type
![landsat_graph_bar](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/3e6e9abf-a04c-4f17-b1ce-7e6b8ebceb10)

# F-1 Score Per Month
![month_graph](https://github.com/Jason-Perez27/SSA-Pivot-Detect/assets/137958043/f378fd05-360c-4113-bcc0-73f8cd55d9d5)
