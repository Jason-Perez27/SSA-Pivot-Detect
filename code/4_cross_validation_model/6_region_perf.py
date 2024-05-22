import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Load CSV file with training data
csv_file_path = '/Users/jasonperez/Desktop/bands_info_training.csv'
data = pd.read_csv(csv_file_path)

# Exclude columns we don't want as features in our model
excluded_columns = ['TIF Name', 'Pixel_QA', 'Cloud_QA', 'X Value', 'Y Value']

# Assign Year, Month, and Landsat as category types/Fill NA columns in the dataset for labels from Landsats 4/5 (No TIRS)
data['Year'] = data['Year'].astype('category')
data['Month'] = data['Month'].astype('category')
data['TIRS1'].fillna(0, inplace=True)
data['TIRS2'].fillna(0, inplace=True)
data['Landsat'] = data['Landsat'].astype('category')
data = data.dropna(subset=['Label ID'])
data.reset_index(drop=True, inplace=True)

# Encode label using our "Label ID" column which holds a binary 0 or 1 value
label_encoder = LabelEncoder()
data['Encoded Label'] = label_encoder.fit_transform(data['Label ID'])

# Exclude the Label column which holds a string
data = data.drop(columns=excluded_columns + ['Label'])

# Define features and target
X = data.drop(columns=['Encoded Label', 'Label ID'])
y = data['Encoded Label']

# Include latitude and longitude columns as features
latitude = data['X-Coord']
longitude = data['Y-Coord']
X['Latitude'] = latitude
X['Longitude'] = longitude

# Calculate additional vegetation indices to check for vegeatation on the keypoint annotation
X['EVI'] = 2.5 * (X['Near_Infrared'] - X['Red']) / (X['Near_Infrared'] + 6 * X['Red'] - 7.5 * X['Blue'] + 1)
X['GNDVI'] = (X['Near_Infrared'] - X['Red']) / (X['Near_Infrared'] + X['Red'])

# Calculate LST using TIRS 1 & 2 for Landsats 7, 8, and 9
def calculate_lst_swa_TIRS(TIRS1, TIRS2):
    # Constants for SWA
    K1 = 774.89  # W m^-2 sr^-1 um^-1 for Band 10
    K2 = 1321.08 # K2 = (W m^-2 sr^-1 um^-1) / K for Band 10
    K3 = 480.89  # W m^-2 sr^-1 um^-1 for Band 11
    K4 = 1201.14 # K2 = (W m^-2 sr^-1 um^-1) / K for Band 11
    
    # Convert DN values to at-sensor radiance
    rad10 = K1 / (TIRS1 + 1)
    rad11 = K3 / (TIRS2 + 1)
    
    # Convert radiance to brightness temperature 
    bt10 = K2 / np.log((K1 / rad10) + 1)
    bt11 = K4 / np.log((K3 / rad11) + 1)
    
    # Estimate LST using SWA
    lst = (bt11 * 1.379) + 0.207 * (bt11 - bt10) - 5.42
    
    return lst

# Calculate LST using the thermal band for Landsats 4 and 5
def calculate_lst_swa_thermal(thermal):
    # Constants for SWA
    K1 = 666.09  # W m^-2 sr^-1 um^-1 for Band 6
    K2 = 1282.71 # K2 = (W m^-2 sr^-1 um^-1) / K for Band 6
    
    # Convert DN values to at-sensor radiance
    rad6 = K1 / (thermal + 1)
    
    # Convert radiance to brightness temperature 
    bt6 = K2 / np.log((K1 / rad6) + 1)
    
    # Estimate LST using SWA
    lst = (bt6 * 0.988) - 0.052  # Assuming emissivity is 0.97
    
    return lst

# Define a variable for the band info to be used for the LST functions
thermal_value  = X['Thermal']
TIRS1 = X['TIRS1']  
TIRS2 = X['TIRS2'] 

# Calculate LST using Split Window Algorithm (SWA)
lst_values_swa_TIRS = calculate_lst_swa_TIRS(TIRS1, TIRS2)
X['LST_SWA_TIRS'] = lst_values_swa_TIRS
lst_values_swa_thermal = calculate_lst_swa_thermal(thermal_value)
X['LST_SWA_THERMAL'] = lst_values_swa_thermal


# Define the groups based on the first column of the CSV file
groups = data.iloc[:, 0]

# Construct our Random Forest Classifier
classifier = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=50
)


# Use GroupKFold for cross validation and set splits to 5
group_kfold = GroupKFold(n_splits=5)

# Variables to store classification results
true_labels = []
predicted_probabilities = []
all_predictions = np.zeros(len(data))
all_true_labels = np.zeros(len(data), dtype=int)
# Choose a confidence threshold
threshold = 0.7

# Perform 5 fold cross validation for each fold of the dataset
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    classifier.fit(X_train, y_train)
    calibrated_classifier = CalibratedClassifierCV(classifier, method='sigmoid', cv='prefit')
    calibrated_classifier.fit(X_train, y_train)
    
    y_pred_prob = calibrated_classifier.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob >= threshold).astype(int)  
    
    # Store the predicted probabilities, predicted labels, and the true labels from all 5 iterations of the cross validation
    all_predictions[test_idx] = y_pred_prob
    all_true_labels[test_idx] = y_test

# Convert interger stored values back into labels
label_encoder.fit(y)
true_labels = label_encoder.inverse_transform(true_labels)


# Filter all of our predictions from cross validation based on our threshold
adjusted_predictions = [(prob >= float(threshold)).astype(int) for prob in predicted_probabilities]

# Convert the predicted labels from cross validation back into class labels
predicted_labels = label_encoder.inverse_transform(adjusted_predictions)


import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from shapely.geometry import box
# Load GeoJSON file containing outline of Sub-Saharan Africa
ssa_map = gpd.read_file('/Users/jasonperez/Desktop/map.geojson')

# Assign the CRS of the GeoJSON map
ssa_map.crs = 'EPSG:4326'

# Adjust predictions with the chosen threshold, checking that the prob is greater than or equal to the threshold
adjusted_predictions_test = [(prob >= threshold).astype(int) for prob in y_pred_prob]

report_test = classification_report(y_test, adjusted_predictions_test, labels=[0, 1], zero_division='warn')

# Construct the geometry for our map using the Latitude/Longitude coordinates saved in the GeoJSON file
geo_data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(X['Longitude'], X['Latitude']))
geo_data['True_Labels'] = all_true_labels
geo_data['Predicted_Labels'] = (all_predictions >= 0.7).astype(int)
geo_data['centroid'] = geo_data['geometry'].centroid

# Define grid size
grid_size = 3
minx, miny, maxx, maxy = geo_data.total_bounds
x_edges = np.arange(minx, maxx, grid_size)
y_edges = np.arange(miny, maxy, grid_size)

# Create a grid of polygons
grid_polygons = []
for x in x_edges:
    for y in y_edges:
        grid_polygons.append(box(x, y, x + grid_size, y + grid_size))




geo_data['grid'] = geo_data.apply(
    lambda row: (
        int(row.centroid.x / grid_size),
        int(row.centroid.y / grid_size)
    ),
    axis=1
)

predicted_labels = (all_predictions >= threshold).astype(int)

# Convert true_labels and predicted_labels to numpy arrays to plot the points in our visual
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Create a GeoDataFrame for the grid
grid_gdf = gpd.GeoDataFrame({'geometry': grid_polygons}, crs=geo_data.crs)

# Spatial join to associate each point with a grid cell
geo_data['geometry'] = geo_data.centroid  # Ensure points are centroids if not already
joined = gpd.sjoin(geo_data, grid_gdf, how='left', op='within')

# Calculate F-1 scores for each grid cell
f1_scores = joined.groupby(joined.index_right).apply(
    lambda g: f1_score(g['True_Labels'], g['Predicted_Labels']) if len(g['True_Labels'].unique()) > 1 else np.nan
)

# Map F-1 scores back to the grid GeoDataFrame
grid_gdf['F1_Score'] = f1_scores

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
ssa_map.plot(ax=ax, color='lightgray')  # Background map
grid_gdf.dropna(subset=['F1_Score']).plot(column='F1_Score', cmap='RdYlGn', ax=ax, legend=True,
                                          legend_kwds={'label': "F1 Score by Grid Cell"})
ax.set_title('F1 Score Performance by Grid Cell in SSA')
plt.show()