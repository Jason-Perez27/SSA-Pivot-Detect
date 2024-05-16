import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
latitude = data['Y-Coord']
longitude = data['X-Coord']
X['Latitude'] = latitude
X['Longitude'] = longitude


# Add spatial features to give context about surrounding area
band_names = ['Blue', 'Green', 'Red', 'Near_Infrared', 'SWIR1', 'Thermal', 'SWIR2', 'TIRS1', 'TIRS2']
radius = 2  
for idx, row in X.iterrows():
    lat = row['Latitude']
    lon = row['Longitude']
    dist_squared = (X['Latitude'] - lat)**2 + (X['Longitude'] - lon)**2
    pixels_within_radius = dist_squared < radius**2
    for band in band_names:
        # Calculate mean and std dev for pixels within the radius
        X.loc[idx, f'SpatialMean_{band}'] = data.loc[pixels_within_radius, band].mean(skipna=True)
        X.loc[idx, f'SpatialStd_{band}'] = data.loc[pixels_within_radius, band].std(skipna=True)

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

# Perform 5 fold cross validation for each fold of the dataset
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Fit the classifier before using it in the CalibratedClassifierCV
    classifier.fit(X_train, y_train)
    
    calibrated_classifier = CalibratedClassifierCV(classifier, method='sigmoid', cv='prefit')
    calibrated_classifier.fit(X_train, y_train)
    
    y_pred_prob = calibrated_classifier.predict_proba(X_test)[:, 1]
    
    true_labels.extend(y[test_idx])
    predicted_probabilities.extend(y_pred_prob)

# Convert labels back to original class labels
label_encoder.fit(y)
true_labels = label_encoder.inverse_transform(true_labels)

# Choose a confidence threshold for label assignment
threshold = 0.5

# Adjust predictions with the chosen threshold
adjusted_predictions = [(prob >= float(threshold)).astype(int) for prob in predicted_probabilities]

# Convert predicted labels back to original class labels
predicted_labels = label_encoder.inverse_transform(adjusted_predictions)

# Generate and print the classification report
report = classification_report(true_labels, predicted_labels, labels=label_encoder.classes_, zero_division='warn')

import matplotlib.pyplot as plt
import seaborn as sns

# Feature importances from the RandomForestClassifier
feature_importances = classifier.feature_importances_
features = X.columns
indices = np.argsort(feature_importances)[::-1]

# Create a bar plot for feature importances
plt.figure(figsize=(10, 10))
ax = sns.barplot(x=feature_importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Features')
ax.tick_params(axis='y', labelsize=10) 
plt.tight_layout() 
plt.show()