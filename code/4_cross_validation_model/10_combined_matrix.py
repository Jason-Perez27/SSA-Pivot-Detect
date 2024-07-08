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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file with training data
csv_file_path = '/home/waves/data/SSA-Pivot-Detect/data/3_script_data/filtered_no_clouds_label_id.csv'
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
most_frequent_landsat = data['Landsat'].mode()[0]
data['Landsat'].fillna(most_frequent_landsat, inplace=True)
string_labels = data[['Label', 'Unique ID']].copy()

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
all_predictions = []
all_true_labels = []
all_string_labels = []
combined_results = pd.DataFrame(0, index=["Active CP", "No CP Irrigated", "Inactive CP", "No CP"], columns=[0, 1])

# Perform 5 fold cross validation for each fold of the dataset
for train_idx, test_idx in group_kfold.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Use the preserved string labels for the test set, mapped by Unique ID
    string_labels_test = string_labels.iloc[test_idx]

    # Fit the classifier
    classifier.fit(X_train, y_train)
    
    # Predict probabilities and convert to binary predictions
    y_pred_prob = classifier.predict_proba(X_test)[:, 1]
    binary_predictions = [(prob >= 0.5).astype(int) for prob in y_pred_prob]

    # Aggregate results for each subclass
    for subclass in combined_results.index:
        subclass_indices = [i for i, label in enumerate(string_labels_test['Label']) if label == subclass]
        subclass_true_labels = [y_test.iloc[i] for i in subclass_indices]
        subclass_predictions = [binary_predictions[i] for i in subclass_indices]

        # Update the combined results DataFrame
        temp_conf_matrix = confusion_matrix(subclass_true_labels, subclass_predictions, labels=[0, 1])
        temp_df = pd.DataFrame(temp_conf_matrix, index=[0, 1], columns=[0, 1])
        combined_results.loc[subclass, 0] += temp_df.loc[0, 0]
        combined_results.loc[subclass, 1] += temp_df.loc[0, 1]
        combined_results.loc[subclass, 0] += temp_df.loc[1, 0]
        combined_results.loc[subclass, 1] += temp_df.loc[1, 1]

# Plot the combined confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(combined_results, annot=True, fmt="d", cmap="Blues", xticklabels=['Non Irrigated', 'Irrigated'], yticklabels=combined_results.index)
plt.title('Combined Confusion Matrix for All Subclasses')
plt.xlabel('Class Labels')
plt.ylabel('Model Predictions')
plt.show()