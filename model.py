import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import joblib

# age – Age of the patient
# bp – Blood Pressure
# sg – Specific Gravity
# al – Albumin
# su – Sugar
# rbc – Red Blood Cells
# pc – Pus Cells
# pcc – Pus Cell Clumps
# ba – Bacteria
# bgr – Blood Glucose Random
# bu – Blood Urea
# sc – Serum Creatinine
# sod – Sodium
# pot – Potassium
# hemo – Hemoglobin
# pcv – Packed Cell Volume
# wbcc – White Blood Cell Count
# rbcc – Red Blood Cell Count
# htn – Hypertension
# dm – Diabetes Mellitus
# cad – Coronary Artery Disease
# appet – Appetite
# pe – Pedal Edema
# ane – Anemia
# class – Chronic Kidney Disease (CKD) or Non-CKD (Target column)



# Function Definations 
# Function to generate bar graph with custom ranges for numeric features
def create_bar_graph_with_range(column):   
    if column == 'age':
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['1-20', '21-40', '41-60', '61-80', '81-100']
    elif column == 'bp':
        bins = [0, 90, 120, 140, 160, 180, 200]
        labels = ['<90', '90-120', '120-140', '140-160', '160-180', '>180']
    elif column == 'bgr':
        bins = [0, 70, 100, 150, 200, 300, 500]
        labels = ['<70', '70-100', '100-150', '150-200', '200-300', '>300']
    else:
        # creating bins using the min and max values
        min_value = data[column].min()
        max_value = data[column].max()
        bins = [min_value, (min_value + max_value) / 3, 2 * (min_value + max_value) / 3, max_value]
        labels = [f'{round(bins[i], 2)} - {round(bins[i+1], 2)}' for i in range(len(bins)-1)]

    # Sorting the bins in ascending order
    bins = sorted(bins)

    # Bin the data and count the values in each bin
    binned_data = pd.cut(X_encoded[column], bins=bins, labels=labels, include_lowest=True)
    bin_counts = binned_data.value_counts()

    # Creating the bar graph
    plt.figure(figsize=(8, 6))
    sns.barplot(x=bin_counts.index, y=bin_counts.values, palette="Set3")
    plt.title(f"Bar Graph of {get_full_form(column)} with Ranges")  
    plt.xlabel('Range')  
    plt.ylabel('Count')  

    plt.savefig(f'Bar_Graph_{get_full_form(column)}.png')
    

    plt.close()


#Function to get full forms of the short forms
def get_full_form(short_form):
    if short_form == 'age':
        return 'Age of the patient'
    elif short_form == 'bp':
        return 'Blood Pressure'
    elif short_form == 'sg':
        return 'Specific Gravity'
    elif short_form == 'al':
        return 'Albumin'
    elif short_form == 'su':
        return 'Sugar'
    elif short_form == 'rbc':
        return 'Red Blood Cells'
    elif short_form == 'pc':
        return 'Pus Cells'
    elif short_form == 'pcc':
        return 'Pus Cell Clumps'
    elif short_form == 'ba':
        return 'Bacteria'
    elif short_form == 'bgr':
        return 'Blood Glucose Random'
    elif short_form == 'bu':
        return 'Blood Urea'
    elif short_form == 'sc':
        return 'Serum Creatinine'
    elif short_form == 'sod':
        return 'Sodium'
    elif short_form == 'pot':
        return 'Potassium'
    elif short_form == 'hemo':
        return 'Hemoglobin'
    elif short_form == 'pcv':
        return 'Packed Cell Volume'
    elif short_form == 'wbcc':
        return 'White Blood Cell Count'
    elif short_form == 'rbcc':
        return 'Red Blood Cell Count'
    elif short_form == 'htn':
        return 'Hypertension'
    elif short_form == 'dm':
        return 'Diabetes Mellitus'
    elif short_form == 'cad':
        return 'Coronary Artery Disease'
    elif short_form == 'appet':
        return 'Appetite'
    elif short_form == 'pe':
        return 'Pedal Edema'
    elif short_form == 'ane':
        return 'Anemia'
    elif short_form == 'class':
        return 'Chronic Kidney Disease (CKD) or Non-CKD (Target column)'
    else:
        return 'Unknown'
    

## Section 1
# Loading the Dataset
file_path = "Final_outcome.xlsx"
data = pd.read_excel(file_path)

#Dropping the first column as it is the indexing
data = data.iloc[:, 1:]  
print("Data after removing indexing column:")
print(data.head())


# List of numerical columns to scale
numerical_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply scaling to the numerical columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Verify scaling
print(data.head())
## Section 2
#Seperating the Data Features and the Target
# Seperating the targeted column to check if the results of the training works or not
target_column = "class"  

# Separating the ckd columns and other features column
X = data.drop(columns=[target_column])
y = data[target_column]

# Display the shapes to verify separation
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Display the first few rows of X and y
print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())




##Section 3
##Encoding the Categorical(Text and String Values in the data) data so it can be used in Computation to process
# Applying one-hot encoding directly using pandas get_dummies
X_encoded = pd.get_dummies(X, drop_first=True) 

# Applying Label Encoding to the target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Checking if the encoding is done currectly or not
print("Class Mapping: ", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

y_encoded = (y_encoded == 0).astype(int)  # Reversing the orders because ckd is made 0 and non-ckd as 1 by default

# Displaying the encoded data
print("Encoded Features (X):")
print(X_encoded.head())  

print("\nEncoded Target (y):")
print(y_encoded)  


## Section 4: Plotting the features in Bar Graph
# List of numeric columns from the dataset
# numeric_columns = ['age', 'bp', 'al', 'su','bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']

# # Ensure all numeric columns are present in the data
# numeric_columns = [col for col in numeric_columns if col in data.columns]

# # Create bar graphs for all numeric columns
# for column in numeric_columns:
#     create_bar_graph_with_range(column)



## Section 5: Model Creation and Training
# Step 1: Spliting the dataset into training and testing sets
# Using 50% of the data for training and 50% for testing
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.5, random_state=100)
print(f"\n Training set: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"\n Testing set: X_test = {X_test.shape}, y_test = {y_test.shape}")


# Step 2: Creating the GPR model
# Defining the kernel (RBF kernel with constant term)
kernel = RBF(1.0, (1e-4, 1e3))

# Creating the GPR model
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3)

# Step 3: Training the model
gpr.fit(X_train, y_train)



## Section 6: Evaluating the Model
#Making predictions on the test set
y_pred, sigma = gpr.predict(X_test, return_std=True)

# Since GPR gives continuous predictions, we will need to apply a threshold for classification if needed
# Converting predictions to binary if it's a classification task (like CKD detection)
y_pred_class = (y_pred > 0.7).astype(int)  # Setting the threshold to 0.7

cm =confusion_matrix(y_test, y_pred_class)
report = classification_report(y_test, y_pred_class, output_dict=True, zero_division=0)
# Printing the predictions and uncertainty
print("Predictions:", y_pred)
print("Uncertainty (std deviation):", sigma)

# Printing out the evaluation results
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred_class)
print(f"Accuracy: {accuracy:.4f}")



##  PLotting the graphs
#Plotting the accuracy per points in the test data
# Ensuring y_test and y_pred are numpy arrays
if not isinstance(y_test, np.ndarray):
    y_test = np.array(y_test)
if not isinstance(y_pred, np.ndarray):
    y_pred = np.array(y_pred)

accuracy_per_point = 1 - np.abs(y_test - y_pred)  
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), accuracy_per_point, marker='o', linestyle='-', color='blue', label='Accuracy per Point')
plt.title("Model Prediction Accuracy for Each Data Point", fontsize=16)
plt.xlabel("Test Data ", fontsize=12)
plt.ylabel("Accuracy (0 to 1)", fontsize=12)
plt.grid(alpha=0.5)
plt.ylim(0, 1.1)  
plt.text(0.95, 0.95, f'Overall Accuracy: {accuracy:.4f}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.savefig('model_accuracy_plot.png', dpi=300)


# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = X_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap of Features")
plt.savefig('correlation_heatmap.png')


# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-CKD", "CKD"], yticklabels=["Non-CKD", "CKD"])
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.savefig('confusion_matrix.png')


# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("ROC Curve", fontsize=16)
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')


# Classification report
report_df = pd.DataFrame(report).transpose()
plt.figure(figsize=(8, 6))
sns.heatmap(report_df[['precision', 'recall', 'f1-score']].iloc[:-1, :], annot=True, cmap='Blues', fmt='.2f')
plt.title('Classification Report Metrics (Precision, Recall, F1-score)')
plt.xlabel('Metrics')
plt.ylabel('Class')
plt.savefig('Classification_report.png')


##Prediction Distribution
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_class, kde=True, color="blue", bins=30)
plt.title("Prediction Distribution", fontsize=16)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.savefig('Prediction_Distribution.png')


### # Error Analysis
errors = y_test - y_pred_class
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=30, color='blue', edgecolor='black')
plt.title("Error Analysis", fontsize=16)
plt.xlabel("Error (True - Predicted)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.savefig('Error_analysis.png')


##Exporting the model
joblib.dump(gpr, 'gpr_model.pkl')
print("Model saved successfully.")



