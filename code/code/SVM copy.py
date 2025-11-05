import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer

# Load functional connectivity matrix data
correlation_matrices = np.load('all_correlation_matrices103.npy')

# Load label data
labels_df = pd.read_csv('PPMI103.txt', delimiter='\t', header=None)
labels_df = labels_df.iloc[:, [0, 3]]  # Select ID and label columns
labels_df.columns = ['ID', 'Label']

# Convert label information to 0 and 1
labels_df['Label'] = labels_df['Label'].apply(lambda x: 0 if x == 'PD' else 1)

# Ensure the order of matrices and labels is consistent
if len(correlation_matrices) != len(labels_df):
    raise ValueError("The number of correlation matrices does not match the number of labels.")

# Combine functional connectivity matrices and label information
labels_df['Matrix'] = list(correlation_matrices)

# Extract features and labels
X = np.array(labels_df['Matrix'].tolist())
y = labels_df['Label'].values

# Check for NaN values
if np.any(np.isnan(X)):
    print("NaN values detected in the feature matrix. Imputing missing values.")
    
    # Use SimpleImputer to fill missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the data into a 2D array
n_samples, n_features1, n_features2 = X_train.shape
X_train_flat = X_train.reshape(n_samples, n_features1 * n_features2)
n_samples, n_features1, n_features2 = X_test.shape
X_test_flat = X_test.reshape(n_samples, n_features1 * n_features2)

# Standardize features
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Create an SVM model
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train_flat, y_train)

# Predict
y_pred = svm_model.predict(X_test_flat)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')

# Print results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')