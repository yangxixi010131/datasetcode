import pandas as pd
import numpy as np
import pickle

def calculate_pearson_correlation(time_series):
    # Convert the time series array to a Pandas DataFrame
    df = pd.DataFrame(time_series)
    # Calculate the Pearson correlation coefficient matrix
    correlations = df.corr(method='pearson')
    return correlations

# Load the time series array from the pickle file
with open("merged_time_series.pkl", "rb") as f:
    time_series_new = pickle.load(f)

# Initialize an empty list to store all correlation matrices
all_correlation_matrices = []

# Extract each time series from time_series_new and calculate the correlation matrix
for i, time_series in enumerate(time_series_new):
    # print(f"Time Series {i+1}:")
    correlation_matrix = calculate_pearson_correlation(time_series)
    
    # print("Correlation Matrix:")
    # print(correlation_matrix)
    # print("\n-----------------------------------\n")
    
    # Add the correlation matrix to the list
    all_correlation_matrices.append(correlation_matrix.values)

# Merge all correlation matrices into a single numpy array
all_correlation_matrices_np = np.stack(all_correlation_matrices)

# Save as a separate numpy file
np.save('all_correlation_matrices103.npy', all_correlation_matrices_np)