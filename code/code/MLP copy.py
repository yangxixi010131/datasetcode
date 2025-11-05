import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Data Loading
# Load functional connectivity matrices
feature_matrices = np.load('all_correlation_matrices103.npy')

# Load label data
labels_df = pd.read_csv('PPMI103.txt', sep='\t', header=None)
labels_df = labels_df[[0, 3]]  # Select ID and label information
labels_df.columns = ['id', 'label']

# Convert labels to 0 and 1
labels_df['label'] = labels_df['label'].apply(lambda x: 0 if x == 'PD' else 1)

# Align labels and functional connectivity matrices by ID
labels_dict = dict(zip(labels_df['id'], labels_df['label']))
max_id = max(labels_dict.keys())
labels = np.array([labels_dict.get(i, np.nan) for i in range(max_id + 1)])  # IDs start from 0
features = feature_matrices

# 2. Data Preprocessing
# Standardize features
scaler = StandardScaler()
features = np.array([scaler.fit_transform(fm) for fm in features])

# Check for NaN or Inf values in the data and handle them
def clean_data(features, labels):
    # Check and handle NaN or Inf values in features
    features[np.isnan(features)] = 0
    features[np.isinf(features)] = 0
    
    # Handle NaN values in labels
    valid_indices = ~np.isnan(labels)
    features = features[valid_indices]
    labels = labels[valid_indices]
    
    return features, labels

features, labels = clean_data(features, labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 3. Model Construction
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(116*116, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 2)  # Output layer: 2 classes

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, 116*116)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # Adjust learning rate

# 4. Model Training
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # Print loss for each epoch
    print(f'Epoch {epoch + 1:03d} loss {loss.item():.4f}')
    
    # Check if loss is NaN
    if torch.isnan(loss):
        print("Loss is NaN during training, stopping training")
        break

# 5. Model Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
    precision = precision_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    recall = recall_score(y_test_tensor.numpy(), predicted.numpy(), average='weighted')
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')