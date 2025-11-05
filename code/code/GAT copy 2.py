import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

# Define GAT model
class GATModel(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GATModel, self).__init__()
        # Reduce model size to prevent overfitting
        self.conv1 = GATConv(num_node_features, 64, heads=2, dropout=0.1)
        self.conv2 = GATConv(64 * 2, 32, heads=2, dropout=0.1)
        self.conv3 = GATConv(32 * 2, 2, heads=1, dropout=0.1)
        self.norm1 = torch.nn.LayerNorm(64 * 2)
        self.norm2 = torch.nn.LayerNorm(32 * 2)
        self.dropout = torch.nn.Dropout(0.2)  # Increase dropout rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Data preprocessing and loading
def load_and_validate_data():
    # Load functional connectivity matrices (50 samples)
    fc_matrices = np.load('all_correlation_matrices50.npy')
    fc_matrices = np.array([np.nan_to_num(m, nan=np.nanmean(m)) for m in fc_matrices])
    
    # Load clinical data (50 samples)
    clinical_df = pd.read_csv('PPMI50.txt', sep='\t', header=None)
    age = StandardScaler().fit_transform(clinical_df[1].fillna(clinical_df[1].mean()).values.reshape(-1, 1))
    gender = clinical_df[2].map({'M':1, 'F':0}).fillna(0).values.reshape(-1,1)
    
    # Create label array directly: first 30 PD (0), last 20 Control (1)
    labels = np.array([0]*30 + [1]*20)
    
    # Feature standardization and fusion
    fc_features = StandardScaler().fit_transform(fc_matrices.reshape(fc_matrices.shape[0], -1))
    features = np.hstack([fc_features, age, gender])
    
    # Build graph structure (similarity matrix for 50 samples)
    adj_matrix = np.load('fused_similarity_matrix_k20_50.npy')
    edge_index = torch.tensor(np.stack(np.where(adj_matrix > 0)), dtype=torch.long)
    
    return torch.tensor(features, dtype=torch.float32), edge_index, labels, features.shape[1]

# Training and evaluation function
def train_and_evaluate(model, data, device):
    # Use smaller learning rate and higher weight decay to prevent overfitting
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=15, min_lr=1e-6)
    
    best_model = None
    best_loss = float('inf')
    early_stopping = EarlyStopping(patience=25, delta=0.001)  # Increase patience
    
    for epoch in range(200):  # Reduce training epochs
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = model.state_dict()
        
        early_stopping(loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        true = data.y[data.test_mask].cpu().numpy()
        pred = pred[data.test_mask].cpu().numpy()
        
        accuracy = (pred == true).mean()
        precision = precision_score(true, pred, average='macro', zero_division=0)
        recall = recall_score(true, pred, average='macro', zero_division=0)
        
    return accuracy, precision, recall

# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Main function
def main():
    x, edge_index, y, num_features = load_and_validate_data()
    
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    
    # Print dataset information
    print(f"Dataset information:")
    print(f"Samples: {len(y)} (PD: {sum(y==0)}, Control: {sum(y==1)})")
    print(f"Training set: {len(train_idx)} (PD: {sum(y[train_idx]==0)}, Control: {sum(y[train_idx]==1)})")
    print(f"Test set: {len(test_idx)} (PD: {sum(y[test_idx]==0)}, Control: {sum(y[test_idx]==1)})")
    print(f"Feature dimension: {num_features}")
    print(f"Edge count: {edge_index.shape[1]}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data = data.to(device)
    
    model = GATModel(num_features).to(device)
    accuracy, precision, recall = train_and_evaluate(model, data, device)
    
    print("\nGAT model test results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == '__main__':
    main()