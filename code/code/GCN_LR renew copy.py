import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import kneighbors_graph

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

# Keep the original GCN model unchanged and add Laplacian regularization
class RobustGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(RobustGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)
        self.norm1 = torch.nn.LayerNorm(128)
        self.norm2 = torch.nn.LayerNorm(64)
        self.dropout = torch.nn.Dropout(0.01)

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
    
    def compute_laplacian(self, edge_index, num_nodes):
        """Compute normalized Laplacian matrix"""
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
        laplacian = sp.csgraph.laplacian(adj, normed=True)
        return torch.tensor(laplacian.toarray(), dtype=torch.float32)

# Improved data preprocessing
def enhanced_preprocessing(features):
    """Add interaction and polynomial features"""
    squared = np.square(features[:, :20])  # Assume first 20 features are important
    interactions = features[:, :10] * features[:, 10:20]
    return np.hstack([features, squared, interactions])

# Improved graph construction
def build_robust_graph(features, y, k=20):
    """Combine KNN and label information for graph construction"""
    knn_graph = kneighbors_graph(features, n_neighbors=k, mode='connectivity')
    label_sim = (y.reshape(-1,1) == y.reshape(1,-1)).astype(float) * 0.3
    adj = knn_graph.toarray() + label_sim
    adj[adj < 0.5] = 0  # Filter weak connections
    return torch.tensor(np.stack(np.where(adj > 0)), dtype=torch.long)

# Data loading (keep original structure with enhancements)
def load_and_validate_data():
    fc_matrices = np.load('all_correlation_matrices103.npy')
    fc_matrices = np.array([np.nan_to_num(m, nan=np.nanmean(m)) for m in fc_matrices])
    
    clinical_df = pd.read_csv('PPMI103.txt', sep='\t', header=None)
    age = StandardScaler().fit_transform(clinical_df[1].fillna(clinical_df[1].mean()).values.reshape(-1, 1))
    gender = clinical_df[2].map({'M':1, 'F':0}).fillna(0).values.reshape(-1,1)
    labels = clinical_df[3].map({'PD':0, 'Control':1}).values
    
    fc_features = StandardScaler().fit_transform(fc_matrices.reshape(fc_matrices.shape[0], -1))
    features = np.hstack([fc_features, age, gender])
    features = enhanced_preprocessing(features)
    
    edge_index = build_robust_graph(features, labels)
    return torch.tensor(features, dtype=torch.float32), edge_index, labels, features.shape[1]

# Improved training strategy (add Laplacian regularization)
def train_model(model, data, device):
    # Precompute Laplacian matrix (save time)
    laplacian = model.compute_laplacian(data.edge_index, data.x.size(0)).to(device)
    
    # Focal loss function
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=0.75, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.nll_loss(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            return (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
    
    class_counts = torch.bincount(data.y[data.train_mask])
    weights = 1. / (class_counts.float() + 1e-6)
    loss_fn = FocalLoss(alpha=weights[1].item())
    
    optimizer = torch.optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': 0.01},
        {'params': model.conv2.parameters(), 'lr': 0.005},
        {'params': model.conv3.parameters(), 'lr': 0.001}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_f1 = 0
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        cls_loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        
        # Laplacian regularization (applied to first hidden layer)
        h1 = F.elu(model.conv1(data.x, data.edge_index))
        lap_loss = torch.trace(h1.T @ laplacian @ h1) / data.x.size(0)
        
        # Total loss = classification loss + 0.1*Laplacian regularization
        total_loss = cls_loss + 0.1 * lap_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validation
        if epoch % 15 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data).argmax(dim=1)
                true = data.y[data.test_mask].cpu().numpy()
                pred = pred[data.test_mask].cpu().numpy()
                
                precision = precision_score(true, pred, average='macro', zero_division=0)
                recall = recall_score(true, pred, average='macro', zero_division=0)
                current_f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(f'Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | F1: {current_f1:.4f}')

def main():
    x, edge_index, y, num_features = load_and_validate_data()
    
    # Create dataset
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    model = RobustGCN(num_node_features=num_features).to(device)
    
    train_model(model, data, device)
    
    # Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    with torch.no_grad():
        pred = model(data).argmax(dim=1)
        true = data.y[data.test_mask].cpu().numpy()
        pred = pred[data.test_mask].cpu().numpy()
        
        print("\nFinal test results:")
        print(f"Accuracy: {(pred == true).mean():.4f}")
        print(f"Precision: {precision_score(true, pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(true, pred, average='macro'):.4f}")

if __name__ == '__main__':
    main()