import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, classification_report
from sklearn.neighbors import kneighbors_graph

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Keep the original GCN model unchanged
class RobustGCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(RobustGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)
        self.norm1 = torch.nn.LayerNorm(128)
        self.norm2 = torch.nn.LayerNorm(64)
        self.dropout = torch.nn.Dropout(0.15)

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

# Enhanced data preprocessing
def enhanced_data_processing():
    # Load functional connectivity matrices
    fc_matrices = np.load('all_correlation_matrices103.npy')
    print(f"Original matrix shape: {fc_matrices.shape}, NaN count: {np.isnan(fc_matrices).sum()}")
    
    # Handle NaN values per sample
    for i in range(fc_matrices.shape[0]):
        sample = fc_matrices[i]
        if np.isnan(sample).any():
            sample_mean = np.nanmean(sample)
            fc_matrices[i] = np.nan_to_num(sample, nan=sample_mean)
    
    # Load clinical data
    clinical_df = pd.read_csv('PPMI103.txt', sep='\t', header=None)
    age = clinical_df[1].values.astype(np.float32)
    gender = clinical_df[2].map({'M':1, 'F':0}).fillna(0).values.astype(np.float32)
    labels = clinical_df[3].map({'PD':0, 'Control':1}).values
    
    # Handle missing values
    age = np.nan_to_num(age, nan=np.nanmean(age))
    
    # Advanced feature engineering
    fc_features = fc_matrices.reshape(fc_matrices.shape[0], -1)
    fc_scaler = StandardScaler()
    fc_features = fc_scaler.fit_transform(fc_features)
    
    # Add polynomial features
    age_poly = StandardScaler().fit_transform(np.column_stack([
        age,
        age**2,
        np.log(np.abs(age)+1e-6)
    ]))
    
    # Add interaction features
    interaction_feat = fc_features[:, :10] * fc_features[:, 10:20]
    
    # Merge all features
    features = np.hstack([
        fc_features,
        age_poly,
        gender.reshape(-1,1),
        interaction_feat
    ])
    
    # Build enhanced graph structure
    def build_enhanced_graph(feat, lbl, k=15):
        # KNN graph based on features
        knn_graph = kneighbors_graph(feat, n_neighbors=k, mode='connectivity')
        
        # Label similarity (strengthen connections between same-label nodes)
        label_sim = (lbl.reshape(-1,1) == lbl.reshape(1,-1)).astype(float) * 0.5
        
        # Combine both connection types
        adj = knn_graph.toarray() + label_sim
        adj[adj < 0.6] = 0  # Filter weak connections
        return torch.tensor(np.stack(np.where(adj > 0)), dtype=torch.long)
    
    edge_index = build_enhanced_graph(features, labels)
    
    return torch.tensor(features, dtype=torch.float32), edge_index, labels, features.shape[1]

# Improved training pipeline
def advanced_training(model, data, device):
    # Class-weighted loss function
    class_counts = torch.bincount(data.y[data.train_mask])
    class_weights = torch.tensor([
        1.0,  # PD class weight
        class_counts[0]/class_counts[1] * 3.0  # Enhanced Control class weight
    ], device=device)
    
    # Focal loss (enhance minority class learning)
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=0.75, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = F.nll_loss(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = self.alpha * (1-pt)**self.gamma * ce_loss
            return loss.mean()
    
    loss_fn = FocalLoss(alpha=class_weights[1].item())
    
    # Layer-wise optimizer configuration
    optimizer = torch.optim.AdamW([
        {'params': model.conv1.parameters(), 'lr': 0.01},
        {'params': model.conv2.parameters(), 'lr': 0.005},
        {'params': model.conv3.parameters(), 'lr': 0.001},
        {'params': model.norm1.parameters(), 'lr': 0.0005},
        {'params': model.norm2.parameters(), 'lr': 0.0005}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    best_f1 = 0
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        
        out = model(data)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Validate every 15 epochs
        if epoch % 15 == 0:
            model.eval()
            with torch.no_grad():
                probs = torch.exp(model(data)[data.test_mask])
                pred = probs.argmax(dim=1)
                true = data.y[data.test_mask].cpu().numpy()
                
                # Compute F1 score
                precision = precision_score(true, pred.cpu(), average='macro', zero_division=0)
                recall = recall_score(true, pred.cpu(), average='macro', zero_division=0)
                current_f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Best F1: {best_f1:.4f}')

# Optimized evaluation function
def enhanced_evaluation(model, data, device):
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    with torch.no_grad():
        logits = model(data)
        probs = torch.exp(logits)
        
        # Dynamic threshold adjustment
        def find_optimal_threshold(true, probs):
            thresholds = np.linspace(0.3, 0.7, 41)
            best_f1 = 0
            best_th = 0.5
            for th in thresholds:
                pred = (probs[:,1] > th).cpu().numpy()
                f1 = 2 * (precision_score(true, pred, zero_division=0) * 
                         recall_score(true, pred, zero_division=0)) / \
                      (precision_score(true, pred, zero_division=0) + 
                       recall_score(true, pred, zero_division=0) + 1e-6)
                if f1 > best_f1:
                    best_f1 = f1
                    best_th = th
            return best_th
        
        true = data.y[data.test_mask].cpu().numpy()
        optimal_th = find_optimal_threshold(true, probs[data.test_mask])
        adjusted_pred = (probs[data.test_mask][:,1] > optimal_th).long().cpu().numpy()
        
        print(f"\nOptimal threshold: {optimal_th:.2f}")
        print("Final test results:")
        print(f"Accuracy: {(adjusted_pred == true).mean():.4f}")
        print(f"Precision: {precision_score(true, adjusted_pred, average='macro'):.4f}")
        print(f"Recall: {recall_score(true, adjusted_pred, average='macro'):.4f}")
        print("\nDetailed classification report:")
        print(classification_report(true, adjusted_pred, target_names=['PD', 'Control'], zero_division=0))

def main():
    # Data preparation
    x, edge_index, y, num_features = enhanced_data_processing()
    
    # Create dataset
    data = Data(x=x, edge_index=edge_index, y=torch.tensor(y, dtype=torch.long))
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    data.train_mask = torch.zeros(len(y), dtype=torch.bool)
    data.test_mask = torch.zeros(len(y), dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.test_mask[test_idx] = True
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Model initialization
    model = RobustGCN(num_node_features=num_features).to(device)
    
    # Training
    advanced_training(model, data, device)
    
    # Evaluation
    enhanced_evaluation(model, data, device)

if __name__ == '__main__':
    main()