import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
import numpy as np

# 1. Dataset for CBM extracted data
class CBMDataset(Dataset):
    def __init__(self, data_dir='cbm_input', radius=100.0):
        self.files = glob.glob(os.path.join(data_dir, '*_cbm_data.pt'))
        self.data = []
        self.radius = radius # Physical radius to define neighbors
        
        for f in self.files:
            item = torch.load(f)
            # Build adjacency matrix based on Euclidean distance
            coords = item['coordinates']
            dist_matrix = torch.cdist(coords, coords, p=2)
            adj_matrix = (dist_matrix < self.radius).float()
            
            item['adjacency'] = adj_matrix
            self.data.append(item)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['features'], item['adjacency'], item['pathways']

# 2. GATv2 Architecture (Pure PyTorch implementation)
class GATv2Layer(nn.Module):
    """
    Dynamic Graph Attention Network (GATv2) Layer.
    Computes: e_ij = a^T LeakyReLU(W[c_i || c_j])
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self, h, adj):
        # h: [N, in_features], adj: [N, N]
        Wh = self.W(h) # [N, out_features]
        N = Wh.size(0)
        
        # To avoid O(N^2) CUDA OOM, only compute combinations for valid edges
        edges = adj.nonzero(as_tuple=False) # [E, 2]
        i, j = edges[:, 0], edges[:, 1]
        
        Wh_i = Wh[i] # [E, out_features]
        Wh_j = Wh[j] # [E, out_features]
        edge_combinations = torch.cat([Wh_i, Wh_j], dim=-1) # [E, 2 * out_features]
        
        # Calculate attention scores for edges only
        e_edges = self.a(self.leakyrelu(edge_combinations)).squeeze(-1) # [E]
        
        # Scatter back to dense NxN matrix
        e = -9e15 * torch.ones((N, N), device=h.device)
        e[i, j] = e_edges
        
        attention = F.softmax(e, dim=-1)
        
        # Aggregate
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

class SpatiallyGroundedCBM(nn.Module):
    def __init__(self, num_concepts=32, hidden_dim=64, num_pathways=50):
        super().__init__()
        # Since concepts are already extracted (CBM bottleneck is frozen VLM),
        # this network starts immediately at the graph level.
        self.gat1 = GATv2Layer(num_concepts, hidden_dim)
        self.gat2 = GATv2Layer(hidden_dim, hidden_dim)
        
        # Final MLP to predict pathway activation scores (ssGSEA)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Prevent overfitting on massive dataset
            nn.Linear(hidden_dim, num_pathways)
        )
        
    def forward(self, concepts, adjacency):
        h = self.gat1(concepts, adjacency)
        h = self.gat2(h, adjacency)
        pathways_pred = self.regressor(h)
        return pathways_pred

# 3. Training Loop with PCC Evaluation
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    dataset = CBMDataset()
    if len(dataset) == 0:
        print("No formatted CBM data found. Run format_data_cbm.py first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    _, _, sample_pathways = dataset[0]
    num_pathways = sample_pathways.shape[1]
    num_concepts = dataset[0][0].shape[1]
    
    print(f"Number of biological concepts (K): {num_concepts}")
    print(f"Number of pathway targets (P): {num_pathways}")
    
    model = SpatiallyGroundedCBM(num_concepts=num_concepts, hidden_dim=128, num_pathways=num_pathways).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500) # Smooth LR decay
    criterion = nn.MSELoss()
    
    epochs = 500 # Increased for overnight convergence
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for concepts, adj, pathways in dataloader:
            # Batch size is 1, squeeze batch dimension
            concepts = concepts.squeeze(0).to(device)
            adj = adj.squeeze(0).to(device)
            pathways = pathways.squeeze(0).to(device)
            
            # --- NEW: Z-score Normalization ---
            # ssGSEA scores can be in the thousands. The HisToSGE paper uses scaled targets!
            # By standardizing (mean=0, std=1), the MSE will drop below 1.0.
            pathway_mean = pathways.mean(dim=0, keepdim=True)
            pathway_std = pathways.std(dim=0, keepdim=True) + 1e-8
            pathways = (pathways - pathway_mean) / pathway_std
            
            # --- NEW: Train/Test Node Masking (80/20 Split) ---
            N = concepts.shape[0]
            # Create a consistent split for reproducibility (or random per epoch for robustness)
            indices = torch.randperm(N, device=device)
            split_idx = int(0.8 * N)
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            
            optimizer.zero_grad()
            
            preds = model(concepts, adj)
            
            # Calculate loss ONLY on the 80% training spots!
            loss = criterion(preds[train_idx], pathways[train_idx])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Store ONLY the 20% unseen test spots for our final metric evaluation
            all_preds.append(preds[test_idx].detach().cpu().numpy())
            all_targets.append(pathways[test_idx].detach().cpu().numpy())
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step() # Step the LR scheduler
        
        # Calculate Average Pearson Correlation Coefficient (PCC) across pathways
        preds_matrix = np.concatenate(all_preds) # [N, P]
        targets_matrix = np.concatenate(all_targets) # [N, P]
        
        pcc_scores = []
        for p in range(preds_matrix.shape[1]):
            pred_p = preds_matrix[:, p]
            target_p = targets_matrix[:, p]
            
            # Avoid division by zero if a pathway has zero variance
            if np.std(pred_p) > 1e-6 and np.std(target_p) > 1e-6:
                corr, _ = pearsonr(pred_p, target_p)
                if not np.isnan(corr):
                    pcc_scores.append(corr)
                    
        pcc = np.mean(pcc_scores) if len(pcc_scores) > 0 else 0.0
            
        print(f"Epoch {epoch+1}/{epochs} | Train MSE Loss: {avg_loss:.4f} | TEST PCC: {pcc:.4f}")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/cbm_gat_model.pth')
    print("Model saved to checkpoints/cbm_gat_model.pth")

if __name__ == "__main__":
    train()
