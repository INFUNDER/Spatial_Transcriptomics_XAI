import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

# --- 1. Dataset Loading (Exact same as your GAT model) ---
class CBMDataset(Dataset):
    def __init__(self, data_dir='cbm_input'):
        self.files = glob.glob(os.path.join(data_dir, '*_cbm_data.pt'))
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        item = torch.load(self.files[idx])
        return item['features'], item['coordinates'], item['pathways']

def collate_fn(batch):
    return batch[0][0], batch[0][1], batch[0][2]

# --- 2. ST-Net Baseline (Independent Spot Prediction) ---
# ST-Net treats every spot as an independent image without spatial communication
class STNet_Baseline(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=128, out_dim=31):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x, coords=None):
        return self.mlp(x)

# --- 3. HisToGene Baseline (Vision Transformer) ---
# HisToGene uses Self-Attention across all spots (treating the tissue like a sentence)
class HisToGene_Baseline(nn.Module):
    def __init__(self, in_dim=32, hidden_dim=128, out_dim=31, n_heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        # Spatial Position Embedding (X, Y coordinates)
        self.pos_embedding = nn.Linear(2, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        self.predictor = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x, coords):
        # x: [N, 32], coords: [N, 2]
        features = self.embedding(x) + self.pos_embedding(coords)
        # Transformer expects [Batch, Seq_Len, Dim], so we add a batch dimension of 1
        features = features.unsqueeze(0) 
        attended_features = self.transformer(features)
        attended_features = attended_features.squeeze(0)
        return self.predictor(attended_features)

# --- 4. Rigorous Training Loop ---
def train_baseline(model_name, model_class, epochs=300):
    print(f"\n{'='*40}\nTraining {model_name} Baseline\n{'='*40}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = CBMDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # Initialize the specific baseline
    model = model_class(in_dim=32, hidden_dim=128, out_dim=31).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    final_pcc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for concepts, coords, pathways in dataloader:
            concepts = concepts.to(device)
            coords = coords.to(device)
            pathways = pathways.to(device)
            
            # Z-score normalization
            pathway_mean = pathways.mean(dim=0, keepdim=True)
            pathway_std = pathways.std(dim=0, keepdim=True) + 1e-8
            pathways = (pathways - pathway_mean) / pathway_std
            
            # 80/20 Masking (Apples-to-Apples with your GATv2)
            N = concepts.shape[0]
            indices = torch.randperm(N, device=device)
            split_idx = int(0.8 * N)
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            
            optimizer.zero_grad()
            
            # Forward pass
            preds = model(concepts, coords)
            
            loss = criterion(preds[train_idx], pathways[train_idx])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            all_preds.append(preds[test_idx].detach().cpu().numpy())
            all_targets.append(pathways[test_idx].detach().cpu().numpy())
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        # Calculate Test PCC
        all_preds_np = np.concatenate(all_preds, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)
        
        pcc_scores = []
        for p in range(all_targets_np.shape[1]):
            if np.std(all_preds_np[:, p]) > 1e-6 and np.std(all_targets_np[:, p]) > 1e-6:
                corr, _ = pearsonr(all_preds_np[:, p], all_targets_np[:, p])
                if not np.isnan(corr):
                    pcc_scores.append(corr)
                    
        pcc = np.mean(pcc_scores) if len(pcc_scores) > 0 else 0.0
        final_pcc = pcc
        
        if (epoch+1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train MSE Loss: {avg_loss:.4f} | TEST PCC: {pcc:.4f}")
            
    print(f"-> {model_name} Final Test PCC: {final_pcc:.4f}")
    
    # Save the model weights for visualization
    os.makedirs('checkpoints', exist_ok=True)
    safe_name = model_name.split(' ')[0].lower()
    torch.save(model.state_dict(), f'checkpoints/{safe_name}_model.pth')
    
    return final_pcc

if __name__ == "__main__":
    stnet_pcc = train_baseline("ST-Net (No Spatial Context)", STNet_Baseline, epochs=300)
    histogene_pcc = train_baseline("HisToGene (Transformer)", HisToGene_Baseline, epochs=300)
    
    print(f"\n{'='*40}\nFINAL BENCHMARK COMPARISON\n{'='*40}")
    print(f"ST-Net Architecture:   {stnet_pcc:.4f} Test PCC")
    print(f"HisToGene Architecture:{histogene_pcc:.4f} Test PCC")
    print(f"Your CBM-GAT Model:    ~0.71 Test PCC (Reference)")
