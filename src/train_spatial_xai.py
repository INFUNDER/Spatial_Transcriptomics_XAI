import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import sys
sys.path.append('HisToGene')
from vis_model import HisToGene

class ConceptBottleneck(nn.Module):
    """Maps 1024-dim UNI features to K interpretable concepts."""
    def __init__(self, in_features=1024, num_concepts=32, out_features=1024):
        super().__init__()
        # Map 1024-dim UNI features to K concepts
        self.concept_projector = nn.Linear(in_features, num_concepts)
        # Concepts are bounded [0, 1] for interpretability
        self.activation = nn.Sigmoid() 
        
        # Map K concepts back to the dimensionality expected by HisToGene
        self.output_projector = nn.Linear(num_concepts, out_features)
        
    def forward(self, x):
        concepts = self.activation(self.concept_projector(x))
        out = self.output_projector(concepts)
        return out, concepts

# 1. Custom Dataset for our HEST-UNI processed data
class HestUniDataset(Dataset):
    def __init__(self, data_dir='histogene_input'):
        self.files = glob.glob(os.path.join(data_dir, '*.pt'))
        self.data = []
        for f in self.files:
            self.data.append(torch.load(f))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        # HisToGene expects batches of (patches, coordinates, expression)
        # However, HisToGene model takes all spots for a single image at once
        # So one item in this dataset is a full WSI.
        patches = item['features'] # [N, 1024]
        coords = item['coordinates'] # [N, 2]
        expr = item['expression'] # [N, n_genes]
        return patches, coords, expr

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load dataset
    dataset = HestUniDataset()
    if len(dataset) == 0:
        print("No processed data found in 'histogene_input'. Please run feature extraction first.")
        return
        
    # Since each sample has a different number of spots (N), batch_size must be 1
    # unless we use a custom collate_fn to pad them.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get number of genes from the first sample
    _, _, sample_expr = dataset[0]
    n_genes = sample_expr.shape[1]
    print(f"Number of genes to predict: {n_genes}")
    
    # 2. Instantiate HisToGene
    # We pass dim=1024 because UNI features are 1024-dimensional.
    # HisToGene normally takes patches of shape (3, patch_size, patch_size) flattened.
    # We will modify the model initialization slightly to accept 1024-dim features directly
    # by overriding the initial patch embedding layer if needed, or we just pass the features.
    n_pos = 64
    
    model = HisToGene(
        n_genes=n_genes, 
        patch_size=224, # Not strictly used for embedding calculation since we feed 1024 directly
        n_layers=4, 
        dim=1024, 
        learning_rate=1e-5, 
        dropout=0.1, 
        n_pos=n_pos
    )
    
    # IMPORTANT FIX: HisToGene by default expects raw flattened pixels and passes them
    # through a Linear layer `self.patch_embedding = nn.Linear(patch_dim, dim)`.
    # Since our features are ALREADY 1024-dimensional (dim=1024), we replace this layer
    # with an Identity layer so it doesn't try to project from (3*224*224) -> 1024.
    model.patch_embedding = nn.Identity()
    
    model.to(device)
    
    # 3. Instantiate Concept Bottleneck Model
    num_concepts = 32 # Placeholder: 32 interpretable pathology concepts
    cbm = ConceptBottleneck(in_features=1024, num_concepts=num_concepts, out_features=1024).to(device)
    
    # Optimize both CBM and HisToGene
    optimizer = optim.Adam(list(model.parameters()) + list(cbm.parameters()), lr=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 20
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (patches, coords, expr) in enumerate(dataloader):
            # HisToGene expects [B, N, dim] (where B=1 since we have varying N)
            # So we keep the batch dimension added by DataLoader
            patches = patches.to(device)
            expr = expr.to(device)
            
            # Map raw pixel coords to a discrete grid [0, n_pos-1] for HisToGene's nn.Embedding
            coords_min = coords.min(dim=1, keepdim=True)[0]
            coords_max = coords.max(dim=1, keepdim=True)[0]
            coords_norm = (coords - coords_min) / (coords_max - coords_min + 1e-8)
            coords = (coords_norm * (n_pos - 1)).long().to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through Concept Bottleneck
            cbm_features, concept_activations = cbm(patches)
            
            # Forward pass through HisToGene GAT
            pred_expr = model(cbm_features, coords)
            
            # Compute primary prediction loss
            pred_loss = criterion(pred_expr, expr)
            
            # Add an L1 sparsity penalty on concepts to encourage disentangled, interpretable concepts
            sparsity_loss = concept_activations.mean()
            loss = pred_loss + (0.01 * sparsity_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/histogene_uni_model.pth')
    print("Model saved to checkpoints/histogene_uni_model.pth")

if __name__ == "__main__":
    train()
