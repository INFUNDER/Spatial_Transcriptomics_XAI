import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from train_cbm_gat import SpatiallyGroundedCBM
from scipy.stats import pearsonr

def plot_spatial_results():
    print("Loading data for spatial visualization...")
    
    os.makedirs('figures/spatial_predictions', exist_ok=True)
    
    data_files = glob.glob('cbm_input/*_cbm_data.pt')
    if len(data_files) == 0:
        print("No data found.")
        return

    # Load model globally since it's the same for all samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None

    for sample_file in data_files:
        sample_name = os.path.basename(sample_file).split('_')[0]
        print(f"\nProcessing sample: {sample_name}")
        item = torch.load(sample_file)

        features = item['features']
        coords = item['coordinates']
        pathways = item['pathways']
        pathway_names = item.get('pathway_names', [f"Pathway {i+1}" for i in range(pathways.shape[1])])

        # Standardize pathways exactly as in training
        pathway_mean = pathways.mean(dim=0, keepdim=True)
        pathway_std = pathways.std(dim=0, keepdim=True) + 1e-8
        pathways_scaled = (pathways - pathway_mean) / pathway_std

        # Build adjacency
        radius = 100.0
        dist_matrix = torch.cdist(coords, coords, p=2)
        adj = (dist_matrix < radius).float()

        features = features.to(device)
        adj = adj.to(device)

        num_concepts = features.shape[1]
        num_pathways = pathways.shape[1]

        if model is None:
            print("Loading trained GATv2 model...")
            model = SpatiallyGroundedCBM(num_concepts=num_concepts, hidden_dim=128, num_pathways=num_pathways).to(device)
            model.load_state_dict(torch.load('checkpoints/cbm_gat_model.pth', map_location=device))
            model.eval()

        # 3. Predict
        with torch.no_grad():
            preds = model(features, adj)
            
        preds = preds.cpu().numpy()
        pathways_scaled_np = pathways_scaled.cpu().numpy()
        coords_np = coords.numpy()

        # 4. Plot top 3 pathways by PCC
        pcc_scores = []
        for p in range(num_pathways):
            # Check variance to avoid NaN
            if np.std(preds[:, p]) > 1e-6 and np.std(pathways_scaled_np[:, p]) > 1e-6:
                corr, _ = pearsonr(preds[:, p], pathways_scaled_np[:, p])
                if not np.isnan(corr):
                    pcc_scores.append((corr, p))

        # Sort by highest PCC
        pcc_scores.sort(reverse=True)
        top_pathways = pcc_scores[:3]

        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        fig.suptitle(f'Spatial Pathway Predictions ({sample_name})', fontsize=18, fontweight='bold', y=0.95)

        # Style definitions
        axes[0, 0].set_title('Observed Pathway Expression (Ground Truth)', fontsize=14, pad=20)
        axes[0, 1].set_title('Predicted Pathway Expression (Our XAI Model)', fontsize=14, pad=20)

        for i, (corr, p_idx) in enumerate(top_pathways):
            pathway_name = pathway_names[p_idx]
            # True
            sc1 = axes[i, 0].scatter(coords_np[:, 0], coords_np[:, 1], c=pathways_scaled_np[:, p_idx], cmap='viridis', s=15, alpha=0.8)
            axes[i, 0].invert_yaxis() # Image coordinates usually have Y inverted
            
            # Format the pathway name nicely (remove HALLMARK_ prefix if it exists)
            clean_name = pathway_name.replace('HALLMARK_', '').replace('_', ' ')
            axes[i, 0].set_ylabel(f'{clean_name}\nTrue Score', fontsize=12, fontweight='bold')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            plt.colorbar(sc1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Predicted
            sc2 = axes[i, 1].scatter(coords_np[:, 0], coords_np[:, 1], c=preds[:, p_idx], cmap='viridis', s=15, alpha=0.8)
            axes[i, 1].invert_yaxis()
            axes[i, 1].set_ylabel(f'{clean_name}\nPredicted Score', fontsize=12, fontweight='bold')
            axes[i, 1].set_title(f'Prediction PCC: {corr:.3f}', fontsize=12, fontweight='bold', color='darkred')
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            plt.colorbar(sc2, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = f'figures/spatial_predictions/{sample_name}_spatial_predictions.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {sample_name} spatial plot to {save_path}")

if __name__ == "__main__":
    plot_spatial_results()
