import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from train_cbm_gat import SpatiallyGroundedCBM
from scipy.stats import pearsonr
from hest import iter_hest

def plot_spatial_results_with_he():
    print("Loading data for H&E spatial visualization...")
    
    os.makedirs('figures/he_overlay', exist_ok=True)
    
    data_files = glob.glob('cbm_input/*_cbm_data.pt')
    if len(data_files) == 0:
        print("No data found.")
        return

    # Load model globally
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None

    for sample_file in data_files:
        sample_name = os.path.basename(sample_file).split('_')[0]
        print(f"\nProcessing sample: {sample_name}")
        item = torch.load(sample_file)

        features = item['features']
        coords = item['coordinates']
        pathways = item['pathways']

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

        print(f"Loading AnnData for {sample_name} to extract H&E image...")
        st_data = list(iter_hest('hest_data', id_list=[sample_name]))[0]
        
        # Get thumbnail directly from Whole Slide Image API
        width, height = st_data.wsi.get_dimensions()
        target_width = 1000
        target_height = int(target_width * height / width)
        img = np.array(st_data.wsi.get_thumbnail((target_width, target_height)))
        
        # Scale physical coords to thumbnail pixel coords
        scale_factor = target_width / width
        pixel_coords = coords_np * scale_factor

        # 5. Plot top 3 pathways by PCC
        pcc_scores = []
        for p in range(num_pathways):
            if np.std(preds[:, p]) > 1e-6 and np.std(pathways_scaled_np[:, p]) > 1e-6:
                corr, _ = pearsonr(preds[:, p], pathways_scaled_np[:, p])
                if not np.isnan(corr):
                    pcc_scores.append((corr, p))

        pcc_scores.sort(reverse=True)
        top_pathways = pcc_scores[:3]

        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        fig.suptitle(f'Spatial Pathway Predictions with H&E ({sample_name})', fontsize=18, fontweight='bold', y=0.95)

        axes[0, 0].set_title('Observed Pathway Expression', fontsize=14, pad=20)
        axes[0, 1].set_title('Our XAI Model Prediction', fontsize=14, pad=20)

        for i, (corr, p_idx) in enumerate(top_pathways):
            # True
            axes[i, 0].imshow(img)
            sc1 = axes[i, 0].scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=pathways_scaled_np[:, p_idx], cmap='magma', s=10, alpha=0.7)
            axes[i, 0].set_ylabel(f'Pathway {p_idx+1}\nPCC: {corr:.3f}', fontsize=12, fontweight='bold')
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            plt.colorbar(sc1, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Predicted
            axes[i, 1].imshow(img)
            sc2 = axes[i, 1].scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=preds[:, p_idx], cmap='magma', s=10, alpha=0.7)
            axes[i, 1].set_title(f'PCC: {corr:.3f}', fontsize=12)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            plt.colorbar(sc2, ax=axes[i, 1], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = f'figures/he_overlay/{sample_name}_he_overlay.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {sample_name} H&E plot to {save_path}")

if __name__ == "__main__":
    plot_spatial_results_with_he()
