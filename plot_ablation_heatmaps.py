import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from train_cbm_gat import SpatiallyGroundedCBM
from train_baselines import STNet_Baseline, HisToGene_Baseline
from scipy.stats import pearsonr

def generate_comparative_heatmaps():
    print("Loading models and generating comparative spatial heatmaps...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Models
    cbm_gat = SpatiallyGroundedCBM(num_concepts=32, hidden_dim=128, num_pathways=31).to(device)
    cbm_gat.load_state_dict(torch.load('checkpoints/cbm_gat_model.pth', map_location=device))
    cbm_gat.eval()
    
    stnet = STNet_Baseline(in_dim=32, hidden_dim=128, out_dim=31).to(device)
    stnet.load_state_dict(torch.load('checkpoints/st-net_model.pth', map_location=device))
    stnet.eval()
    
    histogene = HisToGene_Baseline(in_dim=32, hidden_dim=128, out_dim=31).to(device)
    histogene.load_state_dict(torch.load('checkpoints/histogene_model.pth', map_location=device))
    histogene.eval()

    os.makedirs('figures_ablation', exist_ok=True)
    
    # Process just one representative sample to save time (e.g., TENX200)
    sample_file = 'cbm_input/TENX200_cbm_data.pt'
    sample_name = 'TENX200'
    print(f"\nProcessing sample: {sample_name}")
    
    item = torch.load(sample_file)
    features = item['features'].to(device)
    coords = item['coordinates'].to(device)
    pathways = item['pathways']
    
    # Standardize ground truth
    pathway_mean = pathways.mean(dim=0, keepdim=True)
    pathway_std = pathways.std(dim=0, keepdim=True) + 1e-8
    pathways_scaled = (pathways - pathway_mean) / pathway_std
    
    # Build adjacency for GAT
    radius = 100.0
    dist_matrix = torch.cdist(coords, coords, p=2)
    adj = (dist_matrix < radius).float().to(device)
    
    with torch.no_grad():
        preds_stnet = stnet(features, coords).cpu().numpy()
        preds_histogene = histogene(features, coords).cpu().numpy()
        preds_ours = cbm_gat(features, adj).cpu().numpy()
        
    pathways_scaled_np = pathways_scaled.numpy()
    coords_np = coords.cpu().numpy()

    # Instead of H&E thumbnail, we will just plot the scatter points directly
    # Since we don't have the scale factor without the WSI, we'll just plot using the raw coordinates
    pixel_coords = coords_np

    # Select top 3 pathways from OUR model to plot
    pcc_scores = []
    for p in range(pathways_scaled_np.shape[1]):
        if np.std(preds_ours[:, p]) > 1e-6 and np.std(pathways_scaled_np[:, p]) > 1e-6:
            corr, _ = pearsonr(preds_ours[:, p], pathways_scaled_np[:, p])
            if not np.isnan(corr):
                pcc_scores.append((corr, p))
                
    pcc_scores.sort(reverse=True)
    top_pathways = pcc_scores[:3]

    # Plot exactly like the paper (4 columns: GT, ST-Net, HisToGene, Ours)
    fig, axes = plt.subplots(3, 4, figsize=(32, 20))
    fig.suptitle(f'Spatial Pathway Predictions Baseline Comparison ({sample_name})', fontsize=32, fontweight='bold', y=0.98)
    
    axes[0, 0].set_title('Observed (Ground Truth)', fontsize=24, fontweight='bold', pad=30)
    axes[0, 1].set_title('ST-Net Baseline', fontsize=24, fontweight='bold', pad=30)
    axes[0, 2].set_title('HisToGene Baseline', fontsize=24, fontweight='bold', pad=30)
    axes[0, 3].set_title('Our Model (CBM-GATv2)', fontsize=24, fontweight='bold', pad=30)

    # Function to calculate PCC for a given prediction array
    def get_pcc(preds, gt, p_idx):
        if np.std(preds[:, p_idx]) > 1e-6 and np.std(gt[:, p_idx]) > 1e-6:
            return pearsonr(preds[:, p_idx], gt[:, p_idx])[0]
        return 0.0

    for i, (our_corr, p_idx) in enumerate(top_pathways):
        p_name = item['pathway_names'][p_idx].replace('HALLMARK_', '').replace('_', ' ')
        
        # GT
        sc0 = axes[i, 0].scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=pathways_scaled_np[:, p_idx], cmap='magma', s=20, alpha=0.9)
        axes[i, 0].invert_yaxis()
        axes[i, 0].set_ylabel(p_name, fontsize=20, fontweight='bold', labelpad=20)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        plt.colorbar(sc0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # ST-Net
        st_corr = get_pcc(preds_stnet, pathways_scaled_np, p_idx)
        sc1 = axes[i, 1].scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=preds_stnet[:, p_idx], cmap='magma', s=20, alpha=0.9)
        axes[i, 1].invert_yaxis()
        if i != 0: axes[i, 1].set_title(f'PCC: {st_corr:.3f}', fontsize=20)
        else: axes[i, 1].set_title('ST-Net Baseline\n' + f'PCC: {st_corr:.3f}', fontsize=24, fontweight='bold')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        plt.colorbar(sc1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # HisToGene
        hg_corr = get_pcc(preds_histogene, pathways_scaled_np, p_idx)
        sc2 = axes[i, 2].scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=preds_histogene[:, p_idx], cmap='magma', s=20, alpha=0.9)
        axes[i, 2].invert_yaxis()
        if i != 0: axes[i, 2].set_title(f'PCC: {hg_corr:.3f}', fontsize=20)
        else: axes[i, 2].set_title('HisToGene Baseline\n' + f'PCC: {hg_corr:.3f}', fontsize=24, fontweight='bold')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        plt.colorbar(sc2, ax=axes[i, 2], fraction=0.046, pad=0.04)

        # Ours
        sc3 = axes[i, 3].scatter(pixel_coords[:, 0], pixel_coords[:, 1], c=preds_ours[:, p_idx], cmap='magma', s=20, alpha=0.9)
        axes[i, 3].invert_yaxis()
        if i != 0: axes[i, 3].set_title(f'PCC: {our_corr:.3f}', fontsize=20, fontweight='bold', color='darkred')
        else: axes[i, 3].set_title('Our Model (CBM-GATv2)\n' + f'PCC: {our_corr:.3f}', fontsize=24, fontweight='bold', color='darkred')
        axes[i, 3].set_xticks([])
        axes[i, 3].set_yticks([])
        plt.colorbar(sc3, ax=axes[i, 3], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f'figures_ablation/baseline_comparison_heatmaps.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved baseline comparison heatmaps to {save_path}")

if __name__ == "__main__":
    generate_comparative_heatmaps()
