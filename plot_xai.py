import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from train_cbm_gat import SpatiallyGroundedCBM

def generate_xai_plot():
    print("Generating XAI Explanation Plot...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cbm_gat = SpatiallyGroundedCBM(num_concepts=32, hidden_dim=128, num_pathways=31).to(device)
    cbm_gat.load_state_dict(torch.load('checkpoints/cbm_gat_model.pth', map_location=device))
    cbm_gat.eval()
    
    sample_name = 'TENX200'
    item = torch.load(f'cbm_input/{sample_name}_cbm_data.pt')
    features = item['features'].to(device)
    coords = item['coordinates'].to(device)
    pathways = item['pathways']
    pathway_names = item['pathway_names']
    
    concepts = [
        "dense lymphocyte infiltration", "nuclear pleomorphism", "coagulative tissue necrosis", 
        "fibrotic stroma", "vascular proliferation", "normal epithelial cells", "stromal desmoplasia", 
        "apoptotic cells", "macrophage infiltration", "mitotic figures", "erythrocytes", 
        "collagen fibers", "adipose tissue", "smooth muscle", "tumor nests", "glandular formation", 
        "mucin pools", "calcification", "hypercellularity", "spindle cells", "granulation tissue", 
        "eosinophilic cytoplasm", "basophilic nuclei", "prominent nucleoli", "hyperchromatic nuclei", 
        "cellular atypia", "squamous differentiation", "keratin pearls", "myxoid stroma", 
        "necroinflammatory debris", "lymphoid follicles", "plasma cells"
    ]
    
    radius = 100.0
    dist_matrix = torch.cdist(coords, coords, p=2)
    adj = (dist_matrix < radius).float().to(device)
    
    with torch.no_grad():
        preds = cbm_gat(features, adj).cpu().numpy()
        
    features_np = features.cpu().numpy()
    coords_np = coords.cpu().numpy()
    
    # Find HALLMARK_APOPTOSIS or a related pathway
    target_pathways = ['HALLMARK_APOPTOSIS', 'HALLMARK_INFLAMMATORY_RESPONSE', 'HALLMARK_ALLOGRAFT_REJECTION']
    p_idx = -1
    for p in target_pathways:
        if p in pathway_names:
            p_idx = pathway_names.index(p)
            break
            
    if p_idx == -1: p_idx = 0 # fallback
    
    p_name = pathway_names[p_idx].replace('HALLMARK_', '').replace('_', ' ')
    
    # Concepts to visualize
    concept_idx_1 = concepts.index("apoptotic cells")
    concept_idx_2 = concepts.index("coagulative tissue necrosis")
    concept_idx_3 = concepts.index("dense lymphocyte infiltration")
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'Explainable AI Output for Doctor (Sample: {sample_name})', fontsize=22, fontweight='bold', y=1.05)
    
    # 1. Predicted Pathway
    sc0 = axes[0].scatter(coords_np[:, 0], coords_np[:, 1], c=preds[:, p_idx], cmap='magma', s=25, alpha=0.9)
    axes[0].invert_yaxis()
    axes[0].set_title(f'Predicted Pathway\n[{p_name}]', fontsize=18, fontweight='bold', color='darkred')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # 2. Concept 1
    sc1 = axes[1].scatter(coords_np[:, 0], coords_np[:, 1], c=features_np[:, concept_idx_1], cmap='viridis', s=25, alpha=0.9)
    axes[1].invert_yaxis()
    axes[1].set_title(f'Input Visual Concept:\n"{concepts[concept_idx_1]}"', fontsize=18, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Concept 2
    sc2 = axes[2].scatter(coords_np[:, 0], coords_np[:, 1], c=features_np[:, concept_idx_2], cmap='viridis', s=25, alpha=0.9)
    axes[2].invert_yaxis()
    axes[2].set_title(f'Input Visual Concept:\n"{concepts[concept_idx_2]}"', fontsize=18, fontweight='bold')
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    plt.colorbar(sc2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Concept 3
    sc3 = axes[3].scatter(coords_np[:, 0], coords_np[:, 1], c=features_np[:, concept_idx_3], cmap='viridis', s=25, alpha=0.9)
    axes[3].invert_yaxis()
    axes[3].set_title(f'Input Visual Concept:\n"{concepts[concept_idx_3]}"', fontsize=18, fontweight='bold')
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    plt.colorbar(sc3, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    os.makedirs('figures_supplementary', exist_ok=True)
    out_path = 'figures_supplementary/xai_explanation.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    generate_xai_plot()
