import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from scipy.stats import pearsonr

def generate_xai_proofs():
    print("Generating XAI Interpretability Proofs...")
    
    # 1. Load the biological concepts we defined
    pathology_concepts = [
        "dense lymphocyte infiltration", "nuclear pleomorphism", "coagulative tissue necrosis", 
        "fibrotic stroma", "vascular proliferation", "normal epithelial cells",
        "stromal desmoplasia", "apoptotic cells", "macrophage infiltration", "mitotic figures",
        "erythrocytes", "collagen fibers", "adipose tissue", "smooth muscle", "tumor nests",
        "glandular formation", "mucin pools", "calcification", "hypercellularity", "spindle cells",
        "granulation tissue", "eosinophilic cytoplasm", "basophilic nuclei", "prominent nucleoli",
        "hyperchromatic nuclei", "cellular atypia", "squamous differentiation", "keratin pearls",
        "myxoid stroma", "necroinflammatory debris", "lymphoid follicles", "plasma cells"
    ]
    
    # 2. Aggregate data across all samples to find global correlations
    data_files = glob.glob('cbm_input/*_cbm_data.pt')
    all_features = []
    all_pathways = []
    pathway_names = None
    
    for file in data_files:
        item = torch.load(file)
        all_features.append(item['features'].numpy())
        all_pathways.append(item['pathways'].numpy())
        if pathway_names is None:
            pathway_names = item['pathway_names']
            
    features = np.concatenate(all_features, axis=0)
    pathways = np.concatenate(all_pathways, axis=0)
    
    # Standardize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    pathways = (pathways - pathways.mean(axis=0)) / (pathways.std(axis=0) + 1e-8)
    
    # 3. Calculate Concept-to-Pathway Correlation Matrix
    print("Calculating Concept-Pathway Correlation Matrix...")
    num_concepts = features.shape[1]
    num_pathways = pathways.shape[1]
    
    corr_matrix = np.zeros((num_concepts, num_pathways))
    for c in range(num_concepts):
        for p in range(num_pathways):
            corr, _ = pearsonr(features[:, c], pathways[:, p])
            corr_matrix[c, p] = corr if not np.isnan(corr) else 0.0
            
    # Save the raw matrix
    os.makedirs('figures/xai', exist_ok=True)
    
    # 4. Generate Interpretability Bar Charts for top 3 most predictable pathways
    # Find pathways with the strongest concept signals (highest max absolute correlation)
    max_corrs = np.max(np.abs(corr_matrix), axis=0)
    top_pathway_indices = np.argsort(max_corrs)[::-1][:3]
    
    for p_idx in top_pathway_indices:
        p_name = pathway_names[p_idx].replace('HALLMARK_', '').replace('_', ' ')
        p_corrs = corr_matrix[:, p_idx]
        
        # Get top 5 positive and top 5 negative driving concepts
        sorted_indices = np.argsort(p_corrs)
        bottom_5_idx = sorted_indices[:5]
        top_5_idx = sorted_indices[-5:][::-1]
        
        selected_indices = np.concatenate([top_5_idx, bottom_5_idx])
        selected_corrs = p_corrs[selected_indices]
        selected_concepts = [pathology_concepts[i].title() for i in selected_indices]
        
        # Plot
        plt.figure(figsize=(12, 6))
        colors = ['#d62728' if x > 0 else '#1f77b4' for x in selected_corrs]
        
        y_pos = np.arange(len(selected_concepts))
        plt.barh(y_pos, selected_corrs, color=colors)
        plt.yticks(y_pos, selected_concepts, fontsize=12)
        plt.axvline(0, color='black', linewidth=1)
        
        plt.xlabel('Pearson Correlation (Concept Driver Strength)', fontsize=12, fontweight='bold')
        plt.title(f'XAI Explanation for Pathway: {p_name}', fontsize=16, fontweight='bold', pad=20)
        
        # Add labels
        for i, v in enumerate(selected_corrs):
            if v > 0:
                plt.text(v + 0.01, i, f'+{v:.2f}', va='center', fontweight='bold', color='#d62728')
            else:
                plt.text(v - 0.05, i, f'{v:.2f}', va='center', fontweight='bold', color='#1f77b4')
                
        # Invert y axis to have highest at top
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        safe_name = p_name.replace(' ', '_')
        save_path = f'figures/xai/XAI_{safe_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved XAI Proof for {p_name} to {save_path}")

    # 5. Global Heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                xticklabels=[p.replace('HALLMARK_', '') for p in pathway_names],
                yticklabels=pathology_concepts)
    plt.title('Global Concept-Pathway Bottleneck Correlation', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/xai/Global_CBM_Heatmap.png', dpi=300)
    plt.close()
    print("Saved Global CBM Heatmap to figures/xai/Global_CBM_Heatmap.png")

if __name__ == "__main__":
    generate_xai_proofs()
