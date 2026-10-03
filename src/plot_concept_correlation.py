import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def plot_concept_correlation():
    print("Generating Concept Independence Correlation Matrix...")
    
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
    
    # Capitalize for aesthetics
    display_concepts = [c.title() for c in pathology_concepts]
    
    data_files = glob.glob('cbm_input/*_cbm_data.pt')
    all_features = []
    
    for file in data_files:
        item = torch.load(file)
        all_features.append(item['features'].numpy())
        
    features = np.concatenate(all_features, axis=0)
    
    # Standardize
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Calculate Correlation Matrix of Concepts vs Concepts
    # features is (N, 32)
    # np.corrcoef expects variables as rows, so we transpose
    corr_matrix = np.corrcoef(features, rowvar=False)
    
    # Plotting
    plt.figure(figsize=(24, 20))
    # Use a diverging colormap
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                xticklabels=display_concepts,
                yticklabels=display_concepts,
                vmin=-1, vmax=1,
                linewidths=.5, cbar_kws={"shrink": .7})
                
    plt.title('Concept Independence Matrix (Addressing Concept Entanglement)', fontsize=24, fontweight='bold', pad=20)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    
    os.makedirs('figures/supplementary', exist_ok=True)
    save_path = 'figures/supplementary/Concept_Correlation_Matrix.png'
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Concept Correlation Matrix to {save_path}")

if __name__ == "__main__":
    plot_concept_correlation()
