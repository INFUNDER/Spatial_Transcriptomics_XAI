import torch
import numpy as np
import scanpy as sc
import os
import glob
from hest import iter_hest

# This script formats the extracted UNI features and HEST gene expression
# data into the format that HisToGene expects to be trained on.
def prepare_histogene_data(hest_dir='hest_data', features_dir='extracted_features', out_dir='histogene_input'):
    os.makedirs(out_dir, exist_ok=True)
    
    # Get all extracted feature files
    feature_files = glob.glob(os.path.join(features_dir, '*_uni_features.pt'))
    
    for f_path in feature_files:
        sample_id = os.path.basename(f_path).replace('_uni_features.pt', '')
        print(f"Formatting sample: {sample_id}")
        
        # Load UNI features
        features = torch.load(f_path) # [N, 1024]
        
        # Load HEST data for this sample to get gene expression & coordinates
        st_data = list(iter_hest(hest_dir, id_list=[sample_id]))[0]
        adata = st_data.adata
        
        # We need expression matrix (raw counts usually, HisToGene normalizes it in dataset.py)
        # But we can save it as is.
        expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X # [N, n_genes]
        
        # We need spatial coordinates
        coords = adata.obsm['spatial'] # [N, 2]
        
        # Select highly variable genes or top 1000 genes as HisToGene does
        if expr.shape[1] > 1000:
            sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor='seurat_v3')
            hvg_mask = adata.var['highly_variable'].values
            expr = expr[:, hvg_mask]
            genes = adata.var_names[hvg_mask].tolist()
        else:
            genes = adata.var_names.tolist()
            
        # Save as dictionary
        sample_data = {
            'features': features, # [N, 1024]
            'expression': torch.tensor(expr, dtype=torch.float32), # [N, n_genes]
            'coordinates': torch.tensor(coords, dtype=torch.float32), # [N, 2]
            'genes': genes
        }
        
        torch.save(sample_data, os.path.join(out_dir, f"{sample_id}_histogene.pt"))
        print(f"Saved {sample_id} - Features: {features.shape}, Expression: {expr.shape}, Coords: {coords.shape}")

if __name__ == "__main__":
    prepare_histogene_data()
