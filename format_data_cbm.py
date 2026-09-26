import torch
import numpy as np
import scanpy as sc
import pandas as pd
import os
import glob
from hest import iter_hest
import gseapy as gp

def prepare_cbm_data(hest_dir='hest_data', features_dir='extracted_concepts', out_dir='cbm_input'):
    os.makedirs(out_dir, exist_ok=True)
    
    feature_files = glob.glob(os.path.join(features_dir, '*_cbm_features.pt'))
    if len(feature_files) == 0:
        print(f"No concept features found in {features_dir}. Please run extract_concepts.py first.")
        return
        
    for f_path in feature_files:
        sample_id = os.path.basename(f_path).replace('_cbm_features.pt', '')
        print(f"\nFormatting sample: {sample_id}")
        
        # 1. Load VLM Concept Features
        concepts = torch.load(f_path) # [N, K] where K=32
        print(f"Loaded {concepts.shape[1]} concepts for {concepts.shape[0]} spots.")
        
        # 2. Load Spatial Transcriptomics Data
        st_data = list(iter_hest(hest_dir, id_list=[sample_id]))[0]
        adata = st_data.adata
        
        # Extract raw expression matrix
        expr = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X # [N, n_genes]
        genes = adata.var_names.tolist()
        
        # 3. Perform ssGSEA to convert sparse genes into pathway activation scores
        print("Running single-sample Gene Set Enrichment Analysis (ssGSEA)...")
        # gseapy expects genes as index (rows) and samples/spots as columns
        expr_df = pd.DataFrame(expr.T, index=genes, columns=[f"spot_{i}" for i in range(expr.shape[0])])
        
        try:
            # We use a standard pathway set (e.g., MSigDB Hallmark gene sets)
            # Offline note: gseapy downloads these automatically, so if we are strictly offline,
            # we should provide a local .gmt file. For now, we use a string identifier.
            ss = gp.ssgsea(data=expr_df, gene_sets='MSigDB_Hallmark_2020', outdir=None, min_size=5, max_size=2000, n_jobs=4)
            # Pivot the molten dataframe into a dense matrix [N_samples, P_pathways]
            pivot_df = ss.res2d.pivot(index='Name', columns='Term', values='ES')
            
            # Ensure the row order perfectly matches the original spot coordinates
            spot_names = [f"spot_{i}" for i in range(expr.shape[0])]
            pivot_df = pivot_df.reindex(spot_names).fillna(0)
            
            pathways = pivot_df.columns.tolist()
            pathway_scores = pivot_df.values # [N, n_pathways]
            print(f"Successfully converted {len(genes)} genes into {len(pathways)} pathway scores.")
        except Exception as e:
            print(f"ssGSEA failed (possibly due to network/offline mode): {e}")
            print("Falling back to raw highly variable genes for demonstration...")
            sc.pp.highly_variable_genes(adata, n_top_genes=50, flavor='seurat_v3')
            hvg_mask = adata.var['highly_variable'].values
            pathway_scores = expr[:, hvg_mask]
            pathways = adata.var_names[hvg_mask].tolist()
            print(f"Using {len(pathways)} highly variable genes instead.")
            
        # 4. Extract Spatial Coordinates
        coords = adata.obsm['spatial'] # [N, 2]
        
        # Save as dictionary
        sample_data = {
            'features': concepts, # [N, K]
            'pathways': torch.tensor(pathway_scores, dtype=torch.float32), # [N, P]
            'coordinates': torch.tensor(coords, dtype=torch.float32), # [N, 2]
            'pathway_names': pathways
        }
        
        torch.save(sample_data, os.path.join(out_dir, f"{sample_id}_cbm_data.pt"))
        print(f"Saved {sample_id} - Concepts: {concepts.shape}, Pathways: {pathway_scores.shape}, Coords: {coords.shape}")

if __name__ == "__main__":
    prepare_cbm_data()
