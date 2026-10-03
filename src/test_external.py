import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from PIL import Image
import gseapy as gp
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from macenko import macenko_stain_normalization
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from train_cbm_gat import SpatiallyGroundedCBM

def process_external_dataset():
    print("Testing External Generalization on 10x Genomics Visium Cohort...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    data_dir = 'external_data/10x_Breast_Cancer'
    h5_path = f"{data_dir}/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5"
    spatial_dir = f"{data_dir}/spatial"
    
    print("Loading AnnData...")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    
    # Load coordinates
    positions = pd.read_csv(f"{spatial_dir}/tissue_positions_list.csv", header=None)
    positions.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    positions = positions[positions['in_tissue'] == 1]
    positions.set_index('barcode', inplace=True)
    
    adata = adata[adata.obs_names.intersection(positions.index)].copy()
    coords = positions.loc[adata.obs_names, ['pxl_col_in_fullres', 'pxl_row_in_fullres']].values
    adata.obsm['spatial'] = coords
    
    print(f"Loaded {len(adata)} spots.")
    
    # 2. Run ssGSEA
    print("Running ssGSEA on raw gene expression...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    expr = adata.X.toarray()
    genes = adata.var_names.tolist()
    expr_df = pd.DataFrame(expr.T, index=genes, columns=[f"spot_{i}" for i in range(expr.shape[0])])
    
    ss = gp.ssgsea(data=expr_df,
                   gene_sets='MSigDB_Hallmark_2020',
                   outdir=None,
                   sample_norm_method='rank',
                   no_plot=True)
                   
    pivot_df = ss.res2d.pivot(index='Name', columns='Term', values='ES')
    pivot_df = pivot_df.loc[[f"spot_{i}" for i in range(expr.shape[0])]]
    
    # Load original model's pathway names to align them
    original_data = torch.load('cbm_input/TENX200_cbm_data.pt', map_location='cpu')
    target_pathways = original_data['pathway_names']
    
    # Ensure all target pathways exist (fill with 0 if missing)
    for p in target_pathways:
        if p not in pivot_df.columns:
            pivot_df[p] = 0.0
            
    # Filter and reorder
    pathways_np = pivot_df[target_pathways].values
    
    # 3. Extract CONCH Concepts
    print("Loading CONCH VLM...")
    import transformers
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    local_conch_path = "/home/ronit.28010/.cache/huggingface/hub/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357"
    model = CLIPModel.from_pretrained(local_conch_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
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
    
    text_inputs = processor(text=pathology_concepts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(**text_inputs), p=2, dim=-1)
        
    print("Loading massive WSI TIF...")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    wsi = Image.open(f"{data_dir}/V1_Breast_Cancer_Block_A_Section_1_image.tif").convert("RGB")
    
    patch_size = 256
    half_patch = patch_size // 2
    concept_matrix = []
    
    print("Extracting patches and applying Macenko + CONCH...")
    with torch.no_grad():
        for coord in tqdm(coords):
            x, y = int(coord[0]), int(coord[1])
            try:
                left, top = max(0, x - half_patch), max(0, y - half_patch)
                patch = wsi.crop((left, top, left + patch_size, top + patch_size))
                norm_patch_arr = macenko_stain_normalization(patch)
                norm_patch = Image.fromarray(norm_patch_arr)
            except:
                concept_matrix.append(torch.zeros(32))
                continue
                
            image_inputs = processor(images=norm_patch, return_tensors="pt").to(device)
            image_features = F.normalize(model.get_image_features(**image_inputs), p=2, dim=-1)
            
            similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            concept_matrix.append(similarities.squeeze().cpu())
            
    features_np = torch.stack(concept_matrix).numpy()
    
    # Standardize
    features_scaled = (features_np - features_np.mean(axis=0)) / (features_np.std(axis=0) + 1e-8)
    pathways_scaled = (pathways_np - pathways_np.mean(axis=0)) / (pathways_np.std(axis=0) + 1e-8)
    
    features_t = torch.FloatTensor(features_scaled).to(device)
    coords_t = torch.FloatTensor(coords).to(device)
    pathways_t = torch.FloatTensor(pathways_scaled)
    
    # 4. Run Pre-Trained CBM-GATv2 Model
    print("Running Zero-Shot Inference using Pre-Trained CBM-GATv2...")
    cbm = SpatiallyGroundedCBM(num_concepts=32, hidden_dim=128, num_pathways=31).to(device)
    cbm.load_state_dict(torch.load('checkpoints/cbm_gat_model.pth', map_location=device))
    cbm.eval()
    
    # Note: 10x Visium spots are exactly 100 micrometers apart physically
    radius = 200.0 # Adjust radius for 10x full-res pixels
    dist_matrix = torch.cdist(coords_t, coords_t, p=2)
    adj = (dist_matrix < radius).float().to(device)
    
    with torch.no_grad():
        preds = cbm(features_t, adj).cpu().numpy()
        
    pcc_scores = []
    for p in range(pathways_scaled.shape[1]):
        if np.std(preds[:, p]) > 1e-6 and np.std(pathways_scaled[:, p]) > 1e-6:
            corr, _ = pearsonr(preds[:, p], pathways_scaled[:, p])
            if not np.isnan(corr):
                pcc_scores.append(corr)
                
    final_pcc = np.mean(pcc_scores) if len(pcc_scores) > 0 else 0.0
    print(f"\n=============================================")
    print(f"EXTERNAL GENERALIZATION TEST COMPLETE")
    print(f"Dataset: 10x Genomics Visium Breast Cancer")
    print(f"Zero-Shot Test PCC: {final_pcc:.4f}")
    print(f"=============================================\n")

if __name__ == "__main__":
    process_external_dataset()
