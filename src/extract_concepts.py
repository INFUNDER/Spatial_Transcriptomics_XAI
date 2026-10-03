import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import scanpy as sc
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from hest import iter_hest
from macenko import macenko_stain_normalization
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load the CONCH VLM (using CLIP architecture)
print("Loading CONCH Vision-Language Model...")
local_conch_path = "/home/ronit.28010/.cache/huggingface/hub/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357"
model = CLIPModel.from_pretrained(local_conch_path, local_files_only=True)
processor = CLIPProcessor.from_pretrained(local_conch_path, local_files_only=True)
model.eval()
model.to(device)

# 2. Define the Biologically Relevant Pathology Concepts (The CBM Dictionary)
# As per the project report, these must be text-aligned interpretable concepts
pathology_concepts = [
    "dense lymphocyte infiltration",
    "nuclear pleomorphism",
    "coagulative tissue necrosis",
    "fibrotic stroma",
    "vascular proliferation",
    "normal epithelial cells",
    "stromal desmoplasia",
    "apoptotic cells",
    "macrophage infiltration",
    "mitotic figures",
    # We define K=32 concepts for the bottleneck
    "erythrocytes", "collagen fibers", "adipose tissue", "smooth muscle", "tumor nests",
    "glandular formation", "mucin pools", "calcification", "hypercellularity", "spindle cells",
    "granulation tissue", "eosinophilic cytoplasm", "basophilic nuclei", "prominent nucleoli",
    "hyperchromatic nuclei", "cellular atypia", "squamous differentiation", "keratin pearls",
    "myxoid stroma", "necroinflammatory debris", "lymphoid follicles", "plasma cells"
]

print(f"Initialized {len(pathology_concepts)} interpretable pathology concepts.")

# Pre-compute text embeddings for the concepts
with torch.no_grad():
    text_inputs = processor(text=pathology_concepts, return_tensors="pt", padding=True).to(device)
    text_features = model.get_text_features(**text_inputs)
    # Normalize text features
    text_features = F.normalize(text_features, p=2, dim=-1)

# 3. Setup paths
data_dir = 'hest_data'
output_dir = 'extracted_concepts'
os.makedirs(output_dir, exist_ok=True)

# Select all Breast Cancer (IDC) samples from the dataset
meta_df = pd.read_csv("HEST_v1_3_0_local.csv")
meta_df = meta_df[meta_df['oncotree_code'] == 'IDC']
meta_df = meta_df[meta_df['organ'] == 'Breast']
ids_to_query = meta_df['id'].values

print(f"Total samples to process: {len(ids_to_query)}")

# 4. Iterate through HEST samples and extract concepts
for sample_id in ids_to_query:
    print(f"\nProcessing sample: {sample_id}")
    st_list = list(iter_hest(data_dir, id_list=[sample_id]))
    if len(st_list) == 0: continue
    
    st = st_list[0]
    coords = st.adata.obsm['spatial']
    
    concept_matrix = []
    patch_size = 256
    half_patch = patch_size // 2
    
    print(f"Extracting patches, applying Macenko normalization, and computing VLM concepts for {len(coords)} spots...")
    
    with torch.no_grad():
        for coord in tqdm(coords):
            x, y = int(coord[0]), int(coord[1])
            top_left_x, top_left_y = max(0, x - half_patch), max(0, y - half_patch)
            
            try:
                patch = st.wsi.read_region((top_left_x, top_left_y), 0, (patch_size, patch_size)).convert('RGB')
            except:
                concept_matrix.append(torch.zeros(len(pathology_concepts)))
                continue
            
            # --- STEP 1: MACENKO STAIN NORMALIZATION ---
            # Filter out institutional batch effects using our custom algorithm
            try:
                norm_patch_arr = macenko_stain_normalization(patch)
                norm_patch = Image.fromarray(norm_patch_arr)
            except Exception as e:
                # Fallback if SVD fails on empty/weird patch
                norm_patch = patch
                
            # --- STEP 2: ZERO-SHOT VLM CONCEPT EXTRACTION ---
            image_inputs = processor(images=norm_patch, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            # Compute cosine similarity between image patch and all K text concepts
            # Equation: s_ik = (E_I(x_i) * E_T(c_k)) / (||E_I|| * ||E_T||)
            similarity = torch.matmul(image_features, text_features.T).squeeze(0)
            
            # Use Sigmoid or ReLU to act as the activation score
            activation = torch.sigmoid(similarity * 10) # scale factor for sharper sigmoid
            concept_matrix.append(activation.cpu())
            
    concept_tensor = torch.stack(concept_matrix) # [N, K]
    
    out_path = os.path.join(output_dir, f"{sample_id}_cbm_features.pt")
    torch.save(concept_tensor, out_path)
    print(f"Saved {concept_tensor.shape} concept bottleneck features to {out_path}")

print("\nConcept Bottleneck Extraction completed!")
