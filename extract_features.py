import os
import torch
import scanpy as sc
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import timm
from torchvision import transforms
# In offline mode, the model will be loaded from the cache.
# HF_HUB_OFFLINE=1 should be set in the qsub script.
from hest import iter_hest

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. Load the UNI model
print("Loading UNI model...")
model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)
model.eval()
model.to(device)

# Standard transforms for UNI
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# 2. Setup paths
data_dir = 'hest_data'
output_dir = 'extracted_features'
os.makedirs(output_dir, exist_ok=True)

# Select the same 5 samples we downloaded
meta_df = pd.read_csv("HEST_v1_3_0_local.csv")
meta_df = meta_df[meta_df['oncotree_code'] == 'IDC']
meta_df = meta_df[meta_df['organ'] == 'Breast']
ids_to_query = meta_df['id'].values[:5]

print(f"Extracting features for samples: {ids_to_query}")

# 3. Iterate through HEST samples and extract features
for sample_id in ids_to_query:
    print(f"\nProcessing sample: {sample_id}")
    
    st_list = list(iter_hest(data_dir, id_list=[sample_id]))
    if len(st_list) == 0:
        print(f"Could not find sample {sample_id}")
        continue
    st = st_list[0]
    
    adata = st.adata
    wsi = st.wsi
    
    # We will use the ST spot coordinates on the full resolution image
    # Note: st.adata.obsm['spatial'] contains the [x, y] coordinates
    coords = adata.obsm['spatial']
    
    features = []
    patch_size = 256
    half_patch = patch_size // 2
    
    # Extract patch for each ST spot
    print(f"Extracting patches and running through UNI for {len(coords)} spots...")
    
    with torch.no_grad():
        for coord in tqdm(coords):
            x, y = int(coord[0]), int(coord[1])
            
            # Read region from WSI (using openslide/HEST wrapper)
            # wsi.read_region takes (x, y) at top-left corner
            top_left_x = max(0, x - half_patch)
            top_left_y = max(0, y - half_patch)
            
            # read_region returns a PIL image
            try:
                patch = wsi.read_region((top_left_x, top_left_y), 0, (patch_size, patch_size)).convert('RGB')
            except Exception as e:
                print(f"Warning: could not read region at {x},{y}: {e}")
                # Append zero vector if read fails
                features.append(torch.zeros(1024))
                continue
            
            # Preprocess and pass through UNI
            img_tensor = transform(patch).unsqueeze(0).to(device)
            feature = model(img_tensor).cpu().squeeze(0) # [1024]
            features.append(feature)
            
    # Stack all features for this slide
    features_tensor = torch.stack(features) # [N, 1024]
    
    # Save the feature vector tensor
    out_path = os.path.join(output_dir, f"{sample_id}_uni_features.pt")
    torch.save(features_tensor, out_path)
    print(f"Saved {features_tensor.shape} features to {out_path}")

print("\nFeature extraction completed!")
