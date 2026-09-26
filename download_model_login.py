import timm
import pandas as pd
from huggingface_hub import login

import os
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)

print("Downloading/Caching UNI model...")
model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True)

print("Downloading/Caching HEST CSV...")
meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv", storage_options={'token': token})
meta_df.to_csv("HEST_v1_3_0_local.csv", index=False)

print("Pre-download completed!")

from huggingface_hub import snapshot_download

print("\nDownloading/Caching CONCH VLM model snapshot...")
print("NOTE: Make sure you have requested access at https://huggingface.co/MahmoodLab/CONCH")
try:
    path = snapshot_download(repo_id="MahmoodLab/conch")
    print(f"CONCH successfully cached! Full snapshot path:\\n{path}")
except Exception as e:
    print(f"Failed to download CONCH. Did you get access approval on HuggingFace? Error: {e}")
