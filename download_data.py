import os
import zipfile
import pandas as pd
from huggingface_hub import snapshot_download, login
from tqdm import tqdm

def download_hest(patterns, local_dir):
    repo_id = 'MahmoodLab/hest'
    snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

    seg_dir = os.path.join(local_dir, 'cellvit_seg')
    if os.path.exists(seg_dir):
        print('Unzipping cell vit segmentation...')
        for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
            path_zip = os.path.join(seg_dir, filename)
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extractall(seg_dir)

if __name__ == "__main__":
    # Log in to huggingface (token comes from user)
    token = os.environ.get("HF_TOKEN")
    login(token=token)

    local_dir = 'hest_data'
    os.makedirs(local_dir, exist_ok=True)

    # Download the metadata CSV first to find IDs
    meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv")
    meta_df = meta_df[meta_df['oncotree_code'] == 'IDC']
    meta_df = meta_df[meta_df['organ'] == 'Breast']
    
    # Select 5 samples for the toy dataset
    ids_to_query = meta_df['id'].values[:5]
    print(f"Downloading samples: {ids_to_query}")

    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    download_hest(list_patterns, local_dir)
    print("Download completed!")
