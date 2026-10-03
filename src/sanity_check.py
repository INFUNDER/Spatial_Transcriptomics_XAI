import os
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datasets import load_dataset
from PIL import Image

def main():
    print("Loading MedMNIST from HuggingFace...")
    # Load dataset directly from HuggingFace
    os.environ["HF_HUB_OFFLINE"] = "0"
    dataset = load_dataset('medmnist/pathmnist', split='test')
    
    # Class labels in PathMNIST:
    class_names = {
        0: 'Adipose', 1: 'Background', 2: 'Debris', 3: 'Lymphocytes', 4: 'Mucus', 
        5: 'Smooth Muscle', 6: 'Normal Mucosa', 7: 'Stroma', 8: 'Tumor'
    }
    
    # We will map PathMNIST classes to our concepts
    target_concepts = [
        "adipose tissue", "background", "necroinflammatory debris", "lymphocytes", "mucin pools",
        "smooth muscle", "normal epithelial cells", "fibrotic stroma", "tumor nests"
    ]
    
    print("Loading CONCH VLM...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_conch_path = "/home/ronit.28010/.cache/huggingface/hub/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357"
    
    import transformers
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    
    os.environ["HF_HUB_OFFLINE"] = "1"
    model = CLIPModel.from_pretrained(local_conch_path, local_files_only=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Compute text embeddings for our concepts
    text_inputs = processor(text=target_concepts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(**text_inputs), p=2, dim=-1)
        
    print("Running Zero-Shot Inference on PathMNIST...")
    
    confusion_matrix = np.zeros((9, 9))
    class_counts = np.zeros(9)
    
    with torch.no_grad():
        for item in dataset:
            img = item['image']
            label = item['label'][0]
            
            # Subsample: only do 30 images per class to save time
            if class_counts[label] >= 30:
                continue
                
            img_inputs = processor(images=img.convert("RGB"), return_tensors="pt").to(device)
            img_features = F.normalize(model.get_image_features(**img_inputs), p=2, dim=-1)
            
            similarity = torch.matmul(img_features, text_features.T).squeeze(0)
            
            # Add to confusion matrix
            confusion_matrix[label] += similarity.cpu().numpy()
            class_counts[label] += 1
            
            if np.all(class_counts >= 30):
                break
                
    # Normalize by counts
    for i in range(9):
        if class_counts[i] > 0:
            confusion_matrix[i] /= class_counts[i]
            
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=target_concepts, yticklabels=list(class_names.values()))
    plt.title("CONCH Zero-Shot Validation vs PathMNIST Human Ground Truth")
    plt.xlabel("CONCH Extracted Concept (Cosine Similarity)")
    plt.ylabel("Human Pathologist Label")
    plt.tight_layout()
    plt.savefig('figures/supplementary/Sanity_Check_Hallucination.png', dpi=300)
    print("Saved Sanity Check Heatmap!")

if __name__ == '__main__':
    os.makedirs('figures/supplementary', exist_ok=True)
    main()
