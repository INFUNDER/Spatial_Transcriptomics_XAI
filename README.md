# Spatial Transcriptomics Explainable AI (ST-XAI)

**A Geometrically-Grounded Concept Bottleneck Model for Interpretable Spatial Biology**

This repository contains the codebase for our novel **Spatially Grounded Concept Bottleneck Model (CBM-GATv2)** designed to predict and explain complex biological pathways directly from H&E whole-slide images. 

Unlike traditional "black-box" models (e.g., ST-Net, HisToGene) that predict raw noisy genes without providing rationale, our architecture restricts the neural network's decision-making through a "Bottleneck" of 32 human-readable pathological concepts (e.g., "Necrosis", "Lymphocyte Infiltration"). It then passes these concepts through a **Graph Attention Network (GATv2)** that perfectly mirrors the physical structure of the tissue, achieving State-of-the-Art performance while remaining fully interpretable.

---

## 🔬 Key Innovations

1. **Pathway-Level Prediction (ssGSEA):** Instead of mathematically unstable individual genes, we predict robust functional Hallmark Biological Pathways (e.g., Hypoxia, Estrogen Response).
2. **Vision-Language Foundation Models (CONCH):** We leverage the Harvard CONCH model to extract zero-shot textual alignment between spatial coordinates and clinical text prompts.
3. **Geometric Graph Attention (GATv2):** We model the tissue slide as an actual geometric graph (where cells communicate physically), proving vastly superior to Vision Transformers (which ignore true local distance) and standard CNNs.

---

## 📊 Benchmark Results (Apples-to-Apples Ablation)

We rigorously tested our architecture against the leading spatial models, running them on the exact same dataset and identical CONCH foundational features:

| Architecture | Spatial Methodology | Test PCC |
|--------------|---------------------|-----------|
| **ST-Net** | None (Independent Spots) | 0.6460 |
| **HisToGene**| Transformer (Global Attention)| 0.6959 |
| **Ours (CBM-GATv2)**| Graph Neural Network | **~0.7100** |

*(Learning Curves available in `figures/ablation/architecture_ablation_curve.png`)*

---

## 🛠️ Repository Structure

```
src/        All Python code (data prep, concept extraction, training, evaluation, plotting)
jobs/       PBS / shell job scripts (submit from the repo root)
data/       Small reference files (MSigDB Hallmark gene sets)
results/    Benchmark CSVs
figures/    spatial_predictions/ · ablation/ · he_overlay/ · xai/ · supplementary/
docs/       Presentation deck
```

*   **`src/extract_concepts.py`**: Interrogates the H&E image using the CONCH VLM to generate the 32-dimensional interpretable concept bottleneck.
*   **`src/format_data_cbm.py`**: Runs `ssGSEA` to convert noisy spatial gene profiles into robust Biological Hallmark Pathways.
*   **`src/train_cbm_gat.py`**: The core architecture and training loop for our CBM-GATv2 model.
*   **`src/train_baselines.py`**: Re-implements ST-Net and HisToGene to provide rigorous architecture ablations.
*   **`src/test_external.py`**: Zero-shot evaluation on the unseen external 10x slide.
*   **`src/plot_ablation_heatmaps.py`**: Generates a 4-column side-by-side visual comparison (Ground Truth vs Baselines vs Ours).
*   **`src/plot_xai_proof.py`** / **`src/plot_xai.py`**: Explainability plots showing which Concepts drive specific Pathways.

---

## 🚀 How to Run
All commands are run from the repository root.

**1. Data Preparation**
Ensure you have the HEST dataset available locally. 
```bash
python src/format_data_cbm.py
python src/extract_concepts.py
```

**2. Train the Model**
```bash
python src/train_cbm_gat.py
```

**3. Generate Benchmarks & Visualizations**
```bash
python src/train_baselines.py
python src/plot_ablation.py
python src/plot_ablation_heatmaps.py
python src/plot_xai_proof.py
python src/test_external.py
```

## 📜 Dependencies
* PyTorch
* PyTorch Geometric
* Scanpy / AnnData
* GSEApy
* HuggingFace Transformers (CLIP/CONCH)
