# Spatial Transcriptomics XAI Pipeline 🔬🧬

![Spatial Transcriptomics](https://img.shields.io/badge/Domain-Spatial%20Transcriptomics-blue)
![Explainable AI](https://img.shields.io/badge/Domain-Explainable%20AI%20(XAI)-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-purple)

This repository hosts an **Explainable AI Pipeline for Spatial Transcriptomics**, designed to replace opaque black-box models with transparent, biologically grounded reasoning. It utilizes the **HEST-1k dataset**, extracts high-dimensional morphological features from H&E whole-slide images using the **UNI Vision Foundation Model**, and maps these visual representations into spatial gene expression predictions via a modified **HisToGene Graph Attention Network (GAT)**.

## 🌟 Key Features

- **Automated Data Pipeline**: Scripts to securely authenticate, download, and cache HEST-1k dataset slides and transcriptomic matrices offline.
- **Vision Foundation Model Integration**: Leverages `MahmoodLab/UNI` to extract rich, 1024-dimensional feature embeddings directly from $224 \times 224$ tissue patches.
- **Modified Graph Attention Architecture**: A fully adapted HisToGene pipeline engineered to accept latent foundation features instead of raw pixels, speeding up training and improving representation quality.
- **HPC Ready**: Optimized for cluster environments using PBS/qsub, with support for fully offline GPU nodes.

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have a Conda environment with Python 3.10 and PyTorch (CUDA 12.1 recommended).
```bash
conda create -n spatial_xai python=3.10
conda activate spatial_xai
pip install -r requirements.txt
```

### 2. Authentication & Data Download

Before running the feature extraction, provide your Hugging Face Access Token to download the necessary models and data locally:
```bash
export HF_TOKEN="your_hugging_face_token"
python download_data.py
python download_model_login.py
```

### 3. Feature Extraction (HPC/GPU)

The extraction script slices the H&E images into localized patches, runs them through the UNI foundation model, and saves the resulting 1024-dimensional features.
```bash
# Submit the extraction job to your cluster
qsub extract_features.qsub
```

### 4. Training the Spatial Model

Once features are formatted and extracted into the `histogene_input/` directory, you can train the spatial expression predictor:
```bash
# Submit the HisToGene training job
qsub train_histogene.qsub
```

The fine-tuned model will be saved automatically to `checkpoints/histogene_uni_model.pth`.

## 🧠 Future Work (XAI Integration)

The upcoming phase of this project will integrate a **Concept Bottleneck Model (CBM)** between the foundation model and the GAT. This will force the network to explicitly identify and map human-interpretable pathology concepts (e.g., cell density, immune infiltration) before predicting gene expression, enabling transparent, biological reasoning for every prediction.

---
*Built as part of advanced research into highly interpretable foundation architectures.*
