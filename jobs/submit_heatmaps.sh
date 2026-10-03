#!/bin/bash
#PBS -N heatmap_baselines
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=02:00:00
#PBS -q gpu
#PBS -e /home/ronit.28010/Spatial_Transcriptomics/heatmap.err
#PBS -o /home/ronit.28010/Spatial_Transcriptomics/heatmap.out

cd /home/ronit.28010/Spatial_Transcriptomics
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_xai

echo "Starting GPU-accelerated Baseline Training..."
python src/train_baselines.py

echo "Starting GPU-accelerated Heatmap Plotting..."
python src/plot_ablation_heatmaps.py

echo "Job Completed Successfully!"
