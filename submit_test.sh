#!/bin/bash
#PBS -N external_validation
#PBS -l nodes=1:ppn=8
#PBS -l mem=32gb
#PBS -l walltime=02:00:00
#PBS -q gpu
#PBS -e /home/ronit.28010/Spatial_Transcriptomics/external_test.err
#PBS -o /home/ronit.28010/Spatial_Transcriptomics/external_test.out

cd /home/ronit.28010/Spatial_Transcriptomics
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spatial_xai

echo "Starting External Validation on 10x Genomics Visium Cohort..."
python test_external.py
echo "Job Completed Successfully!"
