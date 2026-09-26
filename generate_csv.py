import re
import csv

output_csv = 'benchmark_ablation_results.csv'

with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Model', 'Epoch', 'Train_MSE_Loss', 'Test_PCC'])
    
    # 1. Parse CBM-GATv2
    with open('cbm_pipeline.o39101', 'r') as f:
        for line in f:
            match = re.search(r'Epoch (\d+)/\d+ \| Train MSE Loss: ([0-9.]+) \| TEST PCC: ([-0-9.]+)', line)
            if match:
                writer.writerow(['CBM-GATv2', int(match.group(1)), float(match.group(2)), float(match.group(3))])
                
    # 2. Parse ST-Net and HisToGene
    current_model = None
    log_path = '/home/ronit.28010/.gemini/antigravity-ide/brain/a1153140-3c98-4856-b2ee-e7bda6740c9f/.system_generated/tasks/task-675.log'
    with open(log_path, 'r') as f:
        for line in f:
            if 'ST-Net' in line and 'Training' in line:
                current_model = 'ST-Net'
            elif 'HisToGene' in line and 'Training' in line:
                current_model = 'HisToGene'
                
            match = re.search(r'Epoch (\d+)/\d+ \| Train MSE Loss: ([0-9.]+) \| TEST PCC: ([-0-9.]+)', line)
            if match and current_model:
                writer.writerow([current_model, int(match.group(1)), float(match.group(2)), float(match.group(3))])

print(f"Successfully generated {output_csv}")
