import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_ablation_curves():
    print("Loading benchmark_ablation_results.csv...")
    df = pd.read_csv('results/benchmark_ablation_results.csv')
    
    # We want to plot Test PCC over Epochs
    # Set the style to look like a high-end journal paper
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define a clean color palette
    colors = {
        'ST-Net': '#1f77b4',       # Blue
        'HisToGene': '#ff7f0e',    # Orange
        'CBM-GATv2': '#d62728'     # Red (Our model)
    }
    
    # Plot each model
    for model_name in ['ST-Net', 'HisToGene', 'CBM-GATv2']:
        model_data = df[df['Model'] == model_name].copy()
        
        # Sort by epoch just in case
        model_data = model_data.sort_values('Epoch')
        
        # We only want to plot up to epoch 300 so the graph isn't artificially stretched
        # since baselines stopped at 300.
        model_data = model_data[model_data['Epoch'] <= 300]
        
        linewidth = 3 if model_name == 'CBM-GATv2' else 2
        linestyle = '-' if model_name == 'CBM-GATv2' else '--'
        
        # If it's a baseline, they only have points every 50 epochs, so we can add markers
        marker = 'o' if model_name != 'CBM-GATv2' else None
        
        ax.plot(model_data['Epoch'], model_data['Test_PCC'], 
                label=model_name, 
                color=colors[model_name], 
                linewidth=linewidth,
                linestyle=linestyle,
                marker=marker,
                markersize=8)

    # Format the plot
    ax.set_title('Architecture Ablation: Test PCC over Time', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Training Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Pearson Correlation (PCC)', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, 305)
    ax.set_ylim(0, 0.75)
    
    # Add a horizontal line for the absolute maximum achieved
    max_pcc = df[df['Model'] == 'CBM-GATv2']['Test_PCC'].max()
    ax.axhline(y=max_pcc, color='#d62728', linestyle=':', alpha=0.5)
    ax.text(5, max_pcc + 0.01, f'Max: {max_pcc:.2f}', color='#d62728', fontweight='bold')
    
    # Customize legend
    ax.legend(title='Architecture', title_fontsize='13', loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    os.makedirs('figures/spatial_predictions', exist_ok=True)
    save_path = 'figures/ablation/architecture_ablation_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Ablation Plot to {save_path}")

if __name__ == "__main__":
    plot_ablation_curves()
