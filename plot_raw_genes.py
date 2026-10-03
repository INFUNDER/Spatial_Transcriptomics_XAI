import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_top_5_genes(h5ad_path, out_path):
    print(f"Loading {h5ad_path}...")
    adata = sc.read_h5ad(h5ad_path)
    
    # Identify highly variable genes to ensure we pick ones with a strong visual signal
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=100, flavor='seurat')
    
    # Extract top 5 genes by variance
    top_genes = adata.var.sort_values(by='dispersions_norm', ascending=False).head(5).index.tolist()
    print(f"Selected Top 5 Genes: {top_genes}")
    
    coords = adata.obsm['spatial']
    expr = adata[:, top_genes].X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()
        
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f'Raw Gene Expression Heatmaps (Sample: TENX200)', fontsize=24, fontweight='bold', y=1.05)
    
    for i, gene in enumerate(top_genes):
        gene_expr = expr[:, i]
        sc_plot = axes[i].scatter(coords[:, 0], coords[:, 1], c=gene_expr, cmap='magma', s=20, alpha=0.9)
        axes[i].invert_yaxis()
        axes[i].set_title(f'Gene: {gene}', fontsize=20, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        plt.colorbar(sc_plot, ax=axes[i], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    plot_top_5_genes('hest_data/st/TENX200.h5ad', 'figures_supplementary/raw_genes_heatmap.png')
