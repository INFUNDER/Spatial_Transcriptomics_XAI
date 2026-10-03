import matplotlib.pyplot as plt
import numpy as np

# Training loss data from histogene_train.o30504
epochs = np.arange(1, 21)
loss = [
    20220.9520, 20181.0379, 20150.3225, 20123.7493, 20105.1271,
    20088.0641, 20070.3553, 20052.5645, 20035.9990, 20017.5609,
    20000.3996, 19981.6110, 19963.4673, 19946.3333, 19929.1646,
    19910.3283, 19894.1904, 19877.2129, 19860.0713, 19843.6312
]

# Set a clean, professional style for publication/presentation
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6), dpi=300)

plt.plot(epochs, loss, marker='o', linewidth=2.5, markersize=8, color='#1f77b4', label='Training Loss')

# Add labels and title
plt.title('Baseline HisToGene (UNI Features) - Training Loss over Epochs', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Loss (MSE)', fontsize=14, fontweight='bold')

# Customize ticks
plt.xticks(np.arange(1, 21, 1), fontsize=12)
plt.yticks(fontsize=12)

# Add grid and legend
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper right')

# Fill area under the curve slightly for aesthetics
plt.fill_between(epochs, loss, min(loss) - 100, color='#1f77b4', alpha=0.1)
plt.ylim(min(loss) - 50, max(loss) + 50)

# Save to file
plt.tight_layout()
plt.savefig('figures/supplementary/baseline_training_loss.png')
print("Saved figures/supplementary/baseline_training_loss.png successfully.")
