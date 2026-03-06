{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration - Darcy Flow Dataset\n",
    "\n",
    "This notebook explores the Darcy flow dataset used in UA-PBR experiments."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "sys.path.append('..')\n",
    "\n",
    "import torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "from uapbr.data.dataset import RockDataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "dataset = RockDataset(n_samples=1000, resolution=32, seed=42)\n",
    "u, a, labels = dataset.u, dataset.a, dataset.labels\n",
    "\n",
    "print(f\"Pressure field shape: {u.shape}\")\n",
    "print(f\"Permeability field shape: {a.shape}\")\n",
    "print(f\"Labels shape: {labels.shape}\")\n",
    "print(f\"Class distribution: {torch.bincount(labels)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize samples\n",
    "fig, axes = plt.subplots(2, 5, figsize=(15, 6))\n",
    "\n",
    "for i in range(5):\n",
    "    # Class 0 (low permeability)\n",
    "    axes[0, i].imshow(u[labels == 0][i, 0].numpy(), cmap='viridis')\n",
    "    axes[0, i].set_title(f'Class 0 (low) - Sample {i}')\n",
    "    axes[0, i].axis('off')\n",
    "    \n",
    "    # Class 1 (high permeability)\n",
    "    axes[1, i].imshow(u[labels == 1][i, 0].numpy(), cmap='viridis')\n",
    "    axes[1, i].set_title(f'Class 1 (high) - Sample {i}')\n",
    "    axes[1, i].axis('off')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PCA analysis\n",
    "u_flat = u.view(u.size(0), -1).numpy()\n",
    "pca = PCA(n_components=2)\n",
    "u_pca = pca.fit_transform(u_flat)\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "plt.scatter(u_pca[:, 0], u_pca[:, 1], c=labels.numpy(), cmap='bwr', alpha=0.6)\n",
    "plt.colorbar(label='Class')\n",
    "plt.xlabel('PC1')\n",
    "plt.ylabel('PC2')\n",
    "plt.title('PCA of Pressure Fields')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
