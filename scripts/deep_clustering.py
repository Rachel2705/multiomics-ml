import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load latent representations
latent = pd.read_csv("results/latent_representations.csv", index_col=0)
print("Latent representations:", latent.shape)

# KMeans on latent space
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(latent)

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(latent)

# Results
results = pd.DataFrame({
    'Sample': latent.index,
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'Cluster': clusters
})

results.to_csv("results/deep_clustering_results.csv", index=False)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=results,
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='Set1',
    s=100
)
plt.title('Deep Learning Multi-omics Clustering\n(Autoencoder Latent Space)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.savefig("results/deep_clustering_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Cluster distribution:")
print(pd.Series(clusters).value_counts().sort_index())
print("\nDeep clustering complete!")
print("Plot saved to results/deep_clustering_plot.png")
