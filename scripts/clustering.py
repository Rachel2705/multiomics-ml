import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load integrated data
data = pd.read_csv("data/integrated.csv", index_col=0)
print("Loaded integrated data:", data.shape)

# PCA - reduce dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data)

print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data)

# Create results dataframe
results = pd.DataFrame({
    'Sample': data.index,
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'Cluster': clusters
})

results.to_csv("results/clustering_results.csv", index=False)

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
plt.title('Multi-omics Integration - PCA + KMeans Clustering')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.savefig("results/clustering_plot.png", dpi=300, bbox_inches='tight')
plt.close()

print("Clustering complete!")
print("Results saved to results/clustering_results.csv")
print("Plot saved to results/clustering_plot.png")
