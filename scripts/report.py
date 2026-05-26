import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv("results/clustering_results.csv")

# Summary statistics
print("=== Multi-omics Integration Report ===")
print(f"Total samples: {len(results)}")
print(f"\nCluster distribution:")
print(results['Cluster'].value_counts().sort_index())

# Plot cluster distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=results,
    x='Cluster',
    palette='Set1'
)
plt.title('Sample Distribution across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Samples')
plt.savefig("results/cluster_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print("\nReport complete!")
print("Plot saved to results/cluster_distribution.png")
