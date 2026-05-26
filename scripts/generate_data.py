import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic RNA-Seq data
n_samples = 50
n_genes = 100

samples = [f"Sample_{i}" for i in range(1, n_samples+1)]
genes = [f"Gene_{i}" for i in range(1, n_genes+1)]

rnaseq_data = pd.DataFrame(
    np.random.randint(0, 1000, size=(n_samples, n_genes)),
    index=samples,
    columns=genes
)

rnaseq_data.to_csv("data/rnaseq.csv")
print("RNA-Seq data generated:", rnaseq_data.shape)

# Generate synthetic WGS variant data
n_variants = 50

variants = [f"Variant_{i}" for i in range(1, n_variants+1)]

wgs_data = pd.DataFrame(
    np.random.randint(0, 3, size=(n_samples, n_variants)),
    index=samples,
    columns=variants
)

wgs_data.to_csv("data/wgs.csv")
print("WGS data generated:", wgs_data.shape)
