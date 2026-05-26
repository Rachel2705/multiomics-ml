import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
rnaseq = pd.read_csv("data/rnaseq.csv", index_col=0)
wgs = pd.read_csv("data/wgs.csv", index_col=0)

print("RNA-Seq shape:", rnaseq.shape)
print("WGS shape:", wgs.shape)

# Normalise RNA-Seq data
scaler = StandardScaler()
rnaseq_scaled = pd.DataFrame(
    scaler.fit_transform(rnaseq),
    index=rnaseq.index,
    columns=rnaseq.columns
)

# Normalise WGS data
wgs_scaled = pd.DataFrame(
    scaler.fit_transform(wgs),
    index=wgs.index,
    columns=wgs.columns
)

# Integrate both datasets
integrated = pd.concat([rnaseq_scaled, wgs_scaled], axis=1)

# Save integrated data
integrated.to_csv("data/integrated.csv")

print("Integrated data shape:", integrated.shape)
print("Integration complete!")
