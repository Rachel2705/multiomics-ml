import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load integrated data
data = pd.read_csv("data/integrated.csv", index_col=0)
print("Loaded data:", data.shape)

# Convert to PyTorch tensor
X = torch.FloatTensor(data.values)

# Create DataLoader
dataset = TensorDataset(X)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

# Initialize model
input_dim = data.shape[1]  # 150
latent_dim = 8
model = Autoencoder(input_dim, latent_dim)
print("Model created!")
print(model)

# Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []
epochs = 100

print("\nTraining started...")
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x = batch[0]
        optimizer.zero_grad()
        reconstructed, latent = model(x)
        loss = criterion(reconstructed, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

# Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("results/training_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# Save latent representations
model.eval()
with torch.no_grad():
    _, latent = model(X)

latent_df = pd.DataFrame(
    latent.numpy(),
    index=data.index,
    columns=[f"Latent_{i}" for i in range(latent_dim)]
)

latent_df.to_csv("results/latent_representations.csv")
print("\nTraining complete!")
print("Latent representations saved!")
print("Training loss plot saved!")
