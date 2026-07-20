import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
import pickle
from tqdm import tqdm

from src.models.encoder import CrossModalModel
from src.datasets.dataset import CrossModalDataset
from src.datasets.transforms import train_transform


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model
model = CrossModalModel(
    embedding_dim=256
)

model.load_state_dict(
    torch.load(
        "final_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model Loaded Successfully")

# Dataset
dataset = CrossModalDataset(
    "test.csv",
    transform=train_transform
)

print("Dataset Size:", len(dataset))

# Test Sample
sar_img, optical_img, sar_path, optical_path = dataset[0]

print("SAR Shape:", sar_img.shape)
print("Optical Shape:", optical_img.shape)

sar_img = sar_img.unsqueeze(0).to(device)
optical_img = optical_img.unsqueeze(0).to(device)

with torch.no_grad():
    sar_emb, optical_emb = model(
        sar_img,
        optical_img
    )

print("SAR Embedding:", sar_emb.shape)
print("Optical Embedding:", optical_emb.shape)

# Generate SAR Embeddings
sar_embeddings = []
sar_paths = []

for i in tqdm(range(len(dataset))):

    sar_img, optical_img, sar_path, optical_path = dataset[i]

    sar_img = sar_img.unsqueeze(0).to(device)

    with torch.no_grad():

        sar_emb = model.sar_encoder(
            sar_img
        )

    sar_embeddings.append(
        sar_emb.cpu().numpy()[0]
    )

    sar_paths.append(
        sar_path
    )

sar_embeddings = np.array(
    sar_embeddings
)

print("Embeddings Shape:", sar_embeddings.shape)

# Save Embeddings
np.save(
    "sar_embeddings.npy",
    sar_embeddings
)

print("SAR embeddings saved")

# Save Paths
with open(
    "sar_paths.pkl",
    "wb"
) as f:

    pickle.dump(
        sar_paths,
        f
    )

print("SAR paths saved")

print("SCRIPT FINISHED")