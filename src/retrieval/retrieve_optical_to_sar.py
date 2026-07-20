import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import faiss
import pickle
import torch

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

print("Model Loaded")

# Dataset
dataset = CrossModalDataset(
    "test.csv",
    transform=train_transform
)

# Load SAR Index
index = faiss.read_index(
    "sar.index"
)

print("SAR FAISS Loaded")

# Load SAR Paths
with open(
    "sar_paths.pkl",
    "rb"
) as f:
    sar_paths = pickle.load(f)

print("SAR Paths Loaded")

# Query Image
query_idx = 0

sar_img, optical_img, sar_path, optical_path = dataset[query_idx]

optical_img = optical_img.unsqueeze(0).to(device)

with torch.no_grad():

    query_embedding = model.optical_encoder(
        optical_img
    )

query_embedding = (
    query_embedding
    .cpu()
    .numpy()
    .astype("float32")
)

distances, indices = index.search(
    query_embedding,
    5
)

print("\nQUERY OPTICAL IMAGE:")
print(optical_path)

print("\nTOP 5 SAR MATCHES:\n")

for rank, idx in enumerate(indices[0], start=1):

    print(
        f"{rank}. {sar_paths[idx]}"
    )