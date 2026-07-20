import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import faiss
import pickle
import torch
import matplotlib.pyplot as plt
from PIL import Image

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

print("Dataset Loaded")

# Load Optical Index
index = faiss.read_index(
    "optical.index"
)

print("FAISS Loaded")

# Load Optical Paths
with open(
    "optical_paths.pkl",
    "rb"
) as f:
    optical_paths = pickle.load(f)

print("Paths Loaded")

# Query Image
query_idx = 0

sar_img, optical_img, sar_path, optical_path = dataset[query_idx]

query_roi = sar_path.split("s1_")[1].split("\\")[0]

sar_img = sar_img.unsqueeze(0).to(device)

# Generate Query Embedding
with torch.no_grad():

    query_embedding = model.sar_encoder(
        sar_img
    )

query_embedding = (
    query_embedding
    .cpu()
    .numpy()
    .astype("float32")
)

# Search Top-5
distances, indices = index.search(
    query_embedding,
    5
)

print("\nQUERY ROI:", query_roi)

# Plot Results
plt.figure(figsize=(18, 4))

# Query SAR Image
plt.subplot(1, 6, 1)

img = Image.open(sar_path)

plt.imshow(img)

plt.title(
    f"Query SAR\nROI {query_roi}"
)

plt.axis("off")

# Retrieved Optical Images
for i, idx in enumerate(indices[0]):

    plt.subplot(1, 6, i + 2)

    retrieved_path = optical_paths[idx]

    roi = retrieved_path.split("s2_")[1].split("\\")[0]

    img = Image.open(
        retrieved_path
    )

    plt.imshow(img)

    plt.title(
        f"Rank {i+1}\nROI {roi}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()