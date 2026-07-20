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
from src.datasets.transforms import train_transform

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Load Model
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
query_path = "demo/query_images/query.png"

img = Image.open(
    query_path
).convert("RGB")

query_tensor = train_transform(
    img
)

query_tensor = query_tensor.unsqueeze(0).to(device)

# Generate Embedding
with torch.no_grad():

    query_embedding = model.sar_encoder(
        query_tensor
    )

query_embedding = (
    query_embedding
    .cpu()
    .numpy()
    .astype("float32")
)

# Search
distances, indices = index.search(
    query_embedding,
    5
)

print("\nTop 5 Retrieved Images:\n")

for rank, idx in enumerate(indices[0], start=1):

    print(
        f"{rank}. {optical_paths[idx]}"
    )

# Visualization
plt.figure(figsize=(18,4))

plt.subplot(1,6,1)
plt.imshow(img)
plt.title("Query")
plt.axis("off")

for i, idx in enumerate(indices[0]):

    plt.subplot(1,6,i+2)

    retrieved = Image.open(
        optical_paths[idx]
    )

    plt.imshow(retrieved)

    plt.title(
        f"Rank {i+1}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()