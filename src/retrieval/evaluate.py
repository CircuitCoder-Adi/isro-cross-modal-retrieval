import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import time
import pickle
import faiss
import torch
import pandas as pd
import numpy as np

from src.models.encoder import CrossModalModel
from src.datasets.dataset import CrossModalDataset
from src.datasets.transforms import train_transform

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Model
model = CrossModalModel(embedding_dim=256)

model.load_state_dict(
    torch.load(
        "final_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Dataset
dataset = CrossModalDataset(
    "test.csv",
    transform=train_transform
)

# FAISS
index = faiss.read_index("optical.index")

# Paths
with open("optical_paths.pkl", "rb") as f:
    optical_paths = pickle.load(f)

top1_correct = 0
top5_correct = 0
top10_correct = 0

total_time = 0
num_samples = len(dataset)

for i in range(num_samples):

    sar_img, optical_img, sar_path, optical_path = dataset[i]

    query_roi = sar_path.split("s1_")[1].split("\\")[0]

    sar_img = sar_img.unsqueeze(0).to(device)

    start = time.time()

    with torch.no_grad():
        query_embedding = model.sar_encoder(sar_img)

    query_embedding = (
        query_embedding
        .cpu()
        .numpy()
        .astype("float32")
    )

    distances, indices = index.search(
        query_embedding,
        10
    )

    elapsed = time.time() - start
    total_time += elapsed

    retrieved_rois = []

    for idx in indices[0]:
        path = optical_paths[idx]
        roi = path.split("s2_")[1].split("\\")[0]
        retrieved_rois.append(roi)

    if query_roi == retrieved_rois[0]:
        top1_correct += 1

    if query_roi in retrieved_rois[:5]:
        top5_correct += 1

    if query_roi in retrieved_rois[:10]:
        top10_correct += 1

top1 = 100 * top1_correct / num_samples
top5 = 100 * top5_correct / num_samples
top10 = 100 * top10_correct / num_samples

avg_time = total_time / num_samples

print(f"Top-1 Accuracy : {top1:.2f}%")
print(f"Top-5 Accuracy : {top5:.2f}%")
print(f"Top-10 Accuracy: {top10:.2f}%")
print(f"Avg Retrieval Time: {avg_time:.4f} sec")