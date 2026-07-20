import numpy as np
import faiss

# Load embeddings
embeddings = np.load(
    "optical_embeddings.npy"
)

print("Embeddings Shape:", embeddings.shape)

# FAISS float32 expect karta hai
embeddings = embeddings.astype("float32")

# Dimension
dimension = embeddings.shape[1]

# Index
index = faiss.IndexFlatL2(
    dimension
)

# Add embeddings
index.add(
    embeddings
)

print("Total Indexed Images:", index.ntotal)

# Save index
faiss.write_index(
    index,
    "optical.index"
)

print("FAISS Index Saved")