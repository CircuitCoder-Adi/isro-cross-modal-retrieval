import numpy as np
import faiss

embeddings = np.load(
    "sar_embeddings.npy"
)

embeddings = embeddings.astype(
    "float32"
)

print(
    "Embeddings Shape:",
    embeddings.shape
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(
    embeddings
)

print(
    "Total Indexed Images:",
    index.ntotal
)

faiss.write_index(
    index,
    "sar.index"
)

print(
    "SAR FAISS Index Saved"
)