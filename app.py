import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st

st.set_page_config(
    page_title="ISRO Cross-Modal Retrieval",
    layout="wide"
)
from PIL import Image
import re
import time
import torch
import faiss
import pickle
from src.datasets.transforms import train_transform
import numpy as np
from src.models.encoder import CrossModalModel
@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    return model, device


model, device = load_model()
@st.cache_resource
def load_indexes():

    optical_index = faiss.read_index(
        "optical.index"
    )

    sar_index = faiss.read_index(
        "sar.index"
    )

    with open(
        "optical_paths.pkl",
        "rb"
    ) as f:
        optical_paths = pickle.load(f)

    with open(
        "sar_paths.pkl",
        "rb"
    ) as f:
        sar_paths = pickle.load(f)

    return (
        optical_index,
        sar_index,
        optical_paths,
        sar_paths
    )


(
    optical_index,
    sar_index,
    optical_paths,
    sar_paths
) = load_indexes()
def extract_roi(path):

    match = re.search(
        r"_(\d+)_p",
        path
    )

    if match:
        return match.group(1)

    return "Unknown"
st.title("Satellite Image Retrieval System")
st.markdown(
    "### Cross-Modal Satellite Image Retrieval using Deep Learning + FAISS"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Top-1 Accuracy", "85.12%")
col2.metric("Top-5 Accuracy", "94.51%")
col3.metric("Top-10 Accuracy", "96.82%")
col4.metric("Avg Time", "0.015s")

st.divider()

uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["png", "jpg", "jpeg"]
)

image_type = st.radio(
    "Select Image Type",
    [
        "SAR",
        "Optical"
    ]
)

retrieval_mode = st.radio(
    "Select Retrieval Mode",
    [
        "Same Modal",
        "Cross Modal"
    ]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

if st.button("Retrieve"):

    if uploaded_file is None:
        st.error("Please upload an image first")

    else:
        start_time = time.time()
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        query_tensor = train_transform(
            image
        )

        query_tensor = (
            query_tensor
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():

            # SAME MODAL
            if retrieval_mode == "Same Modal":

                if image_type == "SAR":

                    query_embedding = (
                        model.sar_encoder(
                            query_tensor
                        )
                    )

                    search_index = sar_index
                    search_paths = sar_paths

                else:

                    query_embedding = (
                        model.optical_encoder(
                            query_tensor
                        )
                    )

                    search_index = optical_index
                    search_paths = optical_paths

            # CROSS MODAL
            else:

                if image_type == "SAR":

                    query_embedding = (
                        model.sar_encoder(
                            query_tensor
                        )
                    )

                    search_index = optical_index
                    search_paths = optical_paths

                else:

                    query_embedding = (
                        model.optical_encoder(
                            query_tensor
                        )
                    )

                    search_index = sar_index
                    search_paths = sar_paths

        query_embedding = (
            query_embedding
            .cpu()
            .numpy()
            .astype("float32")
        )

        distances, indices = search_index.search(
            query_embedding,
            5
        )
        retrieval_time = time.time() - start_time
        st.success("Retrieval Complete")
        st.write(
             f"Retrieval Time: {retrieval_time:.4f} sec"
        )
        st.info(
           f"Mode: {image_type} → {retrieval_mode}"
        )

        cols = st.columns(6)

        with cols[0]:

           st.image(
           image,
           caption="Query Image",
           use_container_width=True
         )

        for i, (idx, dist) in enumerate(
           zip(indices[0], distances[0])
        ):

          result_path = search_paths[idx]

          result_img = Image.open(
             result_path
         )

          roi = extract_roi(
              result_path
         )

          with cols[i + 1]:

            st.image(
            result_img,
            caption=f"Rank {i+1}",
            use_container_width=True
         )

            st.write(
            f"Score: {float(dist):.4f}"
          )
          st.write(
          f"ROI: {roi}"
          )  
st.divider()
st.markdown(
    """
    ### Project Features

    ✅ SAR → Optical Retrieval

    ✅ Optical → SAR Retrieval

    ✅ SAR → SAR Retrieval

    ✅ Optical → Optical Retrieval

    ✅ FAISS Fast Search

    ✅ Deep Cross-Modal Embeddings

    """
)