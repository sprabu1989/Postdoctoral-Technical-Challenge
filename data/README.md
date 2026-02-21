
# Data Folder Structure

This folder contains dataset assets and intermediate outputs used across all tasks.

## Structure

- raw/ : Original PneumoniaMNIST dataset
- processed/ : Resized and normalized images (224x224 for ViT/CLIP)
- embeddings/ : Precomputed feature embeddings (ViT, CLIP, BiomedCLIP)
- faiss_indexes/ : Saved FAISS retrieval indexes
- metadata/ : Label mappings and dataset information

## Dataset Used

PneumoniaMNIST (MedMNIST)

To download automatically:
Use the MedMNIST API inside notebooks.

Large datasets are not stored in this repo for reproducibility.
