
import faiss
import numpy as np

def build_faiss_index(features):
    dim = features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(features)
    return index
