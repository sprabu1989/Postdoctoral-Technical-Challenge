# task3_biomedclip_retrieval/faiss_index.py

import faiss


def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index
