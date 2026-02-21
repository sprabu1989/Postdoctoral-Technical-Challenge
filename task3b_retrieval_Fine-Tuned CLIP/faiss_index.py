# task3_clip_finetuned_retrieval/faiss_index.py

import faiss


def build_index(embeddings):

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index
