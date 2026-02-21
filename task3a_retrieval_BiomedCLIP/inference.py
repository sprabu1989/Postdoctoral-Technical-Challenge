# task3_biomedclip_retrieval/inference.py

from task3_biomedclip_retrieval.retrieval import search_image


def retrieve_similar(query_idx, embeddings, labels, index, k=5):

    retrieved, scores = search_image(query_idx, embeddings, index, k)

    results = []
    for idx, score in zip(retrieved, scores):
        results.append({
            "index": int(idx),
            "label": int(labels[idx]),
            "similarity": float(score)
        })

    return results
