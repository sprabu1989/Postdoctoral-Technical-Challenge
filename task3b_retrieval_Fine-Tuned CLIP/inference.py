# task3_clip_finetuned_retrieval/inference.py

from task3_clip_finetuned_retrieval.retrieval import image_search


def retrieve_similar(query_idx, embeddings, labels, index, k=5):

    retrieved, distances = image_search(
        query_idx,
        embeddings,
        index,
        k
    )

    results = []

    for idx, score in zip(retrieved, distances):
        results.append({
            "index": int(idx),
            "label": int(labels[idx]),
            "similarity": float(score)
        })

    return results
