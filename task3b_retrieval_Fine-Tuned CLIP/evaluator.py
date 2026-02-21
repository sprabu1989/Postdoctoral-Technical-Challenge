# task3_clip_finetuned_retrieval/evaluator.py

import numpy as np
from task3_clip_finetuned_retrieval.retrieval import image_search


def precision_at_k(embeddings, labels, index, k=5):

    scores = []

    for i in range(len(embeddings)):
        retrieved, _ = image_search(i, embeddings, index, k)
        correct = sum(
            1 for idx in retrieved
            if labels[idx] == labels[i]
        )
        scores.append(correct / k)

    mean_precision = np.mean(scores)
    print(f"Mean Precision@{k}: {mean_precision:.4f}")

    return mean_precision
