# task3_biomedclip_retrieval/evaluator.py

import numpy as np
from task3_biomedclip_retrieval.retrieval import search_image


def precision_at_k(embeddings, labels, index, k=5):

    all_precisions = []

    for i in range(len(embeddings)):
        retrieved, _ = search_image(i, embeddings, index, k)
        matches = sum(labels[i] == labels[j] for j in retrieved)
        all_precisions.append(matches / k)

    return {
        "average_precision": float(np.mean(all_precisions)),
        "best_case": float(np.max(all_precisions)),
        "worst_case": float(np.min(all_precisions)),
        "std_dev": float(np.std(all_precisions))
    }
