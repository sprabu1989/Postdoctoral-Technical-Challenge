# task3_biomedclip_retrieval/visualization.py

import matplotlib.pyplot as plt
from task3_biomedclip_retrieval.retrieval import search_image


def visualize(query_idx, embeddings, labels, image_store, index, k=5):

    retrieved, _ = search_image(query_idx, embeddings, index, k)

    plt.figure(figsize=(15, 3))

    plt.subplot(1, k + 1, 1)
    plt.imshow(image_store[query_idx].permute(1, 2, 0))
    plt.title(f"Query\nLabel: {labels[query_idx]}")
    plt.axis("off")

    for i, idx in enumerate(retrieved):
        plt.subplot(1, k + 1, i + 2)
        plt.imshow(image_store[idx].permute(1, 2, 0))
        plt.title(f"Top-{i+1}\nLabel: {labels[idx]}")
        plt.axis("off")

    plt.show()
