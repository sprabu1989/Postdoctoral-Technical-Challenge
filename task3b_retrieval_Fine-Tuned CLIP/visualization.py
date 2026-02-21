# task3_clip_finetuned_retrieval/visualization.py

import matplotlib.pyplot as plt
from task3_clip_finetuned_retrieval.retrieval import image_search


def visualize(query_idx, embeddings, index, dataset, k=5):

    retrieved, _ = image_search(query_idx, embeddings, index, k)

    plt.figure(figsize=(15,3))

    plt.subplot(1, k+1, 1)
    plt.imshow(dataset[query_idx][0].permute(1,2,0))
    plt.title("Query")
    plt.axis("off")

    for i, idx in enumerate(retrieved):
        plt.subplot(1, k+1, i+2)
        plt.imshow(dataset[idx][0].permute(1,2,0))
        plt.title(f"Top {i+1}")
        plt.axis("off")

    plt.show()
