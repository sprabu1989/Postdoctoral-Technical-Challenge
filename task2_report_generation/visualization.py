# task2_report_generation/visualization.py

import matplotlib.pyplot as plt


def display_samples(results, dataset, num_samples=6):

    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))

    for i in range(num_samples):
        res = results[i]
        img, _ = dataset[res["index"]]

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(
            f"GT: {res['ground_truth']} | Pred: {res['vlm_prediction']}"
        )
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
