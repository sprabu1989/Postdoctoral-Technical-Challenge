
# Task 3 Retrieval System Report

## Embedding Model Selection and Justification

For this retrieval system, we utilized the **BiomedCLIP** model from Hugging Face, specifically `'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'`. BiomedCLIP is a medical vision-language pre-training model designed for various biomedical tasks. It combines a vision encoder (ViT-B/16) and a language encoder (PubMedBERT).

**Justification:**
Given that the dataset used (`PneumoniaMNIST`) consists of medical images (X-rays), a model pre-trained on biomedical data like BiomedCLIP is highly suitable. It is expected to produce more relevant and semantically meaningful embeddings for medical images compared to general-purpose CLIP models, leading to better retrieval performance in a medical context.

## Vector Database Implementation Details

We used **FAISS (Facebook AI Similarity Search)** for efficient similarity search over the image embeddings.

**Implementation Steps:**
1.  **Embedding Extraction:** Image features were extracted using the BiomedCLIP model, normalized to unit vectors for cosine similarity, and stored as a NumPy array of `float32` type.
2.  **Index Creation:** An `IndexFlatIP` (Flat Index with Inner Product) was chosen from FAISS. This index type is appropriate because our embeddings are L2-normalized, making the inner product equivalent to cosine similarity.
3.  **Index Population:** The extracted `float32` embeddings were added to the FAISS index.

**Code Snippet for Index Construction:**
```python
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)
```

## Retrieval System Architecture and Usage Instructions

**Architecture:**
The retrieval system follows a standard content-based image retrieval (CBIR) architecture:
1.  **Image Preprocessing:** Input images are resized to 224x224 and converted to grayscale (then to 3 channels) to match the model's expected input.
2.  **Embedding Generation:** The preprocessed images are fed into the BiomedCLIP model to obtain high-dimensional vector representations (embeddings).
3.  **Vector Database (FAISS):** These embeddings are indexed using FAISS for fast similarity search.
4.  **Query Processing:** A query image's embedding is generated, and then a similarity search is performed against the FAISS index.
5.  **Retrieval:** The FAISS index returns the indices of the `k` most similar images based on inner product (cosine similarity).

**Usage Instructions:**

The notebook provides functions to:
-   **Extract Embeddings:** The `embeddings` and `labels` arrays are pre-computed and stored globally.
-   **Search for Similar Images:** The `search_image(query_idx, k=5)` function takes the index of a query image from the dataset and returns the indices of the top `k` most similar images (excluding the query itself).
-   **Visualize Retrieval Results:** The `visualize(query_idx, k=5)` function displays the query image alongside its `k` most similar retrieved images, along with their respective labels.

**Example Usage:**
To search and visualize for a specific image:
```python
visualize(query_idx=10, k=5)
```

## Quantitative Evaluation with Precision@k Metrics

We evaluated the retrieval system using **Precision@k**. Precision@k measures the proportion of retrieved items that are relevant among the top `k` retrieved results. In our case, relevance is defined by whether the retrieved image shares the same class label as the query image.

**Evaluation Methodology:**
For each image in the test dataset, we perform a self-retrieval (querying with the image itself and ignoring the self-match). We then calculate the precision based on the labels of the top `k` retrieved images.

**Results for k=5:**
-   **Total Queries:** 624
-   **Average Precision@5:** 0.8538
-   **Best Case Precision:** 1.0000
-   **Worst Case Precision:** 0.0000
-   **Standard Deviation:** 0.2333

## Visualization of Retrieval Results with Analysis

The `visualize` function helps in qualitatively assessing the retrieval performance. An example visualization is shown below, typically displaying the query image and its top-k retrieved neighbors.

*(An example plot would be displayed here after execution, showing a query image and 5 similar images with their labels.)*

**Analysis of Visualization:**
(This section would be filled after observing actual visualizations. For instance:)
-   In the provided example for `query_idx=10`, all 5 retrieved images share the same label as the query image, indicating a perfect precision for this specific query.
-   We can observe how well the model groups similar medical conditions. For instance, images of pneumonia might be retrieved for a pneumonia query, and normal images for a normal query.

## Discussion of Retrieval Quality and Failure Cases

**Retrieval Quality:**
The average Precision@5 of 0.8538 indicates a generally strong retrieval performance. This suggests that the BiomedCLIP embeddings are effective in capturing the visual semantics of medical images, allowing the FAISS index to find relevant neighbors with high accuracy. The model successfully distinguishes between different classes (e.g., normal vs. pneumonia) for a significant portion of the dataset.

**Potential Failure Cases:**
1.  **High Intra-Class Variation:** Medical images can have significant variations even within the same class (e.g., different severities of pneumonia, artifacts, patient positioning). If the embeddings struggle to normalize these variations, it could lead to misretrievals.
2.  **Low Inter-Class Variation:** Some medical conditions might present subtly in images, making it difficult for the model to differentiate them from other conditions or normal cases, especially if features are very similar.
3.  **Dataset Limitations:** The `PneumoniaMNIST` dataset, while useful, might have certain biases or limited diversity that could impact generalization. Performance on more complex or varied medical datasets might differ.
4.  **Embedding Space Collisions:** In some rare cases, distinct images from different classes might have very similar embeddings, leading to incorrect retrievals. This is reflected in the
