
## Vector Database Implementation Details

To enable efficient similarity search over the generated image embeddings, a **FAISS (Facebook AI Similarity Search)** index was employed. FAISS is a library for efficient similarity search and clustering of dense vectors, making it ideal for large-scale retrieval systems.

### Choice of FAISS and IndexFlatIP:
FAISS was chosen for its unparalleled speed and memory efficiency in performing similarity searches, which is crucial for real-time retrieval tasks. Specifically, an `faiss.IndexFlatIP` index was used. The `IndexFlatIP` stores the vectors directly and performs a brute-force search by computing the inner product (IP) between the query vector and all stored vectors. This choice is particularly suitable for our setup because:

*   **Cosine Similarity and Inner Product:** When vectors are L2-normalized (i.e., their Euclidean norm is 1), the inner product between two vectors is equivalent to their cosine similarity. Since our CLIP embeddings are L2-normalized (`image_features / image_features.norm(p=2, dim=-1, keepdim=True)`), using `IndexFlatIP` directly provides cosine similarity scores, which is a standard and effective metric for semantic similarity.
*   **Simplicity and Accuracy:** For datasets of this size (624 embeddings), a flat index provides exact nearest neighbor search without any approximation, ensuring maximum retrieval accuracy. While more complex indices exist for larger datasets, `IndexFlatIP` offers a good balance of simplicity and performance here.

### Adding Embeddings to the Index:
After extracting and normalizing the image embeddings from the test dataset, they were added to the FAISS index. The `embeddings` array, which has a shape of (624, 512) (624 images, each with a 512-dimensional embedding), was directly added to the index using the `index.add()` method:

```python
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
```

This process indexed all 624 test image embeddings, making them available for rapid similarity queries, whether by image-to-image or text-to-image prompts.
