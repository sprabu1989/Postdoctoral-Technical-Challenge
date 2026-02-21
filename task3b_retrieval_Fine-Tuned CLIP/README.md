# Task 3 - Fine-Tuned CLIP Semantic Retrieval

Model: CLIP ViT-B/32 (Fine-tuned)  
Dataset: PneumoniaMNIST  
Similarity: Cosine Similarity (FAISS IndexFlatIP)

Features:
- CLIP fine-tuning
- Embedding normalization
- Image-to-image retrieval
- Text-to-image retrieval
- Precision@k evaluation
- Visualization module

Pipeline:
1. data_loader.py
2. model_loader.py
3. finetune.py
4. embeddings.py
5. faiss_index.py
6. retrieval.py
7. evaluator.py
8. visualization.py

Outputs stored in outputs/
