# Task 3 - BiomedCLIP Semantic Image Retrieval

Dataset: PneumoniaMNIST  
Model: BiomedCLIP (OpenCLIP version)  
Similarity: Cosine (FAISS IndexFlatIP)  

Features:
- Batched embedding extraction
- Feature normalization
- FAISS index building
- Precision@k evaluation
- Retrieval visualization

Pipeline:
1. data_loader.py
2. model_loader.py
3. embedder.py
4. faiss_index.py
5. evaluator.py
6. visualization.py

Outputs can be stored in outputs/
