# 🏥 Medical Multimodal AI Framework  
## Pneumonia Detection, Clinical Report Generation & Semantic Retrieval

This repository presents a comprehensive medical AI framework built on the PneumoniaMNIST dataset. It integrates discriminative and generative deep learning models to address three core tasks:

1. Medical image classification using Vision Transformers  
2. Automated radiology-style report generation using a Vision-Language Model  
3. Semantic image retrieval using CLIP-based multimodal embeddings  

The project demonstrates end-to-end system design, modular deep learning architecture, multimodal reasoning, and retrieval-based evaluation in healthcare AI.

---

# 📌 Table of Contents

- Project Overview  
- System Architecture  
- Dataset Description  
- Task 1: Vision Transformer Classification  
- Task 2: Medical Vision-Language Report Generation  
- Task 3A: BiomedCLIP Retrieval  
- Task 3B: Fine-Tuned CLIP Retrieval  
- Installation Guide  
- Usage Instructions  
- Evaluation Metrics  
- Experimental Design  
- Results Summary  
- Limitations  
- Future Work  
- License  

---

# 🧠 Project Overview

This framework explores both discriminative and generative paradigms in medical imaging:

- Discriminative Learning → Classification (ViT)  
- Generative Multimodal Learning → Clinical Report Generation (MedGemma)  
- Cross-Modal Representation Learning → Semantic Retrieval (CLIP/BiomedCLIP)  

The project emphasizes:

- Transfer learning  
- Multimodal alignment  
- FAISS-based similarity search  
- Prompt engineering  
- Embedding normalization  
- Explainable AI for healthcare  

---

# 🏗 System Architecture

High-Level Flow:

1️⃣ Image Input  
↓  
2️⃣ Feature Extraction  
   - ViT  
   - CLIP / BiomedCLIP  
↓  
3️⃣ Task-Specific Modules  
   - Classification  
   - Report Generation  
   - Embedding Indexing  
↓  
4️⃣ Evaluation & Visualization  

Modular Structure:

project/  
│  
├── task1_classification/  
├── task2_report_generation/  
├── task3_biomedclip_retrieval/  
├── task3_clip_finetuned_retrieval/  
│  
├── data/  
├── notebooks/  
├── outputs/  
└── requirements.txt  

---

# 📊 Dataset Description

Dataset: PneumoniaMNIST  
Source: MedMNIST v2.2.3  

Properties:
- Binary classification  
- Class 0 → Normal  
- Class 1 → Pneumonia  
- Original resolution: 28×28 grayscale  
- Resized to 224×224 for transformer-based models  

Why this dataset?
- Lightweight  
- Standardized benchmark  
- Suitable for reproducible experimentation  

Limitations:
- Low resolution  
- Binary diagnostic labels  
- Not a substitute for full clinical datasets  

---

# 🔬 Task 1: Vision Transformer Classification

Model:
vit_base_patch16_224 (timm)

Approach:
- Transfer learning from ImageNet  
- Final classifier head replaced  
- CrossEntropyLoss  
- AdamW optimizer  
- Cosine Annealing scheduler  

Outputs:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion matrix  
- Misclassification visualization  

Objective:
Evaluate transformer-based discriminative modeling in medical imaging.

---

# 🏥 Task 2: Medical Report Generation (MedGemma 4B-IT)

Model:
google/medgemma-4b-it  

Type:
Vision-Language Model (Image-Text-to-Text)

Pipeline:
- Structured prompt engineering  
- Chat template construction  
- Token-level decoding  
- Clinical narrative generation  

Example Output:

Findings: Increased opacities in lower lung zone.  
Impression: Radiographic evidence consistent with pneumonia.  
Diagnosis: Pneumonia.  

Evaluation:
- Qualitative comparison with ground truth  
- Prediction extraction from text  
- Subset evaluation  

Goal:
Enhance interpretability beyond binary classification.

---

# 🔎 Task 3A: BiomedCLIP Semantic Retrieval

Model:
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224  

Process:
- Batched embedding extraction  
- L2 normalization  
- FAISS IndexFlatIP  
- Cosine similarity search  

Metric:
Precision@k  

Use Case:
Image-to-image retrieval based on semantic similarity.

---

# 🔍 Task 3B: Fine-Tuned CLIP Retrieval

Model:
openai/clip-vit-base-patch32  

Enhancements:
- Fine-tuned classifier head  
- Joint embedding optimization  
- Text-to-image retrieval  
- Image-to-image retrieval  

Features:
- 512-dimensional normalized embeddings  
- FAISS-based retrieval  
- Precision@k evaluation  

Supports:
- Query by image  
- Query by clinical text  

---

# ⚙ Installation Guide

Step 1: Clone Repository

git clone <repository_url>  
cd project  

Step 2: Install Dependencies

pip install -r requirements.txt  

For MedGemma access:

huggingface-cli login  

---

# ▶ Usage Instructions

Run tasks via modular folders:

cd task1_classification  
cd task2_report_generation  
cd task3_biomedclip_retrieval  
cd task3_clip_finetuned_retrieval  

Or use interactive notebooks in:

notebooks/  

---

# 📈 Evaluation Metrics

Classification:
- Accuracy  
- Precision  
- Recall   
- ROC-AUC  

Retrieval:
- Precision@k  
- Cosine similarity  

Report Generation:
- Qualitative clinical reasoning coherence  

---

# 🧪 Experimental Design

- All models trained/tested on PneumoniaMNIST  
- Embeddings normalized before FAISS indexing  
- Cross-entropy loss for classification  
- Structured prompting for MedGemma  
- Cosine similarity for retrieval evaluation  

---

# 📊 Results Summary (Template)

| Task | Metric | Result |
|------|--------|--------|
| ViT Classification | Accuracy | Train Accuracy: 99.87%, Validation Accuracy: 97.90% |
| BiomedCLIP | Precision@5 | 0.8538 |
| Fine-Tuned CLIP | Precision@5 | 0.9157 |
| MedGemma | Qualitative Alignment | High |


---

# ⚠ Limitations

- PneumoniaMNIST is low resolution  
- MedGemma requires high GPU memory  
- Retrieval sensitive to embedding quality  
- Vision-language models may hallucinate  
- Binary diagnostic setting limits generalization  

---

# 🔮 Future Work

- Higher-resolution clinical datasets  
- Multi-class disease classification  
- Retrieval-Augmented Generation (RAG)  
- Hybrid CNN + VLM pipelines  
- Clinical validation studies  
- Domain-specific fine-tuning  

---

# 🎓 Academic Contribution

This project demonstrates:

- Transformer-based medical image analysis  
- Multimodal vision-language reasoning  
- Cross-modal embedding alignment  
- FAISS-based scalable retrieval  
- Explainable AI in healthcare  

Suitable for:
- Advanced coursework  
- Research publication foundation  
- AI in Healthcare portfolio  
- PhD research extension  

---

# 📜 License

This project is intended for academic and research purposes only.  
Not intended for clinical deployment.

---

# 👨‍💻 Author

Developed as part of advanced research in:

- Medical Computer Vision  
- Vision-Language Models  
- Multimodal Retrieval Systems  
- AI for Healthcare  
