# 🏥 Medical Vision-Language & Retrieval Systems  
### Pneumonia Detection, Report Generation & Semantic Image Retrieval

This repository implements a complete medical AI pipeline using PneumoniaMNIST, including:

- 🔹 Vision Transformer (ViT) Classification  
- 🔹 Medical Vision-Language Report Generation (MedGemma)  
- 🔹 BiomedCLIP Semantic Image Retrieval  
- 🔹 Fine-Tuned CLIP Cross-Modal Retrieval  

The project demonstrates both discriminative and generative multimodal AI systems in healthcare.

---

# 📂 Project Structure

project/
│
├── task1_classification/              
├── task2_report_generation/           
├── task3_biomedclip_retrieval/        
├── task3_clip_finetuned_retrieval/    
│
├── data/                              
├── models/                            
├── notebooks/                         
├── outputs/                           
│
├── requirements.txt
└── README.md

---

# 📊 Dataset

**PneumoniaMNIST (MedMNIST v2.2.3)**

- Binary classification
- Classes:
  - 0 → Normal
  - 1 → Pneumonia
- Original image size: 28×28
- Resized to: 224×224 for ViT and CLIP models

The dataset automatically downloads using the MedMNIST API.

---

# 🔹 Task 1: Vision Transformer Classification

Model: vit_base_patch16_224 (timm)

Features:
- Transfer learning
- Cosine Annealing scheduler
- Confusion matrix
- ROC curve
- Precision–Recall curve
- Misclassification analysis

Outputs:
- Best model checkpoint
- Evaluation metrics
- Visualization plots

---

# 🔹 Task 2: Medical Report Generation (MedGemma 4B-IT)

Model: google/medgemma-4b-it

Features:
- Structured prompt engineering
- Radiology-style report generation
- Prediction extraction from generated text
- Qualitative comparison with ground truth

Example output:

Findings: Increased opacity in lower lung field.  
Impression: Features consistent with pneumonia.  
Diagnosis: Pneumonia.

---

# 🔹 Task 3A: BiomedCLIP Semantic Image Retrieval

Model:
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

Features:
- Batched embedding extraction
- Cosine similarity normalization
- FAISS IndexFlatIP
- Precision@k evaluation
- Image-to-image retrieval visualization

Metric:
- Precision@5

---

# 🔹 Task 3B: Fine-Tuned CLIP Retrieval

Model:
openai/clip-vit-base-patch32

Features:
- CLIP fine-tuning with classifier head
- 512-dimensional embeddings
- Image-to-image retrieval
- Text-to-image retrieval
- Precision@k evaluation

Supports:
- Query by image
- Query by text

---

# 🚀 Installation

Install dependencies:

pip install -r requirements.txt

Or manually:

pip install medmnist==2.2.3
pip install transformers
pip install timm
pip install open_clip_torch
pip install faiss-cpu
pip install accelerate bitsandbytes

For MedGemma access:

huggingface-cli login

---

# ▶️ How to Run

Use the modular folders or the notebooks:

cd task1_classification
cd task2_report_generation
cd task3_biomedclip_retrieval
cd task3_clip_finetuned_retrieval

Or run Colab notebooks inside:

notebooks/

---

# 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Precision@k
- Cosine Similarity

---

# 🧠 Research Contributions

This project demonstrates:

- Transfer learning for medical imaging
- Vision-language medical reasoning
- Cross-modal retrieval
- Fine-tuned multimodal embeddings
- FAISS-based large-scale retrieval
- Explainable AI via report generation

---

# ⚠️ Limitations

- PneumoniaMNIST is low resolution (28×28 original)
- MedGemma requires large GPU memory
- Retrieval performance depends on embedding quality
- VLM may hallucinate without careful prompting

---

# 🔮 Future Work

- Hybrid CNN + VLM pipeline
- Multimodal cross-attention refinement
- Retrieval-Augmented Generation (RAG)
- Domain adaptation to real clinical datasets
- Clinical-grade evaluation

---

# 🏆 Author

Developed as part of advanced research in:

- Medical Vision Transformers
- Vision-Language Models
- Multimodal Retrieval Systems
- AI for Healthcare

---

# 📜 License

For academic and research use only.
