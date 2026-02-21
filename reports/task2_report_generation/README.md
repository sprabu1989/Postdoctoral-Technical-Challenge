# MedGemma Medical Report Generation on PneumoniaMNIST

This project implements a multimodal Vision-Language Model (VLM) pipeline using **MedGemma-1.5-4b-it** to classify chest X-rays and generate automated, structured medical reports from the MedMNIST dataset.

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU (T4 or better recommended for quantization)
- Hugging Face Hub account and token (with access to MedGemma)

### Installation
```bash
pip install -q transformers>=4.50.0 accelerate bitsandbytes sentencepiece medmnist==2.2.3
```

## 🚀 Project Overview
- **Model**: `google/medgemma-1.5-4b-it` (4-bit quantized)
- **Dataset**: `PneumoniaMNIST` (Chest X-ray binary classification)
- **Objective**: Evaluate VLM's ability to provide clinical reasoning vs. traditional CNN 'black-box' classification.

## 📊 Key Features
- **Prompt Ablation**: Testing Free-form, Structured, and Differential prompting strategies.
- **Medical Report Generation**: Automated generation of Radiologist-style FINDINGS and IMPRESSION sections.
- **Multimodal Evaluation**: Qualitative analysis comparing Ground Truth labels with VLM narrative justifications.

## 📈 Results Summary
- **Accuracy**: Achieved ~80% accuracy on binary classification tasks.
- **Explainability**: The model identifies specific anatomical features (e.g., vascular markings, pleural spaces) even on low-resolution 28x28 images.
- **Formatting**: Successfully adopts a clinical persona to follow standard reporting guidelines.

## 📄 File Descriptions
- `task2_report_generation.md`: Detailed clinical analysis and qualitative findings.
- `vlm_reports.csv`: Raw outputs of model predictions and generated reports.
