# Task 2: Automated Medical Report Generation with MedGemma

## 1. Model Selection Justification: MedGemma
MedGemma-1.5-4b-it was selected due to its specialized fine-tuning on medical datasets, allowing it to recognize anatomical structures and clinical terminology. Its multimodal architecture (SigLIP + Gemma-2) enables both image classification and structured report generation. The 4B version is efficient enough for T4 GPU environments using 4-bit quantization.

## 2. Prompting Strategies and Effectiveness
Three strategies were tested:
- **Free-form**: Simple description request.
- **Structured**: Role-based (Thoracic Radiologist) with specific anatomical sections.
- **Differential**: Requested findings and differential diagnosis.

**Effectiveness**: Accuracy remained consistent (~80%) across strategies for binary classification, but the **Structured** and **Differential** prompts significantly improved the clinical utility and legibility of the generated reports.

## 3. Sample Generated Reports
As seen in the notebook visualizations, the model generates reports with sections for Lung Fields, Opacities, and Pleural Findings. Even on low-resolution 28x28 images, it maintains a professional radiologist persona, though upscaling artifacts can occasionally impact granular accuracy.

## 4. Qualitative Analysis: VLM vs. Ground Truth & CNNs
- **Explainability**: Unlike CNNs that provide a single score, the VLM provides narrative justification (e.g., 'clear lung fields').
- **Sensitivity**: The VLM shows high sensitivity to subtle markings, occasionally leading to false positives compared to the ground truth.
- **Confidence**: The VLM uses clinical hedging ('appears normal'), reflecting diagnostic uncertainty better than traditional softmax outputs.

## 5. Strengths and Limitations
- **Strengths**: High interpretability, strong zero-shot instruction following, and standardized clinical formatting.
- **Limitations**: High sensitivity to image resolution (28x28 is sub-optimal for SigLIP), inference latency compared to small CNNs, and the risk of 'confident hallucinations' regarding obscured anatomical details.
