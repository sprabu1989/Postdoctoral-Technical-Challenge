# Task 2 - Vision-Language Medical Report Generation

Model: MedGemma 4B-IT  
Dataset: PneumoniaMNIST  

Features:
- HuggingFace login integration
- Structured prompt-based report generation
- Prediction extraction from generated reports
- Subset evaluation
- Sample visualization

Workflow:
1. hf_setup.py → login
2. data_loader.py → load dataset
3. medgemma_model.py → load model
4. report_generator.py → generate reports
5. evaluator.py → evaluate predictions
6. visualization.py → display results

Outputs stored in outputs/
