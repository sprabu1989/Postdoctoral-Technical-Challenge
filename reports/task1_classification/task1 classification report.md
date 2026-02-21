# Task 1: Pneumonia Classification Report (PneumoniaMNIST)

## 1. Model Architecture and Justification

**Model Used**: Vision Transformer (ViT) with `vit_base_patch16_224` pre-trained on ImageNet.
**Modification**: The original classification head of the ViT model was replaced with a new `nn.Linear` layer to output 2 classes, suitable for the binary classification task (Normal vs. Pneumonia).
`model.head = nn.Linear(model.head.in_features, 2)`

**Justification**:
The Vision Transformer (ViT) architecture is chosen due to its proven success in various computer vision tasks, particularly with large datasets. Leveraging a pre-trained ViT model allows for transfer learning, where the model benefits from features learned on a massive dataset (ImageNet) and adapts them to the medical imaging domain of chest X-rays. This approach significantly reduces the need for extensive training data and computational resources, while often leading to superior performance compared to training from scratch, especially on relatively smaller medical imaging datasets like PneumoniaMNIST. The choice of `vit_base_patch16_224` provides a good balance between model complexity and performance.

## 2. Training Methodology and Hyperparameters

**Dataset**: PneumoniaMNIST (part of MedMNIST v2)
**Image Preprocessing**:
*   Resized to (224, 224) pixels.
*   Converted to 3-channel grayscale.
*   Random Horizontal Flip (p=0.5) and Random Rotation (10 degrees) for data augmentation during training.
*   Normalized using ImageNet mean and standard deviation: `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`.

**Training Parameters**:
*   **Epochs**: 25
*   **Batch Size**: 32
*   **Device**: CUDA (GPU) if available, otherwise CPU.
*   **Loss Function**: `nn.CrossEntropyLoss()` - Standard for multi-class classification.
*   **Optimizer**: `optim.AdamW()` - An Adam optimizer variant with weight decay regularization, commonly used for Transformers.
    *   **Learning Rate (lr)**: 3e-5
    *   **Weight Decay**: 1e-4
*   **Learning Rate Scheduler**: `optim.lr_scheduler.CosineAnnealingLR()`
    *   **T_max**: 10 (Maximum number of iterations for the first restart. In this setup, it dictates the period of the cosine annealing cycle.)

**Training Process Summary**:
The model was trained for 25 epochs. Training and validation losses and accuracies were monitored at each epoch. The `CosineAnnealingLR` scheduler adjusted the learning rate during training. The best model parameters, based on the lowest validation loss, were saved as "best_vit_model.pth".

## 3. Evaluation Metrics and Visualizations

The model's performance was evaluated on the unseen test set of the PneumoniaMNIST dataset.

### Confusion Matrix

```
[[167,  67],
 [  2, 388]]
```

The confusion matrix on the test set is as follows:
*   **True Negatives (Normal correctly classified)**: 167
*   **False Positives (Normal misclassified as Pneumonia)**: 67
*   **False Negatives (Pneumonia misclassified as Normal)**: 2
*   **True Positives (Pneumonia correctly classified)**: 388

**Interpretation**:
The model demonstrates a remarkably low number of False Negatives (2), indicating its high ability to correctly identify actual pneumonia cases. This is crucial in medical diagnosis where missing a disease can have severe consequences. However, there is a notable number of False Positives (67), meaning some healthy patients are incorrectly classified as having pneumonia.

### ROC Curve and AUC Score

**AUC Score**: 0.981

**Interpretation**:
The Area Under the Receiver Operating Characteristic (ROC) Curve is 0.981, which is an excellent score. This indicates that the model has a very high capacity to distinguish between the positive (Pneumonia) and negative (Normal) classes across various classification thresholds. The ROC curve visualization (available in the notebook) is positioned close to the top-left corner, further reinforcing its strong discriminative power.

### Precision-Recall Curve

**Interpretation**:
The Precision-Recall curve visualization (available in the notebook) shows the trade-off between precision and recall for different thresholds. Given the high true positive rate and low false negative rate, and the high AUC, the Precision-Recall curve likely maintains high precision as recall increases, especially for the positive class (Pneumonia). This suggests that the model is effective at retrieving a large proportion of positive instances while keeping the number of false positives relatively low, a desirable characteristic for imbalanced datasets or when the positive class is of primary interest.

## 4. Failure Case Analysis with Examples

Misclassified samples were identified and analyzed to understand patterns in the model's errors. There were 69 misclassified samples in total from the test set.

**Observations from Misclassified Samples (Examples from visualization)**:
The visualizations of randomly selected misclassified images (see notebook output "Sample Misclassifications") showed instances where the model predicted 'Pneumonia' for 'Normal' cases or vice versa. These images often appeared to be borderline cases, or contained features that the model might have over-interpreted or failed to distinguish accurately.

**Confidence Distribution of Wrong Predictions**:
A histogram of the confidence scores for wrong predictions (see notebook output "Confidence Distribution of Wrong Predictions") revealed that a significant portion of the misclassifications occurred with high confidence. This suggests that the model is not always uncertain when it makes a mistake, implying that its decision boundary might be suboptimal for some samples, or that the images themselves are genuinely ambiguous even for human interpretation. Conversely, the "Confidence Distribution of Correct Predictions" showed that the model often made correct predictions with very high confidence.

## 5. Model Strengths and Limitations

### Strengths

*   **High Discriminative Power**: An AUC of 0.981 indicates excellent ability to differentiate between normal and pneumonia cases.
*   **Extremely Low False Negatives**: Only 2 out of 390 actual pneumonia cases were missed. This is a critical strength in medical diagnosis, as it minimizes the risk of overlooking a serious condition.
*   **Effective Feature Learning**: Leveraging a pre-trained ViT demonstrates that the model effectively adapted complex visual features for this specific medical task.
*   **Good Overall Accuracy**: High training and validation accuracies suggest robust learning and generalization to unseen validation data.

### Limitations

*   **High False Positives**: The model misclassified 67 normal cases as pneumonia. While prioritizing recall (low false negatives) is often desired in medical screening, a high false positive rate can lead to patient anxiety, unnecessary further diagnostic tests, and increased healthcare costs.
*   **Confident Errors**: The analysis of wrong prediction confidence indicates that the model sometimes makes incorrect predictions with high confidence, which can be problematic if these predictions are directly used in clinical settings without further scrutiny.
*   **Potential Overfitting to Training Data**: Although the best model was saved based on validation loss, slight discrepancies between training and validation loss/accuracy curves in later epochs indicate a tendency towards overfitting if training continued without early stopping or more aggressive regularization.

## Conclusion

The Vision Transformer model shows exceptional performance on the PneumoniaMNIST dataset. It demonstrates strong learning capabilities, high discriminative power (AUC of 0.981), and a particularly impressive ability to correctly identify pneumonia cases with very few false negatives. The main area for potential improvement lies in reducing false positives, which could lead to unnecessary follow-up procedures for patients identified as having pneumonia when they are actually normal. However, for a critical condition like pneumonia, prioritizing recall (minimizing false negatives) is often preferred, making this model's performance highly valuable.
