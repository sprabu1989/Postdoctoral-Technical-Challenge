# task1_classification/inference.py

import torch


def predict(model, image_tensor, device):

    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, 1)

    return pred.item(), probs.max().item()
