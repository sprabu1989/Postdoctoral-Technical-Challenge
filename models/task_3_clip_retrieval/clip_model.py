
from transformers import CLIPProcessor, CLIPModel

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model
