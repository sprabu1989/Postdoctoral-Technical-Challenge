
from transformers import AutoProcessor, AutoModelForImageTextToText

def load_medgemma(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    return processor, model
