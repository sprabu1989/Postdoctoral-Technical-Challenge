
# BiomedCLIP model loader
from transformers import AutoProcessor, AutoModel

def load_biomedclip(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return processor, model
