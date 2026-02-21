# task3_clip_finetuned_retrieval/config.py

DEVICE = "cuda"
DATA_FLAG = "pneumoniamnist"

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-5

MODEL_NAME = "openai/clip-vit-base-patch32"

TOP_K = 5
