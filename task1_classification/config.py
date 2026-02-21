# task1_classification/config.py

BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2
MODEL_NAME = "vit_base_patch16_224"
BEST_MODEL_PATH = "task1_classification/checkpoints/best_vit_model.pth"
FINAL_MODEL_PATH = "task1_classification/checkpoints/final_vit_model.pth"
