import torch
from .model import SignLanguageTransformer

# --- Configuration ---
TARGET_WORDS = [
    'LIBRARY', 'CELERY', 'PUSH', 'YOU', 'PASSPORT',
    'GOVERNMENT', 'BYE', 'THANKYOU', 'HELLO', 'MYSELF'
]
CHECKPOINT_PATH = "./checkpoints/best_model.pth"
INPUT_DIM = 2 * 21 * 3
EMBED_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
FF_DIM = 512
NUM_CLASSES = len(TARGET_WORDS)
DROPOUT = 0.3

# --- Model Loading ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignLanguageTransformer(
    INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FF_DIM, NUM_CLASSES, DROPOUT
).to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()


def get_model():
    return model, device


def get_target_words():
    return TARGET_WORDS
