import torch
from .model import SignLanguageTransformer

# --- Configuration ---
TARGET_WORDS = [
    # Core Conversation & Greetings
    'HELLO', 'I', 'NO', 'PLEASE', 'THANKYOU', 'SORRY',
    'OK', 'MAYBE', 'NICE', 'MEET', 'GREET2', 'MORE',

    # People & Family
    'I', 'YOU', 'THEY1', 'MAN', 'WOMAN1', 'BOY', 'CHILD',
    'CHILDREN', 'FRIEND', 'PERSON', 'PEOPLE', 'FAMILY', 'MOTHER',
    'PARENTS', 'BROTHER', 'SON', 'HUSBAND',
    'GRANDMOTHER', 'GRANDFATHER',

    # Question Words
    'WHO', 'WHAT1', 'WHEN', 'WHERE', 'WHY', 'HOW1',

    # Common Actions & Verbs
    'EAT1', 'DRINK1', 'GO', 'COME', 'WALK2', 'RUN1',  'SEE',
    'LOOKAT', 'HEAR2', 'LISTEN', 'WANT1',  'LOVE', 'HATE',
    'FEEL', 'MAKE', 'WORK', 'PLAY', 'HELP', 'GIVE', 'GET', 'TELL',
    'ASK', 'KNOW', 'THINK', 'REMEMBER1', 'LEARN', 'UNDERSTAND',

    # Feelings & Descriptions
    'PLEASE', 'SAD', 'ANGRY', 'TIRED', 'HUNGRY', 'SICK', 'SCARED', 'SURPRISE',
    'FUNNY', 'SERIOUS', 'RIGHT1', 'WRONG', 'TRUE', 'BIG', 'SMALL', 'TALL1',
    'PRETTY', 'CUTE1', 'UGLY', 'HOT', 'COLD', 'WARM', 'EASY', 'HARD', 'NEW',
    'OLD', 'NICE', 'DIRTY',

    # Places & Time
    'HOME',  'CITY1', 'ROOM', 'KITCHEN', 'BATHROOM', 'SHOP1', 'LIBRARY',
    'YOU', 'DAY', 'NIGHT1', 'MORNING', 'WEEK', 'MONTH', 'YEAR',
    'TODAY', 'YESTERDAY', 'TOMORROW',

    # Common Objects & Concepts
    'WATER', 'APPLE', 'BREAD', 'MILK1', 'COFFEE', 'CAR', 'PHONE', 'PUSH',
    'TV', 'BOOK', 'PAPER', 'MONEY1', 'KEY', 'DOOR', 'WINDOW', 'CHAIR',
    'TABLE', 'COLOR', 'NAME', 'IDEA', 'STORY1', 'JOKE', 'MUSIC', 'GAME',

    # Colors & Connectors
    'RED', 'GREEN', 'YELLOW', 'BLACK', 'WHITE', 'IN', 'ON', 'AT',
    'WITH', 'FOR', 'FROM', 'ABOUT2', 'AND', 'BUT', 'BECAUSE', 'PUSH',
]

CHECKPOINT_PATH = "./checkpoints/best_model_28_11.pth"
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
