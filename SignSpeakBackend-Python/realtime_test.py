# --- START OF FILE realtime_test.py ---

import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque, Counter
import time
import re

# --- CUSTOM IMPORTS ---
from app.llm_handler import AdvancedSentenceCorrector
from app.model import SignLanguageTransformer

# --- SETUP ENV ---
# Saves cache inside the project folder or Docker container
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')

# --- CONFIGURATION ---
CHECKPOINT_PATH = r"checkpoints\best_model_28_11.pth"

# Data & Model Parameters
TARGET_WORDS = [
    # Core Conversation & Greetings
    'HELLO', 'YES', 'NO', 'PLEASE', 'THANKYOU', 'SORRY',
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
    'HAPPY', 'SAD', 'ANGRY', 'TIRED', 'HUNGRY', 'SICK', 'SCARED', 'SURPRISE',
    'FUNNY', 'SERIOUS', 'RIGHT1', 'WRONG', 'TRUE', 'BIG', 'SMALL', 'TALL1',
    'PRETTY', 'CUTE1', 'UGLY', 'HOT', 'COLD', 'WARM', 'EASY', 'HARD', 'NEW',
    'OLD', 'CLEAN', 'DIRTY',

    # Places & Time
    'HOME',  'CITY1', 'ROOM', 'KITCHEN', 'BATHROOM', 'SHOP1', 'LIBRARY',
    'TIME', 'DAY', 'NIGHT1', 'MORNING', 'WEEK', 'MONTH', 'YEAR',
    'TODAY', 'YESTERDAY', 'TOMORROW',

    # Common Objects & Concepts
    'WATER', 'APPLE', 'BREAD', 'MILK1', 'COFFEE', 'CAR', 'PHONE', 'COMPUTER',
    'TV', 'BOOK', 'PAPER', 'MONEY1', 'KEY', 'DOOR', 'WINDOW', 'CHAIR',
    'TABLE', 'COLOR', 'NAME', 'IDEA', 'STORY1', 'JOKE', 'MUSIC', 'GAME',

    # Colors & Connectors
    'RED', 'GREEN', 'YELLOW', 'BLACK', 'WHITE', 'IN', 'ON', 'AT',
    'WITH', 'FOR', 'FROM', 'ABOUT2', 'AND', 'BUT', 'BECAUSE', 'PUSH',
]

# Model Parameters
INPUT_DIM = 2 * 21 * 3
EMBED_DIM = 256
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
FF_DIM = 512
NUM_CLASSES = len(TARGET_WORDS)
DROPOUT = 0.3
NUM_FRAMES = 60

# --- VOTING & ACCURACY SETTINGS ---
# We store the last 22 predictions.
# To accept a word, it must appear at least 16 times in those 22 frames.
VOTING_WINDOW_SIZE = 22
VOTE_THRESHOLD = 16
CONFIDENCE_THRESHOLD = 0.75

# --- HELPERS ---


def clean_word_display(word):
    """Removes numbers for display (HOW1 -> HOW)."""
    return re.sub(r'\d+', '', word)


def normalize_landmarks(landmarks):
    landmarks = landmarks.reshape(-1, 2, 21, 3)
    normalized_landmarks = np.zeros_like(landmarks)
    for frame_idx in range(landmarks.shape[0]):
        for hand_idx in range(landmarks.shape[1]):
            hand = landmarks[frame_idx, hand_idx]
            if hand.any():
                wrist = hand[0]
                centered_hand = hand - wrist
                scale = np.linalg.norm(centered_hand[9])
                if scale > 1e-6:
                    centered_hand /= scale
                normalized_landmarks[frame_idx, hand_idx] = centered_hand
    return normalized_landmarks.reshape(-1, 2 * 21 * 3)


def draw_styled_text(img, text, pos, color=(255, 255, 255), scale=0.8):
    """Draws text with a nice shadow effect."""
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


# --- INITIALIZATION ---
print("\n--- 1. Loading Sign Language Model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sign_model = SignLanguageTransformer(
    INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FF_DIM, NUM_CLASSES, DROPOUT).to(device)

try:
    sign_model.load_state_dict(torch.load(
        CHECKPOINT_PATH, map_location=device))
    sign_model.eval()
    print(">>> Sign Model Loaded.")
except Exception as e:
    print(f"ERROR: Could not load checkpoint at {CHECKPOINT_PATH}")
    exit()

print("\n--- 2. Loading AI Humanizer ---")
llm_handler = AdvancedSentenceCorrector()
print(">>> AI Loaded.")

print("\n--- 3. Starting Camera ---")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("CRITICAL ERROR: No webcam found.")
        exit()

# --- RUNTIME VARIABLES ---
landmark_buffer = []
voting_deque = deque(maxlen=VOTING_WINDOW_SIZE)

current_sentence_words = []
final_sentence = "Start signing..."
last_confirmed_word = ""

print("\n>>> SYSTEM READY. PRESS 'Q' TO EXIT.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_detected = bool(results.multi_hand_landmarks)

    # 1. Collect Landmarks
    if hand_detected:
        current_landmarks = np.zeros((2, 21, 3))
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                continue
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = results.multi_handedness[hand_idx].classification[0].label
            hand_id = 0 if handedness == 'Right' else 1
            for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                current_landmarks[hand_id, lm_idx, :] = [
                    landmark.x, landmark.y, landmark.z]
        landmark_buffer.append(current_landmarks.flatten())
    else:
        landmark_buffer.append(np.zeros(2 * 21 * 3))

    if len(landmark_buffer) > NUM_FRAMES:
        landmark_buffer.pop(0)

    # 2. Prediction & Voting
    if len(landmark_buffer) == NUM_FRAMES and hand_detected:
        features = torch.tensor(normalize_landmarks(
            np.array(landmark_buffer)), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = sign_model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            raw_word = TARGET_WORDS[predicted_idx.item()]
            conf_val = confidence.item()

            if conf_val > CONFIDENCE_THRESHOLD:
                voting_deque.append(raw_word)
            else:
                voting_deque.append("UNCERTAIN")

            cv2.putText(frame, f"Scanning: {clean_word_display(raw_word)} {conf_val:.0%}", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 3. Process Votes
    if len(voting_deque) == VOTING_WINDOW_SIZE:
        vote_counts = Counter(voting_deque)
        top_word, count = vote_counts.most_common(1)[0]

        if top_word != "UNCERTAIN" and count >= VOTE_THRESHOLD:

            # Progress bar green
            progress = min(count / VOTING_WINDOW_SIZE, 1.0)
            bar_width = int(200 * progress)
            cv2.rectangle(frame, (10, 130),
                          (10 + bar_width, 140), (0, 255, 0), -1)

            if top_word != last_confirmed_word:

                if top_word == 'PUSH':
                    print(">>> PUSH DETECTED. TRIGGERING AI...")
                    final_sentence = "Translating..."

                    # Call LLM
                    final_sentence = llm_handler.correct_sentence(
                        current_sentence_words)
                    print(f">>> AI: {final_sentence}")

                    current_sentence_words = []
                    voting_deque.clear()
                    last_confirmed_word = ""
                else:
                    # Clean the word BEFORE adding it to the list (HOW1 -> HOW)
                    clean_word = clean_word_display(top_word)
                    current_sentence_words.append(clean_word)
                    last_confirmed_word = top_word
                    print(f"Added: {clean_word}")

    # 4. Interface
    cv2.rectangle(frame, (0, 0), (640, 85), (40, 40, 40), -1)

    # Show clean words being built
    builder_text = " > ".join(current_sentence_words)
    draw_styled_text(
        frame, f"Building: {builder_text}", (10, 30), (0, 255, 255), 0.7)

    # Show Final AI sentence
    draw_styled_text(
        frame, f"AI Voice: {final_sentence}", (10, 70), (0, 255, 0), 0.8)

    cv2.imshow('SignSpeak Voting System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
