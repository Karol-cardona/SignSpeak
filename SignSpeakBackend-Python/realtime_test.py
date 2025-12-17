# --- START OF FILE realtime_test.py ---
import sys
import os
import time
import re
from collections import deque, Counter

# --- 1. SETUP AMBIENTE ---
# Aggiunge la directory corrente al path per trovare il package 'app'
sys.path.append(os.getcwd())

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print(">>> Warning: python-dotenv not installed.")

# Check OpenCV per GUI
try:
    import cv2
except ImportError:
    print("❌ ERRORE: Per il test locale serve 'opencv-python' standard.")
    print("👉 Esegui: pip install opencv-python")
    sys.exit(1)

import numpy as np
import torch
import mediapipe as mp

# --- CUSTOM IMPORTS ---
# Assicurati che app/ esista e contenga __init__.py
try:
    from app.llm_handler import AdvancedSentenceCorrector
    from app.model import SignLanguageTransformer
    from app.dependencies import TARGET_WORDS, INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FF_DIM, NUM_CLASSES, DROPOUT
except ImportError as e:
    print(f"❌ Errore importazione moduli 'app': {e}")
    print("Assicurati di eseguire lo script dalla cartella principale del progetto.")
    sys.exit(1)

# --- CONFIGURAZIONE ---
# Percorsi relativi alla root del progetto
CHECKPOINT_PATH = os.path.join("checkpoints", "best_model_28_11.pth")
HF_CACHE_DIR = os.path.join(os.getcwd(), 'hf_cache')
os.environ['HF_HOME'] = HF_CACHE_DIR

# Parametri Logica
NUM_FRAMES = 60 # Allineato con il training (era 45, ma il modello sembra aspettarsene 60 nel session_state)
VOTING_WINDOW_SIZE = 10
VOTE_THRESHOLD = 7
CONFIDENCE_THRESHOLD = 0.75

# --- HELPERS ---
def clean_word_display(word):
    return re.sub(r'\d+', '', word)

def normalize_landmarks(landmarks):
    # Logica identica a session_state.py
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
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

# --- INITIALIZATION ---
print("\n--- 1. Loading PyTorch Model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ Checkpoint non trovato in: {os.path.abspath(CHECKPOINT_PATH)}")
    sys.exit(1)

sign_model = SignLanguageTransformer(
    INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FF_DIM, NUM_CLASSES, DROPOUT
).to(device)

try:
    sign_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    sign_model.eval()
    print(">>> Sign Model Loaded.")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    sys.exit(1)

print("\n--- 2. Loading Gemini/LLM Handler ---")
# Nota: Questo cercherà local_model/qwen... assicurati che la cartella esista
llm_handler = AdvancedSentenceCorrector()

print("\n--- 3. Starting Camera ---")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Webcam 0 not found, trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ CRITICAL ERROR: No webcam found.")
        sys.exit(1)

# --- RUNTIME STATE ---
landmark_buffer = [] # Buffer per il modello PyTorch
voting_deque = deque(maxlen=VOTING_WINDOW_SIZE)

current_sentence_words = []
final_sentence = "Start signing..."
last_confirmed_word = ""

print("\n>>> SYSTEM READY. PRESS 'Q' TO EXIT.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirroring e conversione colore
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    hand_detected = bool(results.multi_hand_landmarks)

    # 1. Raccolta Landmarks
    if hand_detected:
        current_landmarks = np.zeros((2, 21, 3))
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2: continue # Supportiamo max 2 mani

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determina Destra/Sinistra
            # Nota: MediaPipe 'Right' è la mano sinistra nell'immagine specchiata se non gestito,
            # ma qui usiamo la label classification.
            if results.multi_handedness:
                handedness_label = results.multi_handedness[hand_idx].classification[0].label
                # Mapping: Right->0, Left->1 (Come nel training)
                hand_id = 0 if handedness_label == 'Right' else 1
            else:
                hand_id = 0 # Fallback

            for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                current_landmarks[hand_id, lm_idx, :] = [landmark.x, landmark.y, landmark.z]

        landmark_buffer.append(current_landmarks.flatten())
    else:
        # Se non ci sono mani, aggiungiamo zeri per mantenere la continuità temporale
        # (o potresti decidere di non aggiungere nulla, dipendente da come è stato addestrato)
        landmark_buffer.append(np.zeros(2 * 21 * 3))

    # Mantieni dimensione buffer fissa
    if len(landmark_buffer) > NUM_FRAMES:
        landmark_buffer.pop(0)

    # 2. Predizione (Solo se buffer pieno e mani rilevate ora)
    if len(landmark_buffer) == NUM_FRAMES and hand_detected:
        # Prepara tensore
        np_buffer = np.array(landmark_buffer)
        norm_buffer = normalize_landmarks(np_buffer)
        features = torch.tensor(norm_buffer, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = sign_model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            raw_word = TARGET_WORDS[predicted_idx.item()]
            conf_val = confidence.item()

            # Debug visuale a schermo
            cv2.putText(frame, f"Raw: {clean_word_display(raw_word)} ({conf_val:.2f})", (10, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if conf_val > CONFIDENCE_THRESHOLD:
                voting_deque.append(raw_word)
            else:
                voting_deque.append("UNCERTAIN")

    # 3. Logica di Voto
    if len(voting_deque) == VOTING_WINDOW_SIZE:
        vote_counts = Counter(voting_deque)
        top_word, count = vote_counts.most_common(1)[0]

        # Barra progresso stabilità
        progress = min(count / VOTING_WINDOW_SIZE, 1.0)
        color_bar = (0, 255, 0) if count >= VOTE_THRESHOLD else (0, 165, 255) # Verde se stabile, Arancio se in corso
        cv2.rectangle(frame, (10, 130), (10 + int(200 * progress), 140), color_bar, -1)
        cv2.putText(frame, f"Stability: {top_word}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        if top_word != "UNCERTAIN" and count >= VOTE_THRESHOLD:
            if top_word != last_confirmed_word:

                # Caso PUSH -> Trigger LLM
                if top_word == 'PUSH':
                    print(">>> 🤖 PUSH DETECTED. Calling LLM...")
                    final_sentence = "Processing..."
                    cv2.imshow('SignSpeak Voting System', frame)
                    cv2.waitKey(1) # Forza refresh GUI

                    # Chiamata bloccante (per test)
                    translation = llm_handler.correct_sentence(current_sentence_words)
                    final_sentence = translation
                    print(f">>> 🗣️ AI: {final_sentence}")

                    # Reset
                    current_sentence_words = []
                    voting_deque.clear()
                    last_confirmed_word = "" # Reset last word per permettere nuova rilevazione immediata

                else:
                    # Aggiunta parola normale
                    clean = clean_word_display(top_word)
                    current_sentence_words.append(clean)
                    last_confirmed_word = top_word
                    print(f"➕ Added Word: {clean}")

    # 4. Interfaccia Grafica
    # Background scuro in alto
    cv2.rectangle(frame, (0, 0), (640, 90), (30, 30, 30), -1)

    # Costruzione frase corrente
    builder_text = " ".join(current_sentence_words)
    draw_styled_text(frame, f"Gloss: {builder_text}", (10, 30), (0, 255, 255), 0.7)

    # Frase finale AI
    draw_styled_text(frame, f"Trans: {final_sentence}", (10, 75), (50, 255, 50), 0.7)

    cv2.imshow('SignSpeak Voting System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()