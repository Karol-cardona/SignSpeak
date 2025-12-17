from collections import deque, Counter
import numpy as np
import torch
import re
from .llm_handler import AdvancedSentenceCorrector
from .dependencies import TARGET_WORDS, INPUT_DIM

# --- PARAMETRI VELOCITÀ ---
NUM_FRAMES = 60
VOTING_WINDOW_SIZE = 10
VOTE_THRESHOLD = 7          # Soglia di stabilità
CONFIDENCE_THRESHOLD = 0.8 # Soglia di sicurezza (leggermente più permissiva per la velocità)
DETECTION_COOLDOWN = 10     # 0.5 secondi di pausa (MOLTO PIÙ VELOCE)

def clean_word_display(word):
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

class UserSession:
    def __init__(self):
        self.llm_handler = AdvancedSentenceCorrector()
        self.reset()

    def reset(self):
        self.landmark_buffer = []
        self.voting_deque = deque(maxlen=VOTING_WINDOW_SIZE)
        self.current_sentence_words = []
        self.last_confirmed_word = ""
        self.cooldown_counter = 0

    def process_frame(self, frame_numpy: np.ndarray, model, device):
        if not hasattr(self, 'cooldown_counter'):
            self.cooldown_counter = 0

        # Fix Zero Vector (Mani Assenti)
        if np.sum(np.abs(frame_numpy)) < 1e-6:
            self.voting_deque.clear()
            return None

        # GESTIONE COOLDOWN
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            # Importante: durante il cooldown NON aggiungiamo dati al buffer
            # Così evitiamo di "sporcare" la memoria col movimento di ritorno
            return None

        # BUFFERING
        self.landmark_buffer.append(frame_numpy)
        if len(self.landmark_buffer) > NUM_FRAMES:
            self.landmark_buffer.pop(0)

        if len(self.landmark_buffer) < NUM_FRAMES:
            return None

        # PREDIZIONE
        np_buffer = np.array(self.landmark_buffer)
        norm_buffer = normalize_landmarks(np_buffer)
        features = torch.tensor(norm_buffer, dtype=torch.float32).unsqueeze(0).to(device)

        raw_word = "UNCERTAIN"
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

            if confidence.item() > CONFIDENCE_THRESHOLD:
                raw_word = TARGET_WORDS[predicted_idx.item()]

        # VOTING
        self.voting_deque.append(raw_word)

        if len(self.voting_deque) == VOTING_WINDOW_SIZE:
            vote_counts = Counter(self.voting_deque)
            top_word, count = vote_counts.most_common(1)[0]

            if top_word != "UNCERTAIN" and count >= VOTE_THRESHOLD:
                if top_word != self.last_confirmed_word:

                    # PUSH
                    if top_word == 'PUSH':
                        translation = self.llm_handler.correct_sentence(self.current_sentence_words)
                        self.current_sentence_words = []
                        self.voting_deque.clear()
                        self.last_confirmed_word = ""
                        self.cooldown_counter = DETECTION_COOLDOWN
                        self.landmark_buffer = [] # Reset buffer movimento

                        return {
                            "status": "end_of_sentence",
                            "sentence": translation,
                            "prediction": "PUSH"
                        }

                    # PAROLA NORMALE
                    else:
                        clean = clean_word_display(top_word)
                        self.current_sentence_words.append(clean)
                        self.last_confirmed_word = top_word

                        self.voting_deque.clear()
                        self.cooldown_counter = DETECTION_COOLDOWN

                        return {
                            "status": "word_added",
                            "current_words": list(self.current_sentence_words),
                            "prediction": clean
                        }

        return None