# --- START OF FILE session_state.py ---
from collections import deque, Counter
import numpy as np
import torch
import re
from .llm_handler import AdvancedSentenceCorrector
# IMPORT TARGET_WORDS FROM DEPENDENCIES TO ENSURE CONSISTENCY
from .dependencies import TARGET_WORDS


NUM_FRAMES = 60
VOTING_WINDOW_SIZE = 22
VOTE_THRESHOLD = 16
CONFIDENCE_THRESHOLD = 0.75


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
        # Buffer to hold raw landmarks for the model (needs 60 frames)
        self.landmark_buffer = []

        # Deque for the voting mechanism (needs 22 predictions)
        self.voting_deque = deque(maxlen=VOTING_WINDOW_SIZE)

        # State variables
        self.current_sentence_words = []
        self.last_confirmed_word = ""
        self.final_llm_sentence = ""
        self.status_message = "Listening..."

    def process_new_landmarks(self, new_landmarks_np, model, device, llm_handler):
        """
        Ingests a single frame of landmarks, runs prediction, voting, 
        and updates the sentence state.
        """
        # 1. Update Landmark Buffer
        self.landmark_buffer.append(new_landmarks_np)

        # Keep buffer at exact size (NUM_FRAMES)
        if len(self.landmark_buffer) > NUM_FRAMES:
            self.landmark_buffer.pop(0)

        # We can only predict if we have enough frames
        if len(self.landmark_buffer) == NUM_FRAMES:
            # Prepare tensor
            landmarks_arr = np.array(self.landmark_buffer)
            normalized = normalize_landmarks(landmarks_arr)
            features = torch.tensor(
                normalized, dtype=torch.float32).unsqueeze(0).to(device)

            # Run Model
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)

                raw_word = TARGET_WORDS[predicted_idx.item()]
                conf_val = confidence.item()

            # Add to Voting Deque
            if conf_val > CONFIDENCE_THRESHOLD:
                self.voting_deque.append(raw_word)
            else:
                self.voting_deque.append("UNCERTAIN")

            # Check Voting Consensus
            if len(self.voting_deque) == VOTING_WINDOW_SIZE:
                vote_counts = Counter(self.voting_deque)
                top_word, count = vote_counts.most_common(1)[0]

                if top_word != "UNCERTAIN" and count >= VOTE_THRESHOLD:
                    # New stable word detected?
                    if top_word != self.last_confirmed_word:

                        if top_word == 'PUSH':
                            self.status_message = "Translating..."
                            # Trigger LLM
                            self.final_llm_sentence = llm_handler.correct_sentence(
                                self.current_sentence_words)

                            # Reset State
                            self.current_sentence_words = []
                            self.voting_deque.clear()
                            self.last_confirmed_word = ""
                            return {"event": "SENTENCE_COMPLETED", "payload": self.final_llm_sentence}

                        else:
                            clean_word = clean_word_display(top_word)
                            self.current_sentence_words.append(clean_word)
                            self.last_confirmed_word = top_word
                            self.status_message = f"Added: {clean_word}"
                            return {"event": "WORD_ADDED", "payload": clean_word}

        return {"event": "PROCESSING", "payload": None}
