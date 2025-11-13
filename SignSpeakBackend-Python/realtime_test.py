import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from app.model import SignLanguageTransformer
from app.llm_handler import SentenceGenerator  # Import the new class

# --- 1. Configuration ---
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
NUM_CLASSES = 10
DROPOUT = 0.3
NUM_FRAMES = 60

# --- Prediction Smoothing Parameters ---
PREDICTION_WINDOW_SIZE = 20
STABILITY_THRESHOLD = 15

# --- 2. Load Models ---
# Sign Language Model
print("--- Loading sign detection model ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sign_model = SignLanguageTransformer(
    INPUT_DIM, EMBED_DIM, NUM_HEADS, NUM_ENCODER_LAYERS, FF_DIM, NUM_CLASSES, DROPOUT).to(device)
sign_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
sign_model.eval()
print("--- Sign detection model loaded successfully ---")

# Sentence Generation LLM
llm_handler = SentenceGenerator()

# Helper function


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


# --- 3. Setup MediaPipe and OpenCV ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- 4. Real-Time Detection Loop ---
landmark_buffer = []
predictions_deque = deque(maxlen=PREDICTION_WINDOW_SIZE)
stable_prediction = ""
previous_stable_prediction = ""

# Variables for sentence construction
current_sentence_words = []
final_sentence = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR_RGB)
    results = hands.process(image_rgb)
    hand_detected = bool(results.multi_hand_landmarks)

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
        # Append zeros if no hand is detected
        landmark_buffer.append(np.zeros(2 * 21 * 3))

    if len(landmark_buffer) > NUM_FRAMES:
        landmark_buffer.pop(0)

    if len(landmark_buffer) == NUM_FRAMES and hand_detected:
        landmarks_to_predict = np.array(landmark_buffer)
        normalized_landmarks = normalize_landmarks(landmarks_to_predict)
        features = torch.tensor(normalized_landmarks,
                                dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = sign_model(features)
            _, predicted_idx = torch.max(outputs, 1)
            raw_prediction = TARGET_WORDS[predicted_idx.item()]
            predictions_deque.append(raw_prediction)

    if len(predictions_deque) == PREDICTION_WINDOW_SIZE:
        most_common_pred = max(set(predictions_deque),
                               key=list(predictions_deque).count)
        if list(predictions_deque).count(most_common_pred) >= STABILITY_THRESHOLD:
            stable_prediction = most_common_pred

    # --- Sentence Logic: Act only when the stable prediction changes ---
    if stable_prediction and stable_prediction != previous_stable_prediction:
        if stable_prediction == 'PUSH':
            if current_sentence_words:
                final_sentence = llm_handler.correct_sentence(
                    current_sentence_words)
                current_sentence_words = []  # Clear for the next sentence
        else:
            current_sentence_words.append(stable_prediction)

        # Update the previous prediction to prevent re-adding the same word
        previous_stable_prediction = stable_prediction

    if not hand_detected:
        previous_stable_prediction = ""  # Reset to allow re-detection of the same word

    # --- Display Information ---
    # Display current words being built
    cv2.putText(frame, " ".join(current_sentence_words), (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # Display the final, corrected sentence
    cv2.putText(frame, final_sentence, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
