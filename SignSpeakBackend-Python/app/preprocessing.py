import numpy as np
import torch
from typing import List
from .schemas import FrameData


def normalize_landmarks(landmarks):
    """
    Normalizes landmarks to be relative to the wrist and scaled by hand size.
    (This function remains the same)
    """
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


def transform_frames_to_numpy(frames: List[FrameData]) -> np.ndarray | None:
    """
    Transforms a list of FrameData objects into a NumPy array.
    (This function remains the same)
    """
    if not frames:
        return None
    processed_frames = []
    for frame in frames:
        frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)
        if not frame.landmarks or not frame.handedness:
            processed_frames.append(frame_landmarks.flatten())
            continue
        for i, hand_landmarks_list in enumerate(frame.landmarks):
            if i >= len(frame.handedness) or not frame.handedness[i]:
                continue
            hand_info = frame.handedness[i][0]
            hand_category = hand_info.categoryName
            hand_id = 0 if hand_category == 'Right' else 1
            if len(hand_landmarks_list) != 21:
                continue
            for lm_idx, landmark_point in enumerate(hand_landmarks_list):
                frame_landmarks[hand_id, lm_idx, :] = [
                    landmark_point.x, landmark_point.y, landmark_point.z]
        processed_frames.append(frame_landmarks.flatten())
    return np.array(processed_frames, dtype=np.float32)

# --- NEW: The Core Function for Sampling and Smoothing ---


def get_stable_prediction_from_sequence(
    landmark_sequence: np.ndarray,
    model,
    device,
    target_words: list,
    model_expected_frames: int = 60,
    num_augmentations: int = 10  # Number of subsequences to vote on
) -> str | None:
    """
    Applies smoothing via Test-Time Augmentation (TTA) to get a stable prediction.

    1. Generates multiple, slightly different subsequences.
    2. Gets a prediction for each one.
    3. Returns the majority vote.
    """
    num_frames_from_input = landmark_sequence.shape[0]

    # If the input is too short, we can't do much. Pad and make one prediction.
    if num_frames_from_input < model_expected_frames:
        padding = np.zeros(
            (model_expected_frames - num_frames_from_input, 126))
        final_sequence = np.vstack((landmark_sequence, padding))
        features = torch.tensor(normalize_landmarks(
            final_sequence), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(features)
            _, predicted_idx = torch.max(outputs, 1)
            return target_words[predicted_idx.item()]

    # --- Main TTA Logic ---
    predictions = []

    # We will generate `num_augmentations` different views of the data
    for i in range(num_augmentations):
        # Create a slightly different starting point for each subsequence
        # This creates variation in the sampled frames
        start_frame = i

        # Ensure the start_frame doesn't make the sequence too short
        if (num_frames_from_input - start_frame) < model_expected_frames:
            break

        # 1. Use linspace to sample frames, just like in training
        frame_indices = np.linspace(
            start_frame,
            num_frames_from_input - 1,
            model_expected_frames,
            dtype=int
        )

        subsequence = landmark_sequence[frame_indices]

        # 2. Normalize and predict for this subsequence
        normalized_landmarks = normalize_landmarks(subsequence)
        features = torch.tensor(normalized_landmarks,
                                dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(features)
            _, predicted_idx = torch.max(outputs, 1)
            predictions.append(target_words[predicted_idx.item()])

    # 3. Take the majority vote
    if not predictions:
        return None

    most_common_prediction = max(set(predictions), key=predictions.count)
    return most_common_prediction
