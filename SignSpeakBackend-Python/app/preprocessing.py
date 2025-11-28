# --- START OF FILE app/preprocessing.py ---
import numpy as np
from typing import List
from .schemas import FrameData


def transform_frame_to_numpy(frame: FrameData) -> np.ndarray:
    """
    Transforms a SINGLE FrameData object into a flattened (126,) numpy array.
    Used to feed the UserSession buffer one frame at a time.
    """
    # 1. Initialize empty array (2 hands * 21 landmarks * 3 coords)
    frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)

    # 2. Safety check: if no landmarks at all, return zeros
    if not frame.landmarks:
        return frame_landmarks.flatten()

    # 3. Iterate through the hands provided in the frame
    for i, hand_landmarks_list in enumerate(frame.landmarks):

        # Check if handedness info exists for this index
        if not frame.handedness or i >= len(frame.handedness):
            continue

        # --- FIX IS HERE: Check if the specific handedness list is empty ---
        current_hand_info_list = frame.handedness[i]
        if not current_hand_info_list:
            # If there is no info for this hand (empty list), skip it
            continue

        hand_info = current_hand_info_list[0]
        hand_category = hand_info.categoryName  # 'Left' or 'Right'

        # Map Right->0, Left->1 (Matches training data)
        hand_id = 0 if hand_category == 'Right' else 1

        # Ensure we have exactly 21 landmarks before processing
        if len(hand_landmarks_list) != 21:
            continue

        for lm_idx, landmark_point in enumerate(hand_landmarks_list):
            frame_landmarks[hand_id, lm_idx, :] = [
                landmark_point.x, landmark_point.y, landmark_point.z
            ]

    return frame_landmarks.flatten()
