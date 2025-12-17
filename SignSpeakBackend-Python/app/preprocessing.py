# # --- START OF FILE app/preprocessing.py ---
# import numpy as np
# from typing import List
# from .schemas import FrameData


# def transform_frame_to_numpy(frame: FrameData) -> np.ndarray:
#     """
#     Transforms a SINGLE FrameData object into a flattened (126,) numpy array.
#     Used to feed the UserSession buffer one frame at a time.
#     """
#     # 1. Initialize empty array (2 hands * 21 landmarks * 3 coords)
#     frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)

#     # 2. Safety check: if no landmarks at all, return zeros
#     if not frame.landmarks:
#         return frame_landmarks.flatten()

#     # 3. Iterate through the hands provided in the frame
#     for i, hand_landmarks_list in enumerate(frame.landmarks):

#         # Check if handedness info exists for this index
#         if not frame.handedness or i >= len(frame.handedness):
#             continue

#         # --- FIX IS HERE: Check if the specific handedness list is empty ---
#         current_hand_info_list = frame.handedness[i]
#         if not current_hand_info_list:
#             # If there is no info for this hand (empty list), skip it
#             continue

#         hand_info = current_hand_info_list[0]
#         hand_category = hand_info.categoryName  # 'Left' or 'Right'

#         # Map Right->0, Left->1 (Matches training data)
#         hand_id = 0 if hand_category == 'Right' else 1

#         # Ensure we have exactly 21 landmarks before processing
#         if len(hand_landmarks_list) != 21:
#             continue

#         for lm_idx, landmark_point in enumerate(hand_landmarks_list):
#             frame_landmarks[hand_id, lm_idx, :] = [
#                 landmark_point.x, landmark_point.y, landmark_point.z
#             ]

#     return frame_landmarks.flatten()
import numpy as np


def transform_frame_to_numpy(frame_data):
    """
    Converts the Pydantic Object input from FastAPI into the exact Numpy format
    expected by the model.
    """
    # 1. Initialize empty container: 2 hands, 21 points, 3 coordinates (x,y,z)
    current_landmarks = np.zeros((2, 21, 3))

    # 2. Access data using DOT notation
    raw_landmarks_groups = frame_data.landmarks
    raw_handedness_groups = frame_data.handedness

    # 3. Iterate through detected hands
    for i in range(len(raw_landmarks_groups)):

        if i >= len(raw_handedness_groups):
            break

        hand_points = raw_landmarks_groups[i]
        hand_info_list = raw_handedness_groups[i]

        if not hand_info_list:
            continue

        # Get label sent from Frontend ("Right" or "Left")
        frontend_label = hand_info_list[0].categoryName

        # --- REVERTED LOGIC (THE FIX) ---
        # In app_ui.py: hand_id = 0 if handedness == 'Right' else 1
        # We must match that exactly.
        if frontend_label == 'Right':
            hand_idx = 0  # Right Hand -> Index 0
        else:
            hand_idx = 1  # Left Hand -> Index 1
        # --------------------------------

        # Fill coordinates
        for lm_idx, point in enumerate(hand_points):

            # --- MIRROR FIX ---
            # app_ui.py uses cv2.flip(img, 1), which flips the X axis.
            # Your Frontend sends raw X, so we MUST flip it here to match the model.
            current_landmarks[hand_idx, lm_idx, 0] = 1.0 - point.x
            # ------------------

            current_landmarks[hand_idx, lm_idx, 1] = point.y
            current_landmarks[hand_idx, lm_idx, 2] = point.z

    # 4. Flatten the result to shape (126,)
    return current_landmarks.flatten()
