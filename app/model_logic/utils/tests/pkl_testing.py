import pickle
import numpy as np

from ....schemas import FrameData, LandmarkPoint, HandednessInfo
from ...PipelineManager import PipelineManager


def load_pkl_as_framedata(pkl_path: str) -> list[FrameData]:
    """
    Loads your recorded .pkl file and converts it into
    a list of FrameData objects compatible with the PipelineManager.

    Used in Development phase - redundant now with API.
    """

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    keypoints = np.array(data["keypoints"])     # (T, 42, 4) or (T, 2, 21, 4)
    T = keypoints.shape[0]

    # Reshape to (T, 2, 21, 4) if needed
    if keypoints.shape[1] == 42:
        keypoints = keypoints.reshape(T, 2, 21, 4)

    timestamps = data.get("timestamps", np.arange(T) * 0.033)

    frames: list[FrameData] = []

    for i in range(T):
        frame_landmarks = keypoints[i]        # (2,21,4)

        # Create landmark objects for BOTH hands
        hands_landmarks = []
        hands_handedness = []

        # Right = index 0, Left = index 1
        for hand_id, hand_points in enumerate(frame_landmarks):
            lm_list = []
            for (x, y, z, vis) in hand_points:
                lm_list.append(LandmarkPoint(
                    x=float(x), y=float(y), z=float(z), visibility=float(vis)
                ))

            hands_landmarks.append(lm_list)

            # Fake handedness (MediaPipe format)
            handedness_label = "Right" if hand_id == 0 else "Left"
            hands_handedness.append([
                HandednessInfo(
                    score=1.0,
                    index=hand_id,
                    categoryName=handedness_label,
                    displayName=handedness_label,
                )
            ])

        # Build FrameData entry
        frames.append(
            FrameData(
                timestamp=float(timestamps[i]),
                sequenceNumber=int(i),
                receivedAt=float(timestamps[i]),
                landmarks=hands_landmarks,
                handedness=hands_handedness,
            )
        )

    return frames


def test_pipeline_on_pkl( pkl_path: str):
    """
    Runs entire pipeline on a saved recording (.pkl)
    and prints the resulting prediction.
    """
    print("Loading pipeline...")
    pipeline = PipelineManager()

    print("Loading frames...")
    frames = load_pkl_as_framedata(pkl_path)

    print(f"Loaded {len(frames)} frames. Running pipeline...")
    result = pipeline.process(frames)

    print("\n=== PIPELINE RESULT ===")
    for r in result:
        print(r)
    return result


# -------------------------
# EXAMPLE USAGE:
# -------------------------
if __name__ == "__main__":
    test_pipeline_on_pkl(
        pkl_path=""
    )
