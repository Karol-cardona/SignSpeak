import torch
from fastapi import FastAPI, HTTPException
from typing import List

# Import our new schemas and the updated functions
from .schemas import FrameData
from .preprocessing import transform_frames_to_numpy, get_stable_prediction_from_sequence

# Import existing dependencies
from .dependencies import get_model, get_target_words
from .llm_handler import SentenceGenerator

app = FastAPI(title="Sign Language API with Landmark Input and Smoothing")

# --- Load Models ---
sign_model, device = get_model()
target_words = get_target_words()
llm_handler = SentenceGenerator()

# --- In-memory buffer for sentence building ---
word_buffer = []


@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "message": "API is running!"}


@app.post("/predict_landmarks", summary="Predict Sign from a Landmark Sequence with Smoothing")
async def predict_landmarks(frames: List[FrameData]):
    """
    Receives a sequence of landmark data, uses smoothing to get a stable prediction,
    and adds the resulting word to the sentence buffer.
    """
    global word_buffer

    # Step 1: Transform the incoming JSON data into a NumPy array
    landmark_sequence = transform_frames_to_numpy(frames)

    if landmark_sequence is None or landmark_sequence.shape[0] == 0:
        raise HTTPException(
            status_code=400, detail="Invalid or empty landmark data provided.")

    # --- Step 2: Get the stable prediction using our new smoothing function ---
    stable_prediction = get_stable_prediction_from_sequence(
        landmark_sequence=landmark_sequence,
        model=sign_model,
        device=device,
        target_words=target_words,
        model_expected_frames=60,  # The size your model expects
        num_augmentations=10      # How many "votes" to generate. Can be tuned.
    )

    if stable_prediction is None:
        raise HTTPException(
            status_code=400, detail="Could not determine a stable prediction from the provided sequence.")

    # Step 3: Apply the sentence-building logic (this part remains the same)
    if stable_prediction == 'PUSH':
        if not word_buffer:
            return {"prediction": "PUSH", "status": "end_of_sentence", "sentence": ""}

        final_sentence = llm_handler.correct_sentence(word_buffer)
        word_buffer = []

        return {
            "prediction": "PUSH",
            "status": "end_of_sentence",
            "sentence": final_sentence
        }
    else:
        word_buffer.append(stable_prediction)
        return {
            "prediction": stable_prediction,
            "status": "word_added",
            "current_words": word_buffer
        }
