# --- START OF FILE main.py ---
import torch
from fastapi import FastAPI, HTTPException
from typing import Dict

from .schemas import PredictRequest
from .preprocessing import transform_frame_to_numpy
from .dependencies import get_model
from .llm_handler import AdvancedSentenceCorrector
from .session_state import UserSession

app = FastAPI(title="Real-Time Sign Language API with Voting Logic")

# --- Load Resources ---
sign_model, device = get_model()
llm_handler = AdvancedSentenceCorrector()

# --- Global Session Storage ---
# Key: session_id (str), Value: UserSession object
# In production, use Redis. For this demo, memory is fine.
sessions: Dict[str, UserSession] = {}


@app.get("/")
def read_root():
    return {"status": "ok", "message": "SignSpeak API is running"}


@app.post("/process_frames", summary="Process a batch of frames for a user session")
async def process_frames(request: PredictRequest):
    """
    Receives a batch of frames (e.g., last 100ms of video).
    Feeds them sequentially into the user's session logic (mimicking realtime_test.py).
    Returns the current state of the sentence.
    """
    session_id = request.session_id

    # 1. Retrieve or Create Session
    if session_id not in sessions:
        sessions[session_id] = UserSession()
        print(f"Created new session: {session_id}")

    session = sessions[session_id]

    response_data = {
        "session_id": session_id,
        "new_word": None,
        "final_sentence": session.final_llm_sentence,
        "current_builder": session.current_sentence_words,
        "status": "processing"
    }

    # 2. Iterate through EVERY frame in the batch
    # This ensures we don't skip the "sliding window" logic
    for frame_data in request.frames:

        # Convert JSON -> Numpy
        landmark_np = transform_frame_to_numpy(frame_data)

        # Process logic
        result = session.process_new_landmarks(
            landmark_np,
            sign_model,
            device,
            llm_handler
        )

        # Update response if something interesting happened
        if result["event"] == "WORD_ADDED":
            response_data["new_word"] = result["payload"]
            response_data["status"] = "word_added"
            # We don't break; we keep processing remaining frames in case PUSH follows immediately

        elif result["event"] == "SENTENCE_COMPLETED":
            response_data["final_sentence"] = result["payload"]
            response_data["status"] = "sentence_completed"
            response_data["current_builder"] = []  # Reset in response too

    # 3. Return the latest state
    # The app should display 'current_builder' and if 'final_sentence' is not empty, show that.
    return response_data


@app.post("/reset_session/{session_id}")
def reset_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "reset", "message": "Session memory cleared"}
    return {"status": "error", "message": "Session not found"}
