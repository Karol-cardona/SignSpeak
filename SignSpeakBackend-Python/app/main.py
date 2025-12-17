# # --- START OF FILE main.py ---
# import torch
# from fastapi import FastAPI, HTTPException
# from typing import Dict

# from .schemas import PredictRequest
# from .preprocessing import transform_frame_to_numpy
# from .dependencies import get_model
# from .llm_handler import AdvancedSentenceCorrector
# from .session_state import UserSession

# app = FastAPI(title="Real-Time Sign Language API with Voting Logic")

# # --- Load Resources ---
# sign_model, device = get_model()
# llm_handler = AdvancedSentenceCorrector()

# # --- Global Session Storage ---
# # Key: session_id (str), Value: UserSession object
# # In production, use Redis. For this demo, memory is fine.
# sessions: Dict[str, UserSession] = {}


# @app.get("/")
# def read_root():
#     return {"status": "ok", "message": "SignSpeak API is running"}


# @app.post("/process_frames", summary="Process a batch of frames for a user session")
# async def process_frames(request: PredictRequest):
#     """
#     Receives a batch of frames (e.g., last 100ms of video).
#     Feeds them sequentially into the user's session logic (mimicking realtime_test.py).
#     Returns the current state of the sentence.
#     """
#     session_id = request.session_id

#     # 1. Retrieve or Create Session
#     if session_id not in sessions:
#         sessions[session_id] = UserSession()
#         print(f"Created new session: {session_id}")

#     session = sessions[session_id]

#     response_data = {
#         "session_id": session_id,
#         "new_word": None,
#         "final_sentence": session.final_llm_sentence,
#         "current_builder": session.current_sentence_words,
#         "status": "processing"
#     }

#     # 2. Iterate through EVERY frame in the batch
#     # This ensures we don't skip the "sliding window" logic
#     for frame_data in request.frames:

#         # Convert JSON -> Numpy
#         landmark_np = transform_frame_to_numpy(frame_data)

#         # Process logic
#         result = session.process_new_landmarks(
#             landmark_np,
#             sign_model,
#             device,
#             llm_handler
#         )

#         # Update response if something interesting happened
#         if result["event"] == "WORD_ADDED":
#             response_data["new_word"] = result["payload"]
#             response_data["status"] = "word_added"
#             # We don't break; we keep processing remaining frames in case PUSH follows immediately

#         elif result["event"] == "SENTENCE_COMPLETED":
#             response_data["final_sentence"] = result["payload"]
#             response_data["status"] = "sentence_completed"
#             response_data["current_builder"] = []  # Reset in response too

#     # 3. Return the latest state
#     # The app should display 'current_builder' and if 'final_sentence' is not empty, show that.
#     return response_data


# @app.post("/reset_session/{session_id}")
# def reset_session(session_id: str):
#     if session_id in sessions:
#         del sessions[session_id]
#         return {"status": "reset", "message": "Session memory cleared"}
#     return {"status": "error", "message": "Session not found"}
# --- START OF FILE main.py ---
# --- START OF FILE main.py ---
import torch
from fastapi import FastAPI, Request
from typing import Dict, List, Any

from .schemas import PredictRequest, FrameData
from .preprocessing import transform_frame_to_numpy
from .dependencies import get_model
from .llm_handler import AdvancedSentenceCorrector
from .session_state import UserSession

app = FastAPI(title="Real-Time Sign Language API with Voting Logic")

# --- Load Resources ---
sign_model, device = get_model()
llm_handler = AdvancedSentenceCorrector()

# --- Global Session Storage ---
sessions: Dict[str, UserSession] = {}


@app.get("/")
def read_root():
    return {"status": "ok", "message": "SignSpeak API is running"}


# ==========================================
#      YOUR NEW LOGIC (Internal)
# ==========================================
@app.post("/process_frames")
async def process_frames(request: PredictRequest):
    session_id = request.session_id

    if session_id not in sessions:
        sessions[session_id] = UserSession()

    session = sessions[session_id]

    response_data = {
        "session_id": session_id,
        "new_word": None,
        "final_sentence": session.final_llm_sentence,
        "current_builder": session.current_sentence_words,
        "status": "processing"
    }

    for frame_data in request.frames:
        landmark_np = transform_frame_to_numpy(frame_data)
        result = session.process_new_landmarks(
            landmark_np, sign_model, device, llm_handler)

        if result["event"] == "WORD_ADDED":
            response_data["new_word"] = result["payload"]
            # FIX: Create a COPY of the list so it stays safe even if the session resets
            response_data["current_builder"] = list(
                session.current_sentence_words)
            response_data["status"] = "word_added"
        elif result["event"] == "SENTENCE_COMPLETED":
            response_data["final_sentence"] = result["payload"]
            response_data["status"] = "sentence_completed"
            response_data["current_builder"] = []

    return response_data


@app.post("/reset_session/{session_id}")
def reset_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "reset", "message": "Session memory cleared"}
    return {"status": "error", "message": "Session not found"}


# ==========================================
#   ROBUST JAVA ADAPTER (Fixes 422 Error)
# ==========================================
@app.post("/ml/process")
@app.post("/api/predict_landmarks")
async def legacy_ml_process(request: Request):
    try:
        raw_body = await request.json()

        # 1. Extract Landmarks
        clean_frames = []
        items_to_process = []
        if isinstance(raw_body, dict):
            items_to_process = [raw_body]
        elif isinstance(raw_body, list):
            items_to_process = raw_body

        for item in items_to_process:
            if "landmarks" in item and "handedness" in item:
                clean_frames.append({
                    "landmarks": item["landmarks"],
                    "handedness": item["handedness"]
                })

        if not clean_frames:
            return {"results": []}

        # 2. Run Logic
        new_request = PredictRequest(
            session_id="java_default_session",
            frames=clean_frames
        )

        # Call internal logic
        result = await process_frames(new_request)

        # 3. FORMAT RESPONSE FOR JAVA
        ml_results = []

        # Case A: Word Added (PARTIAL)
        if result.get("new_word"):
            ml_results.append({
                "status": "word_added",
                "prediction": result.get("new_word"),

                # --- CRITICAL FIX: SEND THE LIST OF WORDS ---
                # Java needs this to show "Building: HELLO WORLD"
                "current_words": result.get("current_builder", []),
                # --------------------------------------------

                "sentence": None
            })
            print(f"📤 PARTIAL: {result.get('current_builder')}")

        # Case B: Sentence Completed (FINAL)
        if result.get("final_sentence"):
            ml_results.append({
                "status": "end_of_sentence",
                "prediction": None,
                "current_words": [],
                "sentence": result.get("final_sentence")
            })
            print(f"🚀 FINAL: {result.get('final_sentence')}")

        return {"results": ml_results}

    except Exception as e:
        print(f"Error in adapter: {e}")
        return {"results": []}


@app.post("/reset_buffer")
async def reset_buffer(request: Request):
    """
    Hard Reset: Clears ML buffers and current sentence builder.
    """
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Default to "java_default_session" if no ID provided
        session_id = body.get("session_id") or request.query_params.get(
            "session_id") or "java_default_session"

        # We always want to clear the sentence in a hard reset
        clear_sentences = True

        if session_id in sessions:
            # Call the new reset method we added in Step 1
            sessions[session_id].reset(clear_sentence=clear_sentences)
            print(f"🧹 SESSION RESET: {session_id}")
            return {"status": "ok", "message": "Session wiped"}

        return {"status": "ok", "message": "Session not found (nothing to clear)"}

    except Exception as e:
        print(f"Error in reset_buffer: {e}")
        return {"status": "error", "detail": str(e)}

# Keep the alias for Java


@app.post("/api/reset_buffer")
async def reset_buffer_api(request: Request):
    return await reset_buffer(request)


@app.post("/simulate_push")
async def simulate_push(request: Request):
    """
    Simulate a 'PUSH' prediction for a session: force the LLM to translate the
    current built glosses and clear the session's builder so a new sentence may start.
    Accepts optional JSON: { "session_id": "..." }
    Returns JSON in the MLResponse-like format: { "results": [ { status: 'end_of_sentence', 'sentence': '...' } ] }
    """
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            body = {}

        session_id = body.get("session_id") or request.query_params.get(
            "session_id") or "java_default_session"

        if session_id not in sessions:
            return {"results": []}

        session = sessions[session_id]
        sentence = session.force_complete(llm_handler)

        results = []
        if sentence:
            results.append({
                "prediction": None,
                "status": "end_of_sentence",
                "sentence": sentence
            })

        response = {"results": results}
        if results:
            print(f"📤 SIMULATE_PUSH: {response}")
        return response
    except Exception as e:
        print(f"Error in simulate_push: {e}")
        return {"results": []}


@app.post("/api/simulate_push")
async def simulate_push_api(request: Request):
    return await simulate_push(request)
