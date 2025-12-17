# --- START OF FILE test_client.py ---
import requests
import json
import random
import time
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8000"
# MEETING ID è cruciale perché il backend usa active_sessions[meetingId]
MEETING_ID = f"room_{random.randint(100, 999)}"
USER_ID = f"user_{random.randint(1000, 9999)}"

def create_dummy_hand_frame(variation=0.0, timestamp_offset=0):
    """
    Crea un frame JSON che rispetta ESATTAMENTE lo schema Pydantic del backend.
    """
    # Landmarks fittizi (mano destra aperta)
    right_hand_landmarks = []
    for i in range(21):
        point = {
            "x": 0.5 + (random.uniform(-0.01, 0.01) * variation),
            "y": 0.5 + (random.uniform(-0.01, 0.01) * variation),
            "z": 0.0,
            "visibility": 1.0 # Importante aggiungerlo se definito nel modello
        }
        right_hand_landmarks.append(point)

    # Info mano
    hand_info = {
        "score": 0.98,
        "index": 1,
        "categoryName": "Right",
        "displayName": "Right"
    }

    # FrameData Structure
    frame = {
        "timestamp": time.time() * 1000 + timestamp_offset,
        "landmarks": [right_hand_landmarks], # Lista di liste di landmarks (una per mano)
        "handedness": [[hand_info]],         # Lista di liste di info (corrispondente ai landmarks)
        "userInfo": {
            "meetingId": MEETING_ID,
            "userStatus": "talking"
        }
    }
    return frame

def send_batch(frames):
    """
    Invia una lista di frame all'endpoint /api/predict_landmarks
    """
    url = f"{BASE_URL}/api/predict_landmarks"

    # Il body della richiesta è direttamente la lista di frame
    payload = frames

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ CRITICAL: Could not connect to server.")
        print("   Make sure the Docker container is running or 'uvicorn app.main:app' is active.")
        sys.exit(1)

def run_simulation():
    print(f"--- Starting API Simulation ---")
    print(f"--- Meeting ID: {MEETING_ID} ---")

    # 1. Warmup (Riempimento Buffer)
    print("\nPhase 1: Warming up buffers (60 frames)...")
    print("Sending noisy frames (model will likely predict UNCERTAIN)...")

    warmup_frames = [create_dummy_hand_frame(variation=0.05, timestamp_offset=i*33) for i in range(60)]

    # Inviamo in batch da 10
    for i in range(0, 60, 10):
        batch = warmup_frames[i:i+10]
        resp = send_batch(batch)
        if resp and resp['results']:
            print(f"   Batch {i//10 + 1}: Received events: {resp['results']}")
        else:
            print(f"   Batch {i//10 + 1}: No events (Processing...)")
        time.sleep(0.1)

    # 2. Static Sign (Simulazione parola stabile)
    print("\nPhase 2: Simulating a Static Sign (Triggering Voting)...")
    print("   Sending 30 identical frames...")

    static_pose = create_dummy_hand_frame(variation=0.0) # Perfettamente immobile
    static_batch = [static_pose for _ in range(30)]

    for i in range(0, 30, 5):
        batch = static_batch[i:i+5]
        resp = send_batch(batch)

        if resp and resp['results']:
            for res in resp['results']:
                if res['status'] == 'word_added':
                    print(f"   ✅ SUCCESS! Server added word: {res['prediction']}")
                    print(f"   📋 Current Sentence: {res['current_words']}")
                elif res['status'] == 'end_of_sentence':
                    print(f"   🎉 SENTENCE COMPLETE: {res['sentence']}")
        else:
            print(f"   ... buffering ...")

        time.sleep(0.1)

    print("\n--- Simulation Complete ---")
    print("Check Docker logs to see the server-side processing.")

if __name__ == "__main__":
    run_simulation()