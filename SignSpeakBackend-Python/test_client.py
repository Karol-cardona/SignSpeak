# --- START OF FILE test_client.py ---
import requests
import json
import random
import time
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = f"tester_{random.randint(1000, 9999)}"


def create_dummy_hand_frame(variation=0.0):
    """
    Creates a fake JSON frame that looks like a hand to the API.
    We add slight 'variation' to simulate movement noise.
    """
    # Create 21 points for a "Right Hand"
    # We just make up coordinates. The model might predict "UNCERTAIN" or a random word,
    # but the API LOGIC (voting/buffering) will still work.
    right_hand_landmarks = []
    for i in range(21):
        point = {
            "x": 0.5 + (random.uniform(-0.01, 0.01) * variation),
            "y": 0.5 + (random.uniform(-0.01, 0.01) * variation),
            "z": 0.0
        }
        right_hand_landmarks.append(point)

    # Empty Left Hand
    left_hand_landmarks = []

    # Construct the FrameData schema
    frame = {
        "landmarks": [right_hand_landmarks, left_hand_landmarks],
        "handedness": [
            [{"score": 0.95, "index": 1, "categoryName": "Right", "displayName": "Right"}],
            []  # No left hand
        ]
    }
    return frame


def send_batch(frames):
    """Sends a batch of frames to the API."""
    url = f"{BASE_URL}/process_frames"
    payload = {
        "session_id": SESSION_ID,
        "frames": frames
    }

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ CRITICAL: Could not connect to server.")
        print("   Did you run 'uvicorn app.main:app'?")
        sys.exit(1)


def run_simulation():
    print(f"--- Starting Simulation for Session: {SESSION_ID} ---")

    # 1. Simulate "Waiting" (Sending random noise/empty frames)
    # This fills the buffer so the model is ready
    print("\nPhase 1: Warming up buffers (60 frames)...")
    warmup_frames = [create_dummy_hand_frame(variation=0.1) for _ in range(60)]
    # Send in batches of 10
    for i in range(0, 60, 10):
        batch = warmup_frames[i:i+10]
        resp = send_batch(batch)
        print(f"   Batch {i//10 + 1}: Status = {resp['status']}")

    # 2. Simulate holding a specific sign (e.g., 'HELLO')
    # We send IDENTICAL frames to force the Voting Mechanism to agree
    print("\nPhase 2: Simulating a Static Sign (Triggering Voting)...")
    print("   Sending 30 identical frames...")

    static_pose = create_dummy_hand_frame(
        variation=0.0)  # No variation = Perfectly still
    static_batch = [static_pose for _ in range(30)]  # 30 frames

    # Send in one big batch (mimicking a long network call) or smaller ones
    # Let's do batches of 5 to watch the progress
    for i in range(0, 30, 5):
        batch = static_batch[i:i+5]
        resp = send_batch(batch)

        # Check what the server says
        if resp['status'] == 'word_added':
            print(f"   ✅ SUCCESS! Server added word: {resp['new_word']}")
            print(f"   📋 Current Sentence: {resp['current_builder']}")
        else:
            print(f"   ... buffer processing ... (Status: {resp['status']})")

        time.sleep(0.1)

    # 3. Simulate PUSH to trigger LLM
    # NOTE: Since we can't easily fake the 'PUSH' landmarks without dataset data,
    # we will rely on the previous word being confirmed.
    # Ideally, you'd send frames that actually look like PUSH here.

    # For this test, let's just print the state.
    # If the previous step printed "SUCCESS", the logic works.

    print("\n--- Simulation Complete ---")
    print("To test 'PUSH' specifically, the model needs to see specific PUSH landmarks.")
    print("However, if Phase 2 added a word (even a wrong one like 'UNCERTAIN' or a random guess),")
    print("then the Voting Logic and Session State are working perfectly.")


if __name__ == "__main__":
    run_simulation()
