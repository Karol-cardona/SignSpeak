Here is a comprehensive, professional, and advanced `README.md` file. It includes architectural details, step-by-step usage guides, API specifications, and the detailed model performance metrics you provided.

Copy the content below into `SignSpeak_TeamRepo/SignSpeakBackend-Python/README.md`.

---

# 🧠 SignSpeak AI Backend (Python)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

This is the core intelligence engine of the SignSpeak ecosystem. It accepts raw 3D hand landmarks, translates them into ASL glosses using a custom Transformer, and reconstructs natural English sentences using a Generative LLM (Flan-T5).

---

## 🏗️ System Architecture

The backend operates as a **Stateful Microservice**. It does not just classify single images; it maintains a "Session Memory" for each user to understand the context of movements over time.

### The 4-Stage Pipeline

1.  **Ingestion:** Receives batches of normalized 3D landmarks (x, y, z) from the Client/MediaPipe.
2.  **Recognition (The Transformer):** A custom trained PyTorch Transformer analyzes the temporal sequence (60 frames) to predict a raw Gloss (e.g., "HELLO", "HUNGRY").
3.  **Stabilization (The Voting Mechanism):** A sliding window algorithm filters out noise. A word is only accepted if it achieves **72% consensus** over the last 22 frames.
4.  **Translation (The LLM):** When the trigger sign **"PUSH"** is detected, the accumulated glosses (e.g., `["ME", "HUNGRY", "NOW"]`) are sent to the LLM to generate natural speech (`"I am hungry now."`).

---

## 🚀 Quick Start (For Developers)

### Option A: Run with Docker (Recommended)

This ensures you have the exact environment as production.

```bash
# From the root of the team repo
docker-compose up --build
```

_The API will be available at:_ `http://localhost:8000`

### Option B: Run Locally

```bash
# 1. Create venv
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 📡 API Documentation

### Endpoint: `POST /process_frames`

This is the primary communication channel. The mobile app should collect landmarks in a small buffer (approx. 100ms - 200ms worth of data) and send them in batches.

#### 📥 Request Payload

**Headers:** `Content-Type: application/json`

```json
{
  "session_id": "unique_user_id_123",
  "frames": [
    {
      "landmarks": [
        [
          { "x": 0.5, "y": 0.5, "z": 0.0 }
          // ... 21 points for Right Hand ...
        ],
        [] // Empty list if Left Hand not detected
      ],
      "handedness": [
        [
          {
            "score": 0.99,
            "categoryName": "Right",
            "index": 1,
            "displayName": "Right"
          }
        ]
      ]
    }
    // ... list of 5 to 10 frames ...
  ]
}
```

#### 📤 Response Payload

The client must check the **`status`** field to determine the UI action.

```json
{
  "session_id": "unique_user_id_123",
  "status": "word_added", // Options: "processing", "word_added", "sentence_completed"
  "new_word": "HELLO", // Not null only if status is "word_added"
  "current_builder": ["HELLO"],
  "final_sentence": "" // Not empty only if status is "sentence_completed"
}
```

---

## 🧪 Detailed Usage Scenario

Here is the exact lifecycle of a user signing: **"HELLO"** -> **"FRIEND"** -> **"PUSH"** (to translate).

| Step  | User Action                        | API Response Status  | Payload Data                          | UI Action                                            |
| :---- | :--------------------------------- | :------------------- | :------------------------------------ | :--------------------------------------------------- |
| **1** | User raises hands (buffer filling) | `processing`         | `new_word: null`                      | Show "Listening..." spinner.                         |
| **2** | User signs **HELLO**               | `word_added`         | `new_word: "HELLO"`                   | Flash "HELLO" on screen. Add to word list.           |
| **3** | User transitions hands             | `processing`         | `new_word: null`                      | Keep current list displayed.                         |
| **4** | User signs **FRIEND**              | `word_added`         | `new_word: "FRIEND"`                  | Update list: `["HELLO", "FRIEND"]`                   |
| **5** | User signs **PUSH**                | `sentence_completed` | `final_sentence: "Hello, my friend."` | **Clear list.** Show green result box with sentence. |

---

## 📊 Model Performance Report

The underlying Transformer model has been trained on the ASL Citizen dataset and fine-tuned for this specific vocabulary.

### 🏆 Overall Performance

- **Accuracy:** 86%
- **Macro Average F1-Score:** 0.84
- **Weighted Average F1-Score:** 0.85

### ✅ Top Performers (Perfect Score 1.0)

The model is extremely reliable (100% Precision & Recall) on these signs. **Use these for demos:**

> YES, THANKYOU, OK, NICE, MEET, MORE, THEY1, BOY, CHILD, CHILDREN, FAMILY, BROTHER, HELP, KNOW, UNDERSTAND, ANGRY, HUNGRY, SURPRISE, SERIOUS, WRONG, BIG, SMALL, UGLY, HOT, COLD, EASY, DIRTY, CITY1, BATHROOM, LIBRARY, TIME, MORNING, WEEK, WATER, APPLE, MILK1, CAR, COMPUTER, TV, CHAIR, IDEA, STORY1, JOKE, WITH, FOR, FROM, BUT, PUSH.

### ⚠️ Common Confusion Pairs

The voting mechanism helps mitigate these, but the frontend team should be aware of these similar signs:

| Actual Word | Predicted As | Errors | Notes                          |
| :---------- | :----------- | :----- | :----------------------------- |
| **CLEAN**   | NICE         | 5      | Similar sliding motion.        |
| **KITCHEN** | NEW          | 5      | Similar hand shape.            |
| **NIGHT1**  | AT           | 4      | Hand contact position similar. |
| **MUSIC**   | WINDOW       | 4      | Rythmic motion confused.       |
| **FROM**    | MONTH        | 4      | Pulling motion overlap.        |
| **YOU**     | TRUE         | 4      | Pointing gesture overlap.      |

### 🚨 The "Red Zone" (Low Confidence)

These words have an F1-score below 0.50. The model struggles with these, often due to dataset limitations or visual similarity.

- **MUSIC** (0.22)
- **NEW** (0.34)
- **PAPER** (0.34)
- **CLEAN** (0.35)
- **BLACK** (0.35)
- **ON** (0.40)
- **YOU** (0.42)

---

## 🛠️ Configuration & Customization

### Adding/Removing Words

The vocabulary list is strictly defined in `app/dependencies.py`.
To add a word:

1.  Retrain the model (Checkpoint generation).
2.  Update the `TARGET_WORDS` list in `app/dependencies.py`.

### Tuning Sensitivity

Located in `app/session_state.py`:

- `VOTING_WINDOW_SIZE = 22`: How many past frames to remember.
- `VOTE_THRESHOLD = 16`: How many frames must agree to confirm a word.
- `CONFIDENCE_THRESHOLD = 0.75`: Minimum AI confidence to count a vote.

---

## 👨‍💻 Troubleshooting

**1. "500 Internal Server Error" on empty hands**

- _Cause:_ Sending an empty array `[]` for handedness without checking inside `preprocessing.py`.
- _Fix:_ This has been patched in the latest commit. The system now safely ignores empty hand data.

**2. "Connection Refused"**

- _Cause:_ Docker container not running or port 8000 blocked.
- _Fix:_ Check `docker ps` and ensure port 8000 is mapped.

**3. Model predicts "UNCERTAIN" constantly**

- _Cause:_ Camera lighting is poor or hand is too close/far.
- _Fix:_ Ensure the user's upper body and hands are clearly visible.

This document is the **Integration Contract** between the **Python AI Team** and the **Frontend/Mobile Team**.

Copy this entire section into a new file named `API_INTEGRATION_GUIDE.md` in your repository, or share it directly on your team's communication channel (Slack/Discord/Teams).

---

# 🤝 API Integration Guide: Frontend <-> Python Backend

**To:** Frontend/Mobile Developers & Backend Integrators
**From:** AI/Python Team
**Subject:** Step-by-Step Implementation of SignSpeak Real-Time Translation

## 🚨 Core Concept: "The Session"

Unlike standard REST APIs where you send data and get a static result, this API has **memory**.

- You must generate a **Session ID** when the user opens the camera.
- You must use that **same Session ID** for every frame sent during that conversation.
- If you change the ID, the AI forgets what the user signed previously.

---

## 🛠 Phase 1: Connection & Health Check

Before trying to send complex video data, ensure the Docker container is reachable from your mobile app or frontend.

**Step 1:** Ensure the Python Docker container is running.
**Step 2:** Frontend sends a GET request.

- **Endpoint:** `GET http://[SERVER_IP]:8000/`
- **Expected Response:** `200 OK`
  ```json
  { "status": "ok", "message": "SignSpeak API is running" }
  ```

> **Frontend Tip:** Call this once when the app launches to verify the AI service is online.

---

## 📦 Phase 2: Data Collection Strategy

**❌ DON'T:** Send an HTTP request for every single frame (30 times per second). This will crash the network.
**✅ DO:** Collect frames in a "Buffer" and send them in chunks (Batches).

### The Recommended Loop (Client-Side Logic)

1.  **Initialize:** `buffer = []`
2.  **Capture:** MediaPipe gives you landmarks for Frame X.
3.  **Store:** Add Frame X to `buffer`.
4.  **Check:** Is `buffer.length >= 10`? (approx every 100-300ms)
    - **Yes:** Send the buffer to the API via POST. Clear `buffer`.
    - **No:** Keep capturing.

---

## 📨 Phase 3: The Request Payload (Strict Format)

The AI is strict. If the JSON structure is wrong, it will return a `422 Validation Error`.

**Endpoint:** `POST /process_frames`

**Structure Rules:**

1.  **Landmarks:** Must be a list of 2 items: `[Right_Hand_Array, Left_Hand_Array]`.
2.  **Handedness:** Must match the order of landmarks.
3.  **Coordinates:** `x, y, z` must be normalized floats (0.0 to 1.0).

### 📋 Copy-Paste JSON for Testing (Postman)

Use this to test if your networking code works.

```json
{
  "session_id": "test_user_001",
  "frames": [
    {
      "landmarks": [
        [
          { "x": 0.5, "y": 0.5, "z": 0.0 },
          { "x": 0.51, "y": 0.52, "z": -0.01 }
          // ... (Need exactly 21 points here for the hand)
        ],
        []
      ],
      "handedness": [
        [
          {
            "score": 0.99,
            "categoryName": "Right",
            "index": 1,
            "displayName": "Right"
          }
        ]
      ]
    }
  ]
}
```

---

## 🔄 Phase 4: Handling the Response (The State Machine)

The API tells you **exactly** what to show on the screen using the `status` field. You need to implement a `switch` or `if/else` logic in your UI code.

### Case A: `status == "processing"`

The buffer is filling up, or the user is moving between signs.

- **Frontend Action:** Do nothing. Or show a subtle "listening" indicator.
- **Response Data:**
  ```json
  { "status": "processing", "new_word": null, "final_sentence": "" }
  ```

### Case B: `status == "word_added"`

The AI has confirmed a specific sign (e.g., "HELLO").

- **Frontend Action:**
  1.  Flash the `new_word` ("HELLO") on the screen for visual feedback.
  2.  Update the "Current Sentence" text view with the `current_builder` list.
- **Response Data:**
  ```json
  {
    "status": "word_added",
    "new_word": "HELLO",
    "current_builder": ["HELLO"]
  }
  ```

### Case C: `status == "sentence_completed"`

The user signed **PUSH**. The LLM has translated the raw words into English.

- **Frontend Action:**
  1.  **Clear** the "Current Sentence" text view (the raw words are gone).
  2.  **Display** the `final_sentence` in a prominent Green Box / Chat Bubble.
  3.  (Optional) Trigger Text-to-Speech.
- **Response Data:**
  ```json
  {
    "status": "sentence_completed",
    "final_sentence": "Hello, how are you?",
    "current_builder": []
  }
  ```

---

## 🧪 Phase 5: Verification Checklist

Before marking the integration as "Done," perform this exact sequence:

1.  **Start App:** Ensure `session_id` is generated.
2.  **Sign "YES":**
    - Hold the fist (S-handshape) and nod it like a head.
    - **Verify:** UI updates to show "YES".
3.  **Sign "PLEASE":**
    - Rub flat hand on chest in a circle.
    - **Verify:** UI updates to "YES PLEASE".
4.  **Sign "PUSH":**
    - Push both open hands forward (like pushing a door).
    - **Verify:** Raw words disappear.
    - **Verify:** Final English sentence appears: _"Yes, please."_

---

## 🐛 Troubleshooting Guide for Frontend

**Q: I get `500 Internal Server Error`.**

- **A:** You are likely sending an empty array for a hand without checking. Ensure if `handedness` says "Right", the `landmarks[0]` array actually contains 21 points.

**Q: I get `422 Unprocessable Entity`.**

- **A:** Your JSON field names are wrong. Check `categoryName` vs `label`. Check `x, y, z` spelling.

**Q: The response takes too long (>1 sec).**

- **A:** You are sending too many frames in one batch. Reduce your batch size to 5 or 10 frames.

**Q: The AI keeps printing "UNCERTAIN".**

- **A:** The user's hand is likely off-screen or lighting is bad. MediaPipe must detect the hand clearly for the AI to work.
