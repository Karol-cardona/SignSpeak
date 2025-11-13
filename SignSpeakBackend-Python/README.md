# SignSpeak API: A Simple Guide

## Table of Contents

1.  [What Does This Backend Do?](#1-what-does-this-backend-do)
2.  [Quick Setup: Get Running in 3 Steps](#2-quick-setup-get-running-in-3-steps)
3.  [How to Use the API (The Important Part!)](#3-how-to-use-the-api-the-important-part)
    - [The Main Endpoint: `/predict_landmarks`](#the-main-endpoint-predict_landmarks)
    - [Example 1: Sending the word "HELLO"](#example-1-sending-the-word-hello)
    - [Example 2: Sending the word "YOU"](#example-2-sending-the-word-you)
    - [Example 3: Finishing the Sentence with "PUSH"](#example-3-finishing-the-sentence-with-push)
    - [Full User Journey: An Example Conversation](#full-user-journey-an-example-conversation)
4.  [Testing the API Yourself (with Postman or code)](#4-testing-the-api-yourself-with-postman-or-code)

---

## 1. What Does This Backend Do?

Think of our backend as a brain that understands sign language. It doesn't need to see video. Instead, it just needs the **(x, y, z) coordinates** of the hands, frame by frame.

Here is the entire process:

1.  Another part of our system (the "MediaPipe Backend") watches the user's camera and generates a stream of hand landmark data.
2.  When the user performs a sign (like "HELLO"), the MediaPipe backend bundles up all the landmark data for that sign (e.g., 5 seconds worth of frames).
3.  It sends this big bundle of data to our AI backend's `/predict_landmarks` endpoint.
4.  Our backend intelligently analyzes this data, figures out the most likely word ("HELLO"), and saves it.
5.  This repeats for every sign. When the special sign "PUSH" is sent, the backend takes all the saved words, forms a proper sentence using AI, and sends it back.

---

## 2. Quick Setup: Get Running in 3 Steps

Follow these steps to run the project on your own computer.

**Prerequisites:**

- Python 3.8+ installed
- Git installed

**Step 1: Get the Code**
Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/your-username/SignSpeak.git
cd SignSpeak
```

**Step 2: Set Up the Environment**
This will create a virtual space for our project's libraries so they don't interfere with anything else on your computer.

- **On Windows:**
  ```bash
  # Create the virtual environment
  python -m venv venv
  # Activate it
  .\venv\Scripts\activate
  ```
- **On Mac/Linux:**
  ```bash
  # Create the virtual environment
  python3 -m venv venv
  # Activate it
  source venv/bin/activate
  ```

**Step 3: Install Libraries and Run the Server**
Now, install all the required Python packages and start the server. The first time you run this, it will download the AI models (about 1 GB), so you'll need an internet connection.

```bash
# Install everything from the requirements file
pip install -r requirements.txt

# Run the server!
uvicorn app.main:app --reload
```

If everything is successful, you will see a message like this in your terminal:
`Uvicorn running on http://127.0.0.1:8000`

**The server is now running and ready to receive requests!**

---

## 3. How to Use the API (The Important Part!)

Our application has one main endpoint that does all the work.

### The Main Endpoint: `/predict_landmarks`

This is where the MediaPipe backend will send its data.

- **URL:** `http://127.0.0.1:8000/predict_landmarks`
- **Method:** `POST`
- **Input:** A JSON array of "frames". Each frame contains the timestamp and the landmark/handedness data for that moment in time. The number of frames you send can be variable (e.g., a 2-second sign or a 5-second sign).
- **Output:** A JSON object that tells you what happened.

Let's walk through some examples.

### Example 1: Sending the word "HELLO"

The user signs "HELLO". The MediaPipe backend captures 150 frames of data and sends it to our endpoint.

**➡️ INPUT (Request Body):**
A big JSON array. Here's a tiny sample of what one frame in that array looks like:

```json
[
  {
    "timestamp": 31034.3,
    "sequenceNumber": 1,
    "receivedAt": 31034.3,
    "landmarks": [
      [{ "x": 0.24, "y": 0.94, "z": 0.0 } /* ...20 more points... */]
    ],
    "handedness": [[{ "score": 0.98, "categoryName": "Right" }]]
  }
  /* ...149 more frames... */
]
```

**⬅️ OUTPUT (Response from our API):**
Our AI backend analyzes the 150 frames, confidently determines the word is "HELLO", and adds it to its memory. It then replies with:

```json
{
  "prediction": "HELLO",
  "status": "word_added",
  "current_words": ["HELLO"]
}
```

- `status: "word_added"` tells the client that the word was successfully added.
- `current_words` shows the sentence being built so far.

### Example 2: Sending the word "YOU"

Next, the user signs "YOU". The MediaPipe backend sends another batch of frames.

**➡️ INPUT (Request Body):**
A new JSON array of frames for the sign "YOU".

**⬅️ OUTPUT (Response from our API):**
The backend analyzes the new frames, determines the word is "YOU", and adds it to the list it's keeping in memory. It replies with:

```json
{
  "prediction": "YOU",
  "status": "word_added",
  "current_words": ["HELLO", "YOU"]
}
```

- Notice how `current_words` now contains both words.

### Example 3: Finishing the Sentence with "PUSH"

Finally, the user signs "PUSH". This is our special command to finish the sentence.

**➡️ INPUT (Request Body):**
A JSON array of frames for the sign "PUSH".

**⬅️ OUTPUT (Response from our API):**
The backend detects "PUSH". It takes the words it has saved (`["HELLO", "YOU"]`), sends them to the grammar AI to form a proper sentence, and then clears its memory for the next conversation. It replies with:

```json
{
  "prediction": "PUSH",
  "status": "end_of_sentence",
  "sentence": "Hello you."
}
```

- `status: "end_of_sentence"` tells the client that the conversation is finished.
- `sentence` contains the final, corrected text.

### Full User Journey: An Example Conversation

| User Action        | Data Sent to `/predict_landmarks` | Response from API                                                            | UI Should Show                       |
| :----------------- | :-------------------------------- | :--------------------------------------------------------------------------- | :----------------------------------- |
| Signs "HELLO"      | Frames for "HELLO"                | `status: "word_added"`, `current_words: ["HELLO"]`                           | Current: `HELLO`                     |
| Signs "GOVERNMENT" | Frames for "GOVERNMENT"           | `status: "word_added"`, `current_words: ["HELLO", "GOVERNMENT"]`             | Current: `HELLO GOVERNMENT`          |
| Signs "PASSPORT"   | Frames for "PASSPORT"             | `status: "word_added"`, `current_words: ["HELLO", "GOVERNMENT", "PASSPORT"]` | Current: `HELLO GOVERNMENT PASSPORT` |
| Signs "PUSH"       | Frames for "PUSH"                 | `status: "end_of_sentence"`, `sentence: "Hello government passport."`        | Final: `Hello government passport.`  |

---

## 4. Testing the API Yourself (with Postman or code)

You don't need the MediaPipe backend to test this API. You can simulate it!

1.  **Get Sample Data:** Find a file with sample landmark data that you can use for testing.
2.  **Use a Tool like Postman or Insomnia:**
    - Set the request type to `POST`.
    - Set the URL to `http://127.0.0.1:8000/predict_landmarks`.
    - Go to the "Body" tab and select "raw" and "JSON".
    - Paste the entire JSON array of frames into the body.
    - Click "Send".

This will let you test the API's responses and see how it builds sentences without needing the full application to be connected.
