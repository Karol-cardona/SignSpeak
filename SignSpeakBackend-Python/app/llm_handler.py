import os
from openai import OpenAI
from itertools import groupby
import re
import time

# --- CONFIGURATION ---
# We use the Groq API Key you provided
API_KEY = os.environ.get(
    "GEMINI_API_KEY", "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")


class AdvancedSentenceCorrector:
    """
    Uses Groq (via OpenAI SDK) to translate noisy ASL Glosses into fluent English.
    """

    def __init__(self):
        print("--- Initializing AI Handler (Groq) ---")

        if not API_KEY:
            print("⚠️ WARNING: API Key is missing!")

        try:
            # Configure OpenAI Client to point to Groq's servers
            self.client = OpenAI(
                api_key=API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )

            # We use Llama 3 because it is supported by your 'gsk_' key
            self.model_name = "openai/gpt-oss-20b"

            print(
                f"--- Connected to Groq Successfully (Model: {self.model_name}) ---")

        except Exception as e:
            print(f"CRITICAL ERROR connecting to AI: {e}")
            self.client = None

    def _clean_gloss(self, words: list) -> list:
        """Removes numbers (HOW1 -> HOW) and duplicates."""
        cleaned = []
        for word in words:
            clean_word = re.sub(r'\d+', '', word)
            cleaned.append(clean_word)
        return [key for key, group in groupby(cleaned)]

    def correct_sentence(self, words: list) -> str:
        if not words:
            return ""

        if self.client is None:
            return "Error: AI not connected"

        # 1. Clean Input
        clean_words = self._clean_gloss(words)
        raw_sequence = " ".join(clean_words)

        print(f"Sending to AI: {raw_sequence}")

        # 2. THE MASTER PROMPT
        prompt = f"""
        Role: You are an intelligent ASL-to-English interpreter.
        Input: Noisy "Gloss" words from a computer vision model.
        Output: A single, natural, polite English sentence.

        --- USER CONTEXTS ---
        - University Student: "UNDERSTAND IDEA", "HELP ME", "ASK QUESTION".
        - Work Call: "WORK MORNING", "HEAR ME", "LOOK COMPUTER".
        - Family: "LOVE YOU", "COME HOME", "EAT APPLE".

        --- RULES ---
        1. If a word is hallucinated/nonsense, DELETE IT.
        2. Fix grammar (is, am, are, the).
        3. Output ONLY the natural sentence.

        Raw Input: "{raw_sequence}"
        Translation:
        """

        # 3. Call Groq/OpenAI
        try:
            t0 = time.time()

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=0.1,  # Low temperature for consistent translations
            )

            elapsed = time.time() - t0
            result = chat_completion.choices[0].message.content.strip()

            # Clean up quotes if the model added them
            result = result.replace('"', '').replace("'", "")

            print(f"[llm_handler] AI call {elapsed:.2f}s")
            return result

        except Exception as e:
            print(f"AI Error: {e}")
            # Fallback to raw words if API fails
            return raw_sequence
