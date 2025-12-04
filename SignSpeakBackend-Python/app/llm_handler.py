import os
import google.generativeai as genai
from itertools import groupby
import re

# --- CONFIGURATION ---
API_KEY = os.environ.get(
    "GEMINI_API_KEY", "AIzaSyCIKk2S9MCS45IMb9JpbZMUClCA3HIMV2Q")


class AdvancedSentenceCorrector:
    """
    Uses Google Gemini to translate noisy ASL Glosses into fluent English.
    Includes Smart Model Selection to prevent 404 errors.
    """

    def __init__(self):
        print("--- Initializing Gemini AI Handler ---")

        if "PASTE_YOUR" in API_KEY or not API_KEY:
            print("⚠️ WARNING: Gemini API Key is missing!")

        try:
            genai.configure(api_key=API_KEY)

            # --- SMART MODEL SELECTION ---
            # 1. List all models available to your API Key
            all_models = list(genai.list_models())

            # 2. Filter for models that support text generation ('generateContent')
            my_models = [
                m.name for m in all_models if 'generateContent' in m.supported_generation_methods]

            # 3. Priority list: Try Flash first (fastest), then 1.5 Pro, then standard Pro
            target_model = None

            # Check for Flash
            for m in my_models:
                if 'flash' in m.lower():
                    target_model = m
                    break

            # If no Flash, check for 1.5 Pro
            if not target_model:
                for m in my_models:
                    if '1.5' in m and 'pro' in m.lower():
                        target_model = m
                        break

            # Fallback to generic gemini-pro
            if not target_model:
                target_model = 'models/gemini-pro'

            print(f"--- Selected Model: {target_model} ---")
            self.model = genai.GenerativeModel(target_model)
            print("--- Gemini Connected Successfully ---")

        except Exception as e:
            print(f"CRITICAL ERROR connecting to Gemini: {e}")
            self.model = None

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

        if self.model is None:
            return "Error: AI not connected"

        # 1. Clean Input
        clean_words = self._clean_gloss(words)
        raw_sequence = " ".join(clean_words)

        print(f"Sending to Gemini: {raw_sequence}")

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

        # 3. Call Gemini
        try:
            # Generate
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            # Clean up quotes
            result = result.replace('"', '').replace("'", "")
            return result
        except Exception as e:
            print(f"Gemini Error: {e}")
            return raw_sequence
