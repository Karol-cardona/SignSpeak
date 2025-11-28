# # ==============================================================================
# #                      Advanced Post-Processing Script
# #
# # Description:
# #   This script takes a list of raw, potentially noisy word predictions from a
# #   sign language recognition model and transforms it into a coherent,
# #   grammatically correct, and semantically plausible English sentence.
# #
# # Method:
# #   1. Pre-processes the raw word list to remove consecutive duplicates.
# #   2. Constructs a detailed, context-aware prompt for an instruction-tuned LLM.
# #      This prompt includes specific examples of the recognition model's
# #      most common confusion errors, taken directly from its performance report.
# #   3. Uses the powerful google/flan-t5-large model to interpret the noisy
# #      input and generate a high-quality, corrected sentence.
# #
# # ==============================================================================

# # --- Required Libraries ---
# # pip install transformers torch sentencepiece
# from transformers import T5ForConditionalGeneration, T5Tokenizer
# from itertools import groupby
# import os

# # تحديد مجلد جديد في الـ D لحفظ النماذج
# # سيقوم بإنشاء المجلد تلقائياً إذا لم يكن موجوداً
# os.environ['HF_HOME'] = 'D:/huggingface_cache'


# class AdvancedSentenceCorrector:
#     """
#     A class to correct and refine sentences from a noisy ASL transcription model.
#     It uses a powerful LLM guided by a context-rich prompt that includes the
#     transcription model's known weaknesses.
#     """

#     def __init__(self, model_name="google/flan-t5-small"):
#         """
#         Initializes the tokenizer and the instruction-tuned model.
#         This is a one-time setup cost.
#         """
#         print("--- Initializing AdvancedSentenceCorrector ---")
#         try:
#             print(f"--- Loading LLM tokenizer: {model_name} ---")
#             self.tokenizer = T5Tokenizer.from_pretrained(
#                 model_name, cache_dir='D:/huggingface_cache', legacy=False)

#             print(
#                 f"--- Loading LLM model: {model_name} (this may take a moment) ---")
#             self.model = T5ForConditionalGeneration.from_pretrained(
#                 model_name, cache_dir='D:/huggingface_cache')

#             print("--- Advanced LLM loaded successfully ---")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             print(
#                 "Please ensure you have an internet connection and the 'transformers' library is installed.")
#             raise

#         # This dictionary is the "brain" of the corrector. It is derived
#         # DIRECTLY from your model's performance report.
#         # Key: The word the model incorrectly PREDICTED.
#         # Value: The word that was the ACTUAL ground truth.
#         self.confusion_map = {
#             "ASK": "NEED", "GOOD": "THANKYOU", "SON": "DAUGHTER", "WALK1": "NIGHT1",
#             "SISTER": "BROTHER", "NICE": "CLEAN", "SIT": "CHAIR", "MONTH": "FROM",
#             "TRUE": "YOU", "SCHOOL": "PAPER", "BROTHER": "SISTER", "DAUGHTER": "SON",
#             "WHEN": "ABOUT1", "LOOKAT": "TV", "PEOPLE": "PERSON", "THANKYOU": "GOOD",
#             "BLACK": "UGLY", "YEAR": "COFFEE", "WANT1": "SAD", "HAPPY": "FEEL"
#         }

#     def _debounce_words(self, words: list) -> list:
#         """
#         Removes consecutive duplicate words from a list.
#         Example: ['HELLO', 'HELLO', 'MY'] -> ['HELLO', 'MY']
#         """
#         if not words:
#             return []
#         return [key for key, group in groupby(words)]

#     def _build_prompt(self, words: list) -> str:
#         """
#         Constructs a sophisticated prompt with a specific role, context about
#         common errors, and the task for the LLM.
#         """
#         word_sequence = " ".join(words)

#         # We provide the most critical confusion pairs as context in the prompt.
#         # This helps the LLM make more intelligent semantic corrections.
#         error_context = (
#             "- It frequently confuses 'NEED' (actual) and 'ASK' (predicted).\n"
#             "- It often confuses 'THANKYOU' and 'GOOD' with each other.\n"
#             "- It struggles to distinguish family pairs like 'DAUGHTER'/'SON' and 'SISTER'/'BROTHER'.\n"
#             "- It can mistake 'CHAIR' for the action 'SIT'.\n"
#             "- It sometimes confuses 'NIGHT1' with 'WALK1'.\n"
#             "- It may predict 'TV' as 'LOOKAT'."
#         )

#         prompt = (
#             "You are an expert at correcting sentences transcribed from American Sign Language (ASL) by an imperfect AI model. "
#             "The model's transcription is noisy and may contain repeated or semantically incorrect words. "
#             "Your task is to reconstruct the most logical, grammatically correct, and fluent English sentence from the raw transcription, keeping the original intent.\n\n"
#             "Here are some of the AI's most common and predictable errors:\n"
#             f"{error_context}\n\n"
#             f"Raw transcribed words: \"{word_sequence}\"\n\n"
#             "Reconstructed sentence:"
#         )
#         return prompt

#     def correct_sentence(self, words: list) -> str:
#         """
#         Takes a list of raw predicted words, processes them, and uses the LLM
#         with a context-aware prompt to generate a high-quality sentence.

#         Args:
#             words (list): A list of strings, representing the raw output from the
#                           sign language recognition model.

#         Returns:
#             str: A corrected, fluent English sentence.
#         """
#         if not isinstance(words, list):
#             return "Error: Input must be a list of words."
#         if not words:
#             return ""

#         # --- Stage 1: Intelligent Pre-processing ---
#         debounced_words = self._debounce_words(words)
#         if not debounced_words:
#             return ""

#         # --- Stage 2: Context-Aware Sentence Reconstruction ---
#         prompt = self._build_prompt(debounced_words)

#         # Tokenize and generate the corrected sentence
#         inputs = self.tokenizer(
#             prompt, return_tensors="pt", max_length=1024, truncation=True)

#         outputs = self.model.generate(
#             inputs.input_ids,
#             max_length=256,
#             num_beams=5,          # Use beam search for higher quality output
#             early_stopping=True,
#             temperature=0.7,      # Add a bit of creativity for more natural phrasing
#         )

#         corrected_sentence = self.tokenizer.decode(
#             outputs[0], skip_special_tokens=True)

#         return corrected_sentence.strip()


# # ==============================================================================
# #                      DEMONSTRATION OF THE SCRIPT
# # ==============================================================================
# if __name__ == '__main__':
#     # This block will only run when you execute `python post_processing.py` directly.

#     print("\n" + "="*50)
#     print("   Running Demonstration for AdvancedSentenceCorrector")
#     print("="*50 + "\n")

#     # Create an instance of the corrector. This loads the models into memory.
#     corrector = AdvancedSentenceCorrector()

#     print("\n--- Testing with simulated model outputs ---\n")

#     # --- SIMULATION EXAMPLES BASED ON YOUR REPORT ---

#     # Example 1: Model sees "I NEED HELP" but makes a common 'NEED' -> 'ASK' error.
#     raw_prediction_1 = ['MYSELF', 'MYSELF', 'ASK', 'HELP', 'HELP']
#     corrected_1 = corrector.correct_sentence(raw_prediction_1)
#     print(f"Raw Input:  {raw_prediction_1}")
#     print(f"Corrected:  {corrected_1}\n")  # Expected: I need help.

#     # Example 2: Model sees "MY DAUGHTER IS HAPPY" but confuses 'DAUGHTER'->'SON' and 'HAPPY'->'FEEL'.
#     raw_prediction_2 = ['MY', 'SON', 'SON', 'FEEL']
#     corrected_2 = corrector.correct_sentence(raw_prediction_2)
#     print(f"Raw Input:  {raw_prediction_2}")
#     print(f"Corrected:  {corrected_2}\n")  # Expected: My daughter is happy.

#     # Example 3: A more complex sentence. Model sees "THANKYOU MY SISTER SIT ON CHAIR"
#     # but predicts a noisy, repetitive, and confused sequence.
#     raw_prediction_3 = ['GOOD', 'GOOD', 'MY',
#                         'BROTHER', 'SIT', 'SIT', 'ON', 'SIT']
#     corrected_3 = corrector.correct_sentence(raw_prediction_3)
#     print(f"Raw Input:  {raw_prediction_3}")
#     # Expected: Thank you, my sister is sitting on the chair.
#     print(f"Corrected:  {corrected_3}\n")

#     # Example 4: A simple, correct prediction to ensure it doesn't over-correct.
#     raw_prediction_4 = ['HELLO', 'NICE', 'MEET', 'YOU']
#     corrected_4 = corrector.correct_sentence(raw_prediction_4)
#     print(f"Raw Input:  {raw_prediction_4}")
#     print(f"Corrected:  {corrected_4}\n")  # Expected: Hello, nice to meet you.
# --- START OF FILE app/llm_handler.py ---
# --- START OF FILE app/llm_handler.py ---
# --- START OF FILE app/llm_handler.py ---
from transformers import T5ForConditionalGeneration, T5Tokenizer
from itertools import groupby
import os
import torch
import re

# Saves cache inside the project folder or Docker container
os.environ['HF_HOME'] = os.path.join(os.getcwd(), 'hf_cache')


class AdvancedSentenceCorrector:
    """
    Advanced Interpreter that converts robotic ASL Glosses into 
    Natural, Human-Like English.
    """

    def __init__(self, model_name="google/flan-t5-base"):
        print("--- Initializing Natural Language Interpreter ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            print(f"--- Loading Tokenizer: {model_name} ---")
            self.tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir='D:/huggingface_cache',
                legacy=False
            )

            print(f"--- Loading Model: {model_name} ---")
            # Optimization: Use float16 on GPU to save memory (RAM)
            model_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir='D:/huggingface_cache',
                torch_dtype=model_dtype
            ).to(self.device)

            print(f"--- AI Ready on {self.device} ---")
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            print("If you run out of memory, change model_name to 'google/flan-t5-small'")
            raise

    def _clean_gloss(self, words: list) -> list:
        """
        Converts 'HOW1' -> 'HOW', 'TALK1' -> 'TALK'.
        Removes numbers from the raw model output.
        """
        cleaned = []
        for word in words:
            # Regex to remove digits
            clean_word = re.sub(r'\d+', '', word)
            cleaned.append(clean_word)
        return cleaned

    def _debounce_words(self, words: list) -> list:
        """Removes immediate duplicates (YOU YOU -> YOU)."""
        if not words:
            return []
        return [key for key, group in groupby(words)]

    def correct_sentence(self, words: list) -> str:
        if not words:
            return ""

        # 1. Pre-process: Clean numbers and duplicates
        clean_words = self._clean_gloss(words)
        debounced_words = self._debounce_words(clean_words)

        # If the list is empty after cleaning
        if not debounced_words:
            return ""

        raw_sequence = " ".join(debounced_words)

        # 2. Context-Aware Prompt for NATURAL SPEECH
        # This is the secret sauce: We give it EXAMPLES (Few-Shot Learning)
        # so it knows to IGNORE random words.
        prompt = (
            "Task: Rewrite the following Sign Language Glosses into a natural, spoken English sentence.\n"
            "Strict Rules:\n"
            "1. Fix grammar, add pronouns (I, you, he) and verbs (is, am, are).\n"
            "2. DETECT AND REMOVE words that do not fit the context (Hallucinations).\n"
            "3. If the input is just one word, make it a full polite sentence.\n\n"

            "Examples:\n"
            "Input: HELLO NAME WHAT\n"
            "Output: Hello, what is your name?\n\n"

            "Input: ME HUNGRY APPLE CAR\n"
            "Output: I am hungry and I want an apple.\n"
            "(Note: 'CAR' was removed because it did not fit the context of eating)\n\n"

            "Input: YOU GO SCHOOL YESTERDAY\n"
            "Output: Did you go to school yesterday?\n\n"

            "Input: WATER\n"
            "Output: May I have some water please?\n\n"

            "Input: NICE MEET YOU TREE\n"
            "Output: It is nice to meet you.\n"
            "(Note: 'TREE' was removed as noise)\n\n"

            f"Input: {raw_sequence}\n"
            "Output:"
        )

        # 3. Generate
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)

            outputs = self.model.generate(
                inputs.input_ids,
                max_length=128,
                num_beams=5,          # High beams = smarter phrasing
                early_stopping=True,
                temperature=0.6,      # Slightly lower temp for more logical cleaning
                repetition_penalty=1.2
            )

            final_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True)
            return final_text

        except Exception as e:
            print(f"LLM Error: {e}")
            return raw_sequence
