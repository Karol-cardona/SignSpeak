import os
import time
import logging
import re
from itertools import groupby

# Importa la libreria per GGUF
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# Configurazione Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURAZIONE PERCORSO ---
# In Docker è "local_model/...", in locale dipende da dove lanci.
# Usiamo un path relativo sicuro.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "local_model", "qwen2.5-1.5b-instruct-q4_k_m.gguf")

class AdvancedSentenceCorrector:
    def __init__(self):
        logger.info("--- Initializing Local GGUF Handler ---")

        self.llm = None
        if not Llama:
            logger.error("❌ llama-cpp-python not installed!")
            return

        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found at: {MODEL_PATH}")
            return

        try:
            logger.info(f"Loading model from {MODEL_PATH}...")

            # 1. Caricamento Modello
            self.llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=512,       # Contesto ridotto per massima velocità
                n_gpu_layers=-1, # Tutto in GPU
                n_batch=512,     # Processa il prompt in blocchi grandi
                verbose=False    # Riduciamo il rumore nei log
            )

            # 2. --- WARMUP (Il trucco per la velocità) ---
            # Facciamo una richiesta "finta" ora, così il ritardo di 16s
            # avviene adesso e non quando l'utente fa il segno PUSH.
            logger.info("🔥 Warming up CUDA kernels (Please wait ~10s)...")
            start_warm = time.time()
            self.llm.create_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1
            )
            logger.info(f"✅ Warmup Complete! (Took {time.time() - start_warm:.1f}s)")
            logger.info("🚀 System is ready for real-time translation.")

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")

    def _clean_gloss(self, words: list) -> list:
        cleaned = []
        for word in words:
            clean_word = re.sub(r'\d+', '', word)
            cleaned.append(clean_word)
        return [key for key, group in groupby(cleaned)]

    def correct_sentence(self, words: list) -> str:
        if not words:
            return ""

        if not self.llm:
            return "Error: Model not loaded"

        clean_words = self._clean_gloss(words)
        raw_sequence = " ".join(clean_words)

        start_time = time.time()
        # logger.info(f"🚀 Processing: '{raw_sequence}'")

        messages = [
            {"role": "system", "content": f"""
        You are an intelligent ASL-to-English interpreter.
        Input: Noisy "Gloss" words from a computer vision model.
        Output: A single, natural, polite English sentence.

        --- RULES ---
        1. If a word is hallucinated/nonsense, DELETE IT.
        2. Fix grammar (is, am, are, the).
        3. Output ONLY the natural sentence.
        """},
            {"role": "user", "content": raw_sequence}
        ]

        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.1,
                max_tokens=64  # Limitiamo i token per rispondere prima
            )

            result = output['choices'][0]['message']['content'].strip().replace('"', '')

            elapsed = time.time() - start_time
            logger.info(f"⚡ Response: '{result}' | Time: {elapsed:.3f}s")
            return result

        except Exception as e:
            logger.error(f"❌ Inference Error: {e}")
            return raw_sequence