from transformers import T5ForConditionalGeneration, T5Tokenizer


class SentenceGenerator:
    def __init__(self):
        """
        Initializes the tokenizer and model.
        This is done once to avoid reloading the model on every request.
        """
        model_name = "vennify/t5-base-grammar-correction"
        print("--- Loading LLM tokenizer ---")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        print("--- Loading LLM model ---")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("--- LLM model loaded successfully ---")

    def correct_sentence(self, words: list) -> str:
        """
        Takes a list of words, forms a sentence, and uses the T5 model to correct it.
        """
        if not words:
            return ""

        # Join words and format for the model
        raw_sentence = " ".join(words)
        input_text = f"grammar: {raw_sentence}"

        # Tokenize the input
        inputs = self.tokenizer(
            input_text, return_tensors="pt", max_length=256, truncation=True)

        # Generate the corrected sentence
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=256,
            num_beams=5,  # beam search for better results
            early_stopping=True
        )

        # Decode and clean up the output
        corrected_sentence = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        return corrected_sentence.strip()
