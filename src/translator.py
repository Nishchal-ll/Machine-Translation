import torch
from pathlib import Path
import re
import unicodedata

class NepaliTranslator:
    def __init__(self, model_path: str | Path, device=None):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script"""
        devanagari_range = range(0x0900, 0x097F)
        return any(ord(char) in devanagari_range for char in text)

    def remove_artifacts(self, text: str) -> str:
        """Remove common hallucination artifacts and non-Devanagari junk"""
        # Remove English characters (except common words)
        text = re.sub(r'[a-zA-Z]{2,}', '', text)
        
        # Remove digit artifacts
        text = re.sub(r'\d+', '', text)
        
        # Remove special HTML/markup artifacts
        text = re.sub(r'<[^>]+>', '', text)
        
        # Keep only Devanagari, spaces, and basic punctuation
        allowed_chars = set()
        # Devanagari Unicode block
        allowed_chars.update(chr(i) for i in range(0x0900, 0x097F))
        # Common punctuation and spaces
        allowed_chars.update(' \n\t।,.\'"\'')
        
        cleaned = ''.join(char for char in text if char in allowed_chars or ord(char) < 128 and char in ' \n\t।.,')
        return cleaned

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Normalize quotes
        text = text.replace("'", "'").replace("\u2018", "'").replace("\u2019", "'")
        return text

    def postprocess_text(self, text: str) -> str:
        """Clean and normalize output text, remove artifacts"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove artifacts
        text = self.remove_artifacts(text)
        
        # Remove duplicate words/tokens more aggressively
        words = text.split()
        cleaned_words = []
        prev_word = ""
        for word in words:
            # Skip if identical to previous and not a single character
            if word != prev_word or len(word) == 1:
                cleaned_words.append(word)
                prev_word = word
        
        text = ' '.join(cleaned_words).strip()
        
        # Final validation - ensure it's mostly Devanagari
        devanagari_chars = sum(1 for c in text if self.is_devanagari(c))
        if len(text) > 0:
            devanagari_ratio = devanagari_chars / len(text)
            if devanagari_ratio < 0.7:  # If less than 70% Devanagari, something went wrong
                return ""  # Return empty if too much junk
        
        return text.strip()

    @torch.no_grad()
    def translate(self, english_text: str, max_length=64) -> str:
        # Preprocess input
        english_text = self.preprocess_text(english_text)
        
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "npi_Deva"

        inputs = self.tokenizer(english_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Optimized parameters: Beam search for exact domain match
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            min_length=2,
            num_beams=4,
            length_penalty=1.2,
            no_repeat_ngram_size=4,
            early_stopping=True,
            do_sample=False,
            repetition_penalty=2.0,
            # NEW: Enforce stricter generation
            diversity_penalty=0.0,
            num_beam_groups=1,
            temperature=1.0,
            top_p=1.0,  # No sampling, pure beam search
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Postprocess output to remove artifacts
        translation = self.postprocess_text(translation)
        return translation