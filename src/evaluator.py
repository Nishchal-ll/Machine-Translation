import torch
from sacrebleu import corpus_bleu
from tqdm import tqdm
import re

class Evaluator:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def is_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script"""
        devanagari_range = range(0x0900, 0x097F)
        return any(ord(char) in devanagari_range for char in text)

    def remove_artifacts(self, text: str) -> str:
        """Remove hallucination artifacts"""
        text = re.sub(r'[a-zA-Z]{2,}', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        
        allowed_chars = set()
        allowed_chars.update(chr(i) for i in range(0x0900, 0x097F))
        allowed_chars.update(' \n\t।,.\'"')
        
        cleaned = ''.join(char for char in text if char in allowed_chars or (ord(char) < 128 and char in ' \n\t।.,'))
        return cleaned

    @torch.no_grad()
    def generate_translation(self, text: str, max_length=64):
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "npi_Deva"

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Conservative generation for clean output
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
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        translation = self.remove_artifacts(translation)
        return translation.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison (remove extra spaces)"""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.lower()

    def is_similar(self, text1: str, text2: str, threshold=0.8) -> bool:
        """Check if texts are similar using word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return False
        overlap = len(words1 & words2)
        total = max(len(words1), len(words2))
        return (overlap / total) >= threshold

    def evaluate(self, test_data):
        references = []
        predictions = []
        exact_matches = 0
        partial_matches = 0  # Fuzzy matching for close translations

        print("Evaluating on test set...")
        for item in tqdm(test_data):
            pred = self.generate_translation(item["english"])
            predictions.append(pred)
            references.append([item["nepali"]])   # sacrebleu expects list of lists
            
            # Normalize for comparison
            pred_norm = self.normalize_text(pred)
            ref_norm = self.normalize_text(item["nepali"])
            
            # Check for exact match
            if pred_norm == ref_norm:
                exact_matches += 1
                partial_matches += 1
            # Check for partial match (at least 80% similar)
            elif self.is_similar(pred_norm, ref_norm):
                partial_matches += 1

        bleu = corpus_bleu(predictions, references)
        exact_accuracy = (exact_matches / len(test_data)) * 100
        partial_accuracy = (partial_matches / len(test_data)) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"🔵 SacreBLEU Score        : {bleu.score:.2f}")
        print(f"🟢 Exact Match Accuracy   : {exact_accuracy:.2f}% ({exact_matches}/{len(test_data)})")
        print(f"🟡 Partial Match Accuracy : {partial_accuracy:.2f}% ({partial_matches}/{len(test_data)})")
        print(f"{'='*60}")
        
        return {"bleu": bleu.score, "exact": exact_accuracy, "partial": partial_accuracy}