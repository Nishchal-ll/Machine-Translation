# pyrefly: ignore [missing-import]
import torch
from pathlib import Path
import re

try:
    from peft import PeftModel
    HAS_LORA = True
except ImportError:
    HAS_LORA = False

from src.config import MODEL_NAME

class NepaliTranslator:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path | None = None, device=None):
        from transformers import AutoTokenizer
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if isinstance(model_path, (str, Path)) and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            self.model_path = str(model_path)

        if tokenizer_path is None:
            tokenizer_path = self.model_path

        if isinstance(tokenizer_path, (str, Path)) and Path(tokenizer_path).exists():
            tokenizer_path = Path(tokenizer_path)

        # Load tokenizer with fallback to base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as tok_exc:
            print(f"⚠️  Could not load tokenizer from {tokenizer_path}: {tok_exc}")
            print(f"   Loading tokenizer from base model {MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        self.model = self._load_model(self.model_path)
        self.model.eval()
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "npi_Deva"
        try:
            from transformers import GenerationConfig
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
            self.model.generation_config = self.generation_config
        except Exception:
            self.generation_config = None

    def _has_lora_adapter(self, model_path: Path) -> bool:
        if not isinstance(model_path, Path):
            return False
        return (model_path / "adapter_config.json").exists() or (model_path / "adapter_model.bin").exists()

    def _load_model(self, model_path: Path):
        from transformers import AutoModelForSeq2SeqLM

        device_map = "cpu"
        torch_dtype = None
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            device_map = "auto"
            torch_dtype = torch.float16

        # Check if we have a safetensors weights file
        safetensors_path = Path(model_path) / "model.safetensors" if isinstance(model_path, (str, Path)) else None
        if safetensors_path and safetensors_path.exists():
            print(f"🔄 Found safetensors file, loading weights...")
            try:
                from safetensors.torch import load_file
                local_base_path = Path(model_path) / "base_model"
                if local_base_path.exists():
                    print(f"  Loading local base model: {local_base_path}")
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        local_base_path,
                        trust_remote_code=True,
                        use_safetensors=True,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                    )
                else:
                    print(f"  Loading base model: {MODEL_NAME}")
                    base_model = AutoModelForSeq2SeqLM.from_pretrained(
                        MODEL_NAME,
                        trust_remote_code=True,
                        use_safetensors=False,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                    )
                print(f"  Applying weights from {safetensors_path.name}")
                state_dict = load_file(str(safetensors_path))
                result = base_model.load_state_dict(state_dict, strict=False)
                print(f"✅ Model loaded with fine-tuned weights")
                if result.missing_keys:
                    print(f"   (Some keys not loaded: {len(result.missing_keys)})")
                return base_model
            except Exception as weights_exc:
                print(f"⚠️  Failed to load fine-tuned weights: {weights_exc}")
                print(f"   Falling back to base model only...")
                try:
                    return AutoModelForSeq2SeqLM.from_pretrained(
                        MODEL_NAME,
                        trust_remote_code=True,
                        use_safetensors=False,
                        device_map=device_map,
                        torch_dtype=torch_dtype,
                    )
                except Exception as base_exc:
                    raise RuntimeError(f"Failed to load base model: {base_exc}") from base_exc

        # Try loading as a full model directory (if no safetensors file)
        try:
            return AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_safetensors=False,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:
            # Try LoRA adapters
            if HAS_LORA and self._has_lora_adapter(model_path):
                print(f"⚠️  Attempting PEFT LoRA adapter load for {model_path}")
                try:
                    if (Path(model_path) / "base_model").exists():
                        base_model = AutoModelForSeq2SeqLM.from_pretrained(
                            model_path / "base_model",
                            trust_remote_code=True,
                            use_safetensors=False,
                            device_map=device_map,
                            torch_dtype=torch_dtype,
                        )
                    else:
                        base_model = AutoModelForSeq2SeqLM.from_pretrained(
                            MODEL_NAME,
                            trust_remote_code=True,
                            use_safetensors=False,
                            device_map=device_map,
                            torch_dtype=torch_dtype,
                        )
                    return PeftModel.from_pretrained(base_model, model_path, device_map=device_map)
                except Exception as nested_exc:
                    raise RuntimeError(
                        f"Failed to load LoRA-enhanced model from {model_path}: {nested_exc}"
                    ) from nested_exc
            
            # Final fallback to base model
            print(f"⚠️  Could not load model-specific weights, using base model {MODEL_NAME}")
            try:
                return AutoModelForSeq2SeqLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True,
                    use_safetensors=False,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                )
            except Exception as base_exc:
                raise RuntimeError(f"Failed to load base model: {base_exc}") from base_exc

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
        
        # Keep only Devanagari, spaces, and punctuation expected in translations.
        allowed_chars = set()
        # Devanagari Unicode block
        allowed_chars.update(chr(i) for i in range(0x0900, 0x097F))
        # Common punctuation and spaces
        allowed_chars.update(' \n\t।,.?!;:-()"\'')
        
        cleaned = ''.join(
            char for char in text
            if char in allowed_chars or (ord(char) < 128 and char in ' \n\t.,?!;:-()"\'')
        )
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
        
        # Final validation - ensure it's mostly Devanagari among alphabetic characters
        alphabetic_chars = [c for c in text if c.isalpha() or (0x0900 <= ord(c) <= 0x097F)]
        if alphabetic_chars:
            devanagari_chars = sum(1 for c in alphabetic_chars if self.is_devanagari(c))
            devanagari_ratio = devanagari_chars / len(alphabetic_chars)
            if devanagari_ratio < 0.7:  # If less than 70% Devanagari, something went wrong
                return ""  # Return empty if too much junk
        
        return text.strip()

    def split_into_sentences(self, text: str) -> list[str]:
        """Split paragraph into sentences while keeping sentence-ending punctuation."""
        chunks = re.findall(r'[^.!?।]+[.!?।]*', text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    @torch.no_grad()
    def _get_forced_bos_token_id(self, tgt_lang: str):
        if hasattr(self.tokenizer, "lang_code_to_id"):
            return self.tokenizer.lang_code_to_id.get(tgt_lang)
        return None

    def translate_batch(self, english_texts: list[str], max_length=64) -> list[str]:
        if not english_texts:
            return []

        cleaned_inputs = [self.preprocess_text(text) for text in english_texts]

        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "npi_Deva"

        inputs = self.tokenizer(
            cleaned_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        forced_bos_token_id = self._get_forced_bos_token_id(self.tokenizer.tgt_lang)
        generate_kwargs = {
            "generation_config": self.generation_config,
            "max_length": max_length,
            "min_length": 2,
            "num_beams": 4,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "do_sample": False,
            "repetition_penalty": 1.5,
            "diversity_penalty": 0.0,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if forced_bos_token_id is not None:
            generate_kwargs["forced_bos_token_id"] = forced_bos_token_id

        outputs = self.model.generate(**inputs, **generate_kwargs)

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [self.postprocess_text(text) for text in decoded]

    @torch.no_grad()
    def translate(self, english_text: str, max_length=64) -> str:
        sentences = self.split_into_sentences(english_text)
        if not sentences:
            return ""

        # For paragraphs, translate sentence chunks in parallel batches.
        if len(sentences) > 1:
            translated_sentences = self.translate_batch(sentences, max_length=max_length)
            return " ".join(s for s in translated_sentences if s).strip()

        translated = self.translate_batch(sentences, max_length=max_length)
        return translated[0] if translated else ""