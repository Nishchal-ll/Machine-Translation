#!/usr/bin/env python3
"""Minimal interface to compare fine-tuned and Hugging Face NLLB Nepali translations."""

import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

sys.path.insert(0, str(Path(__file__).parent))

from src.config import MODEL_DIR
from src.translator import NepaliTranslator

HF_BASELINE_MODEL = os.getenv("HF_BASELINE_MODEL", "facebook/nllb-200-distilled-600M")
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN")
TRAINED_MODEL_PATH = MODEL_DIR / "best_honorifics_model"

app = Flask(__name__)


def safe_load_trained_model(model_path: Path):
    if not model_path.exists():
        return None, f"Trained model not found at {model_path}."

    try:
        trained = NepaliTranslator(model_path, device="cpu")
        return trained, None
    except Exception as exc:
        return None, f"Failed to load trained model: {exc}"


class HFBaselineTranslator:
    def __init__(self, model_name: str, auth_token: str | None = None):
        self.model_name = model_name
        self.auth_token = auth_token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "npi_Deva"
        self.target_lang_id = self.tokenizer.lang_code_to_id.get("npi_Deva")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=False,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        )
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()
        self.generation_config = GenerationConfig.from_pretrained(self.model_name)
        self.model.generation_config = self.generation_config

    @torch.no_grad()
    def translate(self, text: str, max_length: int = 64) -> str:
        if not text or not text.strip():
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        generation_kwargs = {
            **inputs,
            "generation_config": self.generation_config,
            "max_length": max_length,
            "min_length": 2,
            "num_beams": 4,
            "length_penalty": 1.2,
            "no_repeat_ngram_size": 2,
            "early_stopping": True,
            "do_sample": False,
            "repetition_penalty": 1.5,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.target_lang_id is not None:
            generation_kwargs["forced_bos_token_id"] = self.target_lang_id

        outputs = self.model.generate(**generation_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


trained_translator, trained_load_error = safe_load_trained_model(TRAINED_MODEL_PATH)
if trained_load_error:
    print(f"⚠️  {trained_load_error}")

try:
    baseline_translator = HFBaselineTranslator(HF_BASELINE_MODEL, auth_token=HF_HUB_TOKEN)
    print(f"✅ Loaded Hugging Face baseline model: {HF_BASELINE_MODEL}")
except Exception as exc:
    baseline_translator = None
    print(f"❌ Failed to load Hugging Face baseline model: {exc}")


def compare_sentence(sentence: str) -> dict:
    sentence = sentence.strip()
    if not sentence:
        return {
            "success": False,
            "error": "Input sentence cannot be empty.",
        }

    trained_translation = None
    baseline_translation = None
    errors = {}

    if trained_translator is not None:
        try:
            trained_translation = trained_translator.translate(sentence)
        except Exception as exc:
            errors["trained"] = str(exc)

    if baseline_translator is not None:
        try:
            baseline_translation = baseline_translator.translate(sentence)
        except Exception as exc:
            errors["baseline"] = str(exc)

    return {
        "success": True,
        "input": sentence,
        "trained_translation": trained_translation,
        "baseline_translation": baseline_translation,
        "errors": errors,
    }


@app.route("/api/translate", methods=["POST"])
def api_translate():
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"success": False, "error": "Missing 'text' field."}), 400

    sentence = str(payload["text"]).strip()
    result = compare_sentence(sentence)
    status = 200 if result.get("success", False) else 400
    return jsonify(result), status


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "trained_model_loaded": trained_translator is not None,
        "baseline_model_loaded": baseline_translator is not None,
    }), 200


def main():
    parser = ArgumentParser(description="Minimal interface for trained and NLLB baseline Nepali translation.")
    parser.add_argument("--serve", action="store_true", help="Start the Flask API server.")
    parser.add_argument("--text", type=str, help="Translate a single English sentence through both models.")
    args = parser.parse_args()

    if args.serve:
        print("🚀 Starting translation API server on http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=False)
        return

    if args.text:
        response = compare_sentence(args.text)
        print(response)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
