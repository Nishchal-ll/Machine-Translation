#!/usr/bin/env python3
import sys
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODEL_DIR, MODEL_NAME, DATASET_FILES, DEVICE, SEED
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.evaluator import Evaluator


class HFTranslator:
    def __init__(self, model_path: str | Path, tokenizer_path: str | Path | None = None, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = Path(model_path) if isinstance(model_path, (str, Path)) and Path(model_path).exists() else str(model_path)
        self.tokenizer_path = tokenizer_path or self.model_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = "npi_Deva"
        self.target_lang_id = self.tokenizer.lang_code_to_id.get("npi_Deva")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_path,
            trust_remote_code=False,
            torch_dtype=torch.float16 if self.device.type == "cuda" else None,
        )
        self.model.to(self.device)
        self.model.eval()
        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
            self.model.generation_config = self.generation_config
        except Exception:
            self.generation_config = None

    @torch.no_grad()
    def translate(self, english_text: str, max_length=64) -> str:
        if not english_text or not english_text.strip():
            return ""

        inputs = self.tokenizer(
            english_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generation_kwargs = {
            **inputs,
            "generation_config": self.generation_config,
            "max_length": max_length,
            "min_length": 2,
            "num_beams": 4,
            "length_penalty": 1.0,
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


def format_pair(index: int, item, baseline_pred: str, trained_pred: str | None):
    lines = [f"\n--- Example {index} ---",
             f"English        : {item['english']}",
             f"Reference      : {item['nepali']}",
             f"Baseline       : {baseline_pred}"]
    if trained_pred is not None:
        lines.append(f"Fine-tuned     : {trained_pred}")
    return "\n".join(lines)


def evaluate_model(translation_model, test_data, model_name: str):
    evaluator = Evaluator(translation_model.model, translation_model.tokenizer, DEVICE)
    print(f"\n=== Evaluation for {model_name} ===")
    return evaluator.evaluate(test_data)


def create_compare_app(baseline_translator, trained_translator):
    template_folder = str(Path(__file__).parent.parent / "templates")
    app = Flask(__name__, template_folder=template_folder)

    @app.route('/')
    def compare_page():
        return render_template('compare.html')

    @app.route('/api/compare', methods=['POST'])
    def api_compare():
        data = request.get_json(silent=True)
        if not data or 'text' not in data:
            return jsonify({"success": False, "error": "Missing 'text' field."}), 400

        text = str(data['text']).strip()
        if not text:
            return jsonify({"success": False, "error": "Text cannot be empty."}), 400

        if baseline_translator is None and trained_translator is None:
            return jsonify({"success": False, "error": "No models available for comparison."}), 503

        response = {
            "success": True,
            "input": text,
            "baseline_translation": None,
            "trained_translation": None,
            "errors": {},
        }

        if baseline_translator is not None:
            try:
                response["baseline_translation"] = baseline_translator.translate(text)
            except Exception as exc:
                response["errors"]["baseline"] = str(exc)

        if trained_translator is not None:
            try:
                response["trained_translation"] = trained_translator.translate(text)
            except Exception as exc:
                response["errors"]["trained"] = str(exc)

        return jsonify(response), 200

    return app


def main():
    trained_model_path = Path(MODEL_DIR / "best_honorifics_model")

    baseline_path = MODEL_NAME
    baseline_tokenizer = None
    print(f"📌 Loading baseline model from {baseline_path}")
    baseline_translator = HFTranslator(baseline_path, tokenizer_path=baseline_tokenizer, device=DEVICE)
    print("✅ Baseline model loaded")

    trained_translator = None
    if trained_model_path.exists():
        print(f"📌 Loading trained model from {trained_model_path}:")
        trained_translator = HFTranslator(trained_model_path, device=DEVICE)
        print("✅ Trained model loaded")
    else:
        print(f"⚠️  No trained model found at {trained_model_path}. Only baseline results will be shown.")

    app = create_compare_app(baseline_translator, trained_translator)
    print("🚀 Starting comparison web interface at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
