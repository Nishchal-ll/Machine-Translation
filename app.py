"""
Flask API for Nepali Honorifics Translator
"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys
import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Force CPU-only inference for local runs on low-VRAM systems.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.translator import NepaliTranslator
from src.config import MODEL_DIR

app = Flask(__name__)

# Load trained model once at startup
trained_model_path = MODEL_DIR / "best_honorifics_model"
translator = None
trained_load_error = None
if not trained_model_path.exists():
    trained_load_error = "Model not found. Please train the model first."
    print(f"❌ {trained_load_error}")
else:
    try:
        translator = NepaliTranslator(trained_model_path, device="cpu")
        print("✅ Trained model loaded successfully")
    except Exception as e:
        trained_load_error = str(e)
        print(f"❌ Error loading trained model: {trained_load_error}")


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/translate', methods=['POST'])
def translate():
    """
    API endpoint for translation
    Expected JSON: {"text": "English text here"}
    Returns JSON with both fine-tuned and baseline translations.
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' field in request"
            }), 400

        english_text = data['text'].strip()
        if not english_text:
            return jsonify({
                "success": False,
                "error": "Text cannot be empty"
            }), 400

        if translator is None:
            error_message = "Trained model is not available."
            if trained_load_error:
                error_message += f" {trained_load_error}"
            return jsonify({"success": False, "error": error_message}), 503

        try:
            nepali_translation = translator.translate(english_text)
        except Exception as e:
            return jsonify({"success": False, "error": f"Translation error: {e}"}), 500

        return jsonify({
            "success": True,
            "input": english_text,
            "translation": nepali_translation,
        }), 200

    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({
            "success": False,
            "error": f"Translation error: {str(e)}"
        }), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": translator is not None,
        "trained_load_error": trained_load_error,
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Run on localhost:5000
    print("🚀 Starting Flask API server...")
    print("📖 Visit http://localhost:5000 to access the translator")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
