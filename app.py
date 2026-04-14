"""
Flask API for Nepali Honorifics Translator
"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Force CPU-only inference for local runs on low-VRAM systems.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.translator import NepaliTranslator
from src.config import MODEL_DIR

app = Flask(__name__)

# Load model once at startup
model_path = MODEL_DIR / "best_honorifics_model"
if not model_path.exists():
    print("❌ Model not found. Please train the model first.")
    translator = None
else:
    try:
        translator = NepaliTranslator(model_path, device="cpu")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        translator = None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/translate', methods=['POST'])
def translate():
    """
    API endpoint for translation
    Expected JSON: {"text": "English text here"}
    Returns JSON: {"translation": "Nepali translation", "success": true}
    """
    try:
        if translator is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Please train the model first."
            }), 503

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

        # Translate
        nepali_translation = translator.translate(english_text)

        return jsonify({
            "success": True,
            "input": english_text,
            "translation": nepali_translation
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
        "model_loaded": translator is not None
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
