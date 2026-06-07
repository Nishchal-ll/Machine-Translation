"""
Flask API for Nepali Honorifics Translator
"""
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys
import os
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from langdetect import detect, detect_langs, LangDetectException

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Force CPU-only inference for local runs on low-VRAM systems.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from src.translator import NepaliTranslator
from src.config import MODEL_DIR


def detect_language_and_confidence(text):
    """
    Detect language and return (language, confidence, is_english)
    Returns confidence as 0-100 score
    """
    try:
        # Get all language probabilities
        langs = detect_langs(text)
        
        # Find confidence for English
        english_confidence = 0
        detected_lang = None
        detected_confidence = 0
        
        for lang_prob in langs:
            lang_code = lang_prob.lang
            confidence = lang_prob.prob * 100
            
            if lang_code == 'en':
                english_confidence = confidence
            
            # Track the highest probability language
            if confidence > detected_confidence:
                detected_lang = lang_code
                detected_confidence = confidence
        
        is_english = english_confidence > 20  # If English confidence > 20%, consider it English-like
        
        return {
            'detected_language': detected_lang,
            'detected_confidence': round(detected_confidence, 1),
            'english_confidence': round(english_confidence, 1),
            'is_english_like': is_english
        }
    except LangDetectException:
        # If detection fails, return neutral result
        return {
            'detected_language': 'unknown',
            'detected_confidence': 0,
            'english_confidence': 0,
            'is_english_like': False
        }


def check_obvious_gibberish(text):
    """
    Check for obvious gibberish patterns that should be rejected.
    Returns (is_gibberish, reason)
    """
    if not text or len(text) < 2:
        return True, "Text is too short"
    
    # Check for excessive special characters (>30%)
    special_char_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    if special_char_count > len(text) * 0.3:
        return True, "Too many special characters"
    
    # Check for excessive repeated characters (5+ of same char)
    if re.search(r'(.)\1{4,}', text):
        return True, "Excessive repeated characters detected"
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    if not words:
        return True, "No recognizable words found"
    
    # Check if text is mostly numbers
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < len(text) * 0.3:
        return True, "Text is mostly non-alphabetic"
    
    # Check for too many long gibberish-like words
    gibberish_words = 0
    for word in words:
        word_len = len(word)
        # Long words with few vowels are likely gibberish
        if word_len >= 8:
            vowel_count = sum(1 for c in word.lower() if c in 'aeiou')
            if vowel_count < word_len * 0.2:  # Less than 20% vowels
                gibberish_words += 1
    
    if len(words) > 0 and gibberish_words / len(words) > 0.3:
        return True, "Too many gibberish-like words"
    
    # Check too many consonant-only words (40%+)
    consonant_only_count = sum(1 for w in words if not any(c in 'aeiouAEIOU' for c in w))
    if len(words) > 0 and consonant_only_count / len(words) > 0.4:
        return True, "Too many consonant-only words"
    
    # Check average word length
    avg_word_length = sum(len(w) for w in words) / len(words)
    if avg_word_length > 15:
        return True, "Average word length unusually high"
    
    return False, ""

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
    API endpoint for translation with language detection.
    Expected JSON: {"text": "English text here"}
    Returns JSON with translation, language detection, and confidence scores.
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

        # Check for obvious gibberish (hard reject)
        is_gibberish, gibberish_reason = check_obvious_gibberish(english_text)
        if is_gibberish:
            return jsonify({
                "success": False,
                "error": f"Invalid input: {gibberish_reason}",
                "warning": None
            }), 400

        # Detect language and confidence
        lang_detection = detect_language_and_confidence(english_text)
        
        # Build response with language detection info
        response_data = {
            "success": True,
            "input": english_text,
            "language_detection": lang_detection,
            "warning": None
        }
        
        # Warn if English confidence is low (but still translate)
        if lang_detection['english_confidence'] < 50:
            response_data['warning'] = f"Low English confidence ({lang_detection['english_confidence']}%). Translation quality may be poor."
        
        if translator is None:
            error_message = "Trained model is not available."
            if trained_load_error:
                error_message += f" {trained_load_error}"
            return jsonify({
                **response_data,
                "success": False,
                "error": error_message
            }), 503

        try:
            nepali_translation = translator.translate(english_text)
            response_data['translation'] = nepali_translation
        except Exception as e:
            return jsonify({
                **response_data,
                "success": False,
                "error": f"Translation error: {e}"
            }), 500

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error during translation: {e}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
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