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
# Enable GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

from src.translator import NepaliTranslator
from src.config import MODEL_DIR


def detect_language_and_confidence(text):
    """
    Detect language and return (language, confidence, is_english)
    Returns confidence as 0-100 score
    """
    try:
        langs = detect_langs(text)
        english_confidence = 0
        detected_lang = None
        detected_confidence = 0
        
        for lang_prob in langs:
            lang_code = lang_prob.lang
            confidence = lang_prob.prob * 100
            
            if lang_code == 'en':
                english_confidence = confidence
            
            if confidence > detected_confidence:
                detected_lang = lang_code
                detected_confidence = confidence
        
        is_english = english_confidence > 20
        
        return {
            'detected_language': detected_lang,
            'detected_confidence': round(detected_confidence, 1),
            'english_confidence': round(english_confidence, 1),
            'is_english_like': is_english
        }
    except LangDetectException:
        return {
            'detected_language': 'unknown',
            'detected_confidence': 0,
            'english_confidence': 0,
            'is_english_like': False
        }


def analyze_honorific(nepali_text: str, english_text: str = ""):
    """
    Analyze the honorific register level of the translated Nepali text and English input.
    Returns tone category, label, pronoun representation, and confidence score.
    """
    ne = nepali_text or ""
    en = (english_text or "").lower()
    
    formal_markers = ["तपाईं", "तपाईंले", "तपाईंलाई", "तपाईंको", "हजुर", "हजुरले", "हजुरलाई", "मन्त्रीज्यू", "ज्यू", "हुनुहुन्छ", "हुनुहुन्थ्यो", "गर्नुहोस्", "गर्नुहोला", "गर्नुभयो", "दिनुहोस्", "बस्नुहोस्", "खानुहोस्", "जानुहोस्", "आउनुहोस्", "भन्नुहोस्"]
    semiformal_markers = ["तिमी", "तिमीले", "तिमीलाई", "तिम्रो", "तिमीहरू", "साथी", "भाइ", "बहिनी", "गर्यौ", "गर्छौ", "गर", "देऊ", "खाऊ", "जाऊ", "बस", "हेर", "सक्छौ", "थियौ"]
    informal_markers = ["तँ", "तँलाई", "तैँले", "तेरो", "तेरा", "तेरी", "छस्", "थिइस्", "गर्छस्", "गरिस्", "खास्", "खाइस्", "जास्", "गइस्", "दे", "नछो", "नमेट", "बाबु", "केटा"]

    formal_score = sum(1 for m in formal_markers if m in ne)
    semiformal_score = sum(1 for m in semiformal_markers if m in ne)
    informal_score = sum(1 for m in informal_markers if m in ne)

    # Check English clues if Nepali is ambiguous
    if any(k in en for k in ["sir", "madam", "ma'am", "please", "could you", "would you", "professor", "director", "officer", "mr.", "mrs."]):
        formal_score += 2
    if any(k in en for k in ["friend", "buddy", "colleague", "teammate", "assistant", "bro"]):
        semiformal_score += 2
    if any(k in en for k in ["kid", "child", "son", "little", "dude", "hey you"]):
        informal_score += 1.5

    if formal_score > semiformal_score and formal_score > informal_score:
        return {
            "level": "formal",
            "label": "Formal (उच्च आदरार्थ)",
            "pronoun": "तपाईं / हजुर",
            "suffix": "-होस् / -नुहुन्छ",
            "color": "#1a73e8",
            "description": "Used with elders, superiors, strangers, and formal contexts"
        }
    elif semiformal_score >= formal_score and semiformal_score > informal_score:
        return {
            "level": "semi-formal",
            "label": "Semi-Formal (मध्यम आदरार्थ)",
            "pronoun": "तिमी",
            "suffix": "-ऊ / -छौ",
            "color": "#0d652d",
            "description": "Used with friends, colleagues, siblings, and familiar people"
        }
    elif informal_score > 0:
        return {
            "level": "informal",
            "label": "Informal (निम्न आदरार्थ)",
            "pronoun": "तँ",
            "suffix": "-स् / -छस्",
            "color": "#e37400",
            "description": "Used with close childhood intimates, younger children, or informal speech"
        }
    else:
        # Default respectful Nepali register
        return {
            "level": "formal",
            "label": "Standard Formal (तपाईं)",
            "pronoun": "तपाईं",
            "suffix": "-होस्",
            "color": "#1a73e8",
            "description": "Standard polite register"
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
        if word_len >= 8:
            vowel_count = sum(1 for c in word.lower() if c in 'aeiou')
            if vowel_count < word_len * 0.2:
                gibberish_words += 1
    
    if len(words) > 0 and gibberish_words / len(words) > 0.3:
        return True, "Too many gibberish-like words"
    
    consonant_only_count = sum(1 for w in words if not any(c in 'aeiouAEIOU' for c in w))
    if len(words) > 0 and consonant_only_count / len(words) > 0.4:
        return True, "Too many consonant-only words"
    
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
    trained_load_error = "Model not found in " + str(trained_model_path)
    print(f"❌ {trained_load_error}")
else:
    try:
        translator = NepaliTranslator(trained_model_path)
        print("✅ Trained model loaded successfully on", translator.device)
    except Exception as e:
        trained_load_error = str(e)
        print(f"❌ Error loading trained model: {trained_load_error}")


@app.route('/')
def index():
    """Serve the main Google-style HTML page"""
    return render_template('index.html')


@app.route('/api/translate', methods=['POST'])
def translate():
    """
    API endpoint for translation with language detection and honorific tone analysis.
    Expected JSON: {"text": "English text here", "tone": "auto" | "formal" | "semi-formal" | "informal"}
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' field in request"
            }), 400

        english_text = data['text'].strip()
        desired_tone = data.get('tone', 'auto')

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
        
        response_data = {
            "success": True,
            "input": english_text,
            "language_detection": lang_detection,
            "warning": None,
            "tone_requested": desired_tone,
        }
        
        if lang_detection['english_confidence'] < 40 and not lang_detection['is_english_like']:
            response_data['warning'] = f"Input language may not be English (Confidence: {lang_detection['english_confidence']}%)."
        
        if translator is None:
            error_message = "Trained model is not available."
            if trained_load_error:
                error_message += f" ({trained_load_error})"
            return jsonify({
                **response_data,
                "success": False,
                "error": error_message
            }), 503

        try:
            # If a specific honorific tone is requested and the text has no cue, add subtle guidance
            input_to_translate = english_text
            if desired_tone == "formal" and not any(k in english_text.lower() for k in ["please", "sir", "madam"]):
                # Hint towards formal
                input_to_translate = f"Please {english_text}" if not english_text.lower().startswith("please") else english_text
            elif desired_tone == "semi-formal" and not any(k in english_text.lower() for k in ["friend", "buddy", "brother"]):
                input_to_translate = f"{english_text}, friend"
            elif desired_tone == "informal" and not any(k in english_text.lower() for k in ["bro", "kid", "dude"]):
                input_to_translate = f"{english_text}, kid"

            nepali_translation = translator.translate(input_to_translate)
            
            # Clean injected tags from output if any
            clean_nepali = nepali_translation
            if desired_tone == "semi-formal":
                clean_nepali = re.sub(r'[,،]\s*साथी\s*[।?!]?$', ' ।', clean_nepali).strip()
            elif desired_tone == "informal":
                clean_nepali = re.sub(r'[,،]\s*बाबु\s*[।?!]?$', ' ।', clean_nepali).strip()

            if not clean_nepali:
                clean_nepali = nepali_translation

            response_data['translation'] = clean_nepali
            response_data['honorific_analysis'] = analyze_honorific(clean_nepali, english_text)

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
        "device": str(translator.device) if translator else "none",
        "trained_load_error": trained_load_error,
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("🚀 Starting Honorifics Translator Flask Server...")
    print("📖 Visit http://localhost:5000 in your browser")
    app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)