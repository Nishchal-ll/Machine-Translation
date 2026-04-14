"""
Error Analysis for English-Nepali Translation
Classifies errors into 5 categories with detailed reasoning.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.data_utils import load_honorifics_from_register_files, stratified_split
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm
import re
import pandas as pd
from collections import Counter

# Nepali honorific markers
HONORIFIC_MARKERS = {
    'तपाईं',           # formal "you"
    'तिमी',            # informal "you"
    'गर्नुहोस्',        # formal "do"
    'गर',              # informal "do"
    'नुहुँ',           # formal suffix
    'छ',              # informal suffix
    'हुनुहुन्छ',       # formal verb
    'हुन्छ',           # informal verb
    'छौ',             # informal verb (you)
    'हुनुभयो',        # formal past
    'भयो',            # informal past
    'सर',             # Sir (honorific)
    'मेडम',           # Madam (honorific)
    'कृपया',          # Please (formal)
}

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def get_words(text: str):
    """Extract words from Nepali text"""
    text = normalize_text(text)
    # Split on whitespace and punctuation
    words = re.split(r'[\s।,\.\?!;:\-"\']+', text)
    return [w for w in words if w]  # Remove empty strings

def word_overlap(ref_words, pred_words):
    """Calculate word overlap ratio"""
    if not ref_words or not pred_words:
        return 0.0
    overlap = len(set(ref_words) & set(pred_words))
    total = max(len(ref_words), len(pred_words))
    return overlap / total if total > 0 else 0.0

def jaccard_similarity(ref_words, pred_words):
    """Jaccard similarity between word sets"""
    ref_set = set(ref_words)
    pred_set = set(pred_words)
    if not ref_set or not pred_set:
        return 0.0
    intersection = len(ref_set & pred_set)
    union = len(ref_set | pred_set)
    return intersection / union if union > 0 else 0.0

def has_honorific_mismatch(ref_text, pred_text):
    """Check if honorific usage differs"""
    ref_norm = normalize_text(ref_text)
    pred_norm = normalize_text(pred_text)
    
    ref_honorifics = [m for m in HONORIFIC_MARKERS if m in ref_norm]
    pred_honorifics = [m for m in HONORIFIC_MARKERS if m in pred_norm]
    
    return ref_honorifics != pred_honorifics, ref_honorifics, pred_honorifics

def classify_error(ref_text, pred_text):
    """
    Classify error into 5 categories:
    1. CORRECT - Exact or very close match
    2. LEXICAL_VARIATION - Same meaning, different words
    3. WORD_ORDER_ERROR - Same words, different order
    4. SEMANTIC_ERROR - Wrong meaning/facts
    5. HONORIFIC_ERROR - Wrong politeness level
    """
    
    # Exact match
    if normalize_text(ref_text) == normalize_text(pred_text):
        return "CORRECT", "Exact match"
    
    ref_words = get_words(ref_text)
    pred_words = get_words(pred_text)
    
    overlap = word_overlap(ref_words, pred_words)
    jaccard = jaccard_similarity(ref_words, pred_words)
    
    # Check honorific mismatch
    has_mismatch, ref_hon, pred_hon = has_honorific_mismatch(ref_text, pred_text)
    if has_mismatch and len(ref_hon) > 0 and len(pred_hon) > 0:
        ref_hon_str = ", ".join(ref_hon)
        pred_hon_str = ", ".join(pred_hon)
        return "HONORIFIC_ERROR", f"Honorifics: '{ref_hon_str}' vs '{pred_hon_str}'"
    
    # High word overlap but different order/form = WORD_ORDER_ERROR
    if overlap >= 0.8 and len(ref_words) > 0 and len(pred_words) > 0:
        # Check if word order is actually different
        if ref_words != pred_words:  # Same words but different order
            return "WORD_ORDER_ERROR", f"Word overlap: {overlap:.1%}, Different order/form"
        else:
            return "CORRECT", "Same words, same order"
    
    # Moderate overlap = LEXICAL_VARIATION
    if 0.6 <= overlap < 0.8:
        missing_words = set(ref_words) - set(pred_words)
        extra_words = set(pred_words) - set(ref_words)
        reason = f"Word overlap: {overlap:.1%}"
        if missing_words:
            reason += f" | Missing: {', '.join(list(missing_words)[:2])}"
        if extra_words:
            reason += f" | Extra: {', '.join(list(extra_words)[:2])}"
        return "LEXICAL_VARIATION", reason
    
    # Low overlap = SEMANTIC_ERROR
    if overlap < 0.6:
        missing_count = len(set(ref_words) - set(pred_words))
        return "SEMANTIC_ERROR", f"Low word overlap: {overlap:.1%} | Missing {missing_count} key words"
    
    return "SEMANTIC_ERROR", "Unable to classify"

def remove_artifacts(text: str) -> str:
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
def generate_translations(model, tokenizer, test_data, device):
    """Generate translations"""
    predictions = []
    
    for item in tqdm(test_data, desc="Generating translations"):
        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = "npi_Deva"
        
        inputs = tokenizer(item["english"], return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            min_length=2,
            num_beams=4,
            length_penalty=1.2,
            no_repeat_ngram_size=4,
            early_stopping=True,
            do_sample=False,
            repetition_penalty=2.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = remove_artifacts(pred).strip()
        predictions.append(pred)
    
    return predictions

def main():
    print("📊 Error Analysis for English-Nepali Translation\n")
    
    model_path = MODEL_DIR / "best_honorifics_model"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    # Load data
    print("Loading data...")
    all_data, _, _ = load_honorifics_from_register_files(DATASET_FILES)
    train_data, val_data, test_data = stratified_split(all_data, seed=SEED)
    
    # Take subset for analysis (first 500 for speed)
    test_data = test_data[:500]
    print(f"Analyzing {len(test_data)} samples...")
    
    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(DEVICE)
    model.eval()
    print(f"✅ Model loaded on {DEVICE}\n")
    
    # Generate predictions
    predictions = generate_translations(model, tokenizer, test_data, DEVICE)
    
    # Classify errors
    print("\n🔍 Classifying errors...\n")
    
    results = []
    category_counts = Counter()
    
    for i, (item, pred) in enumerate(tqdm(zip(test_data, predictions), total=len(test_data))):
        en = item["english"]
        ref = item["nepali"]
        
        category, reason = classify_error(ref, pred)
        category_counts[category] += 1
        
        results.append({
            "EN": en,
            "REF": ref,
            "PRED": pred,
            "CATEGORY": category,
            "REASON": reason
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = Path(__file__).parent.parent / "error_analysis.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print statistics
    print("\n" + "="*70)
    print("📈 ERROR CLASSIFICATION SUMMARY")
    print("="*70)
    
    total = len(results)
    for category in ["CORRECT", "LEXICAL_VARIATION", "WORD_ORDER_ERROR", "HONORIFIC_ERROR", "SEMANTIC_ERROR"]:
        count = category_counts[category]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{category:20} : {count:4d} ({percentage:5.1f}%)")
    
    print("="*70)
    
    # Show breakdown by category
    print("\n📋 Sample Errors by Category:\n")
    
    for category in ["CORRECT", "LEXICAL_VARIATION", "WORD_ORDER_ERROR", "HONORIFIC_ERROR", "SEMANTIC_ERROR"]:
        category_df = df[df["CATEGORY"] == category]
        if len(category_df) > 0:
            print(f"\n{'='*70}")
            print(f"🔹 {category} (showing first 2)")
            print(f"{'='*70}")
            
            for idx, row in category_df.head(2).iterrows():
                print(f"\nEN: {row['EN']}")
                print(f"REF: {row['REF']}")
                print(f"PRED: {row['PRED']}")
                print(f"REASON: {row['REASON']}")
    
    print("\n" + "="*70)
    print("✅ Error analysis complete!")
    print(f"Full results saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
