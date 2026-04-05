"""
Data augmentation: Create variations of existing data without adding new examples
This increases effective dataset size by 2-3x
"""
import random
from pathlib import Path
from collections import defaultdict

def augment_dataset(input_file: Path, output_file: Path, multiplier=2):
    """
    Create augmented variations of existing data:
    - Paraphrasing (minimal changes)
    - Back-translation style (reorder slightly)
    - Register mixing (create intermediate formality levels)
    """
    
    # Read original data
    data = []
    with open(input_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            parts = line.rsplit('\t', 2)
            if len(parts) == 3:
                en, ne, reg = parts
                data.append((en.strip(), ne.strip(), reg.strip()))
    
    print(f"📊 Loaded {len(data)} examples")
    
    # Augmentation strategies
    augmented = []
    
    for en, ne, reg in data:
        augmented.append((en, ne, reg))  # Original
        
        # Strategy 1: Paraphrase common patterns
        en_aug = augment_english(en)
        if en_aug != en:
            augmented.append((en_aug, ne, reg))
        
        # Strategy 2: Register variants (if FORMAL, create SEMI-FORMAL variant)
        if reg == "FORMAL":
            ne_semi = formalize_nepali(ne, "SEMI")
            if ne_semi != ne:
                augmented.append((en, ne_semi, "SEMI-FORMAL"))
        elif reg == "INFORMAL":
            ne_semi = formalize_nepali(ne, "SEMI")
            if ne_semi != ne:
                augmented.append((en, ne_semi, "SEMI-FORMAL"))
    
    # Write augmented data
    with open(output_file, 'w', encoding='utf-8') as f:
        for en, ne, reg in augmented:
            f.write(f"{en}\t{ne}\t{reg}\n")
    
    print(f"✅ Created {len(augmented)} augmented examples (+{len(augmented)-len(data)} new)")
    print(f"   Original: {len(data)}, Augmented: {len(augmented)}")

def augment_english(text: str) -> str:
    """Minor paraphrasing of English text"""
    replacements = {
        "please": "kindly",
        "would you": "will you",
        "could you": "can you",
        "may i": "can i",
        "shall i": "should i",
        "i would like to": "i want to",
        "thank you": "thanks",
        "good morning": "hello",
    }
    
    text_lower = text.lower()
    for old, new in replacements.items():
        if old in text_lower:
            # Preserve case
            idx = text_lower.find(old)
            if idx >= 0:
                text = text[:idx] + new + text[idx+len(old):]
                return text
    
    return text  # No change if no match

def formalize_nepali(text: str, style: str) -> str:
    """
    Convert Nepali formality level
    FORMAL: Uses -नुहोस्, तपाई, साहब, आदि
    SEMI: Mixed formality
    INFORMAL: Uses -छु, तिमी, direct forms
    """
    
    if style == "SEMI":
        # Convert very formal markers to semi-formal
        text = text.replace("नुहोस्", "नु")
        text = text.replace("तपाईं", "तपाई")  # Slightly less formal
        text = text.replace("साहब", "")  # Remove title
        text = text.replace("हुन्छ", "छ")  # Less stiff form
    
    return text if text else None

# Run augmentation
if __name__ == "__main__":
    input_file = Path("data/raw/honoorifics.txt")
    output_file = Path("data/raw/honoorifics_augmented.txt")
    
    augment_dataset(input_file, output_file)
