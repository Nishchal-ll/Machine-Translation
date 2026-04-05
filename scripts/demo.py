import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.translator import NepaliTranslator
from src.config import MODEL_DIR

def main():
    model_path = MODEL_DIR / "best_honorifics_model"
    if not model_path.exists():
        print("❌ Model not found. Please train first.")
        return

    translator = NepaliTranslator(model_path)

    print("🇳🇵 English → Nepali Honorifics Translator (Type 'quit' to exit)\n")
    while True:
        text = input("English: ")
        if text.lower() in ['quit', 'exit']:
            break
        nepali = translator.translate(text)
        print(f"Nepali : {nepali}\n")

if __name__ == "__main__":
    main()