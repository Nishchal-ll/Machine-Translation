import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.evaluator import Evaluator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    model_path = MODEL_DIR / "best_honorifics_model"
    if not model_path.exists():
        print("❌ No trained model found. Train first using scripts/train.py")
        return

    all_data, _, _ = load_honorifics_from_register_files(DATASET_FILES)
    _, _, test_data = stratified_split(all_data, seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    evaluator = Evaluator(model, tokenizer, DEVICE)
    evaluator.evaluate(test_data)

if __name__ == "__main__":
    main()