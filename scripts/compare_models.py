#!/usr/bin/env python3
import sys
from pathlib import Path
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).parent.parent))

from src.config import MODEL_DIR, MODEL_NAME, DATASET_FILES, DEVICE, SEED
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.translator import NepaliTranslator
from src.evaluator import Evaluator


def parse_args():
    parser = ArgumentParser(description="Compare baseline and fine-tuned Nepali honorific translations")
    parser.add_argument("--mode", choices=["sample", "test"], default="sample",
                        help="sample: compare a few example sentences; test: evaluate on the full test set")
    parser.add_argument("--examples", type=int, default=8,
                        help="Number of sample sentences to compare when using sample mode")
    parser.add_argument("--trained-model", type=str, default=str(MODEL_DIR / "best_honorifics_model"),
                        help="Path to the fine-tuned model directory")
    return parser.parse_args()


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


def main():
    args = parse_args()
    trained_model_path = Path(args.trained_model)

    print("📌 Loading baseline model:")
    baseline_translator = NepaliTranslator(MODEL_NAME, device=DEVICE)
    print("✅ Baseline model loaded")

    trained_translator = None
    if trained_model_path.exists():
        print(f"📌 Loading trained model from {trained_model_path}:")
        trained_translator = NepaliTranslator(trained_model_path, device=DEVICE)
        print("✅ Trained model loaded")
    else:
        print(f"⚠️  No trained model found at {trained_model_path}. Only baseline results will be shown.")

    all_data, _, _ = load_honorifics_from_register_files(DATASET_FILES)
    _, _, test_data = stratified_split(all_data, seed=SEED)

    if args.mode == "sample":
        print(f"\n📄 Comparing {args.examples} sample test sentences:")
        for idx, item in enumerate(test_data[: args.examples], start=1):
            baseline_pred = baseline_translator.translate(item["english"])
            trained_pred = trained_translator.translate(item["english"]) if trained_translator else None
            print(format_pair(idx, item, baseline_pred, trained_pred))

    if trained_translator:
        baseline_metrics = evaluate_model(baseline_translator, test_data, "Baseline Model")
        trained_metrics = evaluate_model(trained_translator, test_data, "Fine-tuned Model")

        print("\n=== Summary ===")
        print(f"Baseline BLEU   : {baseline_metrics['bleu']:.2f}")
        print(f"Trained BLEU    : {trained_metrics['bleu']:.2f}")
        print(f"Baseline Exact  : {baseline_metrics['exact']:.2f}%")
        print(f"Trained Exact   : {trained_metrics['exact']:.2f}%")
        print(f"Baseline chrF   : {baseline_metrics['chrf']:.2f}")
        print(f"Trained chrF    : {trained_metrics['chrf']:.2f}")
        print(f"Baseline TER    : {baseline_metrics['ter']:.2f}")
        print(f"Trained TER     : {trained_metrics['ter']:.2f}")
    else:
        if args.mode == "test":
            evaluate_model(baseline_translator, test_data, "Baseline Model")

    print("\n✅ Comparison complete.")


if __name__ == "__main__":
    main()
