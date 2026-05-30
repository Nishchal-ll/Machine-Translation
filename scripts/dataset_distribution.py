#!/usr/bin/env python3
import sys
import random
import re
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config import DATASET_FILES, MODEL_DIR, SEED
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.translator import NepaliTranslator

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError(
        "Matplotlib is required to generate dataset charts. "
        "Install it with `pip install matplotlib` or `pip install -r requirements.txt`."
    ) from exc

import numpy as np
import textwrap
import torch

HONORIFIC_TERMS = {
    "तपाईं",
    "तिमी",
    "उहाँ",
    "श्री",
    "आदरणीय",
    "हजुर",
    "जनाब",
    "महोदय",
    "महाशय",
    "बूढा",
    "बुढी",
    "बाबु",
    "आमा",
    "दाई",
    "बहिनी",
    "श्रीमान",
    "श्रीमती",
}


def parse_args():
    parser = ArgumentParser(description="Generate dataset charts and analysis for the honorifics dataset.")
    parser.add_argument(
        "--split-output",
        type=Path,
        default=Path("dataset_distribution.png"),
        help="Path to save the split distribution chart image.",
    )
    parser.add_argument(
        "--lengths-output",
        type=Path,
        default=Path("sentence_length_distribution.png"),
        help="Path to save the sentence length histogram image.",
    )
    parser.add_argument(
        "--error-output",
        type=Path,
        default=Path("error_analysis.png"),
        help="Path to save the error analysis bar chart image.",
    )
    parser.add_argument(
        "--loss-output",
        type=Path,
        default=Path("length_vs_loss.png"),
        help="Path to save the token length vs loss chart image.",
    )
    parser.add_argument(
        "--analysis-image-output",
        type=Path,
        default=Path("dataset_analysis.png"),
        help="Path to save the dataset analysis summary image.",
    )
    parser.add_argument(
        "--chart-type",
        choices=["bar", "pie"],
        default="bar",
        help="Chart style to generate for the split distribution: bar or pie.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used for reproducible splitting and sampling.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_DIR / "best_honorifics_model",
        help="Path to the trained model for translation comparison and loss plotting.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for model translation. Use cpu for low-memory laptops.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of example translation pairs to include in the image.",
    )
    parser.add_argument(
        "--analysis-count",
        type=int,
        default=5,
        help="Number of examples used for translation/error/loss analysis.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=48,
        help="Maximum token length for model translation and loss computation.",
    )
    parser.add_argument(
        "--translate-batch-size",
        type=int,
        default=1,
        help="Batch size for model translation. Use 1 for memory-constrained machines.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable faster laptop-friendly defaults for analysis.",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model-based translation comparison, error analysis, and loss plotting.",
    )
    parser.add_argument(
        "--skip-loss",
        action="store_true",
        help="Skip token length vs loss chart generation.",
    )
    return parser.parse_args()


def build_split_summary(data, seed):
    train_data, val_data, test_data = stratified_split(data, seed=seed)
    return {
        "Train": len(train_data),
        "Validation": len(val_data),
        "Test": len(test_data),
    }


def build_register_counts(dataset_files):
    counts = {}
    for register, path in dataset_files.items():
        line_count = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
        counts[register] = line_count
    return counts


def plot_chart(split_counts, output_path, chart_type="bar"):
    labels = list(split_counts.keys())
    values = list(split_counts.values())
    total = sum(values)
    percentages = [value / total * 100 for value in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    if chart_type == "bar":
        bars = ax.bar(labels, values, color=["#4c78a8", "#f58518", "#e45756"], edgecolor="black")
        ax.set_ylabel("Sentence Pair Count")
        ax.set_title(f"Dataset Split Distribution (Total {total:,} sentence pairs)")
        ax.set_ylim(0, max(values) * 1.15)
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(values) * 0.02,
                f"{height:,} ({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    else:
        ax.pie(
            values,
            labels=[f"{label} ({count:,})" for label, count in zip(labels, values)],
            autopct="%1.1f%%",
            startangle=140,
            colors=["#4c78a8", "#f58518", "#e45756"],
            wedgeprops={"edgecolor": "black"},
            textprops={"fontsize": 10},
        )
        ax.set_title(f"Dataset Split Distribution (Total {total:,} sentence pairs)")
        ax.axis("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def get_sentence_lengths(data):
    english_lengths = [len(item["english"].strip().split()) for item in data]
    nepali_lengths = [len(item["nepali"].strip().split()) for item in data]
    return english_lengths, nepali_lengths


def plot_length_histograms(english_lengths, nepali_lengths, output_path):
    fig, (ax_en, ax_ne) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    max_len = max(max(english_lengths, default=0), max(nepali_lengths, default=0))
    bins = min(30, max_len + 1) if max_len > 0 else 10

    ax_en.hist(english_lengths, bins=bins, color="#4c78a8", edgecolor="black")
    ax_en.set_title("English Sentence Lengths")
    ax_en.set_xlabel("Words per sentence")
    ax_en.set_ylabel("Count")
    ax_en.grid(axis="y", alpha=0.35)

    ax_ne.hist(nepali_lengths, bins=bins, color="#f58518", edgecolor="black")
    ax_ne.set_title("Nepali Sentence Lengths")
    ax_ne.set_xlabel("Words per sentence")
    ax_ne.set_ylabel("Count")
    ax_ne.grid(axis="y", alpha=0.35)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def strip_punctuation(text):
    return re.sub(r"[।?!,.;:\"'()\[\]{}]", "", text).strip()


def punctuation_only_diff(reference, prediction):
    return strip_punctuation(reference) == strip_punctuation(prediction) and reference != prediction


def contains_honorific(text):
    return any(term in text for term in HONORIFIC_TERMS)


def honorific_mismatch(reference, prediction):
    ref_has = contains_honorific(reference)
    pred_has = contains_honorific(prediction)
    return ref_has != pred_has


def word_overlap(reference, prediction):
    ref_words = set(reference.split())
    pred_words = set(prediction.split())
    if not ref_words or not pred_words:
        return 0.0
    return len(ref_words & pred_words) / max(len(ref_words), len(pred_words))


def classify_error(reference, prediction):
    ref_norm = normalize_text(reference)
    pred_norm = normalize_text(prediction)

    if pred_norm == ref_norm:
        return None
    if punctuation_only_diff(ref_norm, pred_norm):
        return "punctuation errors"
    if honorific_mismatch(ref_norm, pred_norm):
        return "honorific mismatch"
    if word_overlap(ref_norm, pred_norm) >= 0.7:
        return "lexical errors"
    return "grammar issues"


def plot_error_analysis(error_counts, output_path):
    labels = list(error_counts.keys())
    values = list(error_counts.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    colors = ["#4c78a8", "#f58518", "#e45756", "#54a24b"]

    total = sum(values)
    bars = ax.bar(labels, values, color=colors[: len(labels)], edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Error Analysis by Category\n(Count and % of analysed examples)")
    ax.set_ylim(0, max(values or [1]) * 1.25)

    for bar, value in zip(bars, values):
        pct = (value / total * 100) if total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values or [1]) * 0.03,
            f"{int(value)}\n({pct:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_length_vs_loss(lengths, losses, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.scatter(lengths, losses, color="#4c78a8", alpha=0.6)
    ax.set_xlabel("Input token length")
    ax.set_ylabel("Loss")
    ax.set_title("Token Length vs Loss")
    ax.grid(alpha=0.3)

    if len(lengths) >= 3:
        bin_count = min(10, max(lengths) - min(lengths) + 1)
        bin_edges = np.linspace(min(lengths), max(lengths), bin_count + 1)
        indices = np.digitize(lengths, bin_edges, right=True)
        avg_loss = [
            np.mean([losses[i] for i, idx in enumerate(indices) if idx == b])
            if any(idx == b for idx in indices)
            else np.nan
            for b in range(1, len(bin_edges))
        ]
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        ax.plot(bin_centers, avg_loss, color="#f58518", marker="o", linestyle="-", label="Avg loss per length bin")
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def wrap_text(text, width=40):
    return "\n".join(textwrap.wrap(str(text), width=width))


def save_analysis_image(output_path, split_counts, length_summary, examples, error_counts=None, dataset_samples=None, analysis_size=None):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    summary_text = [
        "DATASET ANALYSIS SUMMARY",
        "",
        "Dataset split:",
    ]
    total = sum(split_counts.values())
    for subset, count in split_counts.items():
        summary_text.append(f"  {subset}: {count:,} ({count / total * 100:.1f}%)")

    summary_text.extend([
        "",
        "Sentence length summary:",
        f"  English: min={length_summary['english_min']}, max={length_summary['english_max']}, avg={length_summary['english_avg']:.2f}",
        f"  Nepali: min={length_summary['nepali_min']}, max={length_summary['nepali_max']}, avg={length_summary['nepali_avg']:.2f}",
    ])

    if analysis_size is not None:
        summary_text.extend([
            "",
            f"Test analysis sample size: {analysis_size}",
        ])

    if error_counts is not None:
        summary_text.extend([
            "",
            "Error analysis:",
        ])
        for category, count in error_counts.items():
            pct = (count / analysis_size * 100) if analysis_size else 0
            summary_text.append(f"  {category}: {count} ({pct:.0f}%)")

    ax.text(0.01, 0.98, "\n".join(summary_text), va="top", ha="left", fontsize=11, family="monospace")

    if examples:
        table_data = []
        for item in examples:
            table_data.append([
                wrap_text(item["english"], width=30),
                wrap_text(item["nepali"], width=30),
                wrap_text(item["prediction"], width=30),
                item.get("category", "")
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=["English input", "Reference Nepali", "Model output", "Issue"],
            cellLoc="left",
            colLoc="left",
            loc="lower center",
            colColours=["#dfe3ee"] * 4,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)

    if dataset_samples:
        sample_text = ["\nDATASET SAMPLE PAIRS:"]
        for item in dataset_samples:
            sample_text.append(f"  [{item.get('register','')}] {item['english']} -> {item['nepali']}")
        ax.text(0.52, 0.5, "\n".join(sample_text), va="top", ha="left", fontsize=9, family="monospace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def sample_by_register(data, sample_count, seed=SEED):
    groups = defaultdict(list)
    for item in data:
        register = item.get("register", "UNKNOWN")
        groups[register].append(item)

    sampled = []
    rnd = random.Random(seed)
    for register, items in groups.items():
        if items:
            sampled.append(rnd.choice(items))

    remaining = [item for item in data if item not in sampled]
    if len(sampled) < sample_count:
        extra_count = sample_count - len(sampled)
        sampled.extend(rnd.sample(remaining, min(extra_count, len(remaining))))

    return sampled[:sample_count]


def compute_length_vs_loss(test_data, translator, max_length):
    lengths = []
    losses = []
    model = translator.model
    tokenizer = translator.tokenizer
    device = translator.device

    for item in test_data:
        inputs = tokenizer(
            item["english"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        target = tokenizer(
            text_target=item["nepali"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        labels = target["input_ids"]
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        labels[labels == tokenizer.pad_token_id] = -100

        if inputs["input_ids"].dim() == 1:
            inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)
            inputs["attention_mask"] = inputs["attention_mask"].unsqueeze(0)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = float(outputs.loss.item())

        lengths.append(int(inputs["attention_mask"].sum().item()))
        losses.append(loss)

    return lengths, losses


def batch_translate_texts(translator, texts, max_length, batch_size=1):
    translated = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            translated.extend(translator.translate_batch(batch, max_length=max_length))
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                if batch_size > 1:
                    return batch_translate_texts(translator, texts, max_length, batch_size=max(1, batch_size // 2))
                raise RuntimeError(
                    "Translation OOM at batch size 1. Try --device cpu or reduce --max-length."
                ) from exc
            raise
    return translated


def main():
    args = parse_args()

    missing_files = [path for path in DATASET_FILES.values() if not path.exists()]
    if missing_files:
        print("❌ Missing dataset files:")
        for path in missing_files:
            print(f"   - {path}")
        return

    register_counts = build_register_counts(DATASET_FILES)
    print("📁 Dataset file counts:")
    for register, count in register_counts.items():
        print(f"   {register}: {count:,} sentence pairs")

    all_data, skipped, reasons = load_honorifics_from_register_files(DATASET_FILES)
    print(f"\n✅ Loaded {len(all_data):,} valid sentence pairs (skipped {skipped})")
    if reasons:
        print(f"   Skip reasons: {reasons}")

    split_counts = build_split_summary(all_data, seed=args.seed)
    print("\n📊 Split summary:")
    total = sum(split_counts.values())
    for subset, count in split_counts.items():
        print(f"   {subset}: {count:,} ({count / total * 100:.1f}%)")

    plot_chart(split_counts, args.split_output, chart_type=args.chart_type)
    print(f"✅ Split distribution chart saved to: {args.split_output}")

    english_lengths, nepali_lengths = get_sentence_lengths(all_data)
    length_summary = {
        "english_min": min(english_lengths),
        "english_max": max(english_lengths),
        "english_avg": sum(english_lengths) / len(english_lengths),
        "nepali_min": min(nepali_lengths),
        "nepali_max": max(nepali_lengths),
        "nepali_avg": sum(nepali_lengths) / len(nepali_lengths),
    }

    print("📈 Sentence length summary:")
    print(f"   English: min={length_summary['english_min']}, max={length_summary['english_max']}, avg={length_summary['english_avg']:.2f}")
    print(f"   Nepali: min={length_summary['nepali_min']}, max={length_summary['nepali_max']}, avg={length_summary['nepali_avg']:.2f}")

    plot_length_histograms(english_lengths, nepali_lengths, args.lengths_output)
    print(f"✅ Sentence length histogram saved to: {args.lengths_output}")

    translation_examples = []
    error_counts = None
    lengths = []
    losses = []

    if args.fast:
        args.max_length = min(args.max_length, 48)
        args.translate_batch_size = min(args.translate_batch_size, 1)
        args.sample_count = min(args.sample_count, 5)
        args.analysis_count = min(args.analysis_count, 8)
        if args.device == "auto":
            args.device = "cpu"

    if args.skip_model:
        print("\n⚠️  Skipping model-based translation, error analysis, and length-vs-loss charts.")
        analysis_data = []
    elif args.model_path.exists():
        print(f"\n🧠 Loading model for translation comparison from {args.model_path} on device={args.device}")
        translator = NepaliTranslator(args.model_path, device=args.device if args.device != "auto" else None)
        _, _, test_data = stratified_split(all_data, seed=args.seed)
        analysis_data = random.Random(args.seed).sample(test_data, min(args.analysis_count, len(test_data)))

        english_texts = [item["english"] for item in analysis_data]
        predictions = batch_translate_texts(
            translator,
            english_texts,
            max_length=args.max_length,
            batch_size=args.translate_batch_size,
        )

        categories = {
            "punctuation errors": 0,
            "honorific mismatch": 0,
            "lexical errors": 0,
            "grammar issues": 0,
        }

        translation_examples = []
        for item, pred in zip(analysis_data, predictions):
            category = classify_error(item["nepali"], pred)
            if category:
                categories[category] += 1
            translation_examples.append(
                {
                    "english": item["english"],
                    "nepali": item["nepali"],
                    "prediction": pred,
                    "category": category or "exact match",
                }
            )

        error_counts = categories
        plot_error_analysis(error_counts, args.error_output)
        print(f"✅ Error analysis chart saved to: {args.error_output}")

        if not args.skip_loss:
            lengths, losses = compute_length_vs_loss(analysis_data, translator, max_length=args.max_length)
            plot_length_vs_loss(lengths, losses, args.loss_output)
            print(f"✅ Token length vs loss chart saved to: {args.loss_output}")
        else:
            print("⚠️  Skipping length-vs-loss chart generation.")

        sample_indices = random.Random(args.seed).sample(range(len(translation_examples)), min(args.sample_count, len(translation_examples)))
        translation_examples = [translation_examples[i] for i in sample_indices]
    else:
        print(f"⚠️  Model not found at {args.model_path}. Skipping translation comparison and loss plotting.")

    dataset_samples = sample_by_register(all_data, min(args.sample_count, len(all_data)), seed=args.seed)

    save_analysis_image(
        args.analysis_image_output,
        split_counts,
        length_summary,
        translation_examples,
        error_counts=error_counts,
        dataset_samples=dataset_samples,
        analysis_size=len(analysis_data),
    )
    print(f"✅ Dataset analysis image saved to: {args.analysis_image_output}")


if __name__ == "__main__":
    main()
