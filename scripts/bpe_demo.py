#!/usr/bin/env python3
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.bpe import BytePairEncoder


def load_lines(file_paths):
    lines = []
    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    lines.append(text)
    return lines


def extract_english(lines):
    english_texts = []
    for line in lines:
        if "\t" in line:
            english, _ = line.split("\t", 1)
            english_texts.append(english.strip())
        else:
            match = re.search(r"[\u0900-\u097F]", line)
            if match:
                english_texts.append(line[: match.start()].strip())
    return [text for text in english_texts if text]


def main():
    parser = ArgumentParser(description="Train and demo the scratch BPE tokenizer")
    parser.add_argument("--merges", type=int, default=200, help="Number of BPE merge operations")
    parser.add_argument("--samples", type=int, default=8, help="Number of examples to display")
    args = parser.parse_args()

    data_files = [
        Path("data/raw/formal.txt"),
        Path("data/raw/semi-formal.txt"),
        Path("data/raw/informal.txt"),
    ]
    lines = load_lines(data_files)
    english_sentences = extract_english(lines)

    print(f"Training BPE from scratch on {len(english_sentences)} English lines...")
    bpe = BytePairEncoder(num_merges=args.merges)
    bpe.fit(english_sentences)

    print(f"Learned {len(bpe.get_merge_rules())} merge rules")
    print("Top 10 merge rules:")
    for i, merge in enumerate(bpe.get_merge_rules()[:10], start=1):
        print(f"  {i}. {merge[0]} + {merge[1]}")

    print("\nExample tokenization:\n")
    for text in english_sentences[: args.samples]:
        encoded = bpe.encode(text)
        decoded = bpe.decode(encoded)
        print(f"Input   : {text}")
        print(f"Encoded : {encoded}")
        print(f"Decoded : {decoded}")
        print("---")

    print("\nThis simple BPE code is written from scratch in Python.")
    print("It learns the most frequent character pairs and merges them repeatedly.")


if __name__ == "__main__":
    main()
