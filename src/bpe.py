from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

Word = Tuple[str, ...]
Vocabulary = Dict[Word, int]
Pair = Tuple[str, str]


def get_vocab(corpus: Iterable[str]) -> Vocabulary:
    """Build a simple vocabulary from raw text.

    Each word is split into characters and a special end-of-word marker.
    This is the starting vocabulary for byte-pair encoding.
    """
    vocab: Vocabulary = Counter()
    for line in corpus:
        for token in line.strip().split():
            if not token:
                continue
            word = tuple(token) + ("</w>",)
            vocab[word] += 1
    return dict(vocab)


def get_stats(vocab: Vocabulary) -> Counter[Pair]:
    """Count how often each adjacent pair appears in the vocabulary."""
    pairs: Counter[Pair] = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += freq
    return pairs


def merge_vocab(pair: Pair, vocab: Vocabulary) -> Vocabulary:
    """Merge the most frequent pair throughout the vocabulary."""
    merged_vocab: Vocabulary = {}
    first, second = pair

    for word, freq in vocab.items():
        new_word: List[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        merged_vocab[tuple(new_word)] = freq

    return merged_vocab


class BytePairEncoder:
    """A very simple BPE tokenizer implementation."""

    def __init__(self, num_merges: int = 1000):
        self.num_merges = num_merges
        self.merges: List[Pair] = []
        self.vocab: Vocabulary = {}

    def fit(self, corpus: Iterable[str]) -> None:
        """Train the BPE model on raw text."""
        self.vocab = get_vocab(corpus)
        for merge_index in range(self.num_merges):
            pairs = get_stats(self.vocab)
            if not pairs:
                break
            best_pair, best_count = max(pairs.items(), key=lambda item: item[1])
            if best_count < 2:
                break
            self.vocab = merge_vocab(best_pair, self.vocab)
            self.merges.append(best_pair)

    def encode_word(self, word: str) -> List[str]:
        """Encode a single word with the learned merge rules."""
        symbols: List[str] = list(word) + ["</w>"]
        for pair in self.merges:
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def encode(self, text: str) -> str:
        """Encode a whole text line word by word."""
        tokens: List[str] = []
        for word in text.strip().split():
            if not word:
                continue
            encoded_word = self.encode_word(word)
            tokens.append(" ".join(encoded_word))
        return "   ".join(tokens)

    def decode(self, tokenized_text: str) -> str:
        """Decode tokenized text back into plain words."""
        decoded_words: List[str] = []
        for encoded_word in tokenized_text.split("   "):
            pieces = encoded_word.split()
            word = "".join(piece.replace("</w>", "") for piece in pieces)
            decoded_words.append(word)
        return " ".join(decoded_words)

    def get_merge_rules(self) -> List[Pair]:
        """Return the list of learned merge pairs."""
        return self.merges
