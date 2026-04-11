# import re
# from collections import defaultdict
# import random
# from pathlib import Path

# REGISTER_MAP = {
#     "formal": "FORMAL",
#     "semi-formal": "SEMI-FORMAL",
#     "semiformal": "SEMI-FORMAL",
#     "semi_formal": "SEMI-FORMAL",
#     "informal": "INFORMAL",
# }

# def parse_honorifics_file(file_path: Path):
#     pairs = []
#     skipped = 0
#     reasons = defaultdict(int)

#     with open(file_path, encoding="utf-8") as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             parts = line.split("\t")
#             if len(parts) != 3:
#                 skipped += 1
#                 reasons["wrong_column_count"] += 1
#                 continue

#             en, ne, reg = [x.strip() for x in parts]
#             reg_norm = reg.lower().replace(" ", "-")

#             if not en: 
#                 skipped += 1; reasons["empty_english"] += 1; continue
#             if not ne: 
#                 skipped += 1; reasons["empty_nepali"] += 1; continue
#             if not re.search(r"[\u0900-\u097F]", ne):
#                 skipped += 1; reasons["no_devanagari"] += 1; continue
#             if reg_norm not in REGISTER_MAP:
#                 skipped += 1; reasons["unknown_register"] += 1; continue

#             pairs.append({
#                 "english": en,
#                 "nepali": ne,
#                 "register": REGISTER_MAP[reg_norm]
#             })

#     return pairs, skipped, reasons


# def stratified_split(data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
#     random.seed(seed)
#     groups = defaultdict(list)
#     for item in data:
#         groups[item["register"]].append(item)

#     train_data, val_data, test_data = [], [], []

#     for items in groups.values():
#         shuffled = items[:]
#         random.shuffle(shuffled)
#         n = len(shuffled)

#         n_test = max(5, int(n * test_ratio))
#         n_val = max(5, int(n * val_ratio))
#         n_train = n - n_test - n_val

#         test_data.extend(shuffled[:n_test])
#         val_data.extend(shuffled[n_test:n_test + n_val])
#         train_data.extend(shuffled[n_test + n_val:])

#     random.shuffle(train_data)
#     random.shuffle(val_data)
#     random.shuffle(test_data)

#     return train_data, val_data, test_data

# src/data_utils.py
# src/data_utils.py
import re
from collections import defaultdict
import random
from pathlib import Path

REGISTER_MAP = {
    "formal": "FORMAL",
    "semi-formal": "SEMI-FORMAL",
    "semiformal": "SEMI-FORMAL",
    "semi_formal": "SEMI-FORMAL",
    "informal": "INFORMAL",
}

def parse_honorifics_file(file_path: Path):
    """Simple but effective parser for your current dataset"""
    pairs = []
    skipped = 0
    reasons = defaultdict(int)

    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Split from the right - last word should be the register
            parts = line.rsplit(maxsplit=1)
            if len(parts) != 2:
                skipped += 1
                reasons["wrong_format"] += 1
                continue

            eng_nep_part = parts[0].strip()
            register = parts[1].strip().upper()

            if register not in ["FORMAL", "SEMI-FORMAL", "INFORMAL"]:
                skipped += 1
                reasons["unknown_register"] += 1
                continue

            # Find where Nepali starts (first Devanagari character)
            match = re.search(r'[\u0900-\u097F]', eng_nep_part)
            if not match:
                skipped += 1
                reasons["no_devanagari"] += 1
                continue

            split_pos = match.start()
            english = eng_nep_part[:split_pos].strip()
            nepali = eng_nep_part[split_pos:].strip()

            if not english or not nepali:
                skipped += 1
                reasons["empty_field"] += 1
                continue

            pairs.append({
                "english": english,
                "nepali": nepali,
                "register": REGISTER_MAP.get(register.lower(), register)
            })

    return pairs, skipped, reasons


def parse_honorifics_file_with_register(file_path: Path, register_hint: str):
    """Parse a single-register file where each line is either EN\tNE or EN NE."""
    pairs = []
    skipped = 0
    reasons = defaultdict(int)

    register = register_hint.strip().upper()
    if register not in ["FORMAL", "SEMI-FORMAL", "INFORMAL"]:
        raise ValueError(f"Unsupported register hint: {register_hint}")

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            english = ""
            nepali = ""

            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    english = parts[0].strip()
                    nepali = parts[1].strip()
                else:
                    skipped += 1
                    reasons["wrong_format"] += 1
                    continue
            else:
                # Fallback: detect first Devanagari char and split there.
                match = re.search(r"[\u0900-\u097F]", line)
                if not match:
                    skipped += 1
                    reasons["no_devanagari"] += 1
                    continue

                split_pos = match.start()
                english = line[:split_pos].strip()
                nepali = line[split_pos:].strip()

            if not english or not nepali:
                skipped += 1
                reasons["empty_field"] += 1
                continue

            if not re.search(r"[\u0900-\u097F]", nepali):
                skipped += 1
                reasons["no_devanagari"] += 1
                continue

            pairs.append({
                "english": english,
                "nepali": nepali,
                "register": register,
            })

    return pairs, skipped, reasons


def load_honorifics_from_register_files(dataset_files: dict):
    """Load and merge multiple register-specific files into one training list."""
    merged = []
    total_skipped = 0
    all_reasons = defaultdict(int)

    for register, file_path in dataset_files.items():
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found for {register}: {file_path}")

        data, skipped, reasons = parse_honorifics_file_with_register(file_path, register)
        merged.extend(data)
        total_skipped += skipped
        for key, value in reasons.items():
            all_reasons[key] += value

    return merged, total_skipped, dict(all_reasons)


def stratified_split(data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    groups = defaultdict(list)
    for item in data:
        groups[item["register"]].append(item)

    train_data, val_data, test_data = [], [], []

    for items in groups.values():
        shuffled = items[:]
        random.shuffle(shuffled)
        n = len(shuffled)

        n_test = max(5, int(n * test_ratio))
        n_val = max(5, int(n * val_ratio))
        n_train = n - n_test - n_val

        test_data.extend(shuffled[:n_test])
        val_data.extend(shuffled[n_test : n_test + n_val])
        train_data.extend(shuffled[n_test + n_val :])

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data