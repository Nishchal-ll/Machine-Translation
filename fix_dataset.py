# fix_dataset.py
import re
from pathlib import Path

input_file = Path("data/raw/honoorifics.txt")
output_file = Path("data/raw/honoorifics_fixed.txt")

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed = []
bad_lines = 0

for i, line in enumerate(lines, 1):
    line = line.strip()
    if not line:
        continue

    # Split from the right: last word = register, second last = last part of Nepali, rest = English
    parts = line.rsplit(maxsplit=2)

    if len(parts) == 3:
        english = parts[0].strip()
        nepali = (parts[1] + " " + parts[2]).strip() if parts[1].endswith(('।', '?', '!', '.')) else parts[1].strip()
        register = parts[2] if len(parts) == 3 and parts[2].isupper() else parts[2]

        # Final correction - register should be the very last word
        if register.upper() in ["FORMAL", "SEMI-FORMAL", "INFORMAL"]:
            fixed.append(f"{english}\t{nepali}\t{register}\n")
        else:
            # Try again with different split
            parts2 = line.rsplit(maxsplit=1)
            if len(parts2) == 2 and parts2[1].upper() in ["FORMAL", "SEMI-FORMAL", "INFORMAL"]:
                eng_nep = parts2[0].rsplit(maxsplit=1)
                if len(eng_nep) == 2:
                    fixed.append(f"{eng_nep[0]}\t{eng_nep[1]}\t{parts2[1]}\n")
                else:
                    bad_lines += 1
            else:
                bad_lines += 1
    else:
        bad_lines += 1

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(fixed)

print(f" Fixed dataset saved as: {output_file}")
print(f"   Total lines fixed : {len(fixed)}")
print(f"   Bad/skipped lines : {bad_lines}")