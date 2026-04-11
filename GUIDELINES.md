# 📖 NLLB Honorifics Translation - Guidelines & Best Practices

Comprehensive guidelines for fine-tuning, evaluating, and deploying the NLLB-200 Nepali honorifics translation model.

---

## 📋 Table of Contents
1. [Dataset Guidelines](#dataset-guidelines)
2. [Honorific Forms in Nepali](#honorific-forms-in-nepali)
3. [Training Guidelines](#training-guidelines)
4. [Evaluation Guidelines](#evaluation-guidelines)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Common Mistakes](#common-mistakes)
7. [Best Practices](#best-practices)

---

## 📊 Dataset Guidelines

### Format Requirements

**Files:** `data/raw/formal.txt`, `data/raw/semi-formal.txt`, `data/raw/informal.txt`

Each line must have exactly 3 tab-separated fields:
```
[English Text] \t [Nepali Text] \t [REGISTER]
```

### Field Definitions

#### 1. English Text
- **Natural English sentence** in standard English
- Sentence case (capitalize first word)
- Can include punctuation (?, !, .)
- Length: 1-20 words (optimal: 6-10 words)

**Examples:**
```
✅ Good: "Please sit down, sir."
✅ Good: "How are you today?"
❌ Bad:  "pls sit dwn" (non-standard)
❌ Bad:  "PLEASE SIT DOWN" (all caps)
```

#### 2. Nepali Text
- **Pure Devanagari script** (no English/transliteration)
- Natural Nepali with appropriate grammar
- Include danda (।) at end of sentence
- Can include Nepali punctuation

**Examples:**
```
✅ Good: "सर, कृपया बस्नुहोस् ।"
✅ Good: "आज तपाईं कस्तो हुनुहुन्छ ?"
❌ Bad:  "Kr paya basnuhos" (transliteration)
❌ Bad:  "सर कृपया बस्नुहोस्" (missing danda)
```

#### 3. Register
Must be **exactly** one of:
- `FORMAL` - Respectful/formal tone
- `SEMI-FORMAL` - Mixed formality  
- `INFORMAL` - Casual/direct tone

**Do NOT use:** "formal", "Formal", "SEMI_FORMAL", "Semi-Formal"

### Register Definitions

#### FORMAL (**~33% of dataset**)
**Characteristics:**
- Uses honorific verb forms: -नुहोस्, -हुन्छ
- Uses formal pronouns: तपाई, तपाईं, तपाईंलाई
- Includes titles: सर, मेडम, साहब, जी
- Respectful addressing

**Examples:**
```
English: "Please sit down, sir"
Nepali:  "सर, कृपया बस्नुहोस् ।"

English: "How are you, doctor?"
Nepali:  "डाक्टर साहब, तपाईं कस्तो हुनुहुन्छ ?"
```

#### SEMI-FORMAL (**~33% of dataset**)
**Characteristics:**
- Mixed formal and casual
- Less strict honorifics
- Neutral question forms
- Polite but not overly formal

**Examples:**
```
English: "Can you help me?"
Nepali:  "के तपाईं मलाई मद्दत गर्न सक्नुहुन्छ ?"

English: "Will you come tomorrow?"
Nepali:  "के तपाईं भोली आउनुहुन्छ ?"
```

#### INFORMAL (**~33% of dataset**)
**Characteristics:**
- Direct/casual forms: -छु, -ौ, -छौ
- Casual pronouns: तिमी, मेरो, तेरो
- No titles or honorifics
- Friendly/casual tone

**Examples:**
```
English: "I am going home"
Nepali:  "म घर जाँदैछु ।"

English: "Where are you going?"
Nepali:  "तिमी कहाँ जाँदैछौ ?"
```

### Quality Checklist

Before adding to dataset:

- [ ] Exactly 3 tab-separated fields
- [ ] English is grammatically correct
- [ ] Nepali is in Devanagari script
- [ ] Nepali is grammatically correct for the register
- [ ] Register is one of: FORMAL, SEMI-FORMAL, INFORMAL
- [ ] Nepali ends with danda (।) or appropriate punctuation
- [ ] No mixed scripts (no transliteration/Roman in Nepali field)
- [ ] No machine translation artifacts
- [ ] Length is reasonable (1-20 words in each language)
- [ ] Exact meaning correspondence (not loose translation)

---

## 🎯 Honorific Forms in Nepali

### Key Honorific Markers

#### Verb Forms

| Form | Register | Example | English |
|------|----------|---------|---------|
| -नुहोस् (command/request) | FORMAL | "बस्नुहोस्" | "Please sit" |
| -हुन्छ (present/general) | FORMAL | "आउनुहुन्छ?" | "Do you come?" |
| -छ/-छु (direct/casual) | INFORMAL | "म आउँछु" | "I come" |
| -ौ/-छौ (casual 2nd person) | INFORMAL | "तिमी आउँछौ?" | "Do you come?" |

#### Pronouns

| Pronoun | Register | Usage |
|---------|----------|-------|
| तपाई / तपाईं | FORMAL | Respectful "you" |
| तपाईंलाई | FORMAL | Respectful "you" (object form) |
| तिमी | INFORMAL | Casual "you" |
| तिमीलाई | INFORMAL | Casual "you" (object form) |
| उनी / उहाँ | FORMAL | Respectful "he/she" |
| उ / उसले | INFORMAL | Casual "he/she" |

#### Titles & Honorifics

| Title | Register | Example |
|-------|----------|---------|
| साहब (Sahab) | FORMAL | "डाक्टर साहब" (Doctor sir) |
| जी (ji) | FORMAL | "नमस्ते जी" (Hello sir/madam) |
| ऋषी | FORMAL | "शिक्षक ऋषी" (Dear teacher) |
| आदरणीय (Honorable) | FORMAL | "आदरणीय गुरु" (Respected teacher) |

### Common Patterns

#### Making Sentences Formal

```
INFORMAL: "म घर जाँदैछु"
FORMAL:   "सर, म घर जाँदैछु"
          "कृपया, म घर जाँदैछु"
          "साहब, कृपया मलाई घर जन आत्दिनुहोस्"
```

#### Making Sentences Casual

```
FORMAL: "कृपया बस्नुहोस्"
CASUAL: "बस न"
        "बस गर"
        "बस यहाँ"
```

---

## 🏋️ Training Guidelines

### Before Training

#### 1. Verify Dataset
```bash
python -c "
from src.data_utils import load_honorifics_from_register_files
from src.config import DATASET_FILES

all_data, skipped, reasons = load_honorifics_from_register_files(DATASET_FILES)
print(f'Total pairs: {len(all_data)}')
print(f'Skipped: {skipped}')
print(f'Registers: {set(item[\"register\"] for item in all_data)}')
"
```

#### 2. Check Disk Space
```bash
# Required space:
# - Model: 2.3 GB
# - Datasets: 500 MB
# - Outputs: 1-2 GB
# - Temp: 2-3 GB
# Total: ~6 GB free space
```

#### 3. GPU Memory Check
```bash
# Check GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

**Minimum:** 4 GB VRAM (adjust BATCH_SIZE if needed)

### Training Steps

#### Step 1: Clear Previous Models (Optional)
```bash
# If retraining, remove old model to save space
Remove-Item outputs/models/best_honorifics_model -Recurse -Force
```

#### Step 2: Modify Config (If Needed)
Edit `src/config.py`:
```python
EPOCHS = 20              # (default: good for small datasets)
BATCH_SIZE = 2           # (reduce if OOM)
LEARNING_RATE = 5e-6     # (lower = slower but more stable)
WARMUP_RATIO = 0.3       # (higher = longer stability phase)
```

#### Step 3: Start Training
```bash
python scripts/train.py
```

#### Step 4: Monitor Training
Looking for:
- ✅ **Train Loss Decreasing:** 3.5 → 2.5 → 1.8 (good)
- ✅ **Val Loss Decreasing:** 3.2 → 2.5 → 2.0 (good)
- ✅ **Model Saving:** "✅ Best model saved" messages
- ❌ **Loss Exploding:** 0.5 → 50.0 (reduce LR)
- ❌ **Loss Stagnant:** 2.0 → 1.99 (increase epochs, lower LR)

#### Expected Training Metrics

**Epoch 1:**
```
Train Loss: ~3.2  (high because cold start)
Val Loss:   ~2.9
Perplexity: ~18
```

**Epoch 5:**
```
Train Loss: ~1.8  (good improvement)
Val Loss:   ~1.6
Perplexity: ~5
```

**Epoch 20:**
```
Train Loss: ~0.8  (well-trained)
Val Loss:   ~1.2
Perplexity: ~3.3
```

### Training Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| CUDA OOM | Batch too large | Reduce BATCH_SIZE to 1 |
| Loss NaN | Bad learning rate | Use LEARNING_RATE = 1e-6 |
| Loss not decreasing | Poor data quality | Check dataset format |
| Memory leaks | Bad cleanup | Restart Python kernel |
| Slow training | GPU not used | Check `DEVICE = "cuda"` |

---

## 📊 Evaluation Guidelines

### Metrics Explained

#### 1. BLEU Score (0-100)
**What it measures:** Translation quality compared to reference
**Formula:** Matches n-grams (1-gram, 2-gram, etc.) with reference

**Interpretation:**
```
BLEU 0-20:    Poor translation (incomprehensible)
BLEU 20-40:   Fair translation (some correct parts)
BLEU 40-60:   Good translation (mostly correct)
BLEU 60-80:   Very good translation (near-reference)
BLEU 80-100:  Excellent translation (nearly identical to reference)
```

**For honorifics domain:** Target 45-55 BLEU

#### 2. Exact Match Accuracy (%)
**What it measures:** Percentage of translations that match reference exactly

**Formula:**
```
Accuracy = (Exact Matches / Total Examples) × 100
```

**Why it's strict:**
- Requires identical Nepali text
- One wrong diacrit = no match
- One extra/missing word = no match

**Interpretation:**
```
0-10%:   Very difficult domain (realistic for new fine-tuning)
10-30%:  Basic learning (model grasps structure)
30-50%:  Good fine-tuning (captures honorifics patterns)
50-70%:  Excellent fine-tuning (domain expert level)
70%+:    Near-perfect (human-level for this register)
```

**For honorifics domain:** Target 40%+ accuracy

#### 3. Partial Match Accuracy (%)
**What it measures:** Percentage of translations with 80%+ word overlap

**Formula:**
```
Overlap = |Reference Words ∩ Prediction Words| / max(len(ref), len(pred))
Partial = (Overlap ≥ 0.8) / Total Examples × 100
```

**Why it's useful:**
- Captures "close enough" translations
- Accounts for word order variations
- Shows semantic understanding

**For honorifics domain:** Target 60%+ partial accuracy

### Evaluation Best Practices

#### Before Evaluation
1. ✅ Ensure model is fully trained (min 10 epochs)
2. ✅ Ensure `outputs/models/best_honorifics_model/` exists
3. ✅ Run evaluation on **test set only** (not training data)

#### Interpreting Results
```
Good Results:
✅ BLEU: 45+
✅ Exact: 30%+
✅ Partial: 60%+

Average Results:
⚠️ BLEU: 25-45
⚠️ Exact: 10-30%
⚠️ Partial: 40-60%

Poor Results:
❌ BLEU: <25
❌ Exact: <10%
❌ Partial: <40%
```

---

## 🎛️ Hyperparameter Tuning

### Parameter Sensitivity

| Parameter | Range | Sweet Spot | Effect |
|-----------|-------|-----------|--------|
| Learning Rate | 1e-7 to 1e-4 | 5e-6 | Control convergence speed |
| Batch Size | 1-8 | 2-4 | Trade-off: memory vs stability |
| Epochs | 5-50 | 15-25 | More = better (with early stop) |
| LORA_R | 4-32 | 16 | Model capacity for adaptation |
| Warmup Ratio | 0.0-0.5 | 0.3 | Stabilize early training |
| Weight Decay | 0.0-0.1 | 0.02 | Prevent overfitting |

### Tuning Strategy

#### For Better Accuracy (Current: 20-30%)

**Step 1: Stabilize Training**
```python
LEARNING_RATE = 3e-6  # Lower for stability
WARMUP_RATIO = 0.4    # Longer warmup
WEIGHT_DECAY = 0.02   # Regularization
EPOCHS = 25           # More time to learn
```

**Step 2: Expand Model Capacity**
```python
LORA_R = 24           # From 16
LORA_ALPHA = 48       # From 32
```

**Step 3: Fine-tune Learning Rate**
```python
# Try cyclic learning rate schedule
# (already implemented in new trainer.py)
```

#### For Production Deployment

```python
LEARNING_RATE = 1e-6   # Very low for stability
WARMUP_RATIO = 0.5     # Long warmup
EPOCHS = 50            # Many epochs with early stop
EARLY_STOPPING_PATIENCE = 15  # Allow more iterations
GRADIENT_CLIP = 0.5    # Tighter clipping
```

---

## ⚠️ Common Mistakes

### Dataset Mistakes

#### ❌ Mistake 1: Inconsistent Register Labels
```
❌ BAD:
"Please sit"    "कृपया बस्नुहोस्"    "Formal"
"Please sit"    "कृपया बस्नुहोस्"    "FORMAL"
"Please sit"    "कृपया बस्नुहोस्"    "formal"

✅ CORRECT:
"Please sit, sir"    "सर, कृपया बस्नुहोस्"    "FORMAL"
"Please sit"         "कृपया बस्नुहोस्"        "SEMI-FORMAL"
"Sit"                "बस न"                 "INFORMAL"
```

**Fix:** Use exact register names (FORMAL, SEMI-FORMAL, INFORMAL)

#### ❌ Mistake 2: Mixed Scripts in Nepali
```
❌ BAD:
"Hello"    "Namaste. Tapai kasto ho?"    "INFORMAL"

✅ CORRECT:
"Hello"    "नमस्ते, तपाईं कस्तो हुनुहुन्छ ?"    "FORMAL"
"Hello"    "नमस्ते, कस्तो छौ ?"                "INFORMAL"
```

**Fix:** Use only Devanagari script for Nepali field

#### ❌ Mistake 3: Over-complicated English
```
❌ BAD: "Would you be so kind as to furnish me with a cup of tea, if it's not too much trouble?"

✅ GOOD: "Could you please get me some tea?"
```

**Fix:** Use natural, conversational English

### Training Mistakes

#### ❌ Mistake 4: Training on Same Data as Test
```python
❌ BAD:
train_data, _, _ = stratified_split(all_data)
# Then evaluate on all_data (includes train_data)

✅ CORRECT:
train_data, val_data, test_data = stratified_split(all_data)
# Evaluate on test_data only
```

**Fix:** Always use stratified split and evaluate on validation/test sets

#### ❌ Mistake 5: Not Using Early Stopping
```python
❌ BAD: Train for full EPOCHS even if validation loss increases

✅ CORRECT: Stop early if:
  - Validation loss doesn't improve for N epochs
  - Memory runs out
  - Loss becomes NaN
```

#### ❌ Mistake 6: Wrong Dataset Size
```
❌ DON'T: Train on 100 examples (too few)
❌ DON'T: Train on 1M examples (too many for fine-tuning)

✅ DO: Train on 10k-50k examples (optimal for domain-specific)
```

---

## ✅ Best Practices

### 1. Data Quality First
- **Collect diverse honorific examples**
- **Ensure grammatically correct translations**
- **Match sentiment/tone between English and Nepali**
- **Balance registers equally**

### 2. Iterative Improvement
```
Step 1. Train on 11.5k examples → Evaluate → 20-30% accuracy
Step 2. Add 5k more examples → Evaluate → 40-50% accuracy
Step 3. Add 10k more examples → Evaluate → 60-70% accuracy
Step 4. Add domain-specific examples → Evaluate → 75%+ accuracy
```

### 3. Monitoring During Training
- Print **loss every batch** (not just every epoch)
- Check **validation loss trends** (should decrease)
- Watch **training time** (should be consistent)
- Monitor **GPU memory** (should stay stable)

### 4. Reproducibility
```python
# Always set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

### 5. Save Checkpoints
```python
# Save at every epoch, keep best 3
if val_loss < best_val_loss:
    save_model(epoch)  # Save
    keep_best_3_models()  # Remove old ones
```

### 6. Version Control
```bash
# Tag your models
# nllb-honorifics-v1 (11.5k examples, 25% accuracy)
# nllb-honorifics-v2 (23k examples, 45% accuracy)
# nllb-honorifics-v3 (30k examples, 70% accuracy)
```

---

## 🔍 Quality Assurance Checklist

Before submitting for evaluation:

- [ ] Dataset has no duplicates
- [ ] All Nepali text is in Devanagari script
- [ ] All register labels are: FORMAL, SEMI-FORMAL, or INFORMAL
- [ ] Registers are balanced (roughly 33% each)
- [ ] Model training completes without errors
- [ ] Validation loss decreases over epochs
- [ ] Best model is saved at `outputs/models/best_honorifics_model/`
- [ ] Evaluation metrics are reasonable (BLEU > 20)
- [ ] Demo works and produces sensible output
- [ ] No NaN or Inf values in metrics

---

## 📈 Performance Goals

### Short-term (Current Dataset)
| Goal | Metric | Target |
|------|--------|--------|
| Basic Learning | BLEU | 30-40 |
| Structure Understanding | Exact Match | 15-25% |
| Close Matches | Partial Match | 45-60% |

### Medium-term (After Augmentation)
| Goal | Metric | Target |
|------|--------|--------|
| Good Foundation | BLEU | 45-55 |
| Decent Accuracy | Exact Match | 35-45% |
| Semantic Match | Partial Match | 60-75% |

### Long-term (After Adding Data)
| Goal | Metric | Target |
|------|--------|--------|
| Production Ready | BLEU | 60+ |
| High Accuracy | Exact Match | 60%+ |
| Near-perfect | Partial Match | 80%+ |

---

## 📚 References

### Nepali Language Resources
- [Nepali NLP Dataset](https://github.com/mesaugat/nepali-nltk)
- [Devanagari Unicode](https://en.wikipedia.org/wiki/Devanagari#Unicode)
- [Nepali Grammar](https://en.wikibooks.org/wiki/Nepali)

### Model Documentation
- [NLLB-200 Paper](https://arxiv.org/abs/2207.04672)
- [Hugging Face NLLB](https://huggingface.co/docs/transformers/model_doc/nllb)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

### Evaluation Metrics
- [BLEU Score Explained](https://en.wikipedia.org/wiki/BLEU)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)

---

**Last Updated:** March 31, 2026
**Guidelines Version:** 2.0
**Model:** facebook/nllb-200-distilled-600M with LoRA fine-tuning
