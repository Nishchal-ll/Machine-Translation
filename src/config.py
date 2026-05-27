import os
import torch
from pathlib import Path

# ====================== PROJECT PATHS ======================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = Path(os.getenv("MODEL_DIR", OUTPUT_DIR / "models"))
LOG_DIR = OUTPUT_DIR / "logs"

# Training datasets by register
DATASET_FILES = {
    "FORMAL": DATA_DIR / "formal.txt",
    "SEMI-FORMAL": DATA_DIR / "semi-formal.txt",
    "INFORMAL": DATA_DIR / "informal.txt",
}

def _bool_env(name, default=False):
    value = os.getenv(name, str(default))
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ====================== MODEL CONFIG ======================
MODEL_NAME = os.getenv("MODEL_NAME", "facebook/nllb-200-distilled-600M")
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"

# ====================== TRAINING CONFIG ======================
COLAB_MODE = _bool_env("COLAB_MODE")
EPOCHS = int(os.getenv("EPOCHS", 20))                        # Increased from 15 for better convergence
SESSION_SAVE_EVERY_EPOCHS = int(os.getenv("SESSION_SAVE_EVERY_EPOCHS", 1))      # Save resumable checkpoint every N epochs
RESUME_FROM_SESSION = _bool_env("RESUME_FROM_SESSION") if os.getenv("RESUME_FROM_SESSION") is not None else True
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2 if not COLAB_MODE else 4))                     # Reasonable default for Colab GPU
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-6))               # Lower LR for finer domain-specific tuning
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.02))                # Increased for stronger regularization
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))                    # Reduced for memory efficiency (honorifics typically short)
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", 0.3))                 # Increased for longer warmup
GRADIENT_CLIP = float(os.getenv("GRADIENT_CLIP", 1.0))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 4 if not COLAB_MODE else 2))    # Smaller accumulation for Colab
GRADIENT_CHECKPOINTING = _bool_env("GRADIENT_CHECKPOINTING") if os.getenv("GRADIENT_CHECKPOINTING") is not None else True      # Trade compute for memory
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 8))        # Increased to allow longer training
USE_LORA = _bool_env("USE_LORA") if os.getenv("USE_LORA") is not None else False                    # Disable LoRA by default; enable only explicitly
LORA_R = int(os.getenv("LORA_R", 16))                        # LoRA rank (used only if USE_LORA=True)
LORA_ALPHA = int(os.getenv("LORA_ALPHA", 32))                    # LoRA alpha (used only if USE_LORA=True)
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.05))                # LoRA dropout (used only if USE_LORA=True)
SEED = int(os.getenv("SEED", 42))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0 if COLAB_MODE else 2))

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Device: {DEVICE}")
if DEVICE.type == "cuda":
    try:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

if COLAB_MODE:
    print(f"☁️  Colab mode enabled: NUM_WORKERS={NUM_WORKERS}, BATCH_SIZE={BATCH_SIZE}")