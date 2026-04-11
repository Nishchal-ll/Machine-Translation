import torch
from pathlib import Path

# ====================== PROJECT PATHS ======================
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = OUTPUT_DIR / "logs"

# Training datasets by register
DATASET_FILES = {
    "FORMAL": DATA_DIR / "formal.txt",
    "SEMI-FORMAL": DATA_DIR / "semi-formal.txt",
    "INFORMAL": DATA_DIR / "informal.txt",
}

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ====================== MODEL CONFIG ======================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "npi_Deva"

# ====================== TRAINING CONFIG ======================
EPOCHS = 20                        # Increased from 15 for better convergence
SESSION_SAVE_EVERY_EPOCHS = 1      # Save resumable checkpoint every N epochs
RESUME_FROM_SESSION = True         # Continue from previous session checkpoint if available
BATCH_SIZE = 2                     # Reduced for 4GB GPU
LEARNING_RATE = 5e-6               # Lower LR for finer domain-specific tuning
WEIGHT_DECAY = 0.02                # Increased for stronger regularization
MAX_LENGTH = 64                    # Reduced for memory efficiency (honorifics typically short)
WARMUP_RATIO = 0.3                 # Increased for longer warmup
GRADIENT_CLIP = 1.0
GRADIENT_ACCUMULATION_STEPS = 4    # Accumulate gradients every 4 batches = effective batch 8
GRADIENT_CHECKPOINTING = True      # Trade compute for memory
EARLY_STOPPING_PATIENCE = 8        # Increased to allow longer training
USE_LORA = True                    # Use LoRA for efficient domain-specific fine-tuning
LORA_R = 16                        # LoRA rank (increased from 8 for more capacity)
LORA_ALPHA = 32                    # LoRA alpha (increased from 16)
LORA_DROPOUT = 0.05                # LoRA dropout
SEED = 42

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")