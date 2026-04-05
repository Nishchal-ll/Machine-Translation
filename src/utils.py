# import random
# import numpy as np
# import torch
# import os
# from pathlib import Path

# def set_seed(seed: int = 42):
#     """Set random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     print(f"✅ Random seed set to {seed}")


# def print_training_summary(config):
#     """Print nice training configuration summary"""
#     print("\n" + "="*60)
#     print("🚀 TRAINING CONFIGURATION")
#     print("="*60)
#     print(f"Model          : {config.MODEL_NAME}")
#     print(f"Source Lang    : {config.SRC_LANG}")
#     print(f"Target Lang    : {config.TGT_LANG}")
#     print(f"Device         : {config.DEVICE}")
#     print(f"Epochs         : {config.EPOCHS}")
#     print(f"Batch Size     : {config.BATCH_SIZE}")
#     print(f"Learning Rate  : {config.LEARNING_RATE}")
#     print(f"Max Length     : {config.MAX_LENGTH} tokens")
#     print(f"Output Dir     : {config.MODEL_DIR}")
#     print("="*60 + "\n")


# def save_training_log(log_dir: Path, epoch: int, train_loss: float, val_loss: float, perplexity: float):
#     """Simple logging to text file"""
#     log_file = log_dir / "training_log.txt"
#     with open(log_file, "a", encoding="utf-8") as f:
#         f.write(f"Epoch {epoch:02d} | "
#                 f"Train Loss: {train_loss:.4f} | "
#                 f"Val Loss: {val_loss:.4f} | "
#                 f"Perplexity: {perplexity:.2f}\n")

# src/utils.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✅ Random seed set to {seed}")


def print_training_summary(config):
    """Print training configuration safely"""
    print("\n" + "="*60)
    print("🚀 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model          : {getattr(config, 'MODEL_NAME', 'N/A')}")
    print(f"Source Lang    : {getattr(config, 'SRC_LANG', 'N/A')}")
    print(f"Target Lang    : {getattr(config, 'TGT_LANG', 'N/A')}")
    print(f"Device         : {getattr(config, 'DEVICE', 'N/A')}")
    print(f"Epochs         : {getattr(config, 'EPOCHS', 'N/A')}")
    print(f"Batch Size     : {getattr(config, 'BATCH_SIZE', 'N/A')}")
    print(f"Learning Rate  : {getattr(config, 'LEARNING_RATE', 'N/A')}")
    print(f"Max Length     : {getattr(config, 'MAX_LENGTH', 'N/A')} tokens")
    print("="*60 + "\n")