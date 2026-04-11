
# scripts/train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    DATASET_FILES,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_LENGTH,
    WARMUP_RATIO,
    GRADIENT_CLIP,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CHECKPOINTING,
    EARLY_STOPPING_PATIENCE,
    USE_LORA,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    MODEL_NAME,
    SRC_LANG,
    TGT_LANG,
    MODEL_DIR,
    SEED,
    SESSION_SAVE_EVERY_EPOCHS,
    RESUME_FROM_SESSION,
)
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.dataset import HonorificsDataset
from src.trainer import Trainer
from src.utils import set_seed, print_training_summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader


def main():
    set_seed(SEED)

    print("🇳🇵 Starting NLLB-200 Honorifics Fine-Tuning (English → Nepali)\n")

    missing_files = [path for path in DATASET_FILES.values() if not path.exists()]
    if missing_files:
        print("❌ Missing dataset files:")
        for path in missing_files:
            print(f"   - {path}")
        return

    print("📁 Loading datasets from:")
    file_line_counts = {}
    for register, path in DATASET_FILES.items():
        line_count = sum(1 for _ in open(path, encoding="utf-8") if _.strip())
        file_line_counts[register] = line_count
        print(f"   {register}: {path} ({line_count:,} lines)")
    print("")

    all_data, skipped, reasons = load_honorifics_from_register_files(DATASET_FILES)
    print(f"✅ Loaded {len(all_data):,} valid sentence pairs (skipped {skipped})")

    if len(all_data) == 0:
        print("❌ No valid pairs found!")
        print("Reasons:", dict(reasons))
        return

    train_data, val_data, test_data = stratified_split(all_data, seed=SEED)
    print(f"📊 Split → Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    config_obj = type('Config', (), {
        'MODEL_NAME': MODEL_NAME,
        'SRC_LANG': SRC_LANG,
        'TGT_LANG': TGT_LANG,
        'DEVICE': DEVICE,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'MAX_LENGTH': MAX_LENGTH,
        'WARMUP_RATIO': WARMUP_RATIO,
        'GRADIENT_CLIP': GRADIENT_CLIP,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'GRADIENT_CHECKPOINTING': GRADIENT_CHECKPOINTING,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'USE_LORA': USE_LORA,
        'LORA_R': LORA_R,
        'LORA_ALPHA': LORA_ALPHA,
        'LORA_DROPOUT': LORA_DROPOUT,
    })

    print_training_summary(config_obj)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Datasets and loaders
    train_dataset = HonorificsDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset   = HonorificsDataset(val_data,   tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Create config object properly
    config_obj = type('Config', (), {
        'DEVICE': DEVICE,
        'MODEL_DIR': MODEL_DIR,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'MAX_LENGTH': MAX_LENGTH,
        'WARMUP_RATIO': WARMUP_RATIO,
        'GRADIENT_CLIP': GRADIENT_CLIP,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'GRADIENT_CHECKPOINTING': GRADIENT_CHECKPOINTING,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'USE_LORA': USE_LORA,
        'LORA_R': LORA_R,
        'LORA_ALPHA': LORA_ALPHA,
        'LORA_DROPOUT': LORA_DROPOUT,
    })

    # Start Trainer
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config=config_obj)

    resumed_epoch = 0
    if RESUME_FROM_SESSION:
        resumed_epoch = trainer.load_session_checkpoint()

    print(f"\n🚀 Starting session training for {EPOCHS} epochs on {DEVICE}...\n")
    if resumed_epoch > 0:
        print(f"🔁 Continuing from previous session at epoch {resumed_epoch}\n")

    last_completed_epoch = resumed_epoch

    for epoch in range(1, EPOCHS + 1):
        global_epoch = resumed_epoch + epoch
        print(f"--- Session Epoch {epoch}/{EPOCHS} (Global {global_epoch}) ---")
        train_loss = trainer.train_epoch()
        val_loss, perplexity = trainer.validate()
        last_completed_epoch = global_epoch

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Perplexity : {perplexity:.2f}")

        # Early stopping with best model saving
        should_stop, should_save = trainer.check_early_stopping(val_loss)
        if should_save:
            trainer.save_best_model()
            print(f"✅ Validation improved! Best loss: {trainer.best_val_loss:.4f}")
        else:
            print(f"⚠️  Patience: {trainer.patience_counter}/{trainer.patience}")
        
        if should_stop:
            print(f"\n⛔ Early stopping triggered after {epoch} epochs")
            trainer.save_session_checkpoint(epoch_completed=global_epoch)
            break

        if epoch % SESSION_SAVE_EVERY_EPOCHS == 0:
            trainer.save_session_checkpoint(epoch_completed=global_epoch)

    # Always save at end of run so next session can continue.
    trainer.save_session_checkpoint(epoch_completed=last_completed_epoch)

    print("\n🎉 Training finished successfully!")
    print(f"Best model saved at: {trainer.best_model_path}")

if __name__ == "__main__":
    main()