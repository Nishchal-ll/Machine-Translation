
# scripts/train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import *
from src.data_utils import parse_honorifics_file, stratified_split
from src.dataset import HonorificsDataset
from src.trainer import Trainer
from src.utils import set_seed, print_training_summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

def main():
    set_seed(SEED)

    print("🇳🇵 Starting NLLB-200 Honorifics Fine-Tuning (English → Nepali)\n")

    data_file = DATA_DIR / "honoorifics.txt"
    if not data_file.exists():
        print(f"❌ Dataset not found: {data_file}")
        return

    all_data, skipped, reasons = parse_honorifics_file(data_file)
    print(f"✅ Loaded {len(all_data):,} valid sentence pairs (skipped {skipped})")

    if len(all_data) == 0:
        print("❌ No valid pairs found!")
        print("Reasons:", dict(reasons))
        return

    train_data, val_data, test_data = stratified_split(all_data, seed=SEED)
    print(f"📊 Split → Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")

    print_training_summary(type('Config', (), locals()))

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

    print(f"\n🚀 Starting training for {EPOCHS} epochs on {DEVICE}...\n")

    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        train_loss = trainer.train_epoch()
        val_loss, perplexity = trainer.validate()

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
            break

    print("\n🎉 Training finished successfully!")
    print(f"Best model saved at: {trainer.best_model_path}")

if __name__ == "__main__":
    main()