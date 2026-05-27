
# scripts/train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from argparse import ArgumentParser
from src.config import (
    COLAB_MODE,
    DATASET_FILES,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_LENGTH,
    NUM_WORKERS,
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
from src.bpe import BytePairEncoder
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.dataset import HonorificsDataset
from src.trainer import Trainer
from src.utils import set_seed, print_training_summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader


def parse_args():
    parser = ArgumentParser(description="Train the Nepali honorifics translation model")
    parser.add_argument("--batch-size", type=int, help="Override the batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--max-length", type=int, help="Override maximum token length")
    parser.add_argument("--learning-rate", type=float, help="Override learning rate")
    parser.add_argument("--model-name", type=str, help="Override HuggingFace model name/path")
    parser.add_argument("--model-dir", type=str, help="Override the output directory where the trained model is saved")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA fine-tuning")
    parser.add_argument("--bpe-merges", type=int, default=200, help="Number of scratch BPE merge operations for demonstration")
    parser.add_argument("--num-workers", type=int, help="Override DataLoader num_workers")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint if available")
    parser.add_argument("--colab", action="store_true", help="Enable Colab-friendly defaults")
    return parser.parse_args()


def main():
    args = parse_args()
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

    effective_batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    effective_epochs = args.epochs if args.epochs is not None else EPOCHS
    effective_max_length = args.max_length if args.max_length is not None else MAX_LENGTH
    effective_learning_rate = args.learning_rate if args.learning_rate is not None else LEARNING_RATE
    effective_model_name = args.model_name if args.model_name is not None else MODEL_NAME
    effective_use_lora = False if args.no_lora else USE_LORA
    effective_num_workers = NUM_WORKERS
    effective_model_dir = Path(args.model_dir) if args.model_dir else MODEL_DIR

    if args.colab or COLAB_MODE:
        effective_num_workers = 0

    if args.num_workers is not None:
        effective_num_workers = args.num_workers

    effective_model_dir.mkdir(parents=True, exist_ok=True)

    # Scratch BPE demonstration step. This computes merge rules on the training English corpus
    # so the project contains a from-scratch BPE implementation, but the actual model input
    # still uses the built-in HuggingFace tokenizer BPE internally.
    bpe_demo = BytePairEncoder(num_merges=args.bpe_merges)
    bpe_corpus = [item["english"] for item in train_data]
    bpe_demo.fit(bpe_corpus)

    bpe_merges_path = effective_model_dir / "bpe_merges.txt"
    with open(bpe_merges_path, "w", encoding="utf-8") as f:
        for i, pair in enumerate(bpe_demo.get_merge_rules(), start=1):
            f.write(f"{i}\t{pair[0]} {pair[1]}\n")

    print(f"🧠 Scratch BPE prepared on English training data ({len(bpe_demo.get_merge_rules())} merges saved to {bpe_merges_path})")
    if len(bpe_corpus) > 0:
        print("📘 Scratch BPE example encodings:")
        for sample_text in bpe_corpus[:3]:
            print(f"   Input  : {sample_text}")
            print(f"   Encoded: {bpe_demo.encode(sample_text)}")
        print("⚠️  Note: actual model training still uses the transformer tokenizer's built-in BPE for input IDs.")

    config_obj = type('Config', (), {
        'MODEL_NAME': effective_model_name,
        'SRC_LANG': SRC_LANG,
        'TGT_LANG': TGT_LANG,
        'DEVICE': DEVICE,
        'EPOCHS': effective_epochs,
        'BATCH_SIZE': effective_batch_size,
        'LEARNING_RATE': effective_learning_rate,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'MAX_LENGTH': effective_max_length,
        'NUM_WORKERS': effective_num_workers,
        'WARMUP_RATIO': WARMUP_RATIO,
        'GRADIENT_CLIP': GRADIENT_CLIP,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'GRADIENT_CHECKPOINTING': GRADIENT_CHECKPOINTING,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'USE_LORA': effective_use_lora,
        'LORA_R': LORA_R,
        'LORA_ALPHA': LORA_ALPHA,
        'LORA_DROPOUT': LORA_DROPOUT,
        'MODEL_DIR': effective_model_dir,
    })

    print_training_summary(config_obj)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(effective_model_name, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(effective_model_name)

    # Datasets and loaders
    train_dataset = HonorificsDataset(train_data, tokenizer, effective_max_length)
    val_dataset   = HonorificsDataset(val_data,   tokenizer, effective_max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=effective_num_workers,
        pin_memory=DEVICE.type == 'cuda'
    )
    val_loader   = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=DEVICE.type == 'cuda'
    )

    # Start Trainer
    trainer = Trainer(model, train_loader, val_loader, tokenizer, config=config_obj)

    resumed_epoch = 0
    if RESUME_FROM_SESSION or args.resume:
        resumed_epoch = trainer.load_session_checkpoint()

    remaining_epochs = max(effective_epochs - resumed_epoch, 0)
    print(f"\n🚀 Starting training on {DEVICE}...")
    print(f"🎯 Target total epochs: {effective_epochs}")
    if resumed_epoch > 0:
        print(f"🔁 Continuing from previous session at completed global epoch {resumed_epoch}")

    if remaining_epochs == 0:
        print(f"✅ Training already reached target: completed {resumed_epoch}/{effective_epochs} epochs")
        return

    print(f"⏳ Remaining epochs to run now: {remaining_epochs}\n")

    last_completed_epoch = resumed_epoch

    for session_epoch in range(1, remaining_epochs + 1):
        global_epoch = resumed_epoch + session_epoch
        print(f"--- Epoch {global_epoch}/{effective_epochs} (Session {session_epoch}/{remaining_epochs}) ---")
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
            print(f"\n⛔ Early stopping triggered at global epoch {global_epoch}")
            trainer.save_session_checkpoint(epoch_completed=global_epoch)
            break

        if session_epoch % SESSION_SAVE_EVERY_EPOCHS == 0:
            trainer.save_session_checkpoint(epoch_completed=global_epoch)

    # Always save at end of run so next session can continue.
    trainer.save_session_checkpoint(epoch_completed=last_completed_epoch)
    trainer.save_final_model()

    print("\n🎉 Training finished successfully!")
    print(f"Best model saved at: {trainer.best_model_path}")

if __name__ == "__main__":
    main()