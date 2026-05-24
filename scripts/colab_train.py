#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
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
from src.data_utils import load_honorifics_from_register_files, stratified_split
from src.dataset import HonorificsDataset
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import set_seed, print_training_summary
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader


def parse_args():
    parser = ArgumentParser(description="Colab-ready fine-tuning for Nepali honorifics translation")

    parser.add_argument("--model-dir", type=str, default=str(MODEL_DIR),
                        help="Base output directory for checkpoints and model artifacts")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="Pretrained HuggingFace model name or path")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Maximum token length")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of DataLoader workers")
    parser.add_argument("--no-lora", action="store_true",
                        help="Disable LoRA fine-tuning")
    parser.add_argument("--sample-text", type=str,
                        default="Please sign the international renewable energy treaty, secretary?",
                        help="Sentence to translate after each epoch")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint if available")
    parser.add_argument("--use-val-metrics", action="store_true",
                        help="Calculate validation metrics after each epoch")
    return parser.parse_args()


def create_config(args):
    return type('Config', (), {
        'MODEL_NAME': args.model_name,
        'SRC_LANG': SRC_LANG,
        'TGT_LANG': TGT_LANG,
        'DEVICE': DEVICE,
        'EPOCHS': args.epochs,
        'BATCH_SIZE': args.batch_size,
        'LEARNING_RATE': args.learning_rate,
        'WEIGHT_DECAY': WEIGHT_DECAY,
        'MAX_LENGTH': args.max_length,
        'NUM_WORKERS': args.num_workers,
        'WARMUP_RATIO': WARMUP_RATIO,
        'GRADIENT_CLIP': GRADIENT_CLIP,
        'GRADIENT_ACCUMULATION_STEPS': GRADIENT_ACCUMULATION_STEPS,
        'GRADIENT_CHECKPOINTING': GRADIENT_CHECKPOINTING,
        'EARLY_STOPPING_PATIENCE': EARLY_STOPPING_PATIENCE,
        'USE_LORA': False if args.no_lora else USE_LORA,
        'LORA_R': LORA_R,
        'LORA_ALPHA': LORA_ALPHA,
        'LORA_DROPOUT': LORA_DROPOUT,
        'MODEL_DIR': Path(args.model_dir),
    })


def load_data():
    all_data, skipped, reasons = load_honorifics_from_register_files(DATASET_FILES)
    if len(all_data) == 0:
        raise RuntimeError(f"No valid sentence pairs found. Reasons: {dict(reasons)}")
    return stratified_split(all_data, seed=SEED)


def sample_translation(model, tokenizer, sentence: str, device, max_length=64) -> str:
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        min_length=2,
        num_beams=4,
        length_penalty=1.0,
        no_repeat_ngram_size=2,
        early_stopping=True,
        do_sample=False,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    args = parse_args()
    set_seed(SEED)

    print("🚀 Colab fine-tuning starting")
    model_dir = Path(args.model_dir)
    best_model_path = model_dir / "best_honorifics_model"
    checkpoint_path = model_dir / "session_checkpoint.pt"
    final_model_path = model_dir / "final_honorifics_model"

    model_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_data = load_data()
    print(f"✅ Loaded data → Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    config = create_config(args)
    print_training_summary(config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    train_dataset = HonorificsDataset(train_data, tokenizer, args.max_length)
    val_dataset = HonorificsDataset(val_data, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE.type == 'cuda'
    )

    trainer = Trainer(model, train_loader, val_loader, tokenizer, config=config)
    start_epoch = 0
    if args.resume and checkpoint_path.exists():
        start_epoch = trainer.load_session_checkpoint()

    evaluator = Evaluator(trainer.model, tokenizer, DEVICE)
    model_name = args.model_name

    for epoch in range(start_epoch + 1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss = trainer.train_epoch()
        val_loss, perplexity = trainer.validate()

        print(f"Train Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")
        print(f"Perplexity : {perplexity:.2f}")

        should_stop, should_save = trainer.check_early_stopping(val_loss)
        if should_save:
            trainer.save_best_model()
            print(f"✅ Best model updated at epoch {epoch}")

        if epoch % args.save_every == 0:
            trainer.save_session_checkpoint(epoch_completed=epoch)

        translated = sample_translation(trainer.model, tokenizer, args.sample_text, DEVICE, max_length=args.max_length)
        print(f"Sample translation after epoch {epoch}: {translated}")

        if args.use_val_metrics:
            print("⏱️ Running validation metrics...")
            metrics = evaluator.evaluate(val_data)
            print(f"Validation BLEU: {metrics['bleu']:.2f}, Exact: {metrics['exact']:.2f}%")

        if should_stop:
            print(f"⛔ Early stopping triggered at epoch {epoch}")
            trainer.save_session_checkpoint(epoch_completed=epoch)
            break

    trainer.save_session_checkpoint(epoch_completed=epoch)
    trainer.save_final_model()

    print("\n🎉 Training complete.")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    print("You can evaluate the test set with scripts/evaluate.py or use the compare_models.py script.")


if __name__ == '__main__':
    main()
