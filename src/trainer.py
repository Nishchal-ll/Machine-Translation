import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import math
from pathlib import Path
from torch.amp import autocast, GradScaler

# Try to use 8-bit Adam for memory efficiency
try:
    from bitsandbytes.optim import AdamW8bit
    HAS_8BIT_ADAM = True
except ImportError:
    HAS_8BIT_ADAM = False

# Try to use LoRA for efficient fine-tuning
try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_LORA = True
except ImportError:
    HAS_LORA = False

class Trainer:
    def __init__(self, model, train_loader, val_loader, tokenizer, config):
        self.model = model.to(config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config

        # Enable gradient checkpointing to save memory
        if getattr(config, 'GRADIENT_CHECKPOINTING', False) and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Apply LoRA for domain-specific fine-tuning
        if getattr(config, 'USE_LORA', False) and HAS_LORA:
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=["q_proj", "v_proj"],  # Target attention layers
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            print(f"✅ LoRA enabled (r={config.LORA_R}, alpha={config.LORA_ALPHA})")

        # Use 8-bit Adam if available (saves ~75% optimizer memory)
        weight_decay = getattr(config, 'WEIGHT_DECAY', 0.0)
        if HAS_8BIT_ADAM:
            self.optimizer = AdamW8bit(
                self.model.parameters(), 
                lr=config.LEARNING_RATE,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(), 
                lr=config.LEARNING_RATE,
                weight_decay=weight_decay
            )

        # Use cyclic learning rate for small datasets (better convergence)
        # Cycles between LR and 0.1*LR every epoch
        from torch.optim.lr_scheduler import CyclicLR
        
        self.scheduler = CyclicLR(
            self.optimizer,
            base_lr=config.LEARNING_RATE,
            max_lr=config.LEARNING_RATE * 2,  # Go up to 2x the base LR
            step_size_up=len(train_loader) // 2,
            cycle_momentum=False
        )

        self.best_val_loss = float("inf")
        self.best_model_path = config.MODEL_DIR / "best_honorifics_model"
        # Use new torch.amp API
        self.scaler = GradScaler('cuda') if config.DEVICE.type == 'cuda' else None
        self.gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
        self.patience = getattr(config, 'EARLY_STOPPING_PATIENCE', 3)
        self.patience_counter = 0
        self.session_checkpoint_path = config.MODEL_DIR / "session_checkpoint.pt"

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        accumulation_counter = 0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}

            # Use mixed precision to reduce memory (new torch.amp API)
            if self.config.DEVICE.type == 'cuda':
                with autocast('cuda', dtype=torch.float16):
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.gradient_accumulation_steps  # Scale loss for accumulation
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_counter += 1

            # Optimizer step after accumulation
            if accumulation_counter % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(self.val_loader)
        perplexity = math.exp(avg_loss)
        return avg_loss, perplexity

    def check_early_stopping(self, current_val_loss):
        """Check early stopping criteria"""
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            return False, True  # Continue training, save model
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience, False  # Stop if patience exceeded, don't save

    def save_best_model(self):
        self.best_model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.best_model_path)
        self.tokenizer.save_pretrained(self.best_model_path)
        print(f"💾 Best model saved → {self.best_model_path}")

    def save_session_checkpoint(self, epoch_completed: int):
        """Save checkpoint that can resume training in next run."""
        self.session_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch_completed": epoch_completed,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, self.session_checkpoint_path)
        print(f"💾 Session checkpoint saved → {self.session_checkpoint_path}")

    def load_session_checkpoint(self):
        """Load previous training checkpoint. Returns completed epoch count."""
        if not self.session_checkpoint_path.exists():
            return 0

        checkpoint = torch.load(self.session_checkpoint_path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", self.best_val_loss)
        self.patience_counter = checkpoint.get("patience_counter", 0)

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        epoch_completed = int(checkpoint.get("epoch_completed", 0))
        print(f"♻️  Resumed from checkpoint at epoch {epoch_completed}")
        return epoch_completed