# Local Fine-Tuning Script for Llama 3.2 Financial Sentiment Analysis with QLoRA

import os
import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# Configuration
FINANCIAL_SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. "
    "Given the financial text, reply with exactly one word: Positive, Negative, or Neutral."
)

DEFAULT_MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "bias": "none",
    "use_gradient_checkpointing": True,
    "random_state": 42,
}


# Data Processing
def formatting_prompts_func(examples):
   
    inputs = examples["input"] if isinstance(examples["input"], list) else [examples["input"]]
    outputs = examples["output"] if isinstance(examples["output"], list) else [examples["output"]]

    texts = []
    for inp, out in zip(inputs, outputs):
        text = (
            f"{FINANCIAL_SYSTEM_PROMPT}\n\n"
            f"Financial text:\n{inp}\n\n"
            f"Sentiment: {out.strip().capitalize()}"
        )
        texts.append(text)
    return texts


def load_datasets(data_dir: str):
    """Load training and validation datasets from JSONL files."""
    data_path = Path(data_dir)
    train_path = data_path / "train_dataset.jsonl"
    val_path = data_path / "val_dataset.jsonl"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    train_ds = load_dataset("json", data_files=str(train_path), split="train")
    val_ds = load_dataset("json", data_files=str(val_path), split="train")
    
    print(f"Loaded {len(train_ds):,} training samples")
    print(f"Loaded {len(val_ds):,} validation samples")
    print(f"Columns: {train_ds.column_names}")
    
    return train_ds, val_ds



# Model Setup
def setup_model(model_name: str, max_seq_length: int, load_in_4bit: bool = True):
   
    print(f"\n{'='*60}")
    print("Loading Model with Unsloth Optimizations")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Max Seq Length: {max_seq_length}")
    print(f"  4-bit Quantization: {load_in_4bit}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect (float16 for most GPUs)
        load_in_4bit=load_in_4bit,
    )
    
    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_pct:.2f}%)")
    
    return model, tokenizer



# Training Arguments
def get_training_args(args) -> SFTConfig:
    """Create SFTConfig from command line arguments."""
    
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    use_fp16 = not use_bf16 and torch.cuda.is_available()
    
    report_to = []
    if args.use_wandb and WANDB_AVAILABLE:
        report_to.append("wandb")
    
    return SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=report_to,
        run_name=args.run_name,
        dataloader_num_workers=args.num_workers,
        max_grad_norm=1.0,
        gradient_checkpointing=False,  # Handled by Unsloth
        packing=False,
        optim="adamw_torch",
        seed=args.seed,
    )


def setup_wandb(args, model_name: str, max_seq_length: int, training_args: SFTConfig):

    if not args.use_wandb:
        return None
    
    if not WANDB_AVAILABLE:
        print("W&B requested but not installed. Skipping...")
        return None
    
    # Check for API key
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set. Skipping W&B logging...")
        return None
    
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    
    wandb.login()
    
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "model_name": model_name,
            "max_seq_length": max_seq_length,
            "num_train_epochs": training_args.num_train_epochs,
            "learning_rate": training_args.learning_rate,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
        }
    )
    
    print(f"W&B initialized: {args.wandb_project}/{args.run_name}")
    return run


def train(args):
    
    print("\n" + "="*60)
    print("Llama 3.2 Financial Sentiment Fine-Tuning (QLoRA)")
    print("="*60)
    
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training. No GPU detected.")
    
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Load datasets
    train_ds, val_ds = load_datasets(args.data_dir)
    
    
    model, tokenizer = setup_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.no_4bit,
    )
    
    
    training_args = get_training_args(args)
    
    # Setup W&B
    wandb_run = setup_wandb(args, args.model_name, args.max_seq_length, training_args)
    
    # Create trainer
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
    )
    
    # Train
    trainer.train()
    
    # Save final model
    final_output_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    
    print(f"\nModel saved to: {final_output_dir}")
    
    # Finish W&B
    if wandb_run:
        wandb.finish()
        print("W&B run finished")
    
    print("\n" + "="*60)
    print("Training Completed Successfully")
    print("="*60)
    
    return trainer


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 for Financial Sentiment Analysis using QLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "data_dir", 
        type=str, 
        default="./data",
        help="Directory containing train_dataset.jsonl and val_dataset.jsonl"
    )
    parser.add_argument(
        "output_dir", 
        type=str, 
        default="./outputs/llama3-financial-sentiment-qlora",
        help="Directory to save model checkpoints"
    )
    
    # Model arguments
    parser.add_argument(
        "model_name", 
        type=str, 
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "max_seq_length", 
        type=int, 
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "no_4bit", 
        action="store_true",
        help="Disable 4-bit quantization (use full precision)"
    )
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", help="Learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="llama3-financial-sentiment-qlora", help="W&B project name")
    parser.add_argument("--run_name", type=str, default="llama3-3b-qlora-financial-sentiment", help="W&B run name")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

