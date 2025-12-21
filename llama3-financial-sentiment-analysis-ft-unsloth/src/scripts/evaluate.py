# Evaluation Script for Local Fine-Tuned Llama 3.2 Financial Sentiment Model
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report,
    confusion_matrix,
)
from tqdm.auto import tqdm

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

VALID_LABELS = ["Positive", "Negative", "Neutral"]



# Model Loading
def load_model(model_path: str, max_seq_length: int = 256):
    
    print(f"\n{'='*60}")
    print("Loading Fine-Tuned Model")
    print(f"{'='*60}")
    print(f"  Model Path: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Switch to optimized inference mode
    FastLanguageModel.for_inference(model)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"Device: {device}")
    print(f"Model loaded successfully")
    
    return model, tokenizer



# Inference
def predict_sentiment(
    model, 
    tokenizer, 
    text: str, 
    max_new_tokens: int = 4,
    device: str = "cuda"
) -> tuple[str, str]:
    """
    Predict sentiment for a single text.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        text: Input financial text
        max_new_tokens: Maximum tokens to generate
        device: Device to run inference on
    
    Returns:
        Tuple of (predicted_label, raw_output)
    """
    messages = [
        {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    
    # Post-process to extract label
    raw_output = generated.strip()
    first_word = raw_output.split()[0].replace(".", "") if raw_output else ""
    label = first_word.capitalize()
    
    # Validate label
    if label not in VALID_LABELS:
        label = "Neutral"  # Default fallback
    
    return label, raw_output


def batch_predict(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 1,
    max_new_tokens: int = 4,
    device: str = "cuda",
    show_progress: bool = True,
) -> list[tuple[str, str]]:
    """
    Predict sentiment for multiple texts.
    
    Note: Currently processes one at a time due to variable-length generation.
    Batch processing could be added for encoder-only models.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        texts: List of input texts
        batch_size: Batch size (currently unused, kept for API consistency)
        max_new_tokens: Maximum tokens to generate
        device: Device to run inference on
        show_progress: Whether to show progress bar
    
    Returns:
        List of (predicted_label, raw_output) tuples
    """
    results = []
    iterator = tqdm(texts, desc="Predicting") if show_progress else texts
    
    for text in iterator:
        label, raw = predict_sentiment(model, tokenizer, text, max_new_tokens, device)
        results.append((label, raw))
    
    return results


# Evaluation
def evaluate_model(
    model,
    tokenizer,
    val_dataset,
    device: str = "cuda",
    max_samples: int = None,
) -> dict:
    """
    Evaluate the model on the validation dataset.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        val_dataset: Validation dataset
        device: Device to run inference on
        max_samples: Maximum samples to evaluate (None for all)
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\n{'='*60}")
    print("Evaluating Model")
    print(f"{'='*60}")
    
    # Limit samples if specified
    if max_samples and max_samples < len(val_dataset):
        val_dataset = val_dataset.select(range(max_samples))
        print(f"  Evaluating on {max_samples} samples (limited)")
    else:
        print(f"  Evaluating on {len(val_dataset)} samples")
    
    # Collect predictions
    y_true = []
    y_pred = []
    raw_outputs = []
    
    for example in tqdm(val_dataset, desc="Evaluating"):
        true_label = example["output"].strip().capitalize()
        pred_label, raw_output = predict_sentiment(
            model, tokenizer, example["input"], device=device
        )
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        raw_outputs.append(raw_output)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=VALID_LABELS)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", labels=VALID_LABELS)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=VALID_LABELS)
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        labels=VALID_LABELS,
        output_dict=True,
        zero_division=0,
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=VALID_LABELS)
    
    results = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {
            label: f1 for label, f1 in zip(VALID_LABELS, per_class_f1)
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "num_samples": len(y_true),
        "predictions": {
            "y_true": y_true,
            "y_pred": y_pred,
            "raw_outputs": raw_outputs,
        }
    }
    
    return results


def print_results(results: dict):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Samples Evaluated: {results['num_samples']}")
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    
    print(f"\nPer-Class F1 Scores:")
    for label, f1 in results["per_class_f1"].items():
        print(f"{label:10s}: {f1:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"{'':12s} {'Positive':>10s} {'Negative':>10s} {'Neutral':>10s}")
    cm = results["confusion_matrix"]
    for i, label in enumerate(VALID_LABELS):
        row = " ".join(f"{v:10d}" for v in cm[i])
        print(f"{label:12s} {row}")


def save_results(results: dict, output_path: str):
    """Save evaluation results to JSON file."""
    # Remove non-serializable items for JSON
    save_data = {k: v for k, v in results.items() if k != "predictions"}
    save_data["timestamp"] = datetime.now().isoformat()
    
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


# Interactive Demo
def interactive_demo(model, tokenizer, device: str = "cuda"):
    """Run interactive demo for testing the model."""
    print(f"\n{'='*60}")
    print("Interactive Demo")
    print("Type 'quit' or 'exit' to stop")
    print(f"{'='*60}\n")
    
    while True:
        try:
            text = input("Enter financial text: ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text:
                continue
            
            label, raw = predict_sentiment(model, tokenizer, text, device=device)
            print(f"Predicted: {label}")
            print(f"Raw output: {raw}\n")
            
        except KeyboardInterrupt:
            break
    
    print("\nEvaluation Completed")


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Llama 3.2 Financial Sentiment Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model checkpoint"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing val_dataset.jsonl"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (None for all)"
    )
    
    # Model arguments
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save results JSON (optional)"
    )
    
    # W&B arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log results to W&B"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="llama3-financial-sentiment-qlora",
        help="W&B project name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not provided)"
    )
    
    # Mode arguments
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo mode"
    )
    parser.add_argument(
        "--demo_texts",
        nargs="+",
        type=str,
        default=None,
        help="Sample texts to demonstrate (instead of full evaluation)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("Llama 3.2 Financial Sentiment Model Evaluation")
    print("="*60)
    
    # Verify CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU (slow).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.max_seq_length)
    
    # Interactive mode
    if args.interactive:
        interactive_demo(model, tokenizer, device)
        return
    
    # Demo mode with specific texts
    if args.demo_texts:
        print(f"\n{'='*60}")
        print("Demo Predictions")
        print(f"{'='*60}")
        for text in args.demo_texts:
            label, raw = predict_sentiment(model, tokenizer, text, device=device)
            print(f"\nText: {text}")
            print(f"Predicted: {label}")
            print(f"Raw: {raw}")
        return
    
    # Full evaluation mode
    data_path = Path(args.data_dir)
    val_path = data_path / "val_dataset.jsonl"
    
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    
    val_dataset = load_dataset("json", data_files=str(val_path), split="train")
    print(f"Loaded {len(val_dataset)} validation samples")
    
    # Run evaluation
    results = evaluate_model(
        model, tokenizer, val_dataset,
        device=device,
        max_samples=args.max_samples,
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output_file:
        save_results(results, args.output_file)
    
    # Log to W&B
    if args.use_wandb and WANDB_AVAILABLE:
        if os.environ.get("WANDB_API_KEY"):
            run_name = args.run_name or f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=args.wandb_project, name=run_name)
            wandb.log({
                "val_accuracy": results["accuracy"],
                "val_macro_f1": results["macro_f1"],
                "val_weighted_f1": results["weighted_f1"],
                **{f"val_f1_{k}": v for k, v in results["per_class_f1"].items()},
            })
            wandb.finish()
            print(f"Results logged to W&B: {args.wandb_project}/{run_name}")
        else:
            print("WANDB_API_KEY not set. Skipping W&B logging.")
    
    print("\n" + "="*60)
    print("Evaluation Completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

