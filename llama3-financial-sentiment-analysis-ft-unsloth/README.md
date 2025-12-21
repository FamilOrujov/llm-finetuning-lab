# Llama 3.2 Financial Sentiment Analysis Fine-Tuning

Fine-tuning **Llama-3.2-3B-Instruct** for financial sentiment classification using **QLoRA** and **Unsloth**.

## Overview

| Component | Details |
|-----------|---------|
| **Model** | `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` |
| **Method** | QLoRA (4-bit quantization + LoRA adapters) |
| **Dataset** | [Financial Sentiment Analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) |
| **Samples** | 5,257 train / 585 validation |
| **Task** | 3-class sentiment: Positive, Negative, Neutral |

## 🚀 Quick Start - Google Colab

The full training notebook is available on Google Colab with free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I2_FX7USgFMyROr29oaEljxd1QoubYD-?usp=sharing)

## Project Structure

```
├── data/
│   ├── data.csv                 # Raw dataset
│   ├── train_dataset.jsonl      # Processed training data
│   └── val_dataset.jsonl        # Processed validation data
├── notebooks/
│   ├── dataset_prepare.ipynb    # Data preprocessing notebook
│   └── llama3-ft-colab.ipynb    # Colab link notebook
├── src/scripts/
│   ├── train.py                 # Local training script
│   └── evaluate.py              # Evaluation script
├── assets/
│   └── wb_training_analysis.png # W&B training metrics
└── requirements.txt             # Dependencies
```

## Training Metrics

![Training Analysis](assets/wb_training_analysis.png)

## LoRA Configuration

```python
{
    "r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"]
}
```

## Local Training

```bash
# Install dependencies
pip install -r requirements-torch-cu126.txt
pip install -r requirements.txt

# Run training
python src/scripts/train.py ./data ./outputs --epochs 3 --batch_size 4 --use_wandb
```

## Evaluation

```bash
python src/scripts/evaluate.py --model_path ./outputs/final --data_dir ./data
```

