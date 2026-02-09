<div align="center">

#  LLM Fine-Tuning Lab

### A Collection of Large Language Model Fine-Tuning Experiments

<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"></a>
<a href="https://unsloth.ai/"><img src="https://img.shields.io/badge/Unsloth-2x_Faster-7C3AED?style=for-the-badge" alt="Unsloth"></a>
<a href="https://colab.research.google.com/"><img src="https://img.shields.io/badge/Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"></a>

*Practical, reproducible fine-tuning projects for LLMs*

</div>

---

## What Is This Repository.

This repository is a **living laboratory** for LLM fine-tuning experiments. Each folder contains a complete, self-contained project that fine-tunes a large language model for a specific task, from sentiment analysis to code generation, from summarization to domain-specific assistants.

I will add my open-source fine-tuning experiments here over time. Every project includes full training code, datasets, and detailed documentation so you can reproduce the results on free-tier GPUs like Google Colab T4.

### Philosophy

I built this lab with a few core principles in mind.

**Learn by Doing.** Each project walks through the entire fine-tuning pipeline, from data preparation to evaluation. I explain not just *what* I did but *why* I made each decision.

**Reproducibility First.** All experiments can be replicated on free-tier GPUs. I use parameter-efficient techniques like QLoRA that make LLM training accessible without enterprise hardware.

**Production-Ready Results.** These are not toy experiments. Each fine-tuned model achieves strong performance on real-world tasks and can be deployed in actual applications.

---

## Fine-Tuning Projects

| Project | Model | Task | Method | Accuracy | Status |
|:--------|:------|:-----|:-------|:---------|:-------|
| [Llama 3.2 Financial Sentiment](./llama3-financial-sentiment-analysis-ft-unsloth/) | Llama-3.2-3B-Instruct | Sentiment Classification | QLoRA + Unsloth | 83.42% | ✅ Complete |

---

## Project Highlights

<table>
<tr>
<td width="100%" valign="top">

### Llama 3.2 Financial Sentiment Analysis

I fine-tuned Llama 3.2 3B for 3-class financial sentiment classification using QLoRA. The model learns to read financial text the way a seasoned analyst would, distinguishing bullish optimism from bearish fear and neutral factual statements.

**Results:**
- 83.42% accuracy, 81.57% macro F1
- Trained in ~15 min on free Colab T4
- Only 0.75% parameters trained (24.3M)

[![Open Project](https://img.shields.io/badge/Open_Project-blue?style=for-the-badge)](./llama3-financial-sentiment-analysis-ft-unsloth/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I2_FX7USgFMyROr29oaEljxd1QoubYD-?usp=sharing)

</td>
</tr>
</table>

---

## Techniques and Methods

All projects in this lab utilize modern, efficient fine-tuning techniques that make LLM training accessible.

| Technique | Description | Memory Savings |
|:----------|:------------|:---------------|
| **QLoRA** | 4-bit quantization + Low-Rank Adaptation | ~75% reduction |
| **LoRA** | Train small adapter layers instead of full model | ~90% fewer params |
| **Unsloth** | Optimized training kernels for 2x speed | 2x faster training |
| **Gradient Checkpointing** | Trade compute for memory | ~30% reduction |
| **Mixed Precision (FP16/BF16)** | Half-precision training | ~50% reduction |

### Why These Methods

Traditional fine-tuning requires enterprise GPUs with 40GB+ VRAM. These techniques changed that completely.

```
Full Fine-Tuning (7B model):    ~56 GB VRAM  ❌ Requires A100
QLoRA Fine-Tuning (7B model):   ~6 GB VRAM   ✅ Works on T4/RTX 3060
```

I can now fine-tune billion-parameter models on consumer hardware. The quality difference is negligible for most downstream tasks.

---

## Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Base Models** | Llama 3.2, Mistral, Qwen, Phi |
| **Fine-Tuning** | Unsloth, PEFT, TRL, Transformers |
| **Quantization** | bitsandbytes (NF4, INT8) |
| **Tracking** | Weights & Biases, TensorBoard |
| **Deployment** | vLLM, Ollama, HuggingFace Hub |
| **Hardware** | NVIDIA T4, RTX 3090/4090, A100 |

---

## Results Overview

| Project | Task Type | Base Model | Validation Metric | Score |
|:--------|:----------|:-----------|:------------------|:------|
| Financial Sentiment | Classification | Llama 3.2 3B | Accuracy / F1 | 83.4% / 81.6% |

---

## Getting Started

### Prerequisites

```bash
# Python 3.10+
python --version

# CUDA-capable GPU (recommended)
nvidia-smi
```

### Clone the Repository

```bash
git clone https://github.com/FamilOrujov/llm-finetuning-lab.git
cd llm-finetuning-lab
```

### Explore a Project

```bash
# Navigate to a project
cd llama3-financial-sentiment-analysis-ft-unsloth

# Install dependencies
pip install -r requirements-torch-cu126.txt
pip install -r requirements.txt

# Follow the project's README for training instructions
```

### Or Run in Google Colab

Each project includes a Colab notebook, no local setup required.

| Project | Colab Link |
|:--------|:-----------|
| Llama 3.2 Financial Sentiment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I2_FX7USgFMyROr29oaEljxd1QoubYD-?usp=sharing) |

---

## Repository Structure

```
llm-finetuning-lab/
│
├── llama3-financial-sentiment-analysis-ft-unsloth/
│   ├── README.md                 # Project documentation
│   ├── data/                     # Training & validation data
│   ├── notebooks/                # Jupyter notebooks
│   ├── src/scripts/              # Training & evaluation scripts
│   ├── assets/                   # Images & visualizations
│   └── requirements.txt          # Dependencies
│
└── README.md                     # This file (lab overview)
```

