<div align="center">

# 🔬 LLM Fine-Tuning Lab

### A Collection of Large Language Model Fine-Tuning Experiments

[![GitHub Stars](https://img.shields.io/github/stars/FamilOrujov/llm-finetuning-lab?style=social)](https://github.com/FamilOrujov/llm-finetuning-lab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

*Practical, reproducible fine-tuning projects for LLMs*

</div>

---.

## 🎯 What Is This Repository?

This repository is a **living laboratory** for LLM fine-tuning experiments. Each folder contains a complete, self-contained project that fine-tunes a large language model for a specific task from sentiment analysis to code generation, from summarization to domain-specific assistants.

### Philosophy

- **Learn by Doing**: Every project includes full training code, datasets, and detailed documentation
- **Reproducibility First**: All experiments can be replicated on free-tier GPUs (Google Colab T4)
- **Production-Ready**: Not just experiments these are deployable models with real-world utility
- **Efficient Methods**: Focus on parameter-efficient techniques (LoRA, QLoRA) that democratize LLM training

---

## 📂 Fine-Tuning Projects

Click on any project to explore its full documentation, training notebooks, and results.

| Project | Model | Task | Method | Accuracy | Status |
|:--------|:------|:-----|:-------|:---------|:-------|
| [🦙 **Llama 3.2 Financial Sentiment**](./llama3-financial-sentiment-analysis-ft-unsloth/) | Llama-3.2-3B-Instruct | Sentiment Classification | QLoRA + Unsloth | 83.42% | ✅ Complete |

---

## 🚀 Quick Navigation

<table>
<tr>
<td width="50%" valign="top">

### 🦙 Llama 3.2 Financial Sentiment Analysis

Fine-tuning Llama 3.2 3B for 3-class financial sentiment classification using QLoRA.

**Highlights:**
- 83.42% accuracy, 81.57% macro F1
- Trained in ~15 min on free Colab T4
- Only 0.75% parameters trained (24.3M)

[![Open Project](https://img.shields.io/badge/Open_Project-blue?style=for-the-badge)](./llama3-financial-sentiment-analysis-ft-unsloth/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I2_FX7USgFMyROr29oaEljxd1QoubYD-?usp=sharing)

</td>
<td width="50%" valign="top">

### 🔜 Coming Soon

More fine-tuning projects are in development:

- 🧠 **Code Assistant** — Fine-tuning for Python code generation
- 📄 **Document Summarizer** — Long-form text summarization
- 💬 **Customer Support Bot** — Domain-specific conversational AI
- 🏥 **Medical NER** — Named entity recognition for healthcare

*Star this repo to stay updated!*

</td>
</tr>
</table>

---

## 🧪 Techniques & Methods

All projects in this lab utilize modern, efficient fine-tuning techniques:

| Technique | Description | Memory Savings |
|:----------|:------------|:---------------|
| **QLoRA** | 4-bit quantization + Low-Rank Adaptation | ~75% reduction |
| **LoRA** | Train small adapter layers instead of full model | ~90% fewer params |
| **Unsloth** | Optimized training kernels for 2x speed | 2x faster training |
| **Gradient Checkpointing** | Trade compute for memory | ~30% reduction |
| **Mixed Precision (FP16/BF16)** | Half-precision training | ~50% reduction |

### Why These Methods?

Traditional fine-tuning requires enterprise GPUs with 40GB+ VRAM. These techniques make fine-tuning accessible:

```
Full Fine-Tuning (7B model):    ~56 GB VRAM  ❌ Requires A100
QLoRA Fine-Tuning (7B model):   ~6 GB VRAM   ✅ Works on T4/RTX 3060
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Base Models** | Llama 3.2, Mistral, Qwen, Phi |
| **Fine-Tuning** | Unsloth, PEFT, TRL, Transformers |
| **Quantization** | bitsandbytes (NF4, INT8) |
| **Tracking** | Weights & Biases, TensorBoard |
| **Deployment** | vLLM, Ollama, HuggingFace Hub |
| **Hardware** | NVIDIA T4, RTX 3090/4090, A100 |

---

## 📊 Results Overview

Aggregate performance across all projects:

| Project | Task Type | Base Model | Validation Metric | Score |
|:--------|:----------|:-----------|:------------------|:------|
| Financial Sentiment | Classification | Llama 3.2 3B | Accuracy / F1 | 83.4% / 81.6% |

*Table updates as new projects are added*

---

## 🏃 Getting Started

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

Each project includes a Colab notebook — no local setup required:

| Project | Colab Link |
|:--------|:-----------|
| Llama 3.2 Financial Sentiment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1I2_FX7USgFMyROr29oaEljxd1QoubYD-?usp=sharing) |

---

## 📁 Repository Structure

```
llm-finetuning-lab/
│
├── 📂 llama3-financial-sentiment-analysis-ft-unsloth/
│   ├── README.md                 # Project documentation
│   ├── data/                     # Training & validation data
│   ├── notebooks/                # Jupyter notebooks
│   ├── src/scripts/              # Training & evaluation scripts
│   ├── assets/                   # Images & visualizations
│   └── requirements.txt          # Dependencies
│
├── 📂 [future-project-1]/        # Coming soon
├── 📂 [future-project-2]/        # Coming soon
│
└── README.md                     # This file (lab overview)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Add a New Project**: Fork, create your fine-tuning experiment, submit a PR
2. **Improve Documentation**: Fix typos, clarify explanations, add examples
3. **Report Issues**: Found a bug? Open an issue with reproduction steps
4. **Share Results**: Trained a model? Share your metrics and insights

### Contribution Guidelines

- Each project should be self-contained in its own folder
- Include a detailed README with training parameters and results
- Provide reproducible notebooks (preferably Colab-compatible)
- Document your hyperparameter choices and reasoning

---

## 📈 Roadmap

- [x] Llama 3.2 Financial Sentiment Analysis
- [ ] Code generation fine-tuning (StarCoder/CodeLlama)
- [ ] Multi-turn conversation fine-tuning
- [ ] Instruction following (Alpaca-style)
- [ ] Domain adaptation for legal/medical text
- [ ] Model merging experiments (DARE, TIES)
- [ ] Deployment guides (vLLM, Ollama)

---

## 📝 License

This repository is licensed under the MIT License. Individual projects may use models with their own licenses (e.g., Llama Community License).



