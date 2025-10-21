# Tiny-dLLM

Diffusion-style training for masked language models using ModernBERT.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management.

```bash
# Install base dependencies
uv sync

# Install with Muon optimizer support
uv sync --extra muon

# Or use pip if you prefer
pip install torch>=2.0.0 transformers>=4.30.0 datasets>=2.0.0 sws-config

# Optional: Install Muon optimizer
pip install git+https://github.com/KellerJordan/Muon
```

## Usage

Train with default configuration (AdamW):

```bash
uv run python train.py --config configs/default.py
```

Train with Muon optimizer:

```bash
uv sync --extra muon
uv run python train.py --config configs/muon.py
```

Override config values from command line:

```bash
uv run python train.py --config configs/default.py learning_rate=1e-4 batch_size=32
```

Switch dataset:

```bash
uv run python train.py --config configs/default.py dataset_name=bookcorpus dataset_config=plain_text
```

Switch optimizer on the fly:

```bash
uv run python train.py --config configs/default.py optimizer_name=muon
```

## Configuration

The project uses [sws](https://github.com/lucasb-eyer/sws) for configuration management. Config files are in the `configs/` directory.

### Dataset Configuration
- `dataset_name`: HuggingFace dataset name (default: "wikitext")
- `dataset_config`: Dataset configuration (default: "wikitext-2-raw-v1")

### Model Configuration
- `model_name`: Model to use (default: "answerdotai/ModernBERT-base")
- `max_len`: Maximum sequence length (default: 256)
- `prefix_len`: Number of prefix tokens to never mask (default: 16)

### Training Configuration
- `batch_size`: Training batch size (default: 16)
- `num_epochs`: Number of training epochs (default: 30)
- `n_steps`: Number of diffusion steps (default: 10)

### Optimizer Configuration

The project supports multiple optimizers: `adamw`, `adam`, `sgd`, and `muon`.

#### AdamW (default)
- `optimizer_name`: "adamw"
- `learning_rate`: Learning rate (default: 5e-5)
- `weight_decay`: Weight decay (default: 0.01)
- `adam_beta1`: Beta1 parameter (default: 0.9)
- `adam_beta2`: Beta2 parameter (default: 0.999)
- `adam_epsilon`: Epsilon parameter (default: 1e-8)

#### Muon
Muon is a momentum-based optimizer designed for efficient training of neural networks. See [KellerJordan/Muon](https://github.com/KellerJordan/Muon) for more details.

- `optimizer_name`: "muon"
- `muon_lr`: Muon learning rate (default: 0.02)
- `muon_momentum`: Momentum parameter (default: 0.95)
- `muon_nesterov`: Use Nesterov momentum (default: True)
- `muon_backend`: Backend algorithm (default: "newtonschulz5")
- `muon_auxiliary_lr`: Learning rate for auxiliary params (default: muon_lr * 0.1)

#### SGD
- `optimizer_name`: "sgd"
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `sgd_momentum`: Momentum (default: 0.9)
- `sgd_nesterov`: Use Nesterov momentum (default: True)

#### Adam
- `optimizer_name`: "adam"
- `learning_rate`: Learning rate
- `weight_decay`: Weight decay
- `adam_beta1`: Beta1 parameter (default: 0.9)
- `adam_beta2`: Beta2 parameter (default: 0.999)
- `adam_epsilon`: Epsilon parameter (default: 1e-8)

## Project Structure

```
Tiny-dLLM/
├── train.py           # Main training script
├── data.py            # Data loading and collator
├── optimizer.py       # Optimizer factory
├── configs/           # Configuration files
│   ├── default.py     # Default configuration (AdamW)
│   └── muon.py        # Muon optimizer configuration
├── pyproject.toml     # UV/pip dependencies
├── uv.lock            # UV lock file
└── README.md          # This file
```
