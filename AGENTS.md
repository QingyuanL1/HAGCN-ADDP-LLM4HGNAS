# AGENTS.md

This file contains guidelines for agentic coding agents working in this repository.

## Project Overview
HAGCN-for-Solving-APDP implements a Heterogeneous Attention-based Graph Convolutional Network to solve the Asymmetric Pickup and Delivery Problem (APDP) using REINFORCE with rollout baseline. The project uses PyTorch and integrates real-world geographical information from navigation data.

## Build/Test Commands

### Training
Train the model with 20 nodes using rollout baseline:
```bash
python run.py --graph_size 20 --baseline rollout
```

### Evaluation
Evaluate model using greedy decoding:
```bash
python eval.py --model 'outputs/pdp_20/run_20230413T095145/epoch-49.pt' --decode_strategy greedy
```

Evaluate using sampling with 1280 samples:
```bash
python eval.py --model 'outputs/pdp_20/run_20230413T095145/epoch-49.pt' --decode_strategy sample --width 1280 --eval_batch_size 1
```

### GPU Selection
Use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=1,2 python run.py --graph_size 20 --baseline rollout
```

### Testing
No automated test framework is used. Manual evaluation is performed via eval.py with different decode strategies (greedy, sample, bs for beam search).

## Code Style Guidelines

### Import Order
Standard library → Third-party libraries → Project modules (separated by blank lines):

```python
import os
import sys
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pdp.problem_pdp import PDP
from nets.attention_model import AttentionModel
from utils import move_to
```

### Module Organization
- `nets/`: Neural network models (AttentionModel, GraphAttentionEncoder, GCN layers)
- `pdp/`: Problem definition and state management (PDP, StatePDP)
- `utils/`: Utility functions (data loading, beam search, logging)
- `data/`: Training/validation data (coordinates, distance matrices)
- `outputs/`: Saved model checkpoints

### Naming Conventions
- Classes: PascalCase (AttentionModel, StatePDP, PDP, ResidualGatedGCNLayer)
- Functions/Methods: snake_case (train_epoch, validate, get_inner_model)
- Variables: snake_case (batch_size, hidden_dim, learning_rate)
- Module-level constants: UPPERCASE (rarely used)
- Private members: single underscore prefix (e.g., _eval_dataset)

### PyTorch Models
- Always inherit from nn.Module
- Use super().__init__() in __init__
- Define forward() method for computation
- Initialize with device parameter and move model to device
- Use torch.no_grad() context during evaluation
- Set model.train() / model.eval() appropriately

```python
class AttentionModel(nn.Module):
    def __init__(self, device, embedding_dim, hidden_dim, problem, ...):
        super(AttentionModel, self).__init__()
        self.device = device
        # ... initialization ...
        self.to(device)

    def forward(self, input):
        # ... forward pass ...
```

### State Management
Use typing.NamedTuple for state definitions (e.g., StatePDP in pdp/state_pdp.py):

```python
class StatePDP(NamedTuple):
    coords: torch.Tensor
    ids: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor
    to_delivery: torch.Tensor

    @property
    def visited(self):
        # Property method
```

### Type Hints
- Use typing.NamedTuple for complex data structures
- Type hints are used sparingly in function signatures
- PyTorch tensor types are inferred from context

### Error Handling
- Use assert for preconditions and invariant checks
- Example: `assert self.i.size(0) == 1, "Can only update if state represents single step"`
- Assert error messages are descriptive

### Documentation
- Docstrings use triple quotes (""")
- Class docstrings briefly describe purpose
- Method docstrings only for non-obvious functions
- Comments explain non-obvious logic, not what code does

### Argument Parsing
- Use argparse for command-line arguments
- Define all parameters in options.py with help text
- Use type hints in parser.add_argument() for validation
- Default values reflect typical usage

### Logging
- Use tqdm for progress bars during training/evaluation
- Use tensorboard_logger for training metrics
- Use print() for important status messages
- Configure via options: --log_step, --no_progress_bar, --no_tensorboard

### Code Formatting
- Line length: ~100 characters (no strict limit)
- Indentation: 4 spaces (no tabs)
- Blank lines: Separate logical sections with blank lines
- Use f-strings or .format() for string formatting

### Data Loading
- Extend torch.utils.data.Dataset for custom datasets
- Implement __len__ and __getitem__
- Use DataLoader with batch_size and num_workers
- Move data to target device using move_to() utility

### GPU Handling
- Check CUDA availability: `torch.cuda.is_available()`
- Set device: `torch.device("cuda:0" if use_cuda else "cpu")`
- Use torch.nn.DataParallel for multi-GPU training
- Move tensors to device: `tensor.to(device)`

### Gradient Handling
- Use optimizer.zero_grad() before backward pass
- Clip gradients: `torch.nn.utils.clip_grad_norm_(params, max_norm)`
- Step optimizer after backward: `optimizer.step()`

### Baseline Management
- Baselines defined in reinforce_baselines.py
- Options: NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
- Baseline.epoch_callback() called after each epoch
- Baseline.eval() returns baseline values and optional loss

## Dependencies
- Python >= 3.6
- PyTorch >= 1.1
- NumPy
- tensorboard_logger
- tqdm

## Key Files
- run.py: Main training script
- train.py: Training loop functions
- eval.py: Model evaluation
- options.py: Hyperparameter configuration
- reinforce_baselines.py: REINFORCE baseline implementations
- nets/attention_model.py: Main model architecture
- pdp/state_pdp.py: PDP state management
- pdp/problem_pdp.py: PDP problem definition and dataset
