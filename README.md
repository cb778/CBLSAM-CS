# CBLSAM-CS
Combining BiLSTM and Self-attention mechanism for Code Search

## Dependency
Successfully tested in Ubuntu 
- Python == 3.6 
- PyTorch == 1.5.0 or newer
- tqdm == 4.48.2
- numpy  
- tables
- argparse

## Code Structures

- `attentionNet.py`: Includes the core module of CBLSAM-CS.
- `main.py`: The main function entry of the model.
- `dataset.py`: Dataset loader.
- `configs`: Basic configuration of the entire model. Each function defines the hyper-parameters for the corresponding model.
- `utils.py`: Utilities for models and training.
