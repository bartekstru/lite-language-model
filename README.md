# Language Model Experiments

This repository contains various experiments and implementations related to language modeling and neural networks, with a focus on character-level models.

## Contents

### Notebooks

1. `bigram.ipynb`: Implements a simple bigram language model. It includes:
   - Loading and processing a dataset of names
   - Creating a bigram frequency matrix
   - Implementing a basic neural network for bigram prediction
   - Generating new names using the trained model

2. `mlp.ipynb`: Implements a multi-layer perceptron (MLP) for character-level language modeling. Features:
   - Dataset creation from a list of names
   - MLP architecture with multiple hidden layers
   - Training loop with loss visualization
   - Name generation using the trained model

3. `backprop.ipynb`: Focuses on implementing backpropagation from scratch. Includes:
   - Detailed implementation of forward and backward passes
   - Batch normalization
   - Gradient checking
   - Training loop and loss visualization

4. `wavenet.ipynb`: Experiments with WaveNet-style architectures for language modeling. Contains:
   - Dataset preparation
   - WaveNet-inspired model implementation (incomplete)
   - Training and evaluation code

5. `nanoGPT.ipynb`: Implements a small-scale version of GPT (Generative Pre-trained Transformer). Includes:
   - Data loading and preprocessing
   - Transformer architecture implementation
   - Training loop and loss tracking
   - Text generation using the trained model

6. `gradients.ipynb`: Explores gradient computation and optimization techniques. Features:
   - Various optimization algorithms
   - Gradient visualization
   - Experiments with different learning rates and batch sizes

### Python Scripts

1. `llm.py`: Implements a "Lite Language Model". This script includes:
   - Data downloading and preprocessing
   - A complete Transformer-based language model implementation
   - Training loop with checkpointing
   - Text generation functionality

## TO DO List

- [ ] Implement the forward pass for the WaveNet (GRU)
- [ ] Implement the convolutions for WaveNet
- [ ] Speed things up with GPU
- [ ] Hyperparameter search for the right WaveNet architecture
