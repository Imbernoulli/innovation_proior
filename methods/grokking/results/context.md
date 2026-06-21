## Problem Pressure

Modern neural networks can have enough parameters to interpolate their training sets. Random-label experiments make the point sharper: the same broad class of architectures and optimizers can fit arbitrary labels while failing on held-out examples. The question is therefore not whether the model class can store a finite dataset. The question is when training selects a function that captures structure rather than only the observed pairs.

Natural datasets sit in a regime where the exact rule is unknown, train and test distributions are entangled, and validation changes over training are gradual. The interest here is in a task where memorization and rule recovery make different, exactly checkable predictions.

## Existing Clues

Double-descent work shows that interpolation is not necessarily the end of useful learning. Classical U-shaped risk curves can be followed by improved performance after the interpolation threshold, and later work also observes epoch-wise versions where training longer can improve test error. Those settings usually vary effective capacity, model size, noise, or natural-data complexity.

Algorithmic benchmarks offer another clue. Neural Turing Machines, Neural GPUs, memory networks, sequence-to-sequence arithmetic tasks, and procedurally generated mathematics datasets all test whether networks can learn symbolic procedures. Their usual emphasis is architecture, length generalization, broad reasoning coverage, or unlimited generated data.

## Research Question

The setting is a testbed with a complete answer key, a precise notion of unseen examples, and inputs presented so surface notation cannot become a shortcut: opaque symbols rather than decimal digits, permutation notation, or hand-coded features. Every held-out example is an unobserved query from the same finite object, drawn from the same finite rule rather than a new distribution.

The question is what training does to a fixed overparameterized model after it can already fit a small finite training set. The measurement logs training and held-out accuracy separately over optimization time, keeps model size fixed while varying data fraction, optimizer details, regularization, and noise, and follows the curves well past the point where train accuracy saturates.

## Available Ingredients

Several standard tools are already on the table: a small decoder-only transformer with causal masking, learned token embeddings, cross-entropy loss, Adam or AdamW, learning-rate warmup, weight decay, dropout, minibatches, full-batch optimization, and injected gradient or weight noise. Existing generalization measures suggest checking sensitivity to parameter perturbations as a proxy for sharpness.

The candidate data source is mathematical and finite. It allows easy generation of every possible input-output pair and includes both easy and difficult rules. It also includes symmetries or isomorphisms that give internal sanity checks: if two tasks are the same up to relabeling of symbols, a symbol-only learner should treat them similarly.

## Starting Scaffold

```python
import torch
import torch.nn as nn

def make_symbolic_dataset(rule, train_fraction, rng):
    """Create all finite rule queries, hide a random fraction, and encode symbols.
    The open design choice is how to expose inputs so notation cannot become
    a shortcut."""
    pass

class SmallCausalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=2, n_heads=4):
        super().__init__()
        pass

    def forward(self, tokens):
        pass

def rhs_loss_and_accuracy(logits, tokens):
    """Score only the right-hand side of the equation."""
    pass

def train(model, train_data, validation_data, optimizer, steps):
    """Log train and validation behavior as functions of optimization time."""
    pass
```
