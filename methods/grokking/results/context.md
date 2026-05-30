# Context: a controllable testbed for studying generalization beyond memorization

## Research question

The generalization of overparameterized neural networks defies classical learning theory: networks
with far more parameters than training samples — enough to fit arbitrary, even randomly-labeled data
— nonetheless generalize well when trained on meaningful data. Understanding *why*, and *when*, an
overparameterized network generalizes rather than merely memorizing its training set is one of the
least understood aspects of deep learning.

The obstacle to studying this on natural datasets (ImageNet, language corpora) is that everything is
entangled and slow: the data has rich structure, train and test distributions differ in complicated
ways, experiments take days, and the interesting generalization effects are mild and hard to isolate
from confounds. What is needed is a *testbed*: a task where (i) memorization and true generalization
can be cleanly separated and measured, (ii) the data has a precisely known underlying structure so
"did the network find the real pattern?" has a definite answer, (iii) experiments are fast and
reproducible (a single GPU), and (iv) the generalization phenomena are pronounced enough to study in
detail. The question is how to construct such a setting and what it reveals about generalization in
the data-limited regime — past the point where the network can already memorize the training set.

## Background

**Overparameterization and memorization.** Networks of the sizes used in practice can interpolate
arbitrary training data, including random labels (Zhang et al. 2016), yet generalize when trained on
semantically meaningful labels with appropriate optimization. So the capacity to memorize is not the
obstacle to generalization; something about the data and the optimization decides whether the network
generalizes or just memorizes. This separation — interpolate the training set with or without
generalizing — is exactly what a good testbed should expose and control.

**Algorithmic datasets as probes.** Procedurally generated algorithmic tasks — copying, reversing,
sorting sequences, multi-digit arithmetic — have long been used to probe neural networks' symbolic
and algorithmic reasoning (Neural Turing Machines, memory networks, the Neural GPU, Neural
Programmer-Interpreters, Universal Transformers; and reasoning suites like bAbI, and procedurally
generated mathematics, Saxton et al. 2019). The diagnostic limitation for studying *generalization*:
most of this work emphasizes the unlimited-data regime, generalization with respect to input *length*,
or a point estimate of one architecture's accuracy — not what happens in the *data-limited* regime
past the point where a fixed architecture can fully memorize a finite training set.

**Double descent in the loss.** A relevant prior observation: the classical U-shaped validation-loss
curve is, in some settings, followed by a *second descent* — validation loss falls again past the
capacity needed to interpolate the training data (deep double descent and its precursors). This is
mostly studied as a function of model size or capacity, and it is accompanied by a non-monotonic
accuracy peak. A testbed that exposed a *training-time* second descent, far past first interpolation
and *without* an accuracy peak, would be a distinct and cleaner phenomenon worth isolating.

**Regularization and flat minima.** Weight decay (an ℓ2 penalty pulling weights toward the origin),
gradient/weight noise, and dropout are the standard regularizers. A recurring theory of why noise and
small-norm solutions help is that they bias optimization toward *flat* minima — minima where the loss
is insensitive to parameter perturbations — and flatness-based measures have been found among the most
predictive of generalization (Jiang et al. 2019; Hochreiter & Schmidhuber 1997 on flat minima). These
are the levers and the lens one would bring to a generalization testbed.

**The structure of the tasks.** For modular arithmetic with a prime p, every nonzero residue is a
power of a primitive root, so addition modulo p−1 and multiplication modulo p are the same group up to
relabeling of elements — a fact that matters once elements are presented as structureless symbols. The
symmetric group S₅ supplies a non-abelian algebraic task (composition of permutations) with subgroup /
coset structure.

## Baselines

The points of comparison for "how to study generalization in a controllable setting" are the prior
uses of algorithmic data and the prior generalization phenomena:

- **Algorithmic sequence tasks in the unlimited-data regime (Neural Turing Machine, Neural GPU,
  memory/programmer networks).** Core idea: test algorithmic reasoning by training on procedurally
  generated sequences, measuring length generalization. Gap: not aimed at the data-limited regime, and
  not at the memorize-vs-generalize transition for a fixed task; emphasis is on architecture and
  unlimited data.

- **Reasoning suites (bAbI; procedurally generated mathematics, Saxton et al. 2019).** Core idea:
  large procedurally generated benchmarks for reasoning. Gap: tasks are involved enough to require very
  large sample counts to master, so they do not lend themselves to observing subtle generalization
  dynamics past the memorization point on a small, fully-specified task.

- **Random-label memorization studies (Zhang et al. 2016).** Core idea: show networks can interpolate
  arbitrary labels. Gap: establishes that capacity is not the bottleneck but does not provide a knob to
  watch generalization emerge or fail on structured data.

- **Double descent in model capacity (deep double descent and precursors).** Core idea: validation
  loss second-descends past the interpolation threshold as capacity grows. Gap: studied along the
  model-size axis with a non-monotonic accuracy peak; does not isolate a pure training-time effect on a
  tiny, exactly-structured dataset.

## Evaluation settings

The natural testbed is supervised prediction of the missing entries of a binary operation table,
trained to completion (very long optimization budgets) on a controlled fraction of the table:

- **Tasks (binary operations a∘b=c).** Modular arithmetic with prime p = 97: x+y, x−y, x/y, x²+y²,
  x²+xy+y², x²+xy+y²+x, x³+xy, x³+xy²+y (mod 97); a mixed operation (x/y mod p if y is odd else x−y);
  and composition in the permutation group S₅: x·y, x·y·x⁻¹, x·y·x. Each operation defines a full table
  of all p² (or |S₅|²) equations.
- **Data split.** A random fraction (the *training fraction*) of all equations forms the training set;
  the rest is the validation set — so train and validation are disjoint slots of the *same* table, and
  generalization means correctly filling in unseen slots.
- **Tokenization.** Each equation is the token sequence ⟨x⟩ ⟨op⟩ ⟨y⟩ ⟨=⟩ ⟨x∘y⟩, where every distinct
  element a, b, c is its own abstract symbol with no internal structure (no decimal digits, no
  permutation line notation).
- **Metrics.** Training and validation accuracy and loss as functions of optimization steps; the number
  of optimization steps to reach a target validation accuracy (e.g. 99%); and data-efficiency curves
  (converged accuracy and steps-to-generalize vs. training fraction). Embeddings of the symbols are
  visualized (t-SNE of the output-layer weights).

## Code framework

The primitives that exist: a standard decoder-only transformer, AdamW with learning-rate warmup,
cross-entropy loss, and a training loop that logs train/validation accuracy over steps. What is
missing is the *task construction* — how to turn an algebraic operation into a learnable,
generalization-probing dataset — and the *training regime* (which regularizers, how long) under which
the interesting behavior appears. The scaffold leaves those as empty slots.

```python
import torch
import torch.nn as nn

def make_dataset(operation, modulus, train_fraction, rng):
    """Turn a binary operation into tokenized equations and split into train/val.
    How to encode the elements so that memorization and generalization are cleanly separable
    is the open design question. TODO."""
    # build all equations a∘b=c; tokenize; randomly assign train_fraction to train, rest to val
    pass

class Transformer(nn.Module):
    """Standard decoder-only transformer with causal masking."""
    def __init__(self, n_layers, d_model, n_heads, vocab_size, max_len):
        super().__init__()
        # embedding + positional encoding + decoder blocks + output projection
        # TODO
        pass
    def forward(self, tokens):
        # return logits over the vocabulary at each position
        pass

def loss_and_acc(logits, targets):
    """Compute loss/accuracy ONLY on the answer token (the c after '='). TODO."""
    pass

def train(model, train_data, val_data, optimizer, num_steps):
    """Train for num_steps; log train/val accuracy over steps.
    The optimizer/regularization choice and HOW LONG to train are part of what we must determine."""
    for step in range(num_steps):
        # forward on a minibatch, loss on answer token, backward, step
        # periodically record train_acc, val_acc
        pass
```
