## Research question

Every idea you want to test on a language model is gated by how long one training run takes. If a run
costs hours, you run few experiments, plan each one heavily, and bias yourself toward the moves you
already believe in. If a run costs minutes, you run many, plan less, and let noise surface things you
would never have planned. The task here is a *speedrun* built around that observation: train a small
GPT-style language model from scratch, on a single GPU, to a **fixed validation-loss bar as fast as
possible** — and then keep driving the wall-clock down without ever letting the model fall short of the
bar.

The bar is **~3.8 validation cross-entropy loss (≈44.7 perplexity) on WikiText-103**. That value is
chosen deliberately: it sits right where the training curves start to flatten — low enough that the
network has to learn real temporal and topical dependencies to reach it, high enough that you are not
paying for diminishing returns. It corresponds to roughly a single pass over the data, which mirrors the
large-scale regime where each token is seen by the full network only once. The hardware is held fixed: a
single **40 GB NVIDIA A100**. The metric the ladder is ranked on is **wall-clock seconds (on that one
A100) to reach ~3.8 val loss** — lower is better — with the loss bar held constant throughout. The single
free variable is the *training method*: the architecture, the precision, the optimizer schedule, the data
schedule, the batching. Because compute is capped at one GPU and the bar is fixed, you cannot buy the
target with scale; the only lever is algorithmic and engineering efficiency.

## Background

The substrate is the GPT decoder-only language model. A token sequence is embedded, passed through a
stack of residual blocks each containing causal self-attention and a position-wise MLP, layer-normalized,
and projected back to a vocabulary distribution; the loss is next-token cross-entropy. The reference
small-scale implementation of this is Karpathy's nanoGPT: a clean, single-file decoder GPT with learned
token and absolute-position embeddings, multi-head causal attention, a 4×-expansion GELU MLP, weight
tying between the input embedding and the output projection, AdamW, and a cosine learning-rate schedule
with warmup. nanoGPT is the well-understood baseline the field reaches for when it wants "a correct GPT
that trains," and it is the natural starting point for anyone trying to train one *fast*.

Several facts about the hardware and the regime shape what "fast" can mean. (1) The A100 has tensor cores
that run dramatically faster in reduced precision (fp16/bf16) than in fp32, and bf16 has the dynamic range
of fp32, so it tends to be numerically forgiving where fp16 needs loss scaling. (2) PyTorch 2.0 introduced
`torch.compile`, which fuses and optimizes the computation graph ahead of time, and a fused/flash
scaled-dot-product-attention primitive (`F.scaled_dot_product_attention`) that avoids materializing the
full attention matrix. (3) Attention cost is quadratic in sequence length, and that cost is paid per
token regardless of which part of the run it falls in; the MLP and attention together dominate the FLOP
budget. (4) The gradient signal is not stationary over a run: early steps see large, noisy gradients;
late steps see small ones. (5) Absolute learned position embeddings — one trainable vector per position
index, added to the token embedding — are one of several ways to inject order into an otherwise
permutation-equivariant attention stack. These are pre-existing facts about A100s, PyTorch, and
transformer training dynamics — knowable before any particular speedup is designed.

A connected line of work is the single-GPU *speedrun* methodology itself, used previously on image
classification (train a CNN to a fixed CIFAR-10 accuracy in seconds): pick a short, fixed target that
correlates with the longer-run quantity you care about, then relentlessly stack architecture, precision,
schedule, and data-pipeline changes that each shave the wall-clock while holding the target. The premise —
that short experiments at a flattening-point target carry most of the signal of longer runs, at a tiny
fraction of the cost — is the methodological background this work inherits.

## Baselines

- **nanoGPT-style decoder GPT (the prior art this starts from).** A faithful small GPT: token + learned
  absolute-position embeddings, a stack of pre-norm residual blocks (causal multi-head attention, then a
  4× GELU MLP), weight-tied output projection, AdamW with a warmup-then-decay learning-rate schedule, and
  a fixed sequence length and fixed effective batch size held constant for the whole run. Trained this way
  on WikiText-103 it reaches the ~3.8 val-loss bar, but it does so by running a fixed-length, fixed-batch
  schedule end to end — it pays full quadratic attention cost on every token from step one, runs its
  matmuls at whatever precision the default mixed-precision path gives, holds its effective batch size
  constant even though the gradient statistics change by orders of magnitude over the run, and uses a
  GELU MLP and learned absolute positions because those are the defaults, not because they were chosen for
  speed. The gap it leaves open: *every one of those "held constant because it's the default" choices is a
  place where a run could be spending compute it does not need to spend to reach this particular bar.*

- **Constant-effective-batch accumulation.** The standard way to get a large effective batch on one GPU
  is gradient accumulation: pick a fixed number of microbatches, accumulate their gradients, then step.
  The number is a hand-tuned constant. It is set once, for the whole run, to whatever value works at the
  hardest part of training. Its limitation: a single constant cannot be simultaneously right for the
  large-gradient early steps and the small-gradient late steps.

- **Fixed sequence length.** Training at one sequence length for the whole run is simple and is what every
  baseline does. But attention is quadratic in length, and the cost is paid identically at step 1 and at
  the last step. Its limitation is purely that it pays a length-dependent cost uniformly across a run whose
  needs are not uniform.

## Evaluation settings

The benchmark is WikiText-103 (a ~100M-token corpus of Wikipedia articles), tokenized with the GPT-2 BPE
tokenizer (`tiktoken`, vocab rounded up to 50304 for tensor-core-friendly shapes). Training data and a
held-out validation split live on the GPU as token tensors; batches are drawn by random offset sampling.
The quantity reported is validation cross-entropy loss (and its exponential, perplexity), evaluated
periodically on the held-out split; eval time is excluded from the timed budget. The held-fixed target is
~3.8 val loss / ≈44.7 perplexity. Hardware is one 40 GB A100; the ranking quantity is wall-clock seconds
to reach the target on that A100, reported per release as the run's measured time. Timing includes
graph-compilation time (the network is compiled inside the timed region, to keep the comparison honest).

## Code framework

The pre-existing scaffold is a single-file PyTorch GPT trainer. The pieces that already exist: the data
download/tokenize/pack pipeline, an AdamW optimizer, a cross-entropy loss, a learning-rate scheduler, a
`LayerNorm`, and a residual-block container; bf16 tensor-core math and `torch.compile` are available from
PyTorch. The block internals, the way order is injected, the precision policy of the net, the optimizer
*schedule*, the batching policy, and the sequence-length policy are the slots to be filled.

```python
import torch, torch.nn as nn, torch.nn.functional as F

hyp = {
    'net':  {'residual_depth': 384, 'num_heads': 6, 'num_blocks': 6},
    'misc': {'num_tokens': 50304, 'sequence_length': 256, 'device': 'cuda', 'dtype': torch.bfloat16},
    'opt':  {'lr': 2e-3, 'weight_decay': 1e-3, 'total_train_steps': None, 'warmup_percent': None},
}

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, weight=True, bias=False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features)) if weight else None
        self.bias   = nn.Parameter(torch.zeros(num_features)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, weight=self.weight, bias=self.bias, eps=self.eps)

class AttentionBlock(nn.Module):
    def __init__(self, num_features, sequence_length, num_heads):
        super().__init__()
        self.norm = LayerNorm(num_features, bias=False)
        # TODO: the attention sublayer + the way sequence order is injected
    def forward(self, x):
        pass

class MLPBlock(nn.Module):
    def __init__(self, num_channels, expansion_factor=4):
        super().__init__()
        self.norm = LayerNorm(num_channels, bias=False)
        # TODO: the position-wise feedforward sublayer
    def forward(self, x):
        pass

class SpeedyLangNet(nn.Module):
    def __init__(self, net_dict):
        super().__init__()
        self.net_dict = net_dict
    def forward(self, x):
        # TODO: embed -> inject order -> blocks -> final norm -> output projection
        pass

def make_net():
    # TODO: assemble embedding, position handling, the block stack, weight tying, init, precision policy
    pass

def main():
    net  = make_net()
    opt  = torch.optim.AdamW(net.parameters(), weight_decay=hyp['opt']['weight_decay'], fused=True)
    sched = None  # TODO: the learning-rate schedule
    # TODO: the training loop — batching policy, sequence-length policy, accumulation policy, the step
    pass
```
