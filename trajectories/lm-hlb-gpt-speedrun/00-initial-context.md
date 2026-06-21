## Research question

Train a small GPT-style language model from scratch on a single GPU to a **fixed validation-loss bar as fast as possible**, then keep pushing the wall-clock time down without letting the model fall below the bar.

The target is **~3.8 validation cross-entropy loss (≈44.7 perplexity) on WikiText-103**. This point sits where training curves begin to flatten: the model must learn real temporal and topical dependencies, but it does not require paying for diminishing returns. The hardware is fixed at one **40 GB NVIDIA A100**. The ranking metric is **wall-clock seconds on that A100 to reach ~3.8 val loss**, lower is better, with the loss bar held constant. Because compute is capped at one GPU and the target is fixed, scale cannot buy the win; the only levers are algorithmic and engineering choices in the training method.

## Prior art / Background / Baselines

The model family is the decoder-only GPT. A token sequence is embedded, passed through a stack of residual blocks each containing causal self-attention and a position-wise MLP, layer-normalized, and projected back to a vocabulary distribution; the loss is next-token cross-entropy. The reference small-scale implementation is Karpathy's nanoGPT: a single-file decoder with learned token and absolute-position embeddings, multi-head causal attention, a 4×-expansion GELU MLP, weight tying between input embedding and output projection, AdamW, and a cosine learning-rate schedule with warmup. It is the natural starting point for anyone trying to train a correct small GPT fast.

Several pre-existing facts about the hardware and regime shape what "fast" can mean. The A100 runs tensor-core matmuls much faster in reduced precision (bf16/fp16) than in fp32, and bf16 keeps the dynamic range of fp32. The available PyTorch 2.0 stack supplies `torch.compile` for graph fusion and `F.scaled_dot_product_attention` for fused attention. Attention cost is quadratic in sequence length and is paid per token at every step. The MLP and attention together dominate the FLOP budget. Gradient statistics are not stationary: early steps see large noisy gradients, late steps see small ones. Absolute learned position embeddings are one of several ways to inject order into an otherwise permutation-equivariant attention stack.

A related methodology is the single-GPU *speedrun*, previously used on image classification: train a CNN to a fixed CIFAR-10 accuracy in seconds by choosing a short, fixed target at a flattening point and then stacking architecture, precision, schedule, and data-pipeline changes that each shave wall-clock while holding the target. The premise is that a short run at this target carries most of the signal of a longer run at a small fraction of the cost.

**Baselines:**

- **nanoGPT-style decoder GPT.** A faithful small GPT using the standard defaults: learned absolute positions, pre-norm residual blocks, 4× GELU MLP, weight tying, AdamW with warmup and decay, and fixed sequence length and effective batch size held constant for the whole run. It reaches the ~3.8 val-loss bar.

- **Constant-effective-batch gradient accumulation.** The usual way to reach a large effective batch on one GPU is to accumulate gradients over a fixed number of microbatches, then step. The number is hand-tuned once for the whole run.

- **Fixed sequence length.** Training at one sequence length for the entire run is simple and is what every baseline does.

## Fixed substrate / Code framework

The benchmark is fixed: WikiText-103, tokenized with the GPT-2 BPE tokenizer (`tiktoken`, vocab padded to 50304 for tensor-core-friendly shapes). Training and validation tokens live on the GPU; batches are drawn by random offset sampling. The scaffold is a single-file PyTorch GPT trainer. Already wired: the data download/tokenize/pack pipeline, AdamW, cross-entropy loss, a learning-rate scheduler hook, `LayerNorm`, and a residual-block container. bf16 tensor-core math and `torch.compile` are available from the environment.

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

## Editable interface

The open design decisions are the slots in the scaffold above: the attention and MLP internals, how sequence order is injected, the precision policy of the network, the optimizer schedule, the batching/gradient-accumulation policy, and the sequence-length policy. Any change is judged by the same fixed target: whether it reduces wall-clock seconds to ~3.8 val loss on the single A100 without dropping below the bar.

## Evaluation settings

The benchmark is WikiText-103 (~100M tokens of Wikipedia articles), tokenized with GPT-2 BPE. Training and a held-out validation split are stored as token tensors on the GPU; batches are sampled by random offsets. The reported quantity is validation cross-entropy loss (and perplexity), evaluated periodically on the held-out split; evaluation time is excluded from the timed budget. The fixed target is **~3.8 val loss / ≈44.7 perplexity**. Hardware is one 40 GB A100. The ranking quantity is wall-clock seconds to reach the target on that A100, with timing including graph-compilation overhead inside the timed region.
