# Megatron-LM (Tensor Model Parallelism)

## Problem

Train a transformer whose parameters and optimizer state exceed a single accelerator's memory, by splitting the model across $P$ devices — simply (a few primitives in existing PyTorch, no compiler, no model rewrite), with minimal cross-device communication, and composably with data and pipeline parallelism.

## Key idea

A transformer layer is dominated by GEMMs in its MLP and attention. Shard those GEMMs so the only communication is one all-reduce per block per pass:

- **MLP** $Z = B\,\sigma(A X)$. Split the first weight $A$ **column-wise** ($A=[A_1,A_2]$, $X$ replicated): then $\sigma(XA)=[\sigma(XA_1),\sigma(XA_2)]$ — the (elementwise) GeLU applies locally, no sync (splitting $A$ row-wise would force an all-reduce *before* the nonlinearity, since $\sigma(X_1A_1+X_2A_2)\ne\sigma(X_1A_1)+\sigma(X_2A_2)$). Split the second weight $B$ **row-wise** ($B=[B_1;B_2]$): each device computes $Y_iB_i$ from its local $Y_i$, and the output $\sum_i Y_iB_i$ needs **one all-reduce**.
- **Self-attention.** Heads are independent, so split the $Q,K,V$ projections **column-wise by whole heads** — each device computes full attention for its heads locally — and split the output projection **row-wise**, summing per-device head outputs with **one all-reduce**.
- **Conjugate primitives $f$ and $g$.** $f$ (input boundary): identity forward, all-reduce backward (sum $dL/dX$ over the shards $X$ fanned out to). $g$ (output boundary): all-reduce forward (sum partial outputs), identity backward. They are mirror images; each is a tiny `torch.autograd.Function`. A sharded block is $X\to f\to[\text{col GEMM}\to\sigma\to\text{row GEMM}]\to g$. Total: **4 all-reduces per layer** (2 forward, 2 backward).
- **Duplicated, not communicated:** layer norm, dropout, residual adds are run redundantly on every device; each device optimizes its own (duplicated) parameters, so no parameter communication.
- **Vocabulary-parallel output.** Shard the embedding $E$ over vocabulary ($v$ large). Instead of all-gathering logits ($b\times s\times v$ elements), **fuse the sharded logit GEMM with the cross-entropy loss**, communicating only scalar losses ($b\times s$) — a factor-$v$ communication reduction.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def all_reduce(t):
    torch.distributed.all_reduce(t)
    return t


class _F(torch.autograd.Function):          # input boundary: identity fwd, all-reduce bwd
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad):
        return all_reduce(grad)


class _G(torch.autograd.Function):          # output boundary: all-reduce fwd, identity bwd
    @staticmethod
    def forward(ctx, x):
        return all_reduce(x)
    @staticmethod
    def backward(ctx, grad):
        return grad


f = _F.apply
g = _G.apply


class ColumnShardedLinear(nn.Module):       # split output features; X replicated
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert out_features % world_size == 0
        self.weight = nn.Parameter(torch.empty(out_features // world_size, in_features))

    def forward(self, x):
        return F.linear(f(x), self.weight)   # local X A_i ; f handles backward all-reduce


class RowShardedLinear(nn.Module):          # split input features; outputs summed by g
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert in_features % world_size == 0
        self.weight = nn.Parameter(torch.empty(out_features, in_features // world_size))

    def forward(self, x):
        return g(F.linear(x, self.weight))   # local Y_i B_i ; g all-reduces the sum


class ParallelMLP(nn.Module):
    def __init__(self, hidden, world_size):
        super().__init__()
        self.dense_h_to_4h = ColumnShardedLinear(hidden, 4 * hidden, world_size)
        self.dense_4h_to_h = RowShardedLinear(4 * hidden, hidden, world_size)

    def forward(self, x):
        return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x)))


class ParallelSelfAttention(nn.Module):
    def __init__(self, hidden, n_heads, world_size):
        super().__init__()
        assert n_heads % world_size == 0
        self.n_local_heads = n_heads // world_size
        self.head_dim = hidden // n_heads
        self.qkv = ColumnShardedLinear(hidden, 3 * hidden, world_size)   # split by heads
        self.proj = RowShardedLinear(hidden, hidden, world_size)

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        attn = ((q @ k.transpose(-1, -2)) / self.head_dim ** 0.5).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, -1)
        return self.proj(out)


class VocabParallelCrossEntropy(nn.Module):
    def __init__(self, hidden, vocab, world_size):
        super().__init__()
        assert vocab % world_size == 0
        self.embed = nn.Parameter(torch.empty(vocab // world_size, hidden))  # E_i

    def forward(self, x, target):
        local_logits = F.linear(x, self.embed)          # [b, s, vocab/world_size]
        # fuse with cross-entropy: all-reduce only scalars (local max, sum-exp,
        # target-logit) -> communication O(b*s), never the O(b*s*v) logits.
        return fused_vocab_parallel_cross_entropy(local_logits, target)
```
