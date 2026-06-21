My model has grown past what a single accelerator can hold. The trouble is not only the raw weights: the activations and, worst of all, the optimizer state inflate the footprint, since Adam keeps a momentum buffer and a variance buffer per parameter, so the memory I actually need is several times the parameter count. I cannot shrink the model without surrendering the very capacity that motivated scaling — tying or reusing weights across layers caps representational power and defeats the point. So the parameters and their optimizer state must live across several devices, and the only real question is how to split them. Data parallelism, the standard tool, does nothing here: it replicates the whole model on every worker, splits the batch, and averages gradients, which scales throughput beautifully but presumes the entire model fits on one worker — exactly the assumption that has failed. To break the limit I have to split the model itself, and the two existing ways of doing that are each unsatisfying. Pipeline parallelism puts different layers on different devices and streams activations down the chain, but it wastes compute on pipeline bubbles (later stages sit idle while earlier ones fill the pipe), needs careful scheduling of overlapped communication, and in some variants perturbs the optimizer enough to cost accuracy. General distributed tensor frameworks like Mesh-TensorFlow partition individual tensor operations across a device mesh — the right kind of generality — but they demand a new language and a compiler and a re-expression of the model. I want neither a compiler nor a rewrite; I want to keep my existing PyTorch transformer and insert a handful of primitives.

So the target is to split the heavy computation *inside* each layer across $P$ devices with as few cross-device communications as possible, by editing the layer rather than rewriting the framework. I propose Megatron-LM, a tensor model-parallel transformer. Its premise is that a transformer layer is dominated by the GEMMs in its two-layer MLP and its self-attention, while dropout, layer normalization, and residual adds are cheap elementwise work; so if I shard those GEMMs cleverly, the communication can be reduced to a single all-reduce per block per pass. The cleverness is entirely in *which way* each weight matrix is cut.

Take the MLP first, $Z = \text{dropout}\big(B\,\sigma(A X)\big)$ with $X$ of shape $[b\,s, H]$, the first weight $A$ of shape $[H, 4H]$ expanding the width, GeLU as the nonlinearity $\sigma$, and the second weight $B$ of shape $[4H, H]$ projecting back. There are two ways to cut $A$ in the product $XA$. If I cut $A$ along its rows and $X$ along the matching contraction dimension, $X=[X_1, X_2]$ and $A=[A_1; A_2]$, then $XA = X_1A_1 + X_2A_2$ is a *sum* of per-device partials — but the next operation is the nonlinearity, and $\sigma(X_1A_1 + X_2A_2) \ne \sigma(X_1A_1) + \sigma(X_2A_2)$, so I would have to all-reduce the partials *before* applying GeLU, planting a synchronization point in the middle of the block. That is precisely what I want to avoid. So instead I cut $A$ along its columns, $A=[A_1, A_2]$ with $X$ replicated on every device, giving $XA = [XA_1, XA_2]$: device $i$ owns a block of *output columns*, and because GeLU is elementwise it acts on each column block independently,
$$[Y_1, Y_2] = [\,\sigma(XA_1),\ \sigma(XA_2)\,],$$
so $Y_i = \sigma(XA_i)$ is computed entirely locally with no communication. Now the second GEMM $YB$ already has its input $Y=[Y_1, Y_2]$ living split as output-column blocks, so I split $B$ along its *rows*, $B=[B_1; B_2]$, and device $i$ computes $Y_iB_i$ from exactly the $Y_i$ it holds. The full output is $\sum_i Y_iB_i$, so I all-reduce once, at the very end of the block, then apply dropout. This column-then-row pairing is the load-bearing choice: the column split of the first GEMM hands the second GEMM its input already in the layout the row split consumes, so the GeLU between them needs no sync and the only communication is one all-reduce at the block's output.

The backward pass is the mirror image, and recognizing that symmetry is the whole trick. The summed output $Z$ has the same gradient $dL/dZ$ on every device, which flows back into each $Y_iB_i$ locally with no sync; but the input $X$ was replicated and fed to all $P$ column-shards, so $dL/dX = \sum_i (dL/dX)_i$ must be summed across devices — one all-reduce in the backward pass, at the input. I name these two conjugate communication behaviours. The primitive $f$ sits at the *input* of a sharded region: in the forward pass it is the identity (each device just uses $X$ as-is), and in the backward pass it all-reduces the incoming gradient (because $X$ fanned out to all shards). The primitive $g$ sits at the *output*: in the forward pass it all-reduces (summing $\sum_i Y_iB_i$), and in the backward pass it is the identity (the gradient of a summed output is copied identically to each device). They are exact conjugates,
$$f: (\text{forward identity},\ \text{backward all-reduce}), \qquad g: (\text{forward all-reduce},\ \text{backward identity}),$$
each the other with forward and backward swapped, and each is a tiny `torch.autograd.Function`. A sharded block is then just $X \to f \to [\text{column GEMM} \to \sigma \to \text{row GEMM}] \to g$, with $f$ owning the backward sync and $g$ the forward sync — a few lines, no compiler.

Self-attention falls into the same mold for free, because multi-head attention already has the independence structure I had to engineer for the MLP. Each head has its own slice of the query, key, and value projections, computes $\text{softmax}(QK^\top/\sqrt{d})\,V$ on its own, and the heads do not interact until the output projection mixes them. So I split the $Q,K,V$ projection GEMMs column-wise *by whole heads* — device $i$ owns a subset of the heads with their full projections — and each device computes the entire attention for its heads locally, since a head never needs another head's keys or values. Then the output projection that mixes the heads back is split row-wise, exactly as the MLP's second GEMM, consuming each device's per-head output directly and summing the partial contributions with one all-reduce. Same column-then-row pattern, same single $g$ at the output and $f$ at the input. Tallying the layer: attention contributes one all-reduce forward and one backward, the MLP another forward and backward — four communication operations per transformer layer, with the per-head softmax and the GeLU running locally in between because the fusion eliminated the intermediate sync. The parts I deliberately do not shard — layer norm, dropout, residual adds — are cheap and elementwise, so I duplicate them: every device keeps its own copy of the layer-norm parameters and runs dropout and the residual on the now-reduced, identical-across-devices block output. Because each device's parameters are duplicates and everything it touches is either local or identical across devices, each device simply optimizes its own parameter set with no extra parameter communication at all.

One more GEMM is worth sharding: the embedding. The vocabulary holds tens of thousands of tokens, so the embedding matrix $E$ of shape $[H, v]$ is large, and I split it along the *vocabulary* dimension, $E=[E_1, E_2]$. The input lookup then needs a $g$ all-reduce because each device holds only part of the table. The output (tied to the input weights) computes logits $[Y_1, Y_2] = [XE_1, XE_2]$ sharded over vocabulary, and the naive finish — all-gather the logits so the loss can be computed — would move $b \times s \times v$ numbers, which with $v$ huge becomes the dominant communication in the entire model. Instead I fuse the sharded logit GEMM with the cross-entropy loss: each device computes its partial of the loss (its local max, its sum-of-exponentials, its target-logit) from its vocabulary shard, and I all-reduce only those scalars, $b \times s$ numbers. Communicating scalars instead of full logits cuts that cost by a factor of $v$. The result is a tensor-parallel transformer that is a few primitives bolted onto an existing PyTorch model — compute-bound rather than communication-stalled, and composable with both data parallelism and pipeline parallelism.

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
