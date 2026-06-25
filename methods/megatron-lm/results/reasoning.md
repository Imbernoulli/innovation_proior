OK, let me think this through from scratch. My model has grown past what one accelerator can hold — not just the weights, but the activations and, worst of all, the optimizer state, since Adam keeps a momentum buffer and a variance buffer per parameter, so the memory I need is several times the raw weight count. I can't make the model smaller without giving up the capacity that motivated scaling in the first place. So the parameters and their optimizer state have to live across several devices. The question is *how* to split them.

The obvious tool, data parallelism, doesn't touch this. Replicate the model on each worker, split the batch, average gradients — that scales throughput, but it requires the entire model to fit on *one* worker. My whole problem is that it doesn't. So I need to split the model itself.

There are two ways people split a model, and I want to be clear about why neither off-the-shelf option suits me before I build anything. One is pipeline parallelism: put different *layers* on different devices and stream activations down the chain. It works, but it has pipeline bubbles — while the first stage processes micro-batch two, the later stages processing micro-batch one are idle and vice versa — and it needs careful scheduling, and some versions perturb the optimizer in ways that cost accuracy. The other is general distributed tensor computation: partition individual tensor operations across a mesh of devices. That's the right *kind* of generality, but the frameworks for it want me to adopt a new language and a compiler and re-express my model in it. I don't want a compiler. I want to keep my existing PyTorch transformer and insert a handful of primitives. So my constraint is: split the heavy computation *inside* each layer across $P$ devices, with as few cross-device communications as I can manage, by editing the layer rather than rewriting the framework.

What's actually heavy inside a transformer layer? It's two blocks — self-attention and a two-layer MLP — and both are dominated by big matrix multiplies. The elementwise and normalization pieces (dropout, layer norm, residual add) are cheap. So the real question is: how do I shard a GEMM across $P$ devices so each holds only a slice of the weight matrix, and how much do I have to communicate to stitch the result back together? Let me work the MLP first because it's the cleaner case.

The MLP is, in order, a GEMM that expands the width, $Y = \text{GeLU}(XA)$ with $X$ of shape $[b\,s, H]$ and $A$ of shape $[H, 4H]$, then a second GEMM $Z = YB$ with $B$ of shape $[4H, H]$, then dropout. I want to split $A$ across devices. There are two ways to cut a matrix in a product, and I should weigh both rather than assume one.

First way: cut $A$ along its *rows*, and correspondingly cut $X$ along its *columns* — $X = [X_1, X_2]$, $A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}$. Then $XA = X_1 A_1 + X_2 A_2$: each device computes a partial product on its slice of the contraction dimension, and the full product is the *sum* of the partials. That's fine for a pure matmul. But the next thing I do is apply GeLU, and GeLU is nonlinear, so I worry the partials can't be summed *after* the nonlinearity. Let me actually check rather than wave my hands. Take $H=2$, $4H=4$, three rows of $X$, and compare the true $\text{GeLU}(XA)$ against what I'd get if each device applied GeLU to its own partial and I summed those, $\text{GeLU}(X_1A_1)+\text{GeLU}(X_2A_2)$. Running it, the two differ by about $0.15$ in max-abs — not a rounding artifact, a genuine mismatch. So with a row split I cannot apply the nonlinearity device-locally; I'd have to add the partial products back together across devices *first* — an all-reduce sitting right in the *middle* of the block, between the GEMM and the GeLU. That's a synchronization point I'd rather not pay, and now I have a concrete reason to avoid it.

Second way: cut $A$ along its *columns* — $A = [A_1, A_2]$, with $X$ replicated (not split) on every device. Then $XA = [X A_1,\, X A_2]$: device $i$ computes the $i$-th block of *output columns* from the full $X$. Now the nonlinearity: GeLU is elementwise, so it should act on each output column independently,
$$[Y_1,\, Y_2] = [\text{GeLU}(X A_1),\, \text{GeLU}(X A_2)],$$
device $i$ computing $Y_i = \text{GeLU}(X A_i)$ entirely locally. That's the property I'm hoping for, but "elementwise" is exactly the kind of thing I just saw fail under the row split, so I won't take it on faith — I'll fold it into the end-to-end check below rather than declare it now.

Now the second GEMM, $YB$, with $Y = [Y_1, Y_2]$ already living split across devices as output-column blocks. If I split $B$ along its *rows* — $B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$ — then $YB = Y_1 B_1 + Y_2 B_2$, and device $i$ computes $Y_i B_i$ from exactly the $Y_i$ it already holds, with no communication needed to start. The full output is the sum $\sum_i Y_i B_i$, so I'd all-reduce *once*, at the very end of the block, to add the per-device contributions, and then apply dropout. The appeal of pairing a column split with a row split is that the column split of the first GEMM hands the second GEMM its input already in the layout the row split consumes, so nothing needs to be reshuffled between them — and if the GeLU really is local, there is no sync in the interior at all.

Before I build anything on top of that, I want to confirm the whole composition actually reproduces the unsharded MLP — that splitting $A$ column-wise, applying GeLU locally, splitting $B$ row-wise, and summing really equals $\text{GeLU}(XA)\,B$. Same $H=2$, $4H=4$ setup: I split $A$ into its first two and last two columns, $B$ into its first two and last two rows, compute $Y_1B_1 + Y_2B_2$, and compare to the reference $\text{GeLU}(XA)B$. The max-abs difference comes out at $1.1\times10^{-16}$ — machine epsilon, i.e. they are the same up to floating point. So the column-then-row composition is exact, *and* it confirms the local-GeLU step I was unsure about (had GeLU not commuted with the column split, this end-to-end number would not have been zero). One all-reduce in the forward pass buys me the whole sharded MLP.

What about the backward pass? The gradient flows the other way. The output $Z$ is the summed contribution, so $dL/dZ$ is the same on every device and flows back into each $Y_i B_i$ locally — no sync there. But at the *input*, $X$ was replicated and fed to all $P$ column-shards $A_i$, so the gradient with respect to $X$ should be a *sum* of contributions, one from each shard, $dL/dX = \sum_i (dL/dX)_i$. That is a claim about autodiff, and I've been wrong once already today, so let me finite-difference it. I take the sharded forward, a scalar loss $L=\tfrac12\sum Z^2$, and compute $dL/dX$ two ways: analytically as the sum of the two shards' input-gradient contributions, and numerically by perturbing each entry of $X$. The summed-shards gradient matches finite differences to $1.7\times10^{-9}$ — agreement. As a control I also check shard 1's contribution *alone* against finite differences: that is off by about $1.0$ in max-abs, nowhere close. So the input gradient genuinely requires summing across shards; one device's piece is not the answer. One all-reduce in backward (at the input), one all-reduce in forward (at the output). Symmetric, and now checked rather than assumed.

Let me name these two communication behaviours, because they're going to recur and the symmetry is doing real work. Call the primitive at the *input* of the sharded region $f$: in the forward pass it does nothing — $X$ is just used as-is on each device (identity) — and in the backward pass it all-reduces the gradient (because $X$ fanned out to all shards, and I just checked that the input gradient is the sum over shards). Call the primitive at the *output* of the sharded region $g$: in the forward pass it all-reduces (summing $\sum_i Y_i B_i$), and in the backward pass it does nothing — the gradient of a summed output is copied identically to each device (identity). So
$$f: (\text{forward identity},\ \text{backward all-reduce}), \qquad g: (\text{forward all-reduce},\ \text{backward identity}).$$
Reading those two lines side by side, each is the other with "forward" and "backward" swapped — mirror images. Each is trivial to implement as a custom autograd function: $f$'s forward returns its input untouched, and $f$'s backward all-reduces the incoming gradient and returns it; $g$ is the same with the roles swapped. A sharded region is then just $X \to f \to [\text{column GEMM} \to \text{GeLU} \to \text{row GEMM}] \to g \to \text{output}$, with $f$ taking care of the backward sync and $g$ the forward sync. A few lines, no compiler.

Now the self-attention block. Multi-head attention has a structure I can exploit the same way. The heads are *independent*: each head has its own slice of the query, key, and value projections, computes $\text{softmax}(QK^\top/\sqrt{d})V$ on its own, and the heads don't interact until the final output projection mixes them. So if I split the $Q$, $K$, $V$ projection GEMMs *column-wise by whole heads* — device $i$ gets a subset of the heads, with their full query/key/value projections — then each device should compute the entire attention for its heads locally, with no communication, because a head never needs another head's keys or values. This is the same column-parallel split as the MLP, except the natural granularity is the head: keep whole heads together, split heads across devices. The reason it's safe to apply attention's nonlinearities (the per-head softmax) locally is the same reason the MLP's GeLU was safe — the column split keeps each head's entire output-column block on one device, and softmax over a head's scores only ever touches that head's columns, never another's.

Then the output projection that mixes the heads back into the hidden dimension — split it *row-wise*, exactly as I split the MLP's second GEMM. It takes each device's per-head attention output directly (already in the right layout from the column split of $QKV$) and produces a partial contribution to the full output; sum those with one all-reduce. Same column-then-row pattern, same single $g$ at the output and single $f$ at the input. So attention costs one forward and one backward all-reduce too, by exactly the structure I already verified for the MLP — the only difference is the column granularity is a head rather than a single output feature, which doesn't change the algebra.

Tally the whole layer: the attention block contributes one all-reduce forward (its $g$) and one backward (its $f$); the MLP block contributes one forward and one backward. Four communication operations per transformer layer, total. Everything between the $f$ and the $g$ within each block — the GeLU, the per-head softmax — runs locally with no sync, because the column-then-row fusion left no intermediate synchronization point. That is the cost I wanted to drive down, and four all-reduces per layer is where it landed.

What about the parts I haven't sharded — dropout, layer norm, residual adds? They're cheap and elementwise. I could split them and broadcast, but that's just more communication for no memory win. Simpler to *duplicate* them: every device keeps its own copy of the layer-norm parameters and runs dropout and the residual add on the (now-reduced, identical-across-devices) block output before feeding the next sharded region. Since each device's layer-norm params are duplicates and everything it touches is either local or identical across devices, each device can just optimize its own parameter set with no extra parameter communication at all.

One more GEMM deserves attention: the embedding. The vocabulary is large — tens of thousands of tokens — so the embedding matrix $E$ of shape $[H, v]$ is big and worth sharding. Split it along the *vocabulary* dimension, $E = [E_1, E_2]$. The input embedding lookup then needs an all-reduce (a $g$) because each device holds only part of the table. The output embedding (tied to the input weights) computes logits $[Y_1, Y_2] = [X E_1, X E_2]$, sharded over vocabulary. The naive finish is to all-gather the logits so the loss can be computed — but that gathers $b \times s \times v$ numbers. Let me put real numbers on that to see whether it actually matters: at GPT-2 scale, $b=8$, $s=1024$, $v=50257$, all-gathering the logits moves $b\,s\,v \approx 4.1\times10^{8}$ elements. If instead I fuse the sharded logit GEMM with the cross-entropy loss — each device computing its partial of the loss from its vocabulary shard and communicating only the resulting *scalar* losses, $b\times s = 8192$ numbers — the ratio is $4.1\times10^{8}/8192 = 50257$, i.e. exactly $v$. So the fusion cuts that communication by a factor of $v$, which at this vocabulary size is the difference between hundreds of millions of elements and a few thousand. That is plainly worth doing; without it the logit all-gather would dominate every other collective in the model.

Let me write it, grounded in how the sharded layers are actually built. The two conjugate primitives first — the entire synchronization story lives in these:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def all_reduce(t):
    torch.distributed.all_reduce(t)
    return t


class _F(torch.autograd.Function):
    # input boundary of a sharded region: X fans out to every shard.
    # forward: identity (X used as-is). backward: all-reduce (sum dL/dX over shards).
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad):
        return all_reduce(grad)


class _G(torch.autograd.Function):
    # output boundary of a sharded region: outputs are summed across shards.
    # forward: all-reduce (sum partials). backward: identity (copy grad to each shard).
    @staticmethod
    def forward(ctx, x):
        return all_reduce(x)
    @staticmethod
    def backward(ctx, grad):
        return grad


f = _F.apply
g = _G.apply
```

The column-sharded linear holds a slice of the *output* features and leaves $X$ replicated; the row-sharded linear holds a slice of the *input* features and its caller sums the partials with $g$:

```python
class ColumnShardedLinear(nn.Module):
    # weight split along OUTPUT features: device owns A_i, computes X A_i locally.
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert out_features % world_size == 0
        self.weight = nn.Parameter(torch.empty(out_features // world_size, in_features))

    def forward(self, x):
        x = f(x)                              # backward all-reduce of dL/dX
        return F.linear(x, self.weight)        # local: X A_i  -> [Y_i]


class RowShardedLinear(nn.Module):
    # weight split along INPUT features: device owns B_i, computes Y_i B_i locally.
    def __init__(self, in_features, out_features, world_size):
        super().__init__()
        assert in_features % world_size == 0
        self.weight = nn.Parameter(torch.empty(out_features, in_features // world_size))

    def forward(self, x):                      # x is the local Y_i
        y = F.linear(x, self.weight)           # local: Y_i B_i
        return g(y)                            # forward all-reduce: sum_i Y_i B_i
```

The MLP block is then exactly column-then-row, GeLU in between done locally, one $g$ at the end (inside the row layer):

```python
class ParallelMLP(nn.Module):
    def __init__(self, hidden, world_size):
        super().__init__()
        self.dense_h_to_4h = ColumnShardedLinear(hidden, 4 * hidden, world_size)  # A: column
        self.dense_4h_to_h = RowShardedLinear(4 * hidden, hidden, world_size)      # B: row

    def forward(self, x):
        y = F.gelu(self.dense_h_to_4h(x))      # GeLU on local output-columns, no sync
        return self.dense_4h_to_h(y)            # row GEMM + single all-reduce
```

Self-attention reuses the same primitives — $QKV$ column-sharded by whole heads, output projection row-sharded — so the per-head attention is local and only the output projection all-reduces:

```python
class ParallelSelfAttention(nn.Module):
    def __init__(self, hidden, n_heads, world_size):
        super().__init__()
        assert n_heads % world_size == 0
        self.n_local_heads = n_heads // world_size
        self.head_dim = hidden // n_heads
        # QKV split column-wise by whole heads: each device owns n_local_heads heads.
        self.qkv = ColumnShardedLinear(hidden, 3 * hidden, world_size)
        # output projection split row-wise: sums per-device head outputs.
        self.proj = RowShardedLinear(hidden, hidden, world_size)

    def forward(self, x):
        b, s, _ = x.shape
        qkv = self.qkv(x)                                  # local: heads for this device
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_local_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        attn = scores.softmax(dim=-1)                       # per-head, fully local
        out = (attn @ v).transpose(1, 2).reshape(b, s, -1)
        return self.proj(out)                               # row GEMM + single all-reduce
```

And the vocabulary-parallel output, fusing the sharded logit GEMM with cross-entropy so only scalar losses cross the wire:

```python
class VocabParallelCrossEntropy(nn.Module):
    # logits sharded over vocab; compute the loss locally and reduce scalars (b*s),
    # never all-gather the full b*s*v logits.
    def __init__(self, hidden, vocab, world_size):
        super().__init__()
        assert vocab % world_size == 0
        self.embed = nn.Parameter(torch.empty(vocab // world_size, hidden))  # E_i

    def forward(self, x, target):
        local_logits = F.linear(x, self.embed)              # [b, s, vocab/world_size]
        # (sketch) local max / sum-exp / target-logit are all-reduced as scalars,
        # then combined into the cross-entropy -- communication is O(b*s), not O(b*s*v).
        return fused_vocab_parallel_cross_entropy(local_logits, target)
```

The causal chain, start to end: a multi-billion-parameter model and its Adam state exceed one device's memory, and data parallelism can't help because it needs the whole model on one worker; the model itself must be split, but pipeline parallelism wastes compute on bubbles and tensor-algebra frameworks demand a compiler. Since a transformer layer is dominated by the GEMMs in its MLP and attention, I shard those GEMMs — splitting the first GEMM of each block column-wise so the nonlinearity (GeLU, or the per-head softmax) stays local, and the second GEMM row-wise so it consumes that layout directly and needs only a single all-reduce to sum the partial outputs. I checked that column-then-row reproduces the unsharded block to machine precision while the row-first alternative would force a sync before the nonlinearity, and that the input gradient is the sum over shards (a single shard's piece is off by order one). That communication is captured by two conjugate primitives, $f$ (identity forward / all-reduce backward) at the input and $g$ (all-reduce forward / identity backward) at the output, giving four all-reduces per layer total; layer norm, dropout, and residuals are duplicated rather than communicated; and the large vocabulary embedding is sharded over vocabulary with its output logits fused into the cross-entropy so only $b\times s$ scalars are communicated instead of $b\times s\times v$ logits — a factor-$v$ (here $\approx$ 50k) reduction. The result is a tensor-parallel transformer that is a few primitives bolted onto an existing PyTorch model, compute-bound and composable with data and pipeline parallelism.
