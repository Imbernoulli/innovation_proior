# Context

## Research question

The task is training a deep self-attention sequence model on *long* inputs —
sequences of tens of thousands of tokens (long documents, music with tens of
thousands of events, images flattened to thousands of pixels) — under the
memory budget of a single accelerator. The model is the standard recipe: stack
$N$ layers, each an attention sublayer plus a position-wise feed-forward
sublayer, with residual connections and normalization, trained by
backpropagation. The goal is to make this model fit and train at sequence length
$L = 64\text{K}$ on one device, while matching the quality of the unmodified
architecture.

The difficulty is not the parameter count. A back-of-the-envelope estimate makes
this concrete: the largest reported Transformer layer holds $\approx 0.5\text{B}$
parameters $\approx 2\text{GB}$; the input activations for $64\text{K}$ tokens at
embedding width $1024$ and batch $8$ are $64\text{K}\times 1\text{K}\times 8
\approx 0.5\text{B}$ floats $\approx 2\text{GB}$. Per-layer, a large model on a
$64\text{K}$ sequence should fit comfortably on one device, and the entire BERT
training corpus is only $17\text{GB}$ on disk. Yet such models cannot even be
fine-tuned on a single machine. So the question is sharp: where does the memory
actually go, and is the cost fundamental or merely an artifact of how the model
is implemented and trained?

## Background

The memory of a depth-$N$ Transformer on a length-$L$ sequence is dominated by
three multiplicative factors, each knowable by inspection of the architecture,
none of them present in the naive per-layer estimate above:

- **Stored activations scale with depth.** Backpropagation needs the forward
  activations of every layer to compute that layer's gradient, so a model with
  $N$ layers stores roughly $N$ times the activations of a single layer. This is
  the $n_l$ factor.
- **The feed-forward width dominates per-layer memory.** Each layer's
  position-wise feed-forward sublayer projects up to an intermediate width
  $d_{ff}$ that is typically several times the model width $d_{model}$ (a common
  setting is $d_{ff}=4096$ with $d_{model}=1024$). The intermediate activation
  is $[b, L, d_{ff}]$, so the FFN is often the largest single activation tensor.
- **Attention is quadratic in length.** Scaled dot-product attention forms
  $QK^\top$ of shape $[b, L, L]$. At $L=64\text{K}$, even a single head at batch
  $1$ in $32$-bit floats is a $64\text{K}\times 64\text{K}$ matrix $= 16\text{GB}$.

Two standard observations about the attention computation: the output of
attention is $\mathrm{softmax}(QK^\top)V$, and a softmax is dominated by its
largest arguments — for a given query $q_i$, the keys $k_j$ with the largest dot
products $q_i\cdot k_j$ carry most of the weight. Also, the full $QK^\top$ matrix
need never be materialized: attention can be computed query-by-query,
$\mathrm{softmax}(q_i K^\top/\sqrt{d_k})V$, holding only $O(L)$ memory and
recomputing on the backward pass.

Two further pieces of standard background. **Locality-sensitive hashing
(LSH)** is the standard tool for finding near neighbors quickly in
high-dimensional spaces: a hash $h(x)$ is locality-sensitive if nearby vectors
collide (get the same hash) with high probability and distant ones do not.
**Reversible residual networks** are a technique for training deep nets
without storing per-layer activations, by making each layer analytically
invertible so its inputs can be recomputed from its outputs during backprop.

## Baselines

**Scaled dot-product / multi-head attention (Vaswani et al. 2017).** Given
queries, keys, values packed as $Q, K, V$,
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.$$
Multi-head attention runs $h$ such functions in parallel on learned projections
of the inputs to dimension $d_k, d_k, d_v$, concatenates, and projects out. In a
self-attention layer all three of $Q, K, V$ are separate linear projections of
the same activation tensor $A\in\mathbb{R}^{[b,L,d_{model}]}$. The decoder uses a
causal mask so position $i$ attends only to $j\le i$; standard implementations
allow a position to attend to itself.

**Memory-efficient attention.** The same function, but $QK^\top$ is never stored:
each query's row $\mathrm{softmax}(q_i K^\top/\sqrt{d_k})V$ is computed in turn and
recomputed during the backward pass. Memory drops to $O(L)$.

**Sparse / fixed-pattern attention (Child et al. 2019).** Replaces the dense
$L\times L$ pattern with a hand-designed factorized sparse pattern (e.g.
strided / local blocks), reducing the $O(L^2)$ cost.

**Angular locality-sensitive hashing (Andoni et al. 2015).** A practical, near
optimal LSH family for angular (cosine) distance. Project a vector through a
random rotation and read off a bucket from the geometry of the rotated point;
near-in-angle vectors collide with high probability. Concretely, with a random
matrix $R$ of shape $[d_k, b/2]$, defining $h(x)=\arg\max([xR;\,-xR])$ (the
$\arg\max$ over the $b$ signed projected axes) is a known angular-LSH scheme.

**Reversible residual networks (Gomez et al. 2017).** A normal residual layer
computes $y=x+F(x)$ and must cache $x$ for the backward pass. A reversible block
splits the signal in two, $(x_1,x_2)\mapsto(y_1,y_2)$, and computes
$$y_1=x_1+F(x_2),\qquad y_2=x_2+G(y_1),$$
which is exactly invertible:
$$x_2=y_2-G(y_1),\qquad x_1=y_1-F(x_2).$$
Because inputs are recoverable from outputs using only the layer's own functions,
no intermediate activations need to be stored: during backprop, each layer is
inverted on the fly as the gradient flows from output to input. Demonstrated as a
drop-in replacement for ResNet on image classification.

**Adafactor (Shazeer & Stern 2018).** An Adam-like optimizer that factorizes the
second-moment accumulator to use sublinear memory in the parameter dimensions —
relevant because optimizer state is itself a memory cost at this scale.

## Evaluation settings

The natural long-sequence yardsticks at the time: **enwik8**, byte-level language
modeling on the first $100\text{M}$ bytes of Wikipedia, here chunked into
subsequences of $2^{16}=64\text{K}$ tokens to stress long context, reported in
bits per dim. **imagenet64**, autoregressive generation of $64\times 64$ images
flattened into sequences of $\approx 12\text{K}$ tokens, also in bits per dim.
**WMT 2014 English-to-German** translation (encoder-decoder), reported in BLEU
and detokenized sacreBLEU, following standard base/big hyperparameters. A
controlled **synthetic duplication task** — inputs of the form $0w0w$ with
$w\in\{1,\dots,N\}^*$, predicting the second copy — is a probe for
whether an attention approximation can still perform non-local lookups that a
limited-span sparse pattern cannot. Diagnostic measurements at the time: the
$QK^\top$ memory at $L=64\text{K}$ ($16\text{GB}$ for one head), and the FFN
intermediate memory at $d_{ff}=4\text{K}, n_l=16, L=64\text{K}$ (again
$\approx 16\text{GB}$). Models are trained across $8$ accelerator devices with a
total batch of $8$ sequences, $d_{model}=1024$, $d_{ff}=4096$, $n_{heads}=8$.

## Code framework

The pre-existing scaffold is a standard multi-layer self-attention language
model in PyTorch: a token embedding, a stack of residual blocks each wrapping an
attention sublayer and a feed-forward sublayer, a final norm and output
projection, trained by autograd. The pieces that already exist — the embedding,
the feed-forward module, the optimizer, the loss, the residual/normalization
wrappers — are kept; the attention sublayer and residual block are left as stubs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def default(val, d):
    return d if val is None else val

class FeedForward(nn.Module):
    # position-wise FFN: d_model -> d_ff -> d_model, exists already
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x):
        return self.w2(self.dropout(self.act(self.w1(x))))

class SelfAttention(nn.Module):
    # the attention sublayer
    # TODO: implement.
    def __init__(self, dim, heads=8, causal=False):
        super().__init__()
        # TODO
        pass

    def forward(self, x, **kwargs):
        # TODO
        pass

class ResidualBlock(nn.Module):
    # residual wrapping of a sublayer
    # TODO: implement.
    def __init__(self, dim, f, g):
        super().__init__()
        self.f = f   # attention sublayer
        self.g = g   # feed-forward sublayer

    def forward(self, x):
        # TODO
        pass

class SequenceModel(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8, causal=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        # TODO: position information
        self.layers = nn.ModuleList([
            ResidualBlock(dim, SelfAttention(dim, heads, causal), FeedForward(dim))
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.token_emb(x)
        # TODO: add position info; run the stack; norm; project to logits
        pass

# training loop, optimizer, loss already exist
def train_step(model, batch, opt):
    logits = model(batch[:, :-1])
    loss = F.cross_entropy(logits.transpose(1, 2), batch[:, 1:])
    loss.backward()
    opt.step(); opt.zero_grad()
    return loss
```
