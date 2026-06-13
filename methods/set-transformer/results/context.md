# Context

## Research question

How should a neural network process an input that is a *set* — an unordered collection of a variable number of elements — when the target is a property of the whole set? Two requirements come straight from the definition of a set: the model must be **permutation invariant** (its output cannot change when the elements are reordered), and it must accept **sets of any size**. The harder, open part: how do we model the *interactions among elements* during the computation, so that problems whose answer genuinely depends on how the elements relate to one another (amortized clustering, set anomaly detection, counting) can be solved well — not just problems where each element can be scored in isolation and the scores summed?

Set-input problems are everywhere: multiple-instance learning (a label for a bag of instances), 3D shape recognition from a point cloud, set operations and statistics, and meta-learning / few-shot classification where the input set *is* a task's support dataset. Classical feed-forward nets violate both requirements (fixed-size input, order-sensitive); RNNs are sensitive to order. A solution has to be permutation-symmetric by construction and size-agnostic, while still being expressive enough to capture inter-element structure.

## Background

**Permutation invariance and equivariance.** A set function `f` is permutation invariant if `f({x_1,…,x_n}) = f({x_{π(1)},…,x_{π(n)}})` for every permutation `π` — trivially required, since the two argument multisets are the same object. A layer `f: X^n → Y^n` is permutation *equivariant* if `f(πx) = π f(x)`: reordering the inputs reorders the outputs the same way. Stacking equivariant layers and finishing with an invariant aggregation yields an invariant model — this is the design pattern any set network must respect.

**Set-pooling architectures (Deep Sets; Edwards & Storkey 2017; Zaheer et al. 2017).** The standard solution writes
`net({x_1,…,x_n}) = ρ( pool({ φ(x_1),…,φ(x_n) }) )`,
where `φ` is a per-element feed-forward encoder applied independently to each element, `pool` is a symmetric reduction (sum / mean / max), and `ρ` post-processes the pooled vector. Symmetry of `pool` gives permutation invariance for free, and any element count is handled. Zaheer et al. proved this is a **universal approximator** of permutation-invariant functions when `pool = sum` and `ρ, φ` are arbitrary continuous functions — i.e. `ρ(sum(φ(·)))` (more precisely `rFF(sum(rFF(·)))`) can approximate any set function. They also noted the encoder may itself be a stack of permutation-*equivariant* layers, an example being `f_i(x) = σ(λ x_i + γ · pool({x_1,…,x_n}))` with learnable scalars `λ, γ`. This decomposes neatly into an **encoder** `φ` (acts independently on each element) and a **decoder** `ρ∘pool` (aggregates), the structure most set models follow.

**Attention (Bahdanau 2015; Vaswani et al. 2017).** Given `n` query vectors `Q ∈ R^{n×d_q}` and `n_v` key–value pairs `K ∈ R^{n_v×d_q}`, `V ∈ R^{n_v×d_v}`, attention is
`Att(Q,K,V;ω) = ω(QKᵀ) V`.
`QKᵀ ∈ R^{n×n_v}` scores how similar each query is to each key; `ω` is an activation (typically a **scaled softmax** `ω(·) = softmax(·/√d)`, the `1/√d` keeping the logits in a stable range); the output is a weighted sum of the values, each value weighted by its key's match to the query. **Multi-head attention** projects `Q,K,V` into `h` lower-dimensional subspaces with learnable matrices `W_j^Q, W_j^K, W_j^V`, runs an attention head in each, concatenates the `h` outputs and mixes them with `W^O`: `Multihead(Q,K,V) = concat(O_1,…,O_h) W^O`, `O_j = Att(QW_j^Q, KW_j^K, VW_j^V; ω_j)`. A typical choice is `d_q^M = d_v^M = d/h`, `d = d_q`. The full Transformer encoder block wraps this with residual connections, **layer normalization** (Ba et al. 2016), a position-wise feed-forward sublayer, **positional encodings**, and dropout.

**Inducing-point / low-rank approximations.** Sparse Gaussian processes (Snelson & Ghahramani 2005) and Nyström methods (Williams & Seeger 2001; Fowlkes et al. 2004) summarize a large set of points through a small set of `m` representative "inducing points", reducing the cost of an otherwise `O(n²)` interaction (a full kernel / Gram matrix over `n` points) to `O(nm)`. This is the standard trick when full pairwise computation over a large set is too expensive but its low-rank structure can be captured by a small bottleneck.

**Motivating limitation of pooling.** Because `φ` processes each element *independently*, all information about inter-element interactions is discarded before pooling. For some problems this makes the mapping unnecessarily hard. The sharp example is **amortized clustering**: learn a map from a point set to its cluster centers. The map must assign each point to a cluster while modeling *explaining-away* — clusters should not compete to explain overlapping subsets — which is why clustering is normally done by iterative refinement (EM). A set-pooling net can only learn to *quantize* space, and crucially that quantization is fixed: it cannot depend on the contents of the particular input set. The reported consequence is **under-fitting** of pooling architectures on such tasks.

## Baselines

- **Set-pooling / Deep Sets** (`ρ(sum(φ(·)))`, Zaheer et al. 2017; Edwards & Storkey 2017): permutation invariant, size-agnostic, provably universal. Gap: independent per-element encoding discards interactions; fixed, content-independent aggregation under-fits interaction-heavy tasks like clustering.
- **Pooling variants** — mean-pooling, max-pooling, and feature-augmented per-element nets (`rFFp-mean`, `rFFp-max`, where each element is concatenated with a pooled summary before further processing): still a fixed symmetric reduction; the aggregation weight on each element cannot adapt to the set's contents.
- **Simple dot-product attention pooling** (a non-parameterized attention readout): a content-dependent readout at aggregation time, but the encoding upstream still embeds each element in isolation.
- **Recurrent set models** (e.g. order-augmented RNN readouts, Vinyals et al. 2016): can in principle attend over a set, but are order-sensitive unless carefully symmetrized, and do not give a clean permutation-invariant guarantee.

Across these baselines the same wall recurs: each element is embedded before any combination happens, and the combination itself is a fixed symmetric reduction — so on interaction-heavy tasks they under-fit, while staying permutation invariant, size-agnostic, and tractable on large sets remains a hard constraint to satisfy simultaneously.

## Evaluation settings

- **Max regression (synthetic):** input a set of reals `x_i ∈ [0,100]`, set size `n ∼ Unif{1,…,10}`, target `max_i x_i`; metric mean absolute error. A pure interaction-light probe: the answer lives in a single element.
- **Counting unique characters (Omniglot):** input a set of `n ∼ Unif{6,…,10}` character images containing `c` distinct characters; predict `c`. Output is a Poisson rate via softplus; trained by Poisson log-likelihood.
- **Amortized clustering / maximum likelihood for mixtures of Gaussians:** input a point set drawn from a `k`-component 2D Gaussian mixture (`n ∼ Unif(100,500)`, also a large-scale `n ∼ Unif(1000,5000)` regime); output the mixture parameters (mixing weights, means, variances). Metrics: average per-point log-likelihood (before and after one EM refinement step), and clustering accuracy by Adjusted Rand Index. Also a CIFAR-100 meta-clustering variant over VGG features.
- **Set anomaly detection (CelebA):** each input set is mostly "normal" images sharing attributes plus one anomaly; identify the odd one out.
- **Point-cloud classification (ModelNet40):** classify a 3D object from a set of `n = 100/1000/5000` `(x,y,z)` points into 40 classes; sets randomly rotated/scaled for generalization.
- **Protocol/optimizer:** Adam, learning rates `10⁻³`–`10⁻⁴` (decayed), batch sizes 32–128. Also a controlled **runtime benchmark** measuring per-block processing time as a function of set size, to compare the scaling of candidate set operations.

## Code framework

The pieces that already exist: PyTorch `nn.Linear`, `softmax`, `LayerNorm`, multi-head projection bookkeeping (split a `d`-dim representation into `h` heads), Adam, and the encoder→pool→decoder set-pooling template. How to fill in the encoder and decoder so the model overcomes the limitations above is open. The slots below are empty.

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class EncoderBlock(nn.Module):
    """An encoder block that maps a set of n elements to a set of n features.
    The internal computation is to be designed."""
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        # TODO
        pass
    def forward(self, X):
        # TODO
        pass

class Pooling(nn.Module):
    """Aggregate a set of n feature vectors into k output vectors.
    The aggregation rule is to be designed."""
    def __init__(self, dim, num_heads, num_outputs, ln=False):
        # TODO
        pass
    def forward(self, Z):
        # TODO
        pass

class SetModel(nn.Module):
    def __init__(self, ...):
        self.encoder = nn.Sequential(...)   # stack of EncoderBlock
        self.decoder = nn.Sequential(...)   # Pooling, then row-wise FF to outputs
    def forward(self, X):
        return self.decoder(self.encoder(X))
```
