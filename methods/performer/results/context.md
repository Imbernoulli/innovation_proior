# Context

## Research question

Self-attention is the engine of the modern sequence model, but it carries a cost that scales with the square of the sequence length. For an input of `L` tokens with hidden dimension `d`, the attention operation forms an `L × L` matrix of pairwise interaction scores, normalizes it, and uses it to mix value vectors. Building and storing that matrix costs `O(L²d)` time and `O(L² + Ld)` memory. For `L` in the hundreds this is fine; for `L` in the thousands to tens of thousands — protein chains of length 8192, images flattened to 12288 pixels, book-length text — the quadratic term dominates and the model simply runs out of memory or time.

The question is whether attention can be made to scale **linearly** in `L`, in both time and memory, **without** changing what it computes — that is, while still (provably) approximating the same full, dense softmax attention, and without assuming the attention pattern is sparse or low-rank. A solution would have to: (a) avoid ever materializing the `L × L` score matrix; (b) come with a guarantee that the cheap computation is close to the exact one; (c) be numerically stable enough to train end-to-end; and (d) drop into an existing architecture without bespoke kernels or retraining from scratch.

## Background

**Dot-product attention.** Given query, key, and value matrices `Q, K, V ∈ R^{L×d}` (rows are per-token vectors), bidirectional attention computes

    A = exp(QKᵀ / √d),   D = diag(A 1_L),   Att(Q,K,V) = D⁻¹ A V,

where `exp` is elementwise and `1_L` is the all-ones vector. Row `i` of the output is a convex combination of the value rows, with weights `A(i,j)/Σ_j A(i,j)` — a normalized similarity between query `i` and key `j`. Because every weight `A(i,j)` is an exponential of a real number it is strictly positive, and the row-normalization `D⁻¹` turns each row into a probability distribution over tokens. The unidirectional (causal) variant masks `A` to its lower triangle before normalizing, so token `i` only attends to tokens `≤ i`; this is what autoregressive generation uses. The whole cost lives in forming and storing `A`: `O(L²d)` and `O(L²+Ld)`.

**The function being approximated is a kernel.** The score `A(i,j) = exp(q_iᵀk_j / √d)` is a positive-definite kernel evaluated at the query/key pair. Up to the per-vector rescaling by `d^{-1/4}` (which can be folded into `Q` and `K`), it is the **softmax kernel** `SM(x,y) = exp(xᵀy)`. And `exp(xᵀy)` is, after pulling out the norms, a Gaussian (RBF) kernel: `exp(xᵀy) = exp(‖x‖²/2)·exp(-‖x-y‖²/2)·exp(‖y‖²/2)`. So everything known about approximating kernels applies here.

**Random features for kernels.** Rahimi & Recht (2007) showed that a shift-invariant kernel `K(x-y)` is the Fourier transform of a probability density, hence an expectation `K(x,y) = E_{ω∼p}[ζ_ω(x)ζ_ω(y)]` over random frequencies `ω`. Drawing `m` frequencies and forming a randomized feature map `φ(x) ∈ R^m` gives `K(x,y) ≈ φ(x)ᵀφ(y)`, an unbiased estimate whose variance falls as `1/m`. For the Gaussian kernel the classical map uses trigonometric functions: with `ω ∼ N(0,I)`, `φ(x) = (1/√m)(sin(ω_1ᵀx), cos(ω_1ᵀx), …)` because `E[cos(ωᵀ(x-y))]` reproduces the Gaussian kernel. This is the template: pick `ω_i`, pick scalar functions `f_l`, and set `φ(x) = (h(x)/√m)(f_1(ω_1ᵀx),…,f_l(ω_mᵀx))`.

**Orthogonal random features.** Yu et al. (2016) observed that if the sampling distribution is isotropic, one can entangle the `m` random vectors `ω_i` to be **exactly orthogonal** (e.g. by Gram-Schmidt on a Gaussian block) while leaving each marginal distribution unchanged. This keeps the estimator unbiased but reduces its variance versus independent sampling. Prior guarantees for this variance reduction were asymptotic — they held only for large enough dimension `d`.

**Diagnostic facts about cheap-attention attempts.** Two observations about existing approximations set up the problem. First, methods that swap softmax for an arbitrary feature map `φ(q)ᵀφ(k)` (e.g. `φ = elu(·)+1`) to get linear cost are observed to train unstably — exploding gradients and `NaN` losses — because nothing keeps the implied attention scores well-behaved. Second, a kernel estimator that can take **negative** values is dangerous here specifically because the attention rows are convex combinations: a negative or near-zero estimate of a small score, fed through the normalizer `D⁻¹`, can produce negative or blown-up denominators. The many genuinely small entries of `A` (low-relevance token pairs) are exactly where a high-relative-variance estimator does the most damage.

## Baselines

**Sparse / local attention** (Sparse Transformer, Child et al. 2019; Image Transformer, Parmar et al. 2018; Longformer; Routing Transformer with `k`-means clustering). Idea: restrict each query to attend to a fixed or learned subset of keys (a local window, strided pattern, or cluster), so the score matrix has `O(L)` nonzeros. Cost drops toward `O(L√L)` or `O(L)`. Gaps: these do not approximate full softmax — they replace it with a structurally different, sparser mechanism; the sparsity pattern is a prior that may not match the data; there is no rigorous bound on lost representation power; and the patterns often require hand-written CUDA or TVM kernels to realize the speedup.

**Reformer** (Kitaev et al. 2020). Idea: use locality-sensitive hashing to bucket queries and keys with high dot product into the same bin, computing attention only within bins → `O(L log L)`. Also uses reversible layers to save activation memory. Gaps: requires tying queries and keys (`Q=K`); it is an approximation with no unbiasedness guarantee; it bakes in the assumption that only a few large scores matter (sparsity prior).

**Linformer** (Wang et al. 2020). Idea: project the `L` keys and values down to `k ≪ L` rows with learned linear maps, assuming the attention matrix is approximately low-rank → `O(Lk)`. Gaps: it is a **biased** estimator with large mean-squared error; it is defined only for the non-causal (bidirectional) case because the projection mixes across all positions; and it assumes low rank, which need not hold.

**Linear Transformer / "Transformers are RNNs"** (Katharopoulos et al. 2020). Idea: replace `exp(qᵀk)` outright with `φ(q)ᵀφ(k)` for a fixed positive feature map `φ(x)=elu(x)+1`, then exploit that `Σ_j φ(q_i)ᵀφ(k_j) v_jᵀ = φ(q_i)ᵀ(Σ_j φ(k_j) v_jᵀ)` — the key-value sum is shared across queries, giving `O(Lrd)` cost and an `O(1)`-state recurrence in the causal case. Gap: `φ` is chosen by hand and does **not** approximate the softmax kernel — it changes the attention function — and the resulting models are observed to be numerically unstable, producing exploding gradients.

## Evaluation settings

The natural yardsticks are sequence-modeling tasks where `L` is large enough to expose the quadratic cost, measured by next-token accuracy and perplexity (and bits-per-dim/char, loss divided by `ln 2`):

- **Protein language modeling** on TrEMBL (UniProt), single sequences clipped to `L=1024`, and a long-sequence variant concatenating proteins to `L=8192`; both a held-out IID test split and an out-of-distribution split of held-out Pfam families. Masked (15%) accuracy/perplexity for bidirectional models, next-token for unidirectional.
- **Image generation** on ImageNet64 as a flattened pixel sequence, `L=12288`, bits-per-dim.
- **Long-document language modeling** on PG-19 (Project Gutenberg books), with a 32k-token SentencePiece vocabulary, as a long-range bidirectional pretraining benchmark.
- **Long Range Arena**: ListOps, byte-level text classification and retrieval, pixel-sequence image classification, and the Pathfinder spatial task, all stressing long context.
- **Microbenchmarks**: forward/backward wall-clock time and peak memory as a function of `L`, against the exact-attention baseline and against an "identity" attention that just returns `V` (the maximum achievable speedup).

The comparison points are the exact softmax Transformer and the efficient-attention baselines above (Reformer, Linformer, Linear Transformer), under matched model sizes and training hyperparameters.

## Code framework

The pieces that already exist: a standard Transformer training stack (data pipeline, token embedding, an Adam optimizer, cross-entropy loss, a residual+layernorm block harness), and the kernel-approximation primitives — sampling random projections and forming randomized feature maps in the style of random Fourier features. What does **not** yet exist is the attention computation itself; that is the slot to fill. A pre-method scaffold:

```python
import torch, torch.nn as nn

def random_projection(num_features, dim):
    # draw an m x d projection matrix of random directions
    pass  # TODO: which distribution? independent or entangled rows?

def feature_map(x, projection):
    # map a token vector x in R^d to a feature vector in R^r
    # so that <feature_map(q), feature_map(k)> approximates the
    # similarity score the attention mechanism needs
    pass  # TODO: which scalar nonlinearity / scaling makes this faithful and stable?

def attention(Q, K, V, projection, causal):
    # Q, K, V : [..., L, d]
    # must produce the same thing as the normalized score-weighted
    # combination of V, but WITHOUT building an L x L matrix
    pass  # TODO: rearrange the computation to avoid the L x L object

class FastAttention(nn.Module):
    def __init__(self, dim_heads, num_features, causal=False):
        super().__init__()
        # TODO: store / register the random projection; redraw policy
        pass

    def forward(self, q, k, v):
        # TODO: features for q and k, then the rearranged combination
        pass
```

The final method fills `random_projection`, `feature_map`, and `attention` (and `FastAttention`) so that `attention` provably matches softmax attention while costing `O(L·r·d)`.
