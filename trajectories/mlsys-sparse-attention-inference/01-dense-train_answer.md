The whole ladder asks how to spend a quarter of the attention matrix without losing quality, but I cannot say what "losing quality" means until I know what the full matrix buys. The single editable thing is one `SparseAttention` module patched into every attention layer of a frozen 1.5B instruct model; with the density budget removed, that module's job is simply the model's own attention. So before I argue about *how* to be sparse, I run the reference — every query reading every key — and read off the three numbers the sparse rungs will be graded against.

I propose the **dense oracle**: the unmodified full-attention forward, reporting its true density of $1.0$, and the only baseline allowed to exceed the $0.25$ budget. Each query $q_t$ scores against every key it is permitted to see, the scores pass through a softmax, and the output is the softmax-weighted blend of values,
$$\mathrm{out}_t = \sum_{j} \mathrm{softmax}_j\!\left(\frac{q_t \cdot k_j}{\sqrt{D}}\right) v_j,$$
which stacked over all positions is $\mathrm{softmax}(QK^\top/\sqrt{D} + M)\,V$ with $M$ the additive $-\infty$-above-the-diagonal causal bias that forbids reading the future. The scale $1/\sqrt{D}$ is not a free choice: the dot product of two $D$-dimensional vectors has standard deviation $\sqrt{D}$, so dividing by $\sqrt{D}$ returns the logit scale to unity and keeps the softmax in its responsive region — exactly the scaling the model was trained under. The discipline of the oracle is restraint. The instant I touch the scale or insert a mask I am no longer measuring this model's ceiling but some other model's, and the three numbers stop being a valid target; so the oracle reproduces the trained computation and changes nothing.

What makes the implementation the *empty* edit is that the fixed loop has already done everything that makes this the pretrained model's attention. It replicates GQA before calling me — the backbone has 12 query heads and 2 KV heads, but I receive 12 heads on both $Q$ and $K/V$ and never touch the grouping. It applies RoPE to $q$ and $k$, so the rotary position information is already baked into the vectors I score and I must not re-rotate. It passes `is_causal=True` and the default `scale`. So head layout, positional encoding, causal direction, and trained scale are all settled upstream, and my edit is to hand $q, k, v$ straight to PyTorch's fused scaled-dot-product attention with `attn_mask=None` and `is_causal` forwarded — the kernel builds the lower-triangular $-\infty$ bias internally. Using the fused kernel rather than a hand-written $QK^\top \to \mathrm{softmax} \to V$ is not a shortcut that changes the answer; it is the memory-sane way to realize the identical computation and keeps the oracle from gratuitously OOM-ing on the very $N^2$ matrix that motivates the rest of the work. The one piece of bookkeeping the contract demands, `self.last_density`, is honest and constant: a dense causal forward keeps every admissible $(q,k)$ pair, so the kept fraction over the $N(N{+}1)/2$ causal denominator is exactly $1.0$, and I set it literally rather than counting a phantom mask that would only invite a rounding disagreement. The harness recognizes this rung through `ALLOW_DENSE_FLAG=1`, which forwards `--allow-dense` so `enforce_budget` skips the $0.25 + 0.02$ ceiling for the oracle alone — and that asymmetry is what makes the oracle a *reference* rather than a competitor: it is not trying to satisfy the constraint, it is defining the quality the constrained rungs are measured against.

The reason this rung runs first is the cost it deliberately pays. The score matrix $QK^\top$ is $(N \times N)$ per head per layer; at the 8K context this benchmark evaluates that is tens of millions of entries per head, across 12 heads and 24 layers, and the binding constraint is memory traffic — every one of those $N^2$ logits is written, read back for the softmax, and the full key/value tensors streamed for the weighted sum. That is the $O(N^2)$ quadratic that pins long-context inference and the reason dense can be the oracle but never the deployed module. And the numbers it fixes split the way the ladder needs them to: on `niah_8k`, where a single fact is buried in a long haystack, full attention routes any query to the needle's one key in a single hop, so I expect near-perfect retrieval — the cleanest discriminator on the ladder, because there is no partial credit for being *near* the needle. On `longbench_qasper` and `longbench_multifieldqa_en`, where the metric is F1 and the evidence is distributed across the document, the 1.5B model's own competence caps the score well below $1.0$, so the oracle's F1 there is the *realistic* ceiling, not a perfect one, and these degrade gracefully because a pattern covering *some* relevant spans can still answer partially. That split — NIAH is a position-coverage cliff, QA is distributed-evidence — is the diagnosis I carry into the sparse rungs: the gap between the oracle's numbers and a sparse rung's tells me whether the missing quality is a coverage problem to fix with anchors and windows, or a routing problem to fix by making the pattern depend on the query.

```python
# EDITABLE region of custom_sparse_attn.py — rung 1: dense oracle (density 1.0)
class SparseAttention(nn.Module):
    """Dense attention oracle. Reports true full-attention density.

    The point of this baseline is to give an upper-bound quality reference;
    it is not meant to satisfy a real sparsity constraint.
    """

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)
        # Use PyTorch's fused SDPA for efficient dense attention.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0,
            is_causal=is_causal, scale=scale,
        )
        self.last_density = 1.0
        return out
```
