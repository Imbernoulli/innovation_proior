The slot I have to fill takes a set of $V = 48$ variable tokens at each spatial location — one $D$-dimensional vector per meteorological field — and must return a single token, $x: [B, V, L, D] \to [B, L, D]$. There is a whole family of ways to collapse that set, from a parameter-free average up to a content-dependent cross-attention, and before I reach for any learnable machinery I want a floor: the simplest reduction that is *correct* for the contract, adds no parameters, and that every fancier aggregator must beat to justify its cost. If the average already gets me most of the way, the elaborate aggregators are not earning their parameters; if it falls down, the *manner* in which it falls down tells me exactly what the next rung must fix. So this first rung is the transparent baseline by design, and the only real work is deriving *which* parameter-free reduction is the right floor.

The contract narrows the choice almost completely. At one location I have $V$ vectors in $\mathbb{R}^D$ and I want one vector in $\mathbb{R}^D$, and three properties are non-negotiable. First, the variables have no canonical order — there is no privileged arrangement of temperature, geopotential, humidity, the land-sea mask the aggregation should exploit — so the reduction must be invariant to permuting the $V$ tokens. Second, $V$ is read from the input shape at runtime and the module must accept any $V$ (different reanalysis products and climate archives carry different variable sets — the whole reason variable tokenization exists), so $V$ cannot be baked into a weight matrix. Third, the output feeds a fixed downstream stack — a position embedding, then an 8-block ViT initialized from pretrained ClimaX weights, tuned to features on a particular scale — so if the reduction's output magnitude drifts with how many variables I feed, set size masquerades as content and knocks the pretrained backbone off its operating point.

I propose **mean pooling**: $g = \frac{1}{V}\sum_{v} x_v$, the uniform average over the variable axis. The reasoning that lands there is forced rather than chosen. Permutation invariance is the sharp constraint: a reduction unchanged under every reordering of the tokens can only see them as an unordered multiset, so it must be built from an order-independent, associative-commutative combiner — elementwise sum, max, or things built from them. This immediately rules out concatenation (order- *and* size-dependent, producing width $V \cdot D$ that breaks the $[B, L, D]$ contract outright) and any recurrent reader (which invents an artificial order over an orderless set). It also rules out picking a single designated summary token: the per-variable tokenizer produces just $V$ variable tokens, none privileged, so there is no class-token slot to read off. I want a reduction that uses *all* $V$ tokens and presupposes no special slot.

Among the symmetric reductions, the bare sum $g = \sum_v x_v$ is the simplest, but the third constraint kills it. Treat each token coordinate as a draw with per-coordinate mean $\mu$ and variance $\sigma^2$; the sum has expectation $V\mu$, so whenever $\mu \neq 0$ the pooled output's location moves *linearly* with the variable count, and even the centered part has standard deviation growing like $\sqrt{V}\,\sigma$. The pretrained ClimaX backbone never saw inputs at the summed scale — it expects features on the scale of a single token, and the position embedding added next lives on that single-token scale — so the bare sum would hand it features inflated by a factor of order $V$ and drown the position embedding. The fix is forced by the same algebra: scaling the sum as $\sum_v x_v / V^\alpha$ gives expectation $V^{1-\alpha}\mu$, and the only exponent making the location independent of the count, for any nonzero $\mu$, is $\alpha = 1$. That gives the mean, whose expected location is $\mu$ regardless of $V$ and whose per-coordinate variance is $\sigma^2/V$ — exactly the single-token scale the backbone and position embedding expect. The reduction becomes invisible to the operating point, which is decisive in a fine-tune-from-pretrained setting.

There is a cleaner way to see why the *uniform* mean is the right combiner and not merely a convenient one. Any weighted average $\sum_v w_v x_v$ with $\sum_v w_v = 1$ is scale-stable; among all such weightings, with no information distinguishing one variable from another at this stage, the only fixed permutation-symmetric choice is $w_v = 1/V$ — the maximum-entropy, flattest weighting given no reason to prefer any variable. Anything non-uniform asserts a per-variable preference I have not yet earned. This also fixes how the floor relates to the rungs above it: the mean is the degenerate limit of the whole family. The learned weighted sum collapses to it when equal logits give $w_v = 1/V$; the cross-attention collapses to it when the attention weights are uniform and the projections are identity. The mean is not a third unrelated option — it is the honest floor every learned aggregator must clear.

I weighed the one symmetric reduction that genuinely beats the mean in some settings: elementwise max, $g_d = \max_v x_{v,d}$, which keeps the strongest activation per dimension. That is right when the signal at a location is concentrated in *one* extreme variable the mean would dilute — but it is also its failure mode here. At a weather grid cell the informative signal is plausibly spread across many variables at once (geopotential, temperature, humidity, wind all carry real, comparable information about the local state), not concentrated in a lone outlier. Max would discard everything but the per-dimension winner and is non-smooth — per coordinate the gradient flows to exactly one variable, so most of the set gets no learning signal. The mean uses every variable equally and is smooth in all of them, the safer floor for 48 comparably-informative fields, with max kept on the shelf for extreme-driven sets.

What makes the implementation almost nothing is that the task strips the general machinery down. In the fully general set-pooling problem sets are ragged, so a batch pads to a common length and the honest reduction must *mask*: $g = (\sum_v m_v x_v)/\mathrm{clamp}(\sum_v m_v, \min=\epsilon)$, with a denominator floor so an empty set gives a defined $0$. But this task is not ragged — every spatial location has the full $V = 48$ present, the tokenizer produces a token for every variable at every patch, the mask would be all ones, the clamp never fires. With an all-ones mask the masked mean collapses exactly to the plain average: numerator is the full sum, denominator is $V$ itself. So the masking apparatus, the running-count denominator, the empty-set guard, and the $\sqrt{V}$ variance-stabilizing sibling all drop, and the implementable form is just `x.mean(dim=1)` over the variable axis. The `num_heads` and `num_vars` arguments are part of the fixed contract so I store them, but the mean uses neither: there is no attention, and the average is size-agnostic. There are no learnable parameters at all — that is the point of the floor.

I expect this to produce a genuinely working forecaster across all three targets, and then to be beaten in a specific way: a fixed learned per-variable weighting should pull ahead by letting informative fields count more than near-inert ones (the static land-sea mask, orography), recovering whatever signal uniform weighting smears out. That is the falsifiable handoff to the next rung — if a learned per-variable softmax cannot achieve a lower latitude-weighted RMSE than this mean on each target, then the per-variable contributions really are uniform and the parameters are wasted.

```python
# EDITABLE region of custom_forecast.py (lines 310-351) — step 1: mean pooling
class VariableAggregator(nn.Module):
    """Mean pooling variable aggregation.

    Simply averages all V variable tokens at each spatial location.
    No additional learnable parameters.

    Args:
        embed_dim (int): Embedding dimension D.
        num_heads (int): Number of attention heads (unused).
        num_vars (int): Number of input variables V (unused).
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars

    def forward(self, x):
        """
        Args:
            x: [B, V, L, D] — per-variable patch embeddings.
        Returns:
            [B, L, D] — aggregated representation.
        """
        # Average across variable dimension
        out = x.mean(dim=1)  # B, L, D
        return out
```
