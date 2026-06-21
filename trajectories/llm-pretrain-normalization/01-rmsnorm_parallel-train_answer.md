The default block is the standard Pre-LN sandwich: a `LayerNorm` with learned gain in front of each sublayer, $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$ then $x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x))$. The single largest structural inefficiency it carries is that the two sublayers are strictly sequential, each fronted by its own full normalization — four reductions per block per token (a mean and a variance for each of attention's `ln_1` and the MLP's `ln_2`), and the MLP cannot begin until attention's residual has been added and re-normalized. That serial dependency and that duplicated normalization are costs the architecture pays on every one of the 24 blocks, every token, every one of the 12,030 iterations, and neither is obviously buying final-loss quality. So the first thing I change is the most aggressive structural simplification the edit surface allows, and I bundle it with the cheapest defensible normalization rule.

I propose two changes at once, the RMSNorm + parallel block. The first is the normalization rule. The default `LayerNorm`, within a single token's feature vector $a \in \mathbb{R}^{1024}$, computes the mean $\mu = \frac{1}{n}\sum_i a_i$, subtracts it, divides by the standard deviation $\sigma = \sqrt{\frac{1}{n}\sum_i (a_i - \mu)^2}$, and applies a per-channel gain $\gamma$ and bias $\beta$. That is two operations welded together, and they buy two different invariances. Subtracting the mean buys *re-centering invariance*: shift every coordinate of $a$ by a constant and the output is unchanged, because $\mu$ absorbs the shift and the centered vector does not move. Dividing by $\sigma$ buys *re-scaling invariance*: multiply $a$ by a positive $\alpha$ and both $\mu$ and $\sigma$ scale by $\alpha$, so $(a-\mu)/\sigma$ is untouched. The question that decides whether I can drop the mean is which invariance keeps training well-conditioned, and mechanically the answer is the re-scaling one. Stabilizing a deep stack is about controlling the *spread* of activations and gradients so they neither blow up nor vanish across 24 residual blocks; subtracting the mean recenters the cloud but leaves the spread untouched, since $\mathrm{var}(a - \mu) = \mathrm{var}(a)$. The operation that actually pins the magnitude the next block and the backward pass see is the division by the scale. So I drop the mean and replace $\sigma$ — which is *defined* through $\mu$ — with a measure of spread around the origin that references no mean at all, the root mean square:

$$\bar a_i = \frac{a_i}{\mathrm{RMS}(a)}\,\gamma_i, \qquad \mathrm{RMS}(a) = \sqrt{\tfrac{1}{n}\textstyle\sum_i a_i^2}.$$

When $a$ already has zero mean this is *identical* to the default layer ($\sigma$ collapses to $\mathrm{RMS}$), so this is not a wild departure but the same layer with re-centering switched off. Re-scaling invariance survives because $\mathrm{RMS}(\alpha a) = \alpha\,\mathrm{RMS}(a)$ is linear, exactly the property the argument needs; only re-centering is discarded, the invariance I argued does not matter. The bias goes with it: the config sets `bias=False`, so the default fill carries no $\beta$ anyway, and that is the right call — a per-channel shift on a normalization layer exists only to restore a location after recentering, and RMSNorm does no recentering, so there is nothing to restore. My RMSNorm keeps only the gain $\gamma$, initialized to ones, and ignores the `bias` argument it is contractually handed. What makes this safe rather than merely cheap is the backward pass: because $\mathrm{RMS}$ is quadratic in $a$ and $a = Wx$, the weight enters both numerator and denominator, and $\partial L/\partial W$ comes out invariant to scaling the input and *inversely* proportional to scaling the weights — a layer whose weights grow large automatically receives smaller gradients, an implicit per-layer learning-rate adaptation that damps further growth with no schedule. None of that self-regulation came from the mean I am throwing away; all of it survives in the re-scaling structure I keep. RMSNorm is therefore strictly the load-bearing part of `LayerNorm`, at one reduction instead of two. This is the rule every later rung inherits, which is one reason to price it on the first rung where I can read its number.

The second and larger change is the wiring. In the default block the residual is updated twice in series — the MLP reads the *post-attention* residual, depth 2 per block, 48 sequential sublayers across the model. The parallel block instead reads *one* normalized copy of the pre-block residual and lets both sublayers operate on it independently:

$$h = \mathrm{LN}(x), \qquad x \leftarrow x + \mathrm{Attn}(h) + \mathrm{MLP}(h).$$

The arithmetic payoff is direct: one shared norm per block instead of two, on top of the RMSNorm saving, and the two matmul-heavy sublayers compute from the same input without waiting on each other, shortening the critical path. I want to be precise that this is *this surface's* parallel block, not the generic recipe — it collapses to a single shared RMSNorm feeding both branches and a single summed residual write, but it does *not* fuse the attention and MLP input projections into one matmul the way GPT-J's implementation does for its speed win, because the fixed `CausalSelfAttention` and `MLP` modules are outside the edit surface. So the *quality* consequences of going parallel are fully in play here while only part of the *speed* win — the shared norm and the shortened dependency, not the fused projection — is realized.

What going parallel costs, representationally, is exactly why I expect this to be the *weakest* rung. In the sequential block the MLP sees the residual *after* attention has written to it; it can condition on what attention just produced this layer. In the parallel block both sublayers see the same stale $h$ and cannot react to each other within the block, so whatever intra-block cross-talk the sequential ordering allowed is deferred to the next block. The prior lineage is explicit that this shows up as a *small quality loss at small scale* that vanishes only at very large scale — PaLM (Chowdhery et al. 2022) reports no degradation at 62B; the loss is visible below that — and a 355M model is firmly in the regime where the parallel approximation should bite. There is a second, subtler risk from *combining* the two changes: with one shared RMSNorm and a summed residual, the two branch outputs add into the stream with no intervening normalization, and RMSNorm, unlike the LayerNorm I am removing, does not recenter, so any drift in the mean of $h$ is no longer cleaned up before it feeds two branches at once. In a pre-norm stream whose variance already grows with depth, summing two un-normalized branch outputs per block could let that growth run a little hotter. The one reassurance is the init contract: the substrate scales every `c_proj` weight by $1/\sqrt{2\cdot n_{\text{layer}}}$ precisely because the block writes to the residual *twice*, and the parallel block also writes twice (it sums two branch outputs), so the factor still matches the number of writes and the wiring change does not silently break the variance calibration the substrate depends on. The risk is the representational one, not an init blow-up.

So this rung is the ladder's floor by design: maximally simplified structure, on a model small enough that the simplification should cost something, in exchange for the fastest wall-clock — fewer norms and a shorter critical path. I touch nothing in `CONFIG_OVERRIDES`; the learning rate and schedule were chosen for this model and I have no diagnostic yet that would justify moving them. My falsifiable expectation, against the only anchor I have, is that it trains stably to completion, is the cheapest per iteration, and lands at a validation loss that is *competitive but not the best* — at or above whatever a plain sequential RMSNorm block achieves — because the parallel approximation is paying a small-scale quality tax that the sequential block does not. Running it first prices that tax directly for everything that follows.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — rmsnorm_parallel

# ── Normalization (lines 22–31) ──────────────────────────────────────────────
class LayerNorm(nn.Module):
    """RMSNorm — Root Mean Square Layer Normalization."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-5

    def forward(self, input):
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight


# ── Transformer Block (lines 88–100) ─────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # Parallel: single norm, attention and MLP operate in parallel
        h = self.ln(x)
        x = x + self.attn(h) + self.mlp(h)
        return x
```
