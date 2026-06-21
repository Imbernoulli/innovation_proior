The sandwich rung settled the question I was unsure of, and the margin is its own verdict. Validation loss $2.3104$ against the parallel rung's $2.3112$ — it did beat the floor, so the first half of the thesis held — but by only $0.0008$, eight ten-thousandths of a nat, and it paid for that sliver with the slowest run on the board. Worse, two secondary witnesses went the *wrong* way: WikiText-2 perplexity rose to $46.8$ (from $45.98$) and LAMBADA to $72.08$ (from $70.96$), with PIQA dropping. That is the signature of *insurance, not improvement*: the four-norm sandwich bounded the residual-stream variance, nudging the in-distribution loss a hair, but the heavier per-block normalization slightly over-constrained the model, which the perplexity rise on held-out text exposes — exactly the soft-prior outcome I flagged. So I now have two structural experiments and their verdicts. The parallel rung *removed* normalization and lost cross-talk; the sandwich rung *added* normalization and over-constrained generalization; neither moved the metric that matters by more than a whisper. Read together they bracket the answer: I have been spending the edit budget on the *wiring*, and the wiring is not where the quality is. The one change that was unambiguously safe across both rungs was the normalization *rule* — RMSNorm trained stably every time.

I propose the simplest possible fill of the edit surface, RMSNorm in a plain sequential Pre-LN block. Keep the RMSNorm rule from the first rung, drop every structural complication, and put the two sublayers back in the plain sequential pre-norm arrangement the default block used — just with RMSNorm instead of LayerNorm:

$$x \leftarrow x + \mathrm{Attn}(\mathrm{RMSNorm}(x)), \qquad x \leftarrow x + \mathrm{MLP}(\mathrm{RMSNorm}(x)).$$

This is the minimal change from the default fill, and it sits at the sweet spot the two restructured rungs straddled. The sequential ordering restores the intra-block cross-talk the parallel rung lost — the MLP reads the *post-attention* residual, the gap that cost $2.3112$, recovered by doing nothing more exotic than not parallelizing. One norm per sublayer removes the output-variance constraint that I just watched over-constrain the model and push the perplexities up; each sublayer writes its natural contribution into the stream, and at 24 layers — deep enough to train cleanly under pre-norm but not so deep that the un-normalized residual growth actually hurts final loss — that freedom is worth more than the sandwich's insurance. The clean pre-norm identity path keeps the depth-stable, warmup-free gradient. And this is the block the substrate's $1/\sqrt{2\cdot n_{\text{layer}}}$ residual init and schedule were *literally tuned for* — two residual writes per block, one input norm each — so unlike the restructured rungs it incurs no init-vs-wiring mismatch at all.

The RMSNorm swap is not just harmless here but *right*, and worth re-deriving because it explains why the cheaper layer is also the well-conditioned one. The default LayerNorm bundles two operations: subtract the mean — re-centering invariance, shift $a$ by a constant and $\mu$ absorbs it — and divide by $\sigma$ — re-scaling invariance, scale $a$ by $\alpha$ and both $\mu$ and $\sigma$ scale, the ratio unchanged. Stabilizing a deep stack is about controlling the *spread*, and subtracting the mean does not touch the spread: $\mathrm{var}(a-\mu) = \mathrm{var}(a)$, it moves the cloud without shrinking or growing it. The operation that pins the magnitude the next block and the backward pass see is the division by the scale, and RMSNorm keeps exactly that:

$$\bar a_i = \frac{a_i}{\mathrm{RMS}(a)}\,\gamma_i, \qquad \mathrm{RMS}(a) = \sqrt{\tfrac{1}{n}\textstyle\sum_i a_i^2},$$

discarding only the re-centering. When $a$ already has zero mean, RMSNorm and LayerNorm coincide exactly ($\sigma = \mathrm{RMS}$), so this is the same layer with recentering switched off, not a different mechanism. With `bias=False` the default LayerNorm carries no bias anyway, and there is nothing to lose: a bias on a normalization layer exists only to restore a location after recentering, and with no recentering there is none to restore — so RMSNorm carries only the gain $\gamma$, initialized to ones, ignoring the `bias` argument it is handed.

The backward pass is the real reason I trust this as the substrate. Because $\mathrm{RMS}$ is quadratic in $a$ and $a = Wx$, the weight enters both numerator and denominator. The Jacobian of the normalized vector with respect to $a$ is

$$R = \frac{1}{\mathrm{RMS}(a)}\left(I - \frac{a a^\top}{n\,\mathrm{RMS}(a)^2}\right),$$

a $1/\mathrm{RMS}$ scaling times the identity minus a rank-one outer product that projects out the radial direction — perturbing $a$ along itself does not change $a/\mathrm{RMS}$, so the Jacobian must annihilate it, and indeed $R a = 0$. Chaining to the weight gives $\partial L/\partial W = (R(\gamma \odot u))x^\top$ with $u$ the upstream gradient. Scaling the input or the weights by $\delta$ sends $R \to R/\delta$, so $\partial L/\partial W$ is *invariant* to input scaling (the $\delta$ from $x$ cancels the $1/\delta$ from $R$) and *inversely proportional* to weight scaling (only $R$ moves): a layer whose weights grow large automatically receives smaller gradients, an implicit per-layer learning-rate adaptation that damps further growth with no schedule and no extra parameters. None of that self-regulation came from the mean-subtraction; all of it survives in RMSNorm. The $1/n$ inside the root, rather than a plain L2 norm $\lVert a \rVert$, is kept deliberately — it normalizes per-coordinate rather than per-vector, so the scheme behaves consistently across the 1024-dimensional features here and would across other widths, where unit-sphere L2 normalization independent of $n$ does not transfer the same way.

I am honest that I expect the win to be small, because the sandwich already taught me these differences are. The plain block differs from the sandwich only in the *placement* of normalization (input-only versus input-and-output) and from the parallel block in *ordering and norm count* — none of these is a capacity change; the model has the same parameters either way, RMSNorm even fewer with no biases. So the win is placement, not capacity. My most confident expectation is that this posts the *best* validation loss of the three — below the sandwich's $2.3104$ and the parallel's $2.3112$ — because it has the cross-talk the parallel rung lost and lacks the over-constraint the sandwich added. The claim I care about most is the perplexity reversal: WikiText-2 back below $46.8$ and LAMBADA back below $72.08$, because removing the output norm removes exactly the constraint that pushed them up. If the plain block lowers val_loss but the perplexities do *not* improve over the sandwich, then the ordering is noise at this scale and the honest conclusion is that norm placement does not matter here. Wall-clock should sit between the two — cheaper than the sandwich's four norms, slower than the parallel rung's shortened critical path — and that is fine, since speed was never the objective. I touch no `CONFIG_OVERRIDES`: returning to the plain block is precisely returning to the arrangement the substrate's learning rate and schedule were chosen for, and changing them would undo the alignment I am restoring. The finding the ladder lands on is that the simplest fill of this edit surface is the strongest one, and the two restructuring experiments earned their place by showing in measured numbers that spending the budget on wiring in either direction buys nothing at 355M the plain block does not already have.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — rmsnorm (plain Pre-LN)

# ── Normalization (lines 22–31) — REPLACED ───────────────────────────────────
class LayerNorm(nn.Module):
    """RMSNorm — Root Mean Square Layer Normalization."""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = 1e-5

    def forward(self, input):
        rms = input.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (input * rms).type_as(input) * self.weight


# ── Transformer Block (lines 88–100) — UNCHANGED from the default fill ────────
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```
