The parallel rung told me what I wanted with one number: validation loss $2.3112$, against WikiText-2 perplexity $45.98$ and LAMBADA $70.96$, in the fastest wall-clock of anything I will run. It trained stably to completion, which vindicates the RMSNorm half of the bundle — dropping the mean-subtraction and the bias cost nothing catastrophic, so RMSNorm stays the substrate norm from here. But $2.3112$ is the *worst* validation loss on the board, exactly the small-scale parallel tax I expected: the two sublayers could not see each other within a block, and at 355M that lost cross-talk shows up as loss left on the table. The diagnosis is clean and it is *not* about the normalization rule. The parallel wiring also made vivid the structural problem of pure pre-norm — it summed two *un-normalized* branch outputs straight into the residual, with a non-recentering norm in front and nothing behind. So I have spare quality budget, and the lever is the wiring: stop simplifying the block and instead spend structure on *more* normalization, placed where pre-norm leaves a gap.

Here is that gap, stated precisely. In pure pre-norm, $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$ normalizes the *input* to attention, but whatever magnitude attention's projection produces is written into the residual stream raw. Across 24 blocks those raw writes accumulate and the residual-stream variance grows monotonically with depth — the documented reason pre-norm lands at a slightly higher final loss than a successfully-trained post-norm model, and the thing the parallel rung exaggerated. Pure post-norm, $x \leftarrow \mathrm{LN}(x + \mathrm{Attn}(x))$, is the opposite extreme: it normalizes *after* the add, which controls the variance growth beautifully but puts the norm back on the main residual path and breaks the clean identity shortcut that gave pre-norm its depth-stable, warmup-free gradients. I will not give up the pre-norm gradient behavior — the stack is 24 layers deep and relies on the identity path staying clean — but I want post-norm's variance control. The two desiderata are not in conflict if I place *two* norms per sublayer instead of one.

I propose the sandwich-norm block (CogView-style, Ding et al. 2021). Keep the pre-norm on each sublayer's input for the depth-stable gradient, and add a second norm on the sublayer's *output*, applied before it is added to the residual:

$$x \leftarrow x + \mathrm{LN}_{\text{post}}\!\big(\mathrm{Attn}(\mathrm{LN}_{\text{pre}}(x))\big), \qquad x \leftarrow x + \mathrm{LN}_{\text{post2}}\!\big(\mathrm{MLP}(\mathrm{LN}_{\text{pre2}}(x))\big).$$

Read what each norm now does. $\mathrm{LN}_{\text{pre}}$ controls the distribution the sublayer *sees*, which is what makes the gradient bounded and depth-independent — the pre-norm property I refuse to lose. $\mathrm{LN}_{\text{post}}$ controls the magnitude of what the sublayer *contributes* to the residual, so each block writes a quantity of controlled scale into the stream regardless of how large the sublayer's internal projection grew during training. That directly attacks the monotonic variance growth: the stream still adds a fresh contribution per block, but each contribution is normalized, so the growth is bounded the way post-norm bounds it — without ever placing a norm on the main path between blocks. The identity shortcut $x \leftarrow x + (\text{normalized branch})$ is intact and the gradient still flows straight down the residual without passing through a normalization. This is exactly the raw-branch-sum failure the parallel rung exaggerated, turned around by normalizing the branch output before the add.

I want to be careful this is the block the surface actually builds and not a different animal that shares the name. "Post-LN" in the loose sense means $\mathrm{LN}(x + \mathrm{Attn}(x))$, the norm on the main path *after* the residual add — that is **not** what this rung does. The sandwich normalizes the branch output *before* the add and keeps the residual path itself norm-free between blocks: two norms per sublayer, one in and one out, with the residual add threaded *between* them. So the block is four norms — `ln_pre1`/`ln_post1` around attention, `ln_pre2`/`ln_post2` around the MLP — each an RMSNorm carried over verbatim from the first rung (gain only, no bias, one reduction). I am spending normalization, not saving it: the exact opposite trade from the parallel rung, on the bet that controlled branch contributions are worth more at 355M than the half-norm speed saving was.

I am honest about the cost and about why I expect this rung to land *between* the parallel floor and a plain sequential RMSNorm block. Four RMSNorms per block, where the plain block runs two and the parallel rung ran one, makes this the *most expensive* rung in normalization arithmetic — I should expect the slowest wall-clock, a near-mirror of the parallel rung's speed win, and that is the price of the variance control. The subtler concern is whether the output norm helps *final loss* or only *stability*: the learned gain $\gamma$ on $\mathrm{LN}_{\text{post}}$ can re-inflate the branch contribution back toward whatever magnitude the optimizer wants, so the variance control is a *soft* prior, not a hard cap — the network can learn to undo some of it. At 24 layers, deep enough for variance growth to matter but not so deep that pre-norm is in trouble, the benefit may be modest. It should help relative to the parallel rung, which had *no* output control and lost cross-talk on top; whether it beats a plain sequential RMSNorm block — which already has the cross-talk back and the clean pre-norm gradient, just without the output norm — is the open question this rung answers, and my prior is that the output norm is variance insurance more than a final-loss lever at this depth. The init contract holds: the sandwich still writes to the residual twice per block, so the substrate's $1/\sqrt{2\cdot n_{\text{layer}}}$ factor still matches, and with $\mathrm{LN}_{\text{post}}$ gains starting at 1, each initial contribution is normalized to unit RMS and then scaled *down* — conservative, not larger, than the default block. I touch no `CONFIG_OVERRIDES`; the parallel rung's problem was wiring, not optimization.

So this is a clean fill of the same two-region surface: keep rung 1's RMSNorm class verbatim, and replace the `Block` with the sandwich — `ln_pre1`, attention, `ln_post1`; `ln_pre2`, MLP, `ln_post2`; residual adds threaded between, both sublayers sequential again. My most confident expectation is that it *beats* the parallel rung's $2.3112$, since it restores intra-block cross-talk *and* adds output-variance control — two moves in the quality direction with no regression I can name — with the perplexities following down. The expectation I am genuinely unsure of is that it will *not* be the best rung overall, sitting between the parallel floor and the plainest sequential block, because the soft-prior nature of the output norm makes it insurance more than a lever at 24 layers.

```python
# EDITABLE regions of nanoGPT/custom_pretrain.py — rmsnorm_post (sandwich)

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
    """Sandwich-Norm: Pre-LN + Post-LN with RMSNorm (CogView style)."""
    def __init__(self, config):
        super().__init__()
        self.ln_pre1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_post1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_pre2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.ln_post2 = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x):
        x = x + self.ln_post1(self.attn(self.ln_pre1(x)))
        x = x + self.ln_post2(self.mlp(self.ln_pre2(x)))
        return x
```
