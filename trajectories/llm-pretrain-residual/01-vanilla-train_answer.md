The single object I get to design here is the rule that carries information through the residual stream of a 24-layer Pre-LN GPT-2 Medium across depth — everything else (attention, MLP, normalization, data, optimizer schedule) is frozen. Before designing anything, I have to fix the floor, and the floor has to be the *strongest plain residual rule*, not a crippled strawman, so that every later redesign on this ladder is judged against the honest best-simple-thing. That floor is the vanilla additive Pre-LN residual, and I want to be precise about why it is the right one and exactly where it is still weak, because each of its weaknesses is a seam the climb will pull on.

The method at this rung is just plain Pre-LN itself: each block reads a normalized copy of the stream and writes its branch back with coefficient one,

$$x \leftarrow x + \mathrm{attn}(\mathrm{LN}(x)), \qquad x \leftarrow x + \mathrm{mlp}(\mathrm{LN}(x)),$$

repeated once per sublayer, with a single final LayerNorm before the head. This is not an arbitrary default; it is the resolution of a whole lineage. Plain stacked layers with no skip compose $L$ width-preserving maps directly, and the per-layer signal gain $r$ compounds to $r^L$ — unless $r \approx 1$ at *every* layer the forward signal and backward gradient vanish or explode exponentially in depth, and a deeper plain net reaches *higher* training error than a shallow one, so depth is not trainable at all. The residual connection of He et al. (2015), $x_{l+1} = \sigma(x_l + F(x_l))$, adds an identity skip so the block only has to learn a nudge to identity, and the additive $1$ in the unrolled backward product gives the top gradient a route to every shallow layer that is not multiplied by all the Jacobians above it — that made hundreds of layers trainable. But the branch $F$ fires at full strength at init, so the block is *not* the identity at step zero and the stream variance still compounds; the skip tames the worst of $r^L$ without pinning $r=1$. The Transformer then added normalization, and *where* it goes matters: Post-LN, $x \leftarrow \mathrm{LN}(x + \mathrm{sublayer}(x))$, normalizes after the addition, which puts a LayerNorm Jacobian *on* the highway and makes the backward path a product of normalization Jacobians — exactly the multiplicative structure the skip was meant to avoid — giving large, depth-imbalanced gradients near the output at init, which is why Post-LN needs learning-rate warm-up to survive its first steps. Pre-LN is the fix the floor uses: move the norm *inside* the branch, $x \leftarrow x + \mathrm{sublayer}(\mathrm{LN}(x))$, so each sublayer reads a normalized input but writes its raw output into an unnormalized stream whose through-path is identity-plus-addition again. The leading $1$ is restored, the backward highway is clean, the last-layer gradient shrinks like $1/\sqrt{L}$ rather than staying large and depth-independent, and the single final LayerNorm before the head soaks up the accumulated stream scale.

So Pre-LN is the strongest *plain* residual rule, which is exactly what makes it the honest floor — and I should name the one property that every later rung will attack, because it is load-bearing for the whole climb. The Pre-LN stream is a *fixed unit-weight accumulator*. Unrolling the recurrence with $x_{l+1} = x_l + F_l(\mathrm{LN}(x_l))$ gives

$$x_l = x_1 + \sum_{i<l} F_i(\mathrm{LN}(x_i)),$$

the embedding plus every earlier branch output, each added with coefficient exactly one, the same for every token. Three gaps follow. First, the depth-mixing rule is *rigid*: the sequence axis gets full self-attention and the feature axis gets a learned nonlinearity, but the depth axis gets one unweighted running sum with no knob — no way to say "layer 9 should lean on layer 3's output more than layer 8's," and the same mixing for every token. That asymmetry — learned dynamic mixing on two of three axes, the third frozen — is the most fundamental gap. Second, the stream variance *climbs with depth*: each branch reads a normalized input of fixed scale and writes its raw output into an unnormalized stream, so the accumulated norm grows, and because the branch's LayerNorm divides by that growing scale, the deep blocks' Jacobians shrink toward the identity and the deepest layers become near-identity maps that contribute little — measurable, monotone, mechanical, the proximate cause of deep-layer death. Third, the branch is at full strength from step zero, so the early chaotic updates of a freshly-random deep stack get written into the stream at full force — the temporal gap, the reason warm-up exists at all.

These three are not independent: a fix that holds the deep branches back early (temporal) also slows the variance climb (mechanical), which tells me the cheapest first move is one that touches *when and how much* each branch writes, since that single lever bears on two of the three gaps without any new learned machinery. That is the move the next rung reaches for. At this rung, though, the point is only to establish the floor, so the edit is the trivial one — leave the block loop exactly as the vanilla template writes it, an identity replacement of the loop with itself, adding no parameters, no optimizer change, and no config override. The model that runs is plain additive Pre-LN, the cleanest backward highway the plain residual can offer and no mechanism at all for weighting, scheduling, or re-routing the depth flow — a well-trodden recipe (GPT-2 Medium on 7B FineWeb tokens) for which I expect a healthy validation loss in the low-2.2s, sensible WikiText-2 and LAMBADA perplexities, and mid-50s ARC-Easy / low-30s HellaSwag. That is the bar the next rung has to clear: keep this clean highway, but stop treating the depth axis as a fixed unit-weight sum.

```python
# EDITABLE regions of custom_pretrain.py — step 1: vanilla Pre-LN residual (default)

# Block: scaffold default (unchanged).
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT.__init__ residual region: default (no extra parameters).

# GPT.forward block loop — the only edited region (identity replacement of the loop with itself):
def _gpt_forward_block_loop(self, x):
    # ── Residual stream: iterate through transformer blocks ──
    for block in self.transformer.h:
        x = block(x)
    return x

# GPT.configure_optimizers: default (no new param groups).
# CONFIG_OVERRIDES = {}   (no LR / weight-decay override).
```
