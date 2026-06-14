**Problem (from steps 1–2).** Both adanorm (20.01 / 12.31 / 11.77) and the stronger concat-FiLM (19.39 /
11.50 / 10.52) are the *same operator* — a global, per-channel affine drawn from the class label, applied
identically at every spatial position and blind to local feature content. Re-routing it onto the timestep's
AdaGN helped, but the residual FID (the barely-moving Small number) is the affine's *structural* limit:
"which class" wants to act differently on different parts of the image, and a global per-channel gain
cannot. The next gain must change the operator, not the route.

**Key idea (cross-attention: content-dependent, spatially varying injection).** Insert one cross-attention
layer after each UNet block where the **image feature map provides the queries** and the **class embedding
provides the keys/values**. Flatten the $H\times W$ positions into query tokens; with
$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d})V$, the output at position $p$ is
$\sum_j \mathrm{softmax}_j(q_p\cdot k_j)\,v_j$ with $q_p$ a projection of $h_p$ — so the read depends on the
position's own content (unlike FiLM, whose modulation depends only on $c$). The class flows *entirely*
through these layers, so `prepare_conditioning` returns `time_emb` unchanged (timestep = pure noise-level
signal; attention = "what to draw").

**Three design points (all in the substrate `CrossAttentionLayer`).** (1) Scale $1/\sqrt{d_{\text{head}}}$:
logits $q\cdot k$ have variance $d_{\text{head}}$; without it softmax saturates and gradients vanish.
(2) Multi-head (4 heads, per-head width $C/4$): read several aspects of the condition in parallel; cost ~
one full-width head. (3) **Zero-initialized output projection inside a residual**: the block is the identity
at init, so it can be stapled onto the tuned denoiser without disturbing it; conditioning grows from zero.
GroupNorm(32) precedes it, matching the backbone.

**Honest $M=1$ note.** The condition is one class token, so softmax over a single key is identically 1 — the
content-dependence through $q_p$ degenerates and the layer is, in effect, a *learned, zero-init-gated
residual injection* of the class (its own projection + GroupNorm + capacity), richer than concat-FiLM's
per-channel AdaGN bias but not doing spatial routing. The same operator becomes genuinely content-dependent
once $M>1$ (text/layout); that generality is the reason to build it this way.

**What the harness omits.** The full conditioning block (`GroupNorm → proj_in → [self-attn, cross-attn,
GEGLU FFN] × depth → proj_out → residual`) is *not* built. `ClassConditioner` here is a single bare
`CrossAttentionLayer` per block — no self-attention sublayer, no FFN, no proj_in/out wrapper, no depth. Only
the cross-attention core runs.

**Hyperparameters / scaffold edit.** `prepare_conditioning` = identity on `time_emb`; `ClassConditioner` =
`CrossAttentionLayer(channels, cond_dim, num_heads=4)` per block. Fixed loop unchanged.

**Falsifiable expectation.** Below concat-FiLM at all three scales (strongest on the ladder), but by a
*thin* margin — smaller than concat-FiLM's gain over the floor — because the single class token leaves
attention's content-dependent bandwidth mostly unused; the gain, if any, concentrates at the larger scales.
A thin win confirms a genuinely stronger operator whose remaining headroom is in *what the condition is*
(richer tokens / guidance), not *how it is injected*.

```python
def prepare_conditioning(time_emb, class_emb):
    # Cross-attn: time_emb unchanged, conditioning via ClassConditioner
    return time_emb


class ClassConditioner(nn.Module):
    # Cross-attention: class embedding as key/value
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)

    def forward(self, h, class_emb):
        return self.cross_attn(h, class_emb)
```
