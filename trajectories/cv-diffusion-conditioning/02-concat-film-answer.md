**Problem (from step 1).** The adaptive-norm floor (20.01 / 12.31 / 11.77 FID) routed the class *only*
through a post-block, zero-init-gated conditioner, leaving the timestep's tuned block-internal AdaGN path
class-blind. The numbers read like a *routing* bottleneck, not a missing-bandwidth one. The cheapest thing
to falsify is: the affine was fine; it was sent down the wrong road.

**Key idea (concat-FiLM = additive-bias corner, on the time path).** Send the class through the *same*
socket the timestep uses. Embed the class to width `time_embed_dim` (the substrate's
`class_embed` already does this), **add** it to the time embedding, and let each residual block's own
adaptive group norm carry the sum: `prepare_conditioning` returns `time_emb + class_emb`, and
`ClassConditioner` is a **no-op**. Because the block's AdaGN projection is linear,
$(\gamma,\beta) = P(t_{\text{emb}}+c_{\text{emb}}) = P\,t_{\text{emb}} + P\,c_{\text{emb}}$ — the class
contributes an additive shift to the modulation parameters, i.e. the $\gamma=1$ (concatenation /
additive-bias) corner of FiLM, applied at the block's own modulation point.

**Why this over step 1.** It rides machinery already tuned for a global, per-channel side signal (the
timestep), with no fresh sublayer to climb off a zero gate. It is the *simplest* operator on the ladder —
the cheapest corner of the affine family — but routed where the floor's affine could not reach. A
straight elementwise sum (both vectors are `time_embed_dim`) needs no projection, no concatenation
widening, no fixed split; one existing AdaGN projection produces all modulation from the combined vector,
treating timestep and class symmetrically.

**What the harness lets me skip.** No FiLM layer, no gate, no GroupNorm of my own: the diffusers
`ResnetBlock2D` *is* the regressor that turns the fed embedding into per-channel scale/shift. The entire
edit is the routing.

**Hyperparameters / scaffold edit.** `prepare_conditioning` = `time_emb + class_emb`; `ClassConditioner`
= empty module returning `h`. No new hyperparameters; fixed loop unchanged.

**Falsifiable expectation.** Below the floor at all three scales, with the largest absolute gain at Small
(where the blunt floor hurt most) and a tighter Small–Large spread. If it lands at or above 20.01 /
12.31 / 11.77, the routing hypothesis is wrong and the post-block adaptive norm was buying spatial/feature
bandwidth — pointing the next rung at a content-dependent operator.

```python
def prepare_conditioning(time_emb, class_emb):
    # Concat-FiLM: add projected class_emb to time_emb
    return time_emb + class_emb


class ClassConditioner(nn.Module):
    # No-op: all conditioning is via time_emb
    def __init__(self, channels, cond_dim):
        super().__init__()

    def forward(self, h, class_emb):
        return h
```
