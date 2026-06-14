**Problem.** Turn the scaffold's unconditional CIFAR-10 denoiser into a class-conditional one by choosing
*how* the class label enters the UNet. A class label is a global, structureless, per-example signal — the
same profile as the timestep the backbone already ingests — so the first operator to try is the one whose
cost and structure match that profile: a per-channel adaptive affine, the FiLM / AdaGN family.

**Key idea (adaptive normalization, zero-initialized — on a conv feature map).** Leave the timestep path
alone (`prepare_conditioning` returns `time_emb`) and carry the class entirely through a post-block
`ClassConditioner` that applies one `AdaLNBlock` per UNet block. The block normalizes the feature map,
regresses a per-channel scale, shift, and gate from the class embedding, and applies
$x + \text{gate}\cdot\big((1+\text{scale})\cdot\mathrm{norm}(x) + \text{shift} - x\big)$. The
$(1+\text{scale})$ makes zero-scale neutral; the gate is **zero-initialized** so every conditioner starts
as the exact identity and the tuned backbone is undisturbed at init, with class conditioning growing from
a no-op as training learns it.

**Why this is the floor.** It is the lowest-bandwidth conditioning operator: one scale/shift per channel,
applied identically at every spatial location, drawn from the label alone — content-blind and spatially
uniform. It conditions correctly but cannot vary by spatial position or feature content, and here it is
deliberately confined to the *post-block* path (the class never rides the block-internal AdaGN the way the
timestep does). Both are reasons to expect the highest FID of the operators on the ladder.

**Not the transformer adaLN-Zero.** The canonical adaLN-Zero conditions a ViT diffusion backbone (six
modulation vectors per block, $c = t+y$ summed, a gateless decode head, the whole stack identity at init).
None of that exists in this harness — the backbone is a fixed conv UNet, there is no token stream and no
decode head, and the timestep and class are kept on separate paths. Only the core operation
(normalize → $(1+\text{scale})\hat x + \text{shift}$ → zero-init gate → residual) is ported, onto a
`[B,C,H,W]` feature map via GroupNorm, carrying the class alone.

**Hyperparameters / scaffold edit.** `prepare_conditioning` = identity on `time_emb`. `ClassConditioner`
= one `AdaLNBlock(channels, cond_dim)` per block (substrate utility: GroupNorm(1), zero-initialized
`SiLU → Linear(cond_dim, 3·channels)` → scale/shift/gate). No new hyperparameters; the fixed loop
(AdamW `2e-4`, EMA `0.9995`, 35k steps/scale, 50-step DDIM) is unchanged.

```python
def prepare_conditioning(time_emb, class_emb):
    # AdaNorm: time_emb unchanged, conditioning via ClassConditioner
    return time_emb


class ClassConditioner(nn.Module):
    # Adaptive LayerNorm-Zero: class embedding modulates features
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.adaln = AdaLNBlock(channels, cond_dim)

    def forward(self, h, class_emb):
        return self.adaln(h, class_emb)
```
