**Problem (from step 1).** Pixle's blind random rearrangement found almost nothing on robust models
(mean ASR ≈ 0.011) because it spends no information deciding *which* pixels to touch. The harness allows
gradients, so the next rung should *compute* per-pixel importance and spend the budget on it.

**Key idea (JSMA, adversarial saliency).** From the forward derivative `dF_j/dx_i` (Jacobian of the
*logits* w.r.t. the input — per-feature, per-class, signed), build a targeted saliency: a feature is
useful for target `t` iff increasing it raises `F_t` (`dF_t/dx_i > 0`) and lowers the rest
(`sum_{j!=t} dF_j/dx_i < 0`); score is the product of the two favorable magnitudes. Modify *pairs* of
features (one compensates the other's sign flaw), saturate each to its extreme (`theta=1`), drop
saturated pixels, recompute the Jacobian each step (the net is non-linear), stop when the target is
reached or the budget is spent.

**Why on the logits / why a least-likely target.** On the softmax the two gate conditions are mechanically
dependent (probabilities sum to one) and saturated, ruining the ranking; on the logits they carry
independent information. `torchattacks.JSMA` is targeted-only, so the harness sets
`set_mode_targeted_least_likely` — the model's least-confident class is the most aggressive untargeted
proxy, valid for any `n_classes`.

**Scaffold edit / hyperparameters (budget bookkeeping is load-bearing).** JSMA counts *features*
(`C*H*W = 3072`), touching `num_features*gamma` of them over `ceil(num_features*gamma/2)` iterations; a
spatial pixel changes if any channel moves, so feature count upper-bounds distinct pixels. Set
`gamma = pixels / (C*H*W) = 24/3072` so `max_iters = 12`, `≤ 24` features, `≤ 24` distinct pixels —
exactly the budget. A careless constant `gamma` overshoots and the harness rejects every sample (ASR→0).

**What to watch.** Mean ASR clears the 0.011 floor (it *uses* saliency) but stays low, exposing the
brittleness of greedy first-order saliency against models trained to flatten exactly that signal.

```python
def run_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    pixels: int,
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    import torchattacks

    _ = (device, n_classes)
    model.eval()

    # gamma bounds total perturbed features (C*H*W space) to `pixels`, which
    # is a sufficient upper bound on the number of distinct spatial pixels.
    num_features = int(images.shape[1] * images.shape[2] * images.shape[3])
    gamma = float(pixels) / float(num_features)

    attack = torchattacks.JSMA(model, theta=1.0, gamma=gamma)
    # Least-likely class as target -> strong untargeted proxy, works for any
    # n_classes.
    attack.set_mode_targeted_least_likely(quiet=True)
    return attack(images, labels)
```
