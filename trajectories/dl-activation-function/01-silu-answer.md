**Problem.** Only the `CustomActivation` curve is editable; the optimizer (SGD `lr=0.1`, momentum 0.9,
wd 5e-4), cosine-200 schedule, Kaiming init, augmentation, and the three model definitions are frozen.
The scaffold default is ReLU `max(x,0)`, which zeroes the whole negative half (dead units, zero gradient
there) and has a curvature-free kink at the origin — right where BatchNorm concentrates the inputs.

**Key idea.** Replace the hard sign-gate with the simplest smooth self-gated curve, SiLU `x·σ(x)`:
multiply the input by a sigmoid of itself. Smooth everywhere, small live negative region (no dying
units), unbounded above, bounded below, non-monotonic; recovers `x` for large positive and `0` for large
negative, so it keeps ReLU's training behavior and softens the rest.

**Why this rung.** It is the cheapest defensible step off ReLU: zero learnable parameters and zero new
hyperparameters, so any change is attributable to curve *shape* alone under the frozen pipeline. The
task fixes silu to `nn.SiLU` (fixed `β=1`, `x·σ(x)`), *not* the temperature-`β` learnable Swish — so
nothing is registered in `__init__` and nothing is added to the optimizer.

**Hyperparameters.** None — `β=1` fixed, no learnable parameters.

**What to watch.** Expect at-or-slightly-above ReLU everywhere, with a small and uneven margin: thinner
on the easy/near-saturated runs, possibly larger on the deeper 100-class VGG. A small win from a single
fixed gate motivates trying a *different-shaped* fixed gate next.

```python
# EDITABLE region of pytorch-vision/custom_activation.py (lines 32-49) -- step 1: SiLU
class CustomActivation(nn.Module):
    """SiLU/Swish activation function.

    SiLU(x) = x * sigmoid(x).
    Self-gated activation discovered via automated search.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x)
```
