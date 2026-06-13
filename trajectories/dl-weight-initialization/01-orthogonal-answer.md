**Problem.** The scaffold default draws every weight from `N(0, 0.01²)` — a variance unrelated to layer
shape — so the per-layer signal shrinks geometrically with depth and a deep net starts nearly dead.
Fan-scaling (He) fixes only the *mean* signal norm; the spread of a Gaussian's singular values still
amplifies some directions and annihilates others, compounding over depth.

**Key idea.** Preserve the *whole* norm, not just the second moment: initialize each weight map as a
(semi-)orthogonal matrix so every singular value is one (Saxe et al. 2014). In a deep linear net this is exact
dynamical isometry — forward and backward norms conserved at every depth. A conv tensor
`(out, in, kh, kw)` is flattened to `(out, in·kh·kw)` and filled with a random orthogonal matrix
(`nn.init.orthogonal_`).

**Why it works / its scaling.** A pure orthogonal map preserves norm, but ReLU halves the variance per layer,
so scale the orthogonal matrix by gain `√2` (`calculate_gain('relu')`) — every singular value becomes `√2`,
the layer amplifies by √2, ReLU pulls back to unit, post-activation norm holds across depth. This is He's
target second moment achieved with a *controlled* spectrum instead of a Gaussian's spread. BatchNorm is left
at the neutral `(γ=1, β=0)` so it re-standardizes without fighting the conv scaling; biases zero.

**Hyperparameters / scaffold edit.** One uniform rule, no architecture branching: `Conv2d` and `Linear` →
`orthogonal_(gain=√2)`, zero bias; `BatchNorm2d` → `(weight=1, bias=0)`.

**What to watch.** Best on VGG-16-BN (the plain stack orthogonality was derived for); ordinary on ResNet-56
(orthogonality controls per-branch convs but not residual *accumulation* over 27 blocks); weakest on
MobileNetV2, whose depthwise convs flatten to tall `(C, 9)` matrices that `orthogonal_` cannot make
row-orthogonal — the isometry story breaks exactly there.

```python
def initialize_weights(model, config):
    """Orthogonal initialization.

    Conv2d & Linear: orthogonal matrix (gain=sqrt(2) for ReLU).
    BatchNorm2d: weight=1, bias=0.
    """
    gain = nn.init.calculate_gain('relu')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=gain)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```
