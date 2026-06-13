**Problem (from step 1).** Orthogonal init won only where the topology matched the theory: best on VGG
(72.83), ordinary on ResNet-56 (72.08), *weakest* on MobileNetV2 (93.88) where depthwise convs flatten to
tall `(C,9)` matrices the orthogonal construction can't make row-orthogonal. The expensive full-spectrum
control wasn't doing the work — and BatchNorm already re-standardizes activation variance at every layer, so
preserving the whole forward norm is partly redundant.

**Key idea.** Drop the spectrum target; get the *second moment* exactly right for ReLU, the cheap BN-friendly
way (He et al. 2015): draw conv weights from `N(0, √(2/fan))`, where the factor of two pays back the variance
ReLU's half-rectification removes. Same target as orthogonal-√2, but a plain Gaussian spectrum — which the
step-1 numbers say is the part that actually mattered.

**Why it works / the mode split.** `fan_in` scaling stabilizes the *forward* activation variance; `fan_out`
the *backward* gradient variance. Convs use **`fan_out`** — BN already pins forward variance, so controlling
backward gradient scale is the higher-value lever; the depthwise convs that broke orthogonality just get the
correct per-filter variance. The bare `Linear` head (no BN after it) uses **`fan_in`** to keep the pre-logit
scale sane into the softmax. BN stays neutral `(γ=1, β=0)`; biases zero.

**Hyperparameters / scaffold edit.** One uniform pass, no architecture branching: `Conv2d` →
`kaiming_normal_(mode='fan_out', nonlinearity='relu')`; `Linear` →
`kaiming_normal_(mode='fan_in', nonlinearity='relu')`, zero bias; `BatchNorm2d` → `(1, 0)`.

**What to watch.** Expect to *beat* orthogonal on MobileNetV2 (depthwise handled honestly); a near-tie on VGG
(orthogonal's spectral conditioning may hold a slim edge on its home stack); a wash on ResNet-56 — both fix
per-layer variance, *neither* fixes residual accumulation, which is what the next rung must attack.

```python
def initialize_weights(model, config):
    """Kaiming/He normal initialization (fan_out, ReLU).

    Conv2d: N(0, sqrt(2/fan_out)) — preserves forward-pass variance with ReLU.
    BatchNorm2d: weight=1, bias=0.
    Linear: N(0, sqrt(2/fan_in)).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```
