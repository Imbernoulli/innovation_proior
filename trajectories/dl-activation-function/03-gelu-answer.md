**Problem.** Mish (rung 2) beat SiLU only by hundredths (92.78 / 70.50 / 94.70 vs 92.72 / 70.38 /
94.69) — swapping one fixed smooth gate for another buys almost nothing. Still only the
`CustomActivation` curve is editable; pipeline and models frozen. One principled fixed gate remains
untried.

**Key idea.** Same self-gated form `x·h(x)`, but set the gate to the CDF of the distribution the inputs
actually follow. Every activation here sits right after BatchNorm, which standardizes its inputs toward
`N(0,1)`, so the calibrated "probability this unit fires" is the standard-normal CDF `Φ`, not the
logistic `σ` (SiLU) or `tanh∘softplus` (Mish). That gives GELU `f(x)=x·Φ(x)`,
`Φ(x)=½[1+erf(x/√2)]` — the task fixes gelu to `F.gelu(x)`, parameter-free, same drop-in contract.

**Why it should win among the fixed gates.** Same asymptotics/shape as SiLU and Mish (`f→x` for large
positive, `f→0` for large negative, shallow non-monotonic dip). But `σ` and `Φ` differ most near and
just-below zero — exactly where BN concentrates the mass — and the Gaussian gate weights each input by
its *actual* standout probability under the BN-standardized distribution, while the logistic gate is a
mismatched approximation. So the distribution-matched gate should be the best fixed curve on these
heavily-BN'd nets.

**Hyperparameters.** None — parameter-free.

**What to watch.** Expect resnet20-cifar10 ~92.7–92.8 and mobilenetv2-fmnist ~94.7 (saturated, no gate
moves them), and vgg16bn-cifar100 to clear both SiLU's 70.38 and Mish's 70.50 by more than the hundredths
separating the earlier rungs. The deeper signal: gains between fixed curves are decaying (tenths →
hundredths), so the family is converging — the next move must give the network a *learnable* activation,
not another hand-picked shape.

```python
# EDITABLE region of pytorch-vision/custom_activation.py (lines 32-49) -- step 3: GELU
class CustomActivation(nn.Module):
    """GELU activation function.

    GELU(x) = x * Phi(x) where Phi is the Gaussian CDF.
    Smooth, non-monotonic, allows small negative values.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)
```
