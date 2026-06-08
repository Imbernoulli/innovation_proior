**Problem.** SiLU (rung 1) softened ReLU but, as the plainest fixed gate, left the deep/hard run with
headroom: resnet20-cifar10 92.72 and mobilenetv2-fmnist 94.69 were flat, while vgg16bn-cifar100 only
reached 70.38. Still only the `CustomActivation` curve is editable; pipeline and models frozen.

**Key idea.** Stay in the self-gated family `x·h(x)` but change the gate from `h=σ` to
`h=tanh∘softplus`, giving Mish `f(x)=x·tanh(softplus(x))`. Same asymptotics as SiLU (`f→x` for large
positive, `f→0` for large negative), same smooth, non-monotonic, live-negative shape (min ≈ −0.31), and
the same parameter-free drop-in contract — the task fixes mish to `x*torch.tanh(F.softplus(x))`.

**Why it should beat SiLU where it matters.** With `s=softplus(x)`, `softplus′=σ`, the derivative is
`f′(x)=tanh(s)+x·sech²(s)·σ(x)=Δ(x)·SiLU(x)+f(x)/x`, with `Δ(x)=sech²(softplus(x))`. So Mish's gradient
is SiLU's gradient rescaled by a smooth, input-dependent factor `Δ(x)` — a per-input *preconditioner*
that smooths and better-conditions the loss surface. That pays most where depth/capacity make
conditioning the bottleneck (the VGG-16-BN/CIFAR-100 run) and least where the net is already easy.

**Hyperparameters.** None — parameter-free, like rung 1.

**What to watch.** Expect resnet20-cifar10 (~92.72) and mobilenetv2-fmnist (~94.69) to stay roughly
flat, and vgg16bn-cifar100 to clear SiLU's 70.38. If the VGG run *also* stays flat, the fixed-curve
lever is exhausted and the next move must give the network *learnable* control of the activation.

```python
# EDITABLE region of pytorch-vision/custom_activation.py (lines 32-49) -- step 2: Mish
class CustomActivation(nn.Module):
    """Mish activation function.

    Mish(x) = x * tanh(softplus(x)).
    Self-regularized non-monotonic activation with smooth gradients.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
```
