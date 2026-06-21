Swish, the activation x·σ(βx), improves over ReLU but was discovered by black-box search, so the field has lacked a principled explanation of what it is or how to generalize it. ReLU, Leaky ReLU, PReLU, and the absolute value are all instances of the Maxout family, max(η_a(x), η_b(x)), where each activation is the hard maximum of two linear pieces. Swish and the related fixed gates SiLU, Mish, and GELU smooth out the ReLU kink, but they are single fixed curves: every neuron in the network uses the same shape, and the first-derivative bounds are hardwired. That means the gains from swapping one fixed curve for another shrink as networks get larger and better optimized, and the network has no way to learn how nonlinear each channel should be.

The way forward is to stop choosing a fixed gate and instead smooth the Maxout family itself. The standard smooth maximum S_β(x_1,…,x_n) = (Σ_i x_i e^{βx_i})/(Σ_i e^{βx_i}) is a differentiable surrogate for max, with β acting as a temperature. When applied to two linear pieces, it simplifies to S_β(η_a, η_b) = (η_a − η_b)·σ(β(η_a − η_b)) + η_b. Plugging in η_a = x and η_b = 0 gives x·σ(βx), which is exactly Swish. So Swish is not a mysterious search artifact; it is the smooth maximum of {x, 0}. The same construction can be applied to the rest of the Maxout family, producing a strictly larger family of learnable activations.

The proposed method is ACON-C, short for Activate-or-Not C. Its formula is f(x) = (p_1 − p_2)·x·σ(β(p_1 − p_2)x) + p_2 x, where p_1, p_2, and β are learnable per-channel parameters. At initialization p_1 = 1, p_2 = 0, and β = 1, so the module begins as SiLU and then specializes. The derivative asymptotes to slopes p_1 for large positive inputs and p_2 for large negative inputs, and the local extrema of the derivative are approximately 1.0998·p_1 − 0.0998·p_2 and 1.0998·p_2 − 0.0998·p_1. Unlike Swish, whose bounds are fixed at about 1.0998 and −0.0998, ACON-C learns its own gradient ceiling and floor per channel. The parameter β separates two roles that Swish conflates: p_1 and p_2 set the bounds, while β sets how fast the derivative reaches them. Geometrically, β → ∞ makes the unit behave like max(p_1 x, p_2 x), a nonlinearity that activates, and β → 0 makes it behave like the arithmetic mean ((p_1 + p_2)/2)x, a linear pass-through. Each channel therefore learns whether to activate.

A fixed per-channel β can only choose one nonlinear degree for the whole channel, so it cannot vary by input sample. meta-ACON removes that limitation by generating β from the input itself. A lightweight channel-wise routing function, inspired by Squeeze-and-Excitation, computes β_c = σ(W_2 · BN(W_1 · GAP(x))) using global average pooling followed by a two-layer bottleneck with reduction r = 16 and a final sigmoid. This keeps the parameter overhead negligible while giving each channel a sample-dependent switching factor. Different inputs can therefore push the same channel toward a sharp nonlinear max or toward a linear response.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AconC(nn.Module):
    """ACON-C: smooth maximum of two learnable linear pieces p1*x and p2*x.

    f(x) = (p1 - p2) * x * sigmoid(beta * (p1 - p2) * x) + p2 * x

    p1, p2, beta are per-channel learnable parameters. Initialized at p1=1,
    p2=0, beta=1 so the module starts as SiLU x*sigmoid(x), then learns
    per-channel gradient bounds and nonlinear degree.
    """

    def __init__(self, width):
        super().__init__()
        self.p1 = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1, 1))

    def forward(self, x):
        diff = (self.p1 - self.p2) * x
        return diff * torch.sigmoid(self.beta * diff) + self.p2 * x


class MetaAconC(nn.Module):
    """meta-ACON-C: beta is generated from the input for per-sample control.

    beta_c = sigmoid(BN(FC2(BN(FC1(GAP(x))))))   # SE-style bottleneck, r=16
    f(x)   = (p1 - p2) * x * sigmoid(beta * (p1 - p2) * x) + p2 * x

    p1 and p2 are learnable per-channel bounds; beta adapts per sample and
    per channel at negligible extra parameter cost.
    """

    def __init__(self, width, r=16):
        super().__init__()
        inner = max(r, width // r)
        self.fc1 = nn.Conv2d(width, inner, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(inner)
        self.fc2 = nn.Conv2d(inner, width, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(width)
        self.p1 = nn.Parameter(torch.ones(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, width, 1, 1))

    def forward(self, x):
        ctx = x.mean(dim=(2, 3), keepdim=True)
        beta = torch.sigmoid(self.bn2(self.fc2(self.bn1(self.fc1(ctx)))))
        diff = (self.p1 - self.p2) * x
        return diff * torch.sigmoid(beta * diff) + self.p2 * x
```
