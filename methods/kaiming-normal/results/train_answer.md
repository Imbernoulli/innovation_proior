The thing that actually defeats me is training a very deep convolutional rectifier network from scratch. I stack twenty or thirty conv layers, fill the weights with i.i.d. Gaussian noise at the std everyone copies from the AlexNet recipe, $\sigma = 0.01$, and the run just sits there — the loss does not move and the gradients in the early layers are essentially zero. The identical recipe trains a shallow net fine. So the failure is about depth, and it is there from the very first step, before learning could have gone wrong: it is an *initialization* failure, not an optimization failure. The workarounds of the day confirm the diagnosis without curing it — pre-training a shallow eight-layer model to seed the deeper one, or hanging auxiliary classifiers off the middle of the network to inject gradient nearer the input. Both are admissions that with a fixed-std Gaussian the signal or the gradient cannot survive the trip through that many layers, and both add cost and may land in a poorer optimum. What I want instead is a *data-independent, one-shot* rule that sets each layer's initial weight std from that layer's shape alone, no calibration passes, such that neither the forward responses nor the backward gradients shrink or grow exponentially with depth.

The reason the trip kills the signal is purely multiplicative. Whatever a single layer does to the scale of the signal, the next layer does again, and the next, so if each layer multiplies the response variance by a factor $\beta$, then after $L$ layers it is multiplied by $\beta^L$. The same holds backward for the gradients. And $\beta^L$ is vicious: for $\beta$ a hair above one it explodes, for a hair below one it collapses toward zero, with no gentle middle once $L$ is large. The whole game is therefore to make the per-layer variance factor *exactly* one — not on average over the run, but as a property of the random numbers I write at initialization. If the factor is one, the product is one for any depth and depth stops being the enemy. Xavier/Glorot already tried to enforce this with a fan-dependent std, but its derivation assumes the activation sits in a linear regime where $f'(s)\approx 1$ and the unit is symmetric about zero — precisely the assumption a rectifier breaks, since $\max(0,y)$ deletes the negative half-line. The orthogonal route of Saxe et al. preserves norm exactly in a deep *linear* net, but it too only works once a gain is chosen to offset the nonlinearity, and it never says what that gain should be for a rectifier; operationally it also needs a matrix factorization rather than a single i.i.d. draw. The missing piece in every case is an activation-specific correction for the rectifier, and that is what I derive.

I propose He (Kaiming) initialization: draw each weight from a zero-mean Gaussian with variance $2/\text{fan}$, i.e. std $\sqrt{2/\text{fan}}$, with biases set to zero. The derivation is the variance-propagation argument of Glorot & Bengio redone honestly for the rectifier. Take a conv response $y = Wx + b$, where $x$ stacks the co-located $k\times k$ pixels across the $c$ input channels into a vector of length $n = k^2 c$ (the fan-in) and $W$ is $d \times n$. With the weights i.i.d., zero-mean, symmetric about zero, the inputs i.i.d. and independent of the weights, summing $n$ independent product terms gives $\text{Var}[y_l] = n_l\,\text{Var}[w_l]\,E[x_l^2]$. Here is the exact spot where I cannot copy Glorot: they wrote $E[x^2]=\text{Var}[x]$, which is true only for zero-mean $x$, but $x_l = \max(0, y_{l-1})$ is never negative and has strictly positive mean. So I keep $E[x^2]$ honest. If $w_{l-1}$ is symmetric and $b_{l-1}=0$, then $y_{l-1}$ is itself symmetric about zero, and the rectifier keeps the positive half exactly while zeroing the negative half, so $E[\max(0,y)^2] = \int_0^\infty y^2 p(y)\,dy = \tfrac12\int_{-\infty}^{\infty} y^2 p(y)\,dy = \tfrac12 E[y^2] = \tfrac12\text{Var}[y_{l-1}]$. That factor of one-half is the whole story — it is the literal fraction of its input's second moment that the rectifier passes, because it discards exactly half the symmetric distribution. Substituting,

$$\text{Var}[y_l] = \tfrac12\,n_l\,\text{Var}[w_l]\,\text{Var}[y_{l-1}],$$

so the per-layer forward factor is $\tfrac12 n_l\text{Var}[w_l]$, and chaining $L$ of them multiplies by $\prod_l \tfrac12 n_l\text{Var}[w_l]$. To kill the exponential I force every factor to one, $\tfrac12 n_l\text{Var}[w_l]=1$, which solves to $\text{Var}[w_l]=2/n_l$, a Gaussian of std $\sqrt{2/n_l}$, with $b=0$ — the zero bias I needed anyway for the symmetry argument. Against Glorot's linear result $n\,\text{Var}[w]=1$, i.e. $\sqrt{1/n}$, my std is larger by exactly $\sqrt 2$ per layer: the rectifier ate half the variance at every layer, so I put twice as much in to compensate, and that is the entire correction — the factor of $2$ inside the square root. The first layer technically has no rectifier in front of it, so its honest condition is $n_1\text{Var}[w_1]=1$ without the half, but one layer's factor being off by $2$ shifts the whole product by a single constant, not by anything that compounds with depth, so I apply the same rule everywhere for uniformity.

Forward variance is only half the picture, and the thing dying in my failed runs was the *gradient*, so I run the same analysis backward to be sure I am not preserving the forward signal while strangling the gradient. The back-propagated gradient is $\Delta x_l = \hat W_l \Delta y_l$, where $\hat W$ is the weights rearranged into the shape back-prop uses, a $c\times\hat n$ matrix with $\hat n = k^2 d$. Note the asymmetry: forward, a response gathers from $n=k^2 c$ connections; backward, the gradient at an input pixel gathers from $\hat n = k^2 d$, because the filter count $d$ now plays the role the input-channel count $c$ played forward — this is fan-out versus fan-in, and $\hat n \ne n$ in general. With $w$ and $\Delta y_l$ independent and $w$ symmetric zero-mean, $\Delta x_l$ is zero-mean and $\text{Var}[\Delta x_l]=\hat n_l\,\text{Var}[w_l]\,\text{Var}[\Delta y_l]$. The rectifier reappears through its *derivative*: $\Delta y_l = f'(y_l)\,\Delta x_{l+1}$, and for ReLU $f'$ is $1$ when $y>0$ and $0$ when $y<0$, each with probability one-half by the symmetry of $y$. Since $f'$ is $0$ or $1$, $f'^2=f'$ and $E[f'^2]=\tfrac12$, so $\text{Var}[\Delta y_l]=\tfrac12\text{Var}[\Delta x_{l+1}]$. The same one-half drops out — but for a wholly different reason than forward: there it was the rectifier keeping half the output mass, here it is the derivative being one on half the domain. Hence

$$\text{Var}[\Delta x_l] = \tfrac12\,\hat n_l\,\text{Var}[w_l]\,\text{Var}[\Delta x_{l+1}],$$

and the backward sufficient condition $\tfrac12\hat n_l\text{Var}[w_l]=1$ gives std $\sqrt{2/\hat n_l}$ — identical in form, but with the fan-out $\hat n_l=k^2 d_l$ where forward had the fan-in $n_l=k^2 c_l$.

Now the two conditions disagree, since $n_l\ne\hat n_l$ unless the channel counts match, and one variance per layer cannot satisfy both. Glorot split the difference with $\text{Var}[w]=2/(n+\hat n)$, preserving neither but their average. I ask whether I even need the compromise, because if imposing one condition keeps the *other* product from diverging, I can simply pick one. Impose the backward condition, $\text{Var}[w_l]=2/\hat n_l$, and substitute into the forward factors: $\prod_l \tfrac12 n_l\,(2/\hat n_l) = \prod_l n_l/\hat n_l$. With $n_l=k_l^2 c_l = k_l^2 d_{l-1}$ and $\hat n_l = k_l^2 d_l$, this is $\prod_l d_{l-1}/d_l$, which telescopes to $d_1/d_L = c_2/d_L$ — the ratio of the second layer's input channels to the last layer's filter count, a fixed modest constant that does not shrink with depth and so cannot cause the $\beta^L$ catastrophe. By symmetry, imposing the forward condition leaves the backward product at $d_L/c_2$, equally harmless. So no compromise is needed: either condition alone makes the network trainable. I preserve the backward signal on the convs, since that is the quantity that was actually dying. To see the magnitude, take VGG-style model B with $3\times3$ convs of $64,128,256,512$ filters: the backward rule wants stds $\sqrt{2/(9d)}$, namely $0.059, 0.042, 0.029, 0.021$, all several times larger than the fixed $0.01$. Each layer scales the gradient std by its own std, and these multiply, so under $0.01$ the gradient reaching the early layers is about $1/(5.9\cdot4.2^2\cdot2.9^2\cdot2.1^4)\approx 1/(1.7\times10^4)$ of the variance-matched value — four orders of magnitude too small, exactly the "diminishing gradients" people report. And it explains why Xavier, which does scale with the fan, still falls short: its $\sqrt{1/n}$ is smaller than the rectifier's $\sqrt{2/n}$ by $1/\sqrt2$ per layer, hence $1/2^{L/2}$ over depth, about $1/2000$ at $22$ layers and $1/32768$ at $30$. The $\sqrt2$ is not cosmetic; it is the precise amount needed to cancel the rectifier's lost second moment before depth turns it into an exponential collapse.

The rule generalizes almost for free to the leaky/parametric rectifier $f(y)=\max(0,y)+a\min(0,y)$, which passes the positive half unchanged and the negative half scaled by $a$. Redoing $E[x^2]$, the positive half still gives $\tfrac12 E[y^2]$ and the negative half, no longer thrown away, contributes $a^2\cdot\tfrac12 E[y^2]$, so $E[x^2]=\tfrac12(1+a^2)\text{Var}[y]$ and the condition becomes $\tfrac12(1+a^2)n_l\text{Var}[w_l]=1$, giving $\text{Var}[w_l]=2/((1+a^2)n_l)$ and std $\sqrt{2/(1+a^2)}/\sqrt{n_l}$. This is a clean consistency check: $a=0$ recovers my $\sqrt{2/n}$, and $a=1$ recovers $\sqrt{1/n}$, exactly Glorot's linear result — Glorot was solving the $a=1$ corner of my own formula and applying it where $a=0$. The right way to think of it is std $=\text{gain}/\sqrt{\text{fan}}$, where the gain depends only on the nonlinearity's slope — $\sqrt2$ for ReLU, $\sqrt{2/(1+a^2)}$ for slope $a$, $1$ for a linear/symmetric unit — and $\text{fan}$ is $n$ (fan-in) or $\hat n$ (fan-out) according to which direction I preserve. Two practical notes fall out of the derivation. Biases are zero because the symmetry of $y$ required it; batch-norm affine starts at identity, scale $1$ and shift $0$, so it does not disturb the variance the conv init just set. And because the rule carries the input variance roughly undimmed to the last layer, an unnormalized input (e.g. range $[-128,128]$) can overflow the softmax, so I shrink the classifier's fully-connected layers a touch below $\sqrt{2/n}$ (or fold in a small per-layer factor like $(1/128)^{1/L}$) — a patch on the classifier end, not a change to the variance matching.

The canonical library primitive computes $\text{std}=\text{gain}/\sqrt{\text{fan}}$ from the tensor's shape:

```python
import math
import warnings
import torch
from torch import Tensor


def _calculate_fan_in_and_fan_out(tensor: Tensor):
    # conv weight [out, in, k, k]: receptive_field = k*k
    # linear weight [out, in]:     receptive_field = 1
    if tensor.dim() < 2:
        raise ValueError("fan in and fan out require at least 2 dimensions")
    num_output_fmaps = tensor.size(0)
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size     # n   = k^2 * c
    fan_out = num_output_fmaps * receptive_field_size   # n_hat = k^2 * d
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    mode = mode.lower()
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    if nonlinearity in (
        "linear", "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
        "sigmoid",
    ):
        return 1.0
    if nonlinearity == "relu":
        return math.sqrt(2.0)                           # sqrt(2): the rectifier's lost factor
    if nonlinearity == "leaky_relu":
        slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1.0 + slope ** 2))      # sqrt(2/(1+a^2)) for slope a
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def kaiming_normal_(tensor: Tensor, a: float = 0,
                    mode: str = "fan_in", nonlinearity: str = "leaky_relu",
                    generator: torch.Generator | None = None) -> Tensor:
    """Fill `tensor` from N(0, std^2) with std = gain / sqrt(fan)."""
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op", stacklevel=2)
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)                          # ReLU: sqrt(2/fan); leaky: sqrt(2/((1+a^2)*fan))
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)
```

Applied as a one-shot, shape-only initializer over a built rectifier CNN — convs preserve the backward pass, linear layers the forward pass, biases zero, batch-norm affine at identity:

```python
import torch.nn as nn


def initialize_weights(model, config):
    """He/Kaiming init for a deep ReLU CNN: std = sqrt(2/fan), data-independent."""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # fan_out = out_channels * k^2: preserves the backward-pass variance
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)            # b = 0: keeps y zero-mean symmetric
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)             # affine = identity at init
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # fan_in = in_features: preserves the forward-pass variance
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
```
