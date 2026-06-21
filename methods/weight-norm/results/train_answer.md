# Weight Normalization

The thing that actually bottlenecks first-order training of deep networks is the curvature of the loss. When the Hessian at a solution is ill-conditioned — a large spread between its largest and smallest eigenvalues — a gradient step points mostly along the steep directions, so the iterate bounces across a narrow valley while crawling along its flat floor, and progress per step collapses. The lever I want to pull is the conditioning, and the fact I keep underusing is that the curvature the optimizer sees is not a property of the function being fit but of the *coordinates* the parameters are written in: the same input-output map can be parameterized many equivalent ways, and two of them computing the identical function can have wildly different condition numbers. So "choose better coordinates for the parameters" is a real and nearly free lever, separate from "find a better optimizer." The principled way to fight curvature head-on is the natural gradient — precondition by the inverse Fisher and the step becomes reparameterization-invariant and whitened — but the Fisher is enormous and must be estimated and inverted, which is exactly what KFAC, FANG, and PRONG spend their cost on. The cheaper road, following Raiko et al., is to leave the optimizer as plain SGD and change the model's parameterization so the *ordinary* gradient already comes out whitened.

The obvious incumbent in that space is batch normalization, which standardizes each neuron's pre-activation $t = v \cdot x$ over the minibatch as $t' = (t - \mu[t])/\sigma[t]$ and then reapplies a learnable scale and shift. It improves conditioning and tolerates large learning rates, but the mechanism — statistics estimated over the batch — drags in a cluster of problems that are the reason I will not just use it: the output for one example now depends on the others sharing its batch; $\mu[t]$ and $\sigma[t]$ are estimates, so the layer injects stochastic noise into the gradient, with high variance when the batch is small; train and test compute different functions, since at test time frozen running averages are substituted; recurrence breaks it, because the same weights are reused every timestep and standardizing the cell state bleeds away the information it is meant to carry; and it costs real time and memory. The injected noise alone makes it a poor fit for reinforcement learning and generative models. I want the conditioning benefit without the batch.

I propose *weight normalization*. The starting observation is what the division by $\sigma[t]$ is really doing. Take a single layer with whitened inputs $x$ — independent, zero mean, unit variance. Then $\mu[t] = v \cdot \mathbb{E}[x] = 0$ and $\mathrm{Var}(t) = \sum_i v_i^2 \mathrm{Var}(x_i) = \sum_i v_i^2 = \lVert v \rVert^2$, so $\sigma[t] = \lVert v \rVert$, and batch normalization collapses to $t' = (v \cdot x)/\lVert v \rVert$ — the minibatch has dissolved into a deterministic quantity, the pre-activation divided by the norm of the weight vector. So in this clean case, normalizing the activations over the batch is identical to normalizing the weight by its own norm, which has nothing to do with the batch and nothing stochastic in it. The move is to build that behaviour directly into the parameterization. Let the weight that actually multiplies $x$ be
$$ w = \frac{g}{\lVert v \rVert}\, v, $$
where $v$ is a trainable direction-carrying vector and $g$ a separate trainable scalar, and run SGD directly in $(g, v, b)$ rather than in $w$. For positive $g$, $\lVert w \rVert = g$ exactly (with an unconstrained scalar, $\lVert w \rVert = |g|$ if $g$ crosses zero, the sign merely flipping the direction), so the length scale is owned solely by $g$ and the direction solely by $v/\lVert v \rVert$ — magnitude and direction are torn apart into two parameters that no longer fight over the same coordinate. This is genuinely different from earlier max-norm/weight-clipping schemes, which keep optimizing in $w$ and merely project the norm back after each step: there the optimizer still sees the plain gradient in $w$ and the projection is an afterthought that never changes the geometry the step is taken in. Here the steps are taken in $(g, v)$, so the optimizer experiences the new geometry directly. A log-scale $g = e^s$ is possible and keeps the scale positive over many orders of magnitude, but the conditioning argument came from separating direction and scale, not from curving the scale coordinate again, so the direct scalar $g$ is the simpler thing to optimize.

The payoff has to show up in the gradient the optimizer actually receives, so differentiate $L$ through $w = (g/\lVert v \rVert)v$. Writing $r = \lVert v \rVert$ with $\partial r/\partial v_j = v_j/r$, the scalar picks up
$$ \nabla_g L = \frac{\nabla_w L \cdot v}{\lVert v \rVert}, $$
the projection of the ordinary weight gradient onto the current direction — how much the gradient wants to lengthen or shorten the weight along itself. For $v$, which appears both directly and through $r$, the chain rule gives $\partial w_i/\partial v_j = g(\delta_{ij}/r - v_i v_j/r^3)$, and after recognizing $\nabla_w L \cdot v = \lVert v \rVert\,\nabla_g L$ in the second term this collapses to
$$ \nabla_v L = \frac{g}{\lVert v \rVert}\,\nabla_w L - \frac{g\,\nabla_g L}{\lVert v \rVert^2}\, v = \frac{g}{\lVert v \rVert}\,M_w\,\nabla_w L, \qquad M_w = I - \frac{w w'}{\lVert w \rVert^2}. $$
So differentiating through the reparameterization does two specific things to the ordinary gradient: it *scales* it by $g/\lVert v \rVert$, and it *projects* it onto the subspace orthogonal to the weight, since $M_w$ is the projector onto the complement of $w$ (and $v$ is collinear with $w$, so it can be written with either). That is the whole content of weight normalization on the backward pass, and it is where the conditioning comes from. If the ordinary weight gradient has covariance $C$ across the data, then the $v$-gradient has covariance $D = (g^2/\lVert v \rVert^2)\,M_w C M_w$. The load-bearing empirical fact is that $w$ tends to sit close to a dominant eigenvector of $C$ — the direction the gradient varies most across examples is often roughly the weight direction itself — so the projection removes the single biggest, most variable eigen-direction of the gradient noise, leaving $D$ flatter and closer to a scaled identity than $C$. A flatter gradient covariance is exactly better conditioning, the same whitening the natural-gradient methods buy by inverting the Fisher, here falling out for free from the parameterization.

There is a second benefit hiding in the projection. Because $\nabla_v L$ is always orthogonal to $v$ ($M_w v = 0$, so $\nabla_v L \cdot v = 0$ identically), a plain steepest-descent step $v' = v + \Delta v$ has $\Delta v \perp v$, and by Pythagoras $\lVert v' \rVert = \sqrt{1 + c^2}\,\lVert v \rVert$ with $c = \lVert \Delta v \rVert/\lVert v \rVert$. Since $\sqrt{1+c^2} \ge 1$, the norm of $v$ grows monotonically — and that is a feature. The factor multiplying the direction in the forward pass is $g/\lVert v \rVert$, so as $\lVert v \rVert$ grows the effective step on the direction shrinks; and the growth rate is set by $c$, which is large exactly when the gradients are noisy. Noisy gradients make $\lVert v \rVert$ shoot up and throttle the effective step; quiet gradients leave $c \approx 0$, $\lVert v \rVert$ stops growing, and the step stabilizes. The parameterization self-regulates its own effective learning rate, which is why a network in these coordinates tolerates a wide range of learning rates — pick one too large and $\lVert v \rVert$ simply inflates until the effective rate is sane. Strictly this monotonic-growth argument is for plain steepest descent; with momentum or a per-parameter optimizer like Adam, $\lVert v \rVert$ can grow or shrink, so it holds only qualitatively, but the self-stabilizing pull is still there.

What I lose relative to batch normalization is the automatic pinning of every layer's feature scale to unit variance, which is what makes it forgiving of bad initializations. The analytic init schemes (Glorot; He) set scales right only under distributional assumptions that drift immediately, so I fix the initial scales empirically. Sample the directions $v \sim \mathcal{N}(0, 0.05^2)$ and don't worry about their scale, because $g$ and $b$ will absorb it; then push one minibatch through, and at each neuron measure the mean $\mu[t]$ and population standard deviation $\sigma[t]$ of the normalized pre-activation $t = (v \cdot x)/\lVert v \rVert$ over the batch. Setting
$$ g \leftarrow \frac{1}{\sigma[t]}, \qquad b \leftarrow -\frac{\mu[t]}{\sigma[t]} $$
makes $g \cdot t + b = (t - \mu[t])/\sigma[t]$, so every pre-activation starts at zero mean and unit variance — the same good starting point batch normalization would give, baked into the initial $g$ and $b$ and then left to train freely. It holds exactly only for the batch used, but it puts every layer on the same footing at step zero. This is the data-dependent initialization idea of LSUV and Krähenbühl et al., applied to the $(v, g)$ coordinates; it needs a meaningful feed-forward batch, so recurrent models fall back to standard initialization.

One asymmetry remains: the reparameterization fixes the *scale* of activations but not their *mean*, since nothing in $w = (g/\lVert v \rVert)v$ centers the pre-activation. To recover the centering without dragging the noisy variance estimate back in, I optionally add a stripped-down batch normalization that subtracts the minibatch mean and nothing else — mean-only batch normalization:
$$ t = w \cdot x, \qquad \tilde t = t - \mu[t] + b, \qquad y = \varphi(\tilde t), $$
with $\mu[t]$ running-averaged for test time. On the backward pass each example's own $\tilde t$ keeps a $1$ from the $t$ term and a shared $-1/m$ from the mean over the $m$ examples, so the gradient flowing back is centered,
$$ \nabla_t L = \nabla_{\tilde t} L - \mu[\nabla_{\tilde t} L]. $$
Forward it centers the activations, backward it centers the gradients, and it is cheap. It still injects noise, but only from the *mean* estimate, which by the central limit theorem is approximately Gaussian and light-tailed — unlike full batch normalization's noise, which comes from the highly kurtotic variance estimate with its heavy-tailed spikes. So the reparameterized weights supply the clean conditioning while the mean-only centering supplies only the centering and a mild, near-Gaussian regularizing noise; bolting centering onto the old coordinates would not create the scale-and-project gradient above. And the whole thing is cheap: a convolutional filter is reused across every spatial location and example, so it has far fewer weights than pre-activations, and $\lVert v \rVert$ is computed once per filter and is deterministic — no variance in it at all, versus the full-variance minibatch estimates $\mu[t]$ and $\sigma[t]$.

In code, the forward side is a pre-forward hook that rebuilds the effective weight from its two parts before each call, recomputing $w = g \cdot v / \lVert v \rVert$ with the norm taken over every dimension except the output-channel dimension so each output neuron has its own weight vector and its own $g$; the optimizer only ever sees `weight_g` and `weight_v`. The data-dependent init is a one-shot pass setting $g \leftarrow 1/\mathrm{std}$ and $b \leftarrow -\mathrm{mean}/\mathrm{std}$ measured over the batch and spatial axes, and the optional centering is just an ordinary module on the pre-activation tensor, so the training loop stays unchanged.

```python
import torch
import torch.nn as nn


def _norm_except_dim(v, dim):
    # dim=None uses one global norm; otherwise keep one norm per selected axis.
    if dim is None:
        dim = -1
    if dim == -1:
        return v.norm()
    if dim < 0:
        dim += v.dim()
    perm = [dim] + [d for d in range(v.dim()) if d != dim]
    flat = v.permute(*perm).reshape(v.size(dim), -1)
    n = flat.norm(2, dim=1)
    shape = [1] * v.dim()
    shape[dim] = v.size(dim)
    return n.reshape(shape)


def _compute_effective_weight(module, name, dim):
    g = getattr(module, name + "_g")           # magnitude per output channel
    v = getattr(module, name + "_v")           # direction
    return v * (g / _norm_except_dim(v, dim))  # w = g * v / ||v||


class _WeightReparameterization:
    def __init__(self, name, dim):
        self.name, self.dim = name, dim

    def __call__(self, module, _inputs):       # runs right before forward()
        setattr(module, self.name, _compute_effective_weight(module, self.name, self.dim))


def reparameterize_weight(module, name="weight", dim=0):
    """Reparameterize module.<name> into magnitude (<name>_g) and direction (<name>_v);
    SGD then runs in those. dim=0 -> one magnitude per output channel; dim=None -> one."""
    if dim is None:
        dim = -1
    w = getattr(module, name)
    del module._parameters[name]
    module.register_parameter(name + "_g", nn.Parameter(_norm_except_dim(w, dim).data))
    module.register_parameter(name + "_v", nn.Parameter(w.data))
    setattr(module, name, _compute_effective_weight(module, name, dim))
    module.register_forward_pre_hook(_WeightReparameterization(name, dim))
    return module


@torch.no_grad()
def data_dependent_init(layer, x):
    """One minibatch -> g <- 1/std, b <- -mean/std, per output channel."""
    getattr(layer, "weight_g").fill_(1.0)
    if layer.bias is not None:
        layer.bias.zero_()
    t = layer(x)
    reduce = [d for d in range(t.dim()) if d != 1]
    mean = t.mean(dim=reduce)
    shape = [1, -1] + [1] * (t.dim() - 2)
    centered = t - mean.reshape(shape)
    std = (centered.pow(2).mean(dim=reduce) + 1e-10).sqrt()
    getattr(layer, "weight_g").copy_((1.0 / std).reshape(getattr(layer, "weight_g").shape))
    if layer.bias is not None:
        layer.bias.copy_((-mean / std).reshape(layer.bias.shape))
    return layer


class ActivationTransform(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.momentum = momentum

    def forward(self, t):
        reduce = [d for d in range(t.dim()) if d != 1]
        if self.training:
            mu = t.mean(dim=reduce)
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mu.detach())
        else:
            mu = self.running_mean
        shape = [1, -1] + [1] * (t.dim() - 2)
        return t - mu.reshape(shape) + self.bias.reshape(shape)


# usage
conv = reparameterize_weight(nn.Conv2d(3, 96, 3, padding=1), name="weight", dim=0)
center = ActivationTransform(96)
# data_dependent_init(conv, first_minibatch)   # one-shot before training
# then train with Adam; the pre-forward hook rebuilds `weight` each forward.
```
