The activation function that follows each linear map matters a great deal for how a deep network trains, yet the field keeps falling back on ReLU, $\max(x,0)$, because every hand-designed replacement gives gains that are *inconsistent* across models and datasets. Leaky ReLU adds a small negative slope $\alpha x$ to keep the dead half alive; PReLU learns that slope per channel; ELU and SELU push the mean toward zero with $\alpha(e^x-1)$ on the negative side and fixed self-normalizing constants; softplus $\log(1+e^x)$ smooths the kink. Each fixes one perceived flaw and each fails to displace ReLU — the best non-ReLU baseline changes from one model to the next. That recurring inconsistency is itself the clue: it means we do not actually know *which* properties of an activation matter, so arguing from a property wishlist (don't die on the negative side, saturate gently, center the mean) is guessing. I would rather not guess. The honest move is to let a validation signal choose scalar functions directly, then inspect the shapes it prefers and ask which "essential" ReLU properties were truly necessary.

I propose searching the space of scalar activations and report what falls out of it: the self-gated unit I call Swish, $f(x) = x\,\sigma(\beta x)$, with $\beta=1$ recovering the unit already known as SiLU. The reason to restrict the search to scalar functions — one real input to one real output — is that this is the only family that is a literal drop-in: swap it in for ReLU anywhere and the network is identical except for the pointwise map, no shape changes, no architecture surgery. To make the space both expressive and searchable I build candidates by composing a small library of primitives, borrowing the trick that worked for searching optimizer update rules. One core unit computes $b(u_1(\text{src}_1), u_2(\text{src}_2))$: pick two sources, pass each through a unary function, and combine the two scalars with a binary function. Unaries are things like $x,\ -x,\ x^2,\ \sqrt{x},\ e^x,\ \sin,\ \sigma,\ \tanh,\ \max(x,0),\ \beta x,\ \beta$ (with $\beta$ a trainable scalar, possibly per channel); binaries are $x_1+x_2,\ x_1\cdot x_2,\ x_1-x_2,\ \max(x_1,x_2),\ x_1/x_2$ and so on. Stacking units lets later units read earlier scalar outputs, and everything stays pointwise. One unit is enumerable, but a few units blow the space up to roughly $10^{12}$ candidates, so there are two regimes: for small spaces, exhaustively enumerate — train a child network with each candidate and rank by validation accuracy; for large ones, use an RNN controller that autoregressively emits the tokens of a function string and is trained with policy-gradient RL (PPO, with an exponential-moving-average of past rewards as the baseline to cut variance), where the reward is the validation accuracy of a child network trained with the candidate. The expensive part is that every candidate costs a full child-network run, so the child is kept small and cheap — ResNet-20 on CIFAR-10, around 10K steps — and the work is parallelized: the controller proposes a batch onto a queue, workers train and report back. The transfer risk is real and has to be checked outside the search loop, but cheap-child-then-validate-large is the only affordable design.

What comes back is more instructive than what I expected. The useful candidates are not tangled many-unit expressions but one- or two-unit curves, which makes sense: extra depth produces wilder derivatives, discontinuities from unsafe division, and oscillations a deep net has to optimize through. The stronger pattern is that good candidates keep the raw preactivation alive until the final combine, something of the form $b(x, g(x))$ — and ReLU itself fits that template as $\max(x,0)$ with $g(x)=0$, $b=\max$. The clean path for $x$ is worth preserving even once I stop hand-designing the curve. Now write ReLU the other way, as $x\cdot\mathbb{1}(x>0)$: the input times a hard gate. A hard step is exactly the kind of non-smooth hand choice I wanted to stop assuming, and its smooth bounded substitute is a sigmoid; if the gate should still depend on the preactivation's sign and scale, it should see $\beta x$. In the core-unit notation this is a single unit with $u_1(x)=x$, $u_2(x)=\sigma(\beta x)$, and $b(a,b)=a\cdot b$, which lands on
$$f(x) = x\,\sigma(\beta x),$$
the input gated by a sigmoid of a scaled copy of itself — a self-gated unit. The $\beta$ knob is the whole story. For positive $\beta$, as $\beta\to\infty$ the gate $\sigma(\beta x)$ becomes the step $\mathbb{1}(x>0)$, so $x\,\sigma(\beta x)\to\max(x,0)$ and Swish becomes ReLU; as $\beta\to 0$, $\sigma(\beta x)\to\tfrac12$, so $f\to x/2$ and Swish becomes a scaled line. The same formula thus interpolates between a line and a soft ReLU, and a trainable $\beta$ — constant, or per channel — lets each channel set its own gate sharpness instead of fixing it by hand. At $\beta=1$ this is exactly SiLU, $x\,\sigma(x)$.

The shape is what overturns the ReLU dogma. Like ReLU, Swish is unbounded above and so does not saturate on the positive side; unlike ReLU it is smooth and non-monotonic, dipping slightly below zero for small negative $x$ before returning toward zero as $x\to-\infty$. That undershoot is not something the usual monotonic checklist would suggest, but the formula makes it unavoidable: a negative input times a positive gate is negative, and the gate shrinks only gradually. The derivative is where the dogma actually cracks. With $s=\sigma(\beta x)$ and $f=xs$,
$$f'(x) = s + \beta x\, s(1-s) = \beta f + \sigma(\beta x)\bigl(1 - \beta f\bigr),$$
so $\beta$ controls how fast $f'$ approaches $0$ on the far-negative side and $1$ on the far-positive side. At $\beta=1$ the derivative's magnitude stays below $1$ until roughly $x\approx1.28$; over much of the active range this unit does *not* preserve gradients at unit scale the way ReLU does on its positive half. That separates two ideas I used to conflate: a non-saturating positive tail is genuinely useful, but an exactly unit derivative over all positive inputs is not forced by the search space — and in residual networks the identity path already carries gradients, so the activation need not bear that burden alone. Two practical cautions fall straight out. The forward pass is the one-line pointwise map $x\cdot\sigma(\beta x)$; if BatchNorm precedes it, keep BatchNorm's learnable scale on, because ReLU's positive homogeneity can make that scale look redundant whereas this smooth self-gated curve is not scale-equivariant in the same way. And since the derivative is often smaller than ReLU's unit positive slope, a learning rate tuned around ReLU should be re-tuned rather than blindly reused.

```python
import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish / SiLU: x * sigmoid(beta * x)."""

    def __init__(self, num_channels=None, trainable_beta=False, beta_init=1.0):
        super().__init__()
        if trainable_beta:
            shape = (num_channels,) if num_channels is not None else (1,)
            self.beta = nn.Parameter(torch.full(shape, float(beta_init)))
            self.trainable = True
        else:
            self.register_buffer("beta", torch.tensor(float(beta_init)))
            self.trainable = False

    def forward(self, x):
        if self.trainable and self.beta.numel() > 1:
            shape = [1] * x.dim()
            shape[1] = -1
            b = self.beta.view(*shape)
        else:
            b = self.beta
        return x * torch.sigmoid(b * x)


class SiLU(nn.Module):
    """SiLU == Swish-1: x * sigmoid(x)."""

    def forward(self, x):
        return x * torch.sigmoid(x)
```
