The cost in a Transformer decoder block sits mostly in the position-wise feed-forward network, not in attention: the FFN projects up to a hidden width of $4d$, applies a pointwise nonlinearity, and projects back down, so its two matrices alone are $8d^2$ parameters against attention's $4d^2$. Yet almost all design effort goes to attention, and the FFN is just $\mathrm{act}(xW_1)W_2$ where the only nonlinear element is the single scalar function $\mathrm{act}$ — and that function is invariably inherited rather than chosen: ReLU because the original Transformer used it, or GELU because BERT and the GPT line switched and the codebase followed. If I want a cheap win in autoregressive language-model pretraining, that one pointwise map is the most under-examined knob in the whole architecture. The constraint I impose on myself is strict: change only $\mathrm{act}$, keep the $(B,T,d)\to(B,T,d)$ shape contract, leave attention, normalization, the optimizer, and the data untouched, and ideally add no parameters and no new hyperparameter, so the result is a one-line edit anyone can drop into an existing codebase and simply run.

Looking hard at the incumbents shows they all share a hidden assumption. ReLU is $\max(0,z)$: it kills the negatives — which is real work, since zeroing absent features is what makes the layer sparse — but its positive branch is literally the identity $z$, so above zero the FFN is a plain linear map with a hard kink at the origin, and a strongly-firing unit is treated on the same linear scale as one barely on. GELU, $z\,\Phi(z)$ with $\Phi$ the Gaussian CDF, smooths that kink and adds a gentle dip below zero, but $\Phi(z)\to 1$ as $z$ grows, so $\mathrm{GELU}(z)\to z$: asymptotically it too is the identity. Swish, $z\,\sigma(\beta z)$ — the winner of an actual search over scalar activations, which already tells me searching this slot pays off — is self-gated and smooth, and again $\sigma(\beta z)\to 1$ gives $\mathrm{Swish}(z)\to z$. Every activation that displaced ReLU lives in one family: smooth near zero, and $\mathrm{act}(z)/z\to 1$ on the positive side. They disagree only on how gently to treat the region around the origin and agree completely that a big positive signal should pass through roughly unchanged. The unexamined assumption is that the positive branch should be (nearly) linear, and I see no reason it must be.

I propose Squared ReLU (ReLU²): replace the FFN's activation with $\mathrm{act}(z)=\max(0,z)^2$ — rectify, then square. Question the linearity assumption directly. When a unit is strongly activated, the network is expressing high confidence in a feature; a linear positive branch says "twice the evidence, twice the output" and gives no way for a few loud units to dominate beyond their linear share. A faster-than-linear positive branch instead lets the loud units pull ahead of the quiet ones, sharpening the layer into a more selective, contrastive map while still killing the negatives and keeping the sparsity. There is precedent that this has a principled upside: Krotov and Hopfield's dense associative memories use a rectified-polynomial energy $F_n(x)=x^n$ for $x\ge 0$ and $0$ otherwise, where increasing $n$ sharpens each energy term and lets the network pack more memories — a capacity $K^{\max}=\alpha_n N^{n-1}$ that grows with $n$ — and the open question they pose is exactly whether, above the threshold, the output should grow linearly, sub-linearly, or faster than linearly. That reasoning lives in shallow energy-based models on toy data and never in a deep Transformer FFN; that gap is the opening.

Which polynomial? The natural family is $f_p(z)=\max(0,z)^p$. Since I am about to stack this through dozens of layers, I want the smallest step beyond linear, perturbing the known-good architecture as little as possible. The case $p=1$ is just ReLU; the next integer is $p=2$, giving $\max(0,z)^2$ — the minimal faster-than-linear rectified polynomial. The rectification is not optional. A bare $z^2$ is even: it forgets the sign of $z$, so a strongly negative pre-activation produces the same large output as a strongly positive one, turning "feature absent" ($z<0$) into "very present" and destroying the gating that made ReLU useful. So the order is forced — rectify first to keep off units off, then square the surviving positive part — which is exactly $\max(0,z)^2$, not the parabola. Examining the function confirms the regime is genuinely new: it is $0$ for $z\le 0$ and $z^2$ for $z>0$, with derivative $0$ for $z<0$ and $2z$ for $z>0$, so it is continuous in value at the origin and, unlike ReLU, continuous in slope too ($2z\to 0$ matches the left side), leaving zero smoothly with zero initial slope and then accelerating. Crucially $\mathrm{act}(z)/z = z\to+\infty$: it grows quadratically without bound relative to its input, the deep-network analogue of the "spikier above threshold packs more capacity" story.

What makes the design self-justifying is that this same function is the tied special case of the gated-linear-unit family. ReGLU's $i$-th hidden unit is $\max(0,(xW)_i)\cdot(xV)_i$ — a ReLU-gated projection times a second, independent projection. Force the two affine maps to coincide, $V=W$ (and tie the biases when present), so both use the shared pre-activation $z_i=(xW+b)_i$. Then for $z_i>0$ the gate is $z_i$ and the product is $z_i\cdot z_i=z_i^2$; for $z_i\le 0$ the gate is $0$ and the product is $0$. Together,

$$\max(0,z)\cdot z \;=\; \max(0,z)^2 \quad\text{for all } z,$$

so squared ReLU applied after an affine projection is precisely ReGLU with gate and value tied:

$$\mathrm{ReGLU}(x)=\max(0,xW+b)\otimes(xV+c),\qquad V=W,\ c=b \;\Longrightarrow\; \big(\mathrm{ReLU}(xW+b)\big)^2.$$

That tying is the whole point. ReGLU, GEGLU, and SwiGLU each use three weight matrices — a gate projection $W$, a value projection $V$, and the down-projection — so to match parameters and FLOPs against the two-matrix baseline the GLU configurations shrink the hidden width $d_{\!f\!f}$ by a factor of two-thirds (3072 to 2048 in the T5-base comparison). Tying the gate and value collapses the extra projection, leaving two matrices and the full $4d$ width: no third matrix, no width reduction, no new hyperparameter. I keep a genuine degree-2 multiplicative interaction — a product of two copies of the same pre-activation, the cheapest possible degree-2 term, which strictly enriches the representable function class versus a purely additive layer — and a nonsaturating positive branch rather than a sigmoid gate whose derivative can vanish through depth. And it is cheaper per element than GELU or Swish: no $\mathrm{erf}$, no $\exp$, no $\sigma$, just a rectify and a square.

Stopping at $p=2$ rather than chasing higher powers is deliberate. Higher $p$ grows faster and would dominate harder, but $\max(0,z)^p$ for larger $p$ is numerically meaner — moderately large activations overflow in low precision, small positive ones underflow toward zero, and the dynamic range degrades through a deep bf16 stack. Only $p=2$ is the minimal jump out of the linear-asymptote family and the one that coincides exactly with tied ReGLU; the cubic and higher are products of three or more copies, a different object without the clean single-product reading. So degree 2 is the sweet spot where the asymptotic argument, the ReGLU special case, and numerical stability all point at the same function. The squaring sits exactly where the old activation sat, pointwise on the $4d$ hidden state between the up-projection $c_{\mathrm{fc}}$ (which produces $z=xW_1$) and the down-projection $c_{\mathrm{proj}}$; dropout and biases are inherited from the surrounding FFN. The final form is

$$\mathrm{FFN}(x)=\big(\max(0,\,xW_1+b_1)\big)^2 W_2 + b_2,\qquad W_1\!:\,d\to d_{\!f\!f},\ W_2\!:\,d_{\!f\!f}\to d,\ d_{\!f\!f}=4d,$$

a one-line replacement of the inherited activation, `x = F.relu(x).square()`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Transformer feed-forward sublayer with a squared-ReLU (ReLU squared) activation.

    Up-project to 4x width, apply max(0, z)^2 pointwise, project back down.
    Two matrices and the full 4x hidden width -- no extra parameters versus a
    ReLU/GELU FFN. Equivalent to ReGLU with tied gate/value affine maps.
    Maps (B, T, n_embd) -> (B, T, n_embd).
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)            # z = x W1 : up-projection to the 4x hidden state
        x = F.relu(x).square()      # max(0, z)^2  (rectify keeps sparsity, then square)
        x = self.c_proj(x)          # down-projection back to n_embd
        x = self.dropout(x)
        return x
```
