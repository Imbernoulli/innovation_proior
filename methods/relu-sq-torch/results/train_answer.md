The bill for training Transformer language models is dominated by compute, and that compute scales hard with model size, so the question I care about is not raw step time but compute-to-quality: how much loss I can shave for a fixed number of training tokens. Inside a decoder block the position-wise feed-forward network is the fatter consumer by parameter count — it projects each token up to four times the model width, applies a pointwise nonlinearity, and projects back down. The two matmuls are load-bearing and I cannot make them cheaper without changing what the layer computes, but the activation wedged between them is special: it has no parameters of its own, and swapping it costs essentially nothing in FLOPs. It is the one free lever on quality, and any gain there multiplies across every layer of every model in a scaling sweep. So I hold the FFN structure fixed, treat the middle activation as the design slot, and ask what shape of nonlinearity squeezes more sample efficiency out of the same two matmuls.

The shapes already on the shelf disappoint on inspection. ReLU, $\max(z,0)$, is exactly the identity for $z>0$; GELU, $z\,\Phi(z)$, has $\Phi(z)\to 1$ so $\text{GELU}(z)\to z$; Swish, $z\,\sigma(\beta z)$, has $\sigma(\beta z)\to 1$ so it too approaches $z$. All three are asymptotically linear — a strongly-firing unit and a barely-firing one get pushed through at nearly the same slope $\approx 1$, and the smooth variants differ from ReLU only near the origin, at the knee. That is precisely why GELU only barely beats ReLU on language-model perplexity: reshaping the knee is reshuffling deck chairs around $z=0$ while the bulk of the function stays linear. If I want a real change, the lever has to be the asymptotics, not the knee. A separate tradition does change the game: the gated linear units, which multiply two linear projections of the input instead of squashing one. Writing $p=xW$ and $q=xV$, the local variation of $p\otimes\sigma(q)$ contains a term $dp\otimes\sigma(q)$ — a route through the linear branch scaled only by the gate value, with no derivative-of-nonlinearity factor crushing it, an un-attenuated multiplicative path through depth. Shazeer's GLU variants ran with this for the FFN, e.g. $\text{FFN}_{\text{ReGLU}}(x)=(\max(xW,0)\otimes xV)\,W_2$, and GEGLU/SwiGLU came out ahead of the plain ReLU/GELU/Swish FFNs on held-out perplexity. But the gain is not free: each variant has a *third* weight matrix $V$, and to stay parameter- and FLOP-matched they shrink the inner width $d_{ff}$ by a factor of $2/3$. That violates my one constraint — change only the activation, no new weights, no shrunk dimension, no retune — on the first line.

So I hold two facts side by side: the multiplicative interaction is what buys the perplexity, and its price is a whole extra projection matrix. A product needs two factors, and in ReGLU those factors are two different linear maps $xW$ and $xV$. What if they were the *same* map? I propose squared ReLU — call it ReLU² — the rectified quadratic $\text{act}(z)=\max(z,0)^2$, applied between the FFN's two matmuls, so that
$$\text{FFN}(x, W_{fc}, W_{proj}) = \big(\,\text{relu}(x\,W_{fc}^\top)^2\,\big)\,W_{proj}^\top.$$
The reason this is exactly the GLU benefit with one matrix instead of two is a short derivation. Take a ReGLU hidden unit, $\max(u^\top x,0)\cdot(v^\top x)$, and force the two weight vectors equal, $u=v$. Let $z=u^\top x$ be the pre-activation the up-projection already hands me. If $z>0$, then $\max(z,0)=z$ and the unit is $z\cdot z=z^2$; if $z\le 0$, then $\max(z,0)=0$ and the product is $0$. Tying the weights collapses ReGLU's unit into "$z^2$ when $z>0$, else $0$," which is exactly $\text{relu}(z)^2$. The second multiplicative factor was never something I needed a new matrix for — it was sitting in the up-projection's own output. So I recover the same second-order interaction with one matrix, no $V$, no $2/3$ width shrink, a literal drop-in into the existing two-matrix FFN, parameter-matched by construction and tested under the same learning-rate schedule.

What makes it work, beyond cost, is that I have changed the asymptotics on purpose. As $z\to\infty$, $\text{relu}(z)^2\sim z^2$, super-linear where ReLU/GELU/Swish all top out at slope $\approx 1$. Its derivative is
$$\frac{d}{dz}\,\text{relu}(z)^2 = 2\,\text{relu}(z),$$
where the indicator $\mathbf{1}[z>0]$ is subsumed by $\text{relu}$ — on the dead half $\text{relu}(z)=0$ already, so $2\,\text{relu}(z)\,\mathbf{1}[z>0]=2\,\text{relu}(z)$ exactly. This gradient grows linearly with the pre-activation and is unbounded above: a unit firing hard sends back a proportionally larger gradient, whereas GELU and Swish saturate near a constant slope so confident and timid units are treated alike. The activation is therefore not just shaping the forward pass; it tells the optimizer that confident, strongly-active units matter more and should be pushed harder, and it does so for free because the magnitude is already in $z$. That is a genuine inductive bias.

Two design choices fix the exact form. Power 2, not 3 or 4, because it is the *minimal* super-linear rectified polynomial: a power-$n$ unit has gradient $n\,\text{relu}(z)^{n-1}$, so $n=3$ already gives quadratic gradient growth $3z^2$ that overflows much sooner in bf16, while the forward $z^n$ blows up faster too. Rectified polynomials are sharper for higher $n$ — the associative-memory line of Krotov and Hopfield uses exactly $z^n$ to sharpen each stored pattern and raise capacity — but sharper cuts both ways for an activation I must train in low precision, and $2$ is the smallest, safest step away from linear that still gives the multiplicative term, with gradient growth only linear. And I rectify rather than use a plain $z^2$, because $z^2$ is even, $z^2=(-z)^2$, firing identically for $+z$ and $-z$; it is non-monotonic, dropping then rising through the origin, and it discards the sign of the pre-activation entirely, collapsing two inputs the up-projection deliberately separated by sign onto the same output. $\text{relu}(z)^2$ is monotone non-decreasing — flat-zero on the negative half, rising on the positive half — so it keeps ReLU's "off" behavior and only curves the active half upward. The rectification is what preserves the gate while the square adds the shaping. Seen this way, $\text{relu}(z)^2$ is the leanest possible degree-2 rectified multiplicative interaction — the same higher-order structure that ReGLU, GEGLU, and even the $x^3$ term in the tanh-approximate GELU all carry — stripped to a single self-multiplication with no parameters.

I also want this to be fast, since an activation that swaps in for GELU should avoid GELU's expensive elementwise path. Rectify-and-square is already cheaper per element than an $\text{erf}$ or $\tanh$, but rather than let autograd build a graph through the elementwise chain I hand-write the backward and save exactly what I need. With row-batched $x\in\mathbb{R}^{N\times d}$, $W_{fc}\in\mathbb{R}^{4d\times d}$, $W_{proj}\in\mathbb{R}^{d\times 4d}$, and $h=xW_{fc}^\top$, $r=\text{relu}(h)$, $a=r^2$, $y=aW_{proj}^\top$, given the upstream $g=\partial L/\partial y$, the gradients are
$$\frac{\partial L}{\partial a}=g\,W_{proj},\qquad \frac{\partial L}{\partial h}=2\,\text{relu}(h)\odot\frac{\partial L}{\partial a},$$
$$\frac{\partial L}{\partial W_{proj}}=g^\top a,\qquad \frac{\partial L}{\partial W_{fc}}=\Big(\frac{\partial L}{\partial h}\Big)^\top x,\qquad \frac{\partial L}{\partial x}=\frac{\partial L}{\partial h}\,W_{fc}.$$
The shapes check: $g\,W_{proj}$ is $(N,d)\cdot(d,4d)\to(N,4d)$ matching $a$; $g^\top a$ is $(d,N)\cdot(N,4d)\to(d,4d)$ matching $W_{proj}$; $(\partial L/\partial h)^\top x$ is $(4d,N)\cdot(N,d)\to(4d,d)$ matching $W_{fc}$; and $(\partial L/\partial h)\,W_{fc}$ is $(N,4d)\cdot(4d,d)\to(N,d)$ matching $x$. The identity is exact at the origin as well, where $\text{relu}(0)=0$ selects the subgradient $0$. So the only nonlinear-specific value the backward needs is $r=\text{relu}(h)$; everything else is three matmuls I would pay regardless, plus one multiply-by-two-and-multiply, far cheaper than differentiating an $\text{erf}$. I wrap this in a custom autograd Function whose forward stashes $(x, W_{fc}, W_{proj}, h, \text{relu}(h))$ — mirroring the torch baseline, which keeps both the pre-activation and $\text{relu}(h)$ — and whose backward executes exactly this derivation. One precision detail: the forward runs in bf16 under autocast, so I cast the saved tensors and weights to the gradient's dtype before the backward matmuls and the $2\,\text{relu}(h)$ multiply, to avoid silently mixing a bf16 weight with an fp32 grad. I keep the activation clean and let the surrounding optimizer handle scale rather than clamping, since a clamp would change the function I just derived; power 2 is already the controlled step, with gradient growth only linear in $\text{relu}(h)$.

The activation itself, as used in the FFN — `square(relu(x))`, the annotated PyTorch form identical to the mesh-tensorflow primitive:

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class SquaredReLU(nn.Module):
    """act(z) = max(z, 0)^2."""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        return x * x
```

And the working FFN core with the hand-written backward, a drop-in for a GELU FFN that is faster and adds no parameters:

```python
import torch
from torch.nn import functional as F


def fused_mlp_forward(x, w_fc, w_proj):
    """FFN core: up-project -> squared ReLU -> down-project, with a custom backward.

    x:      (B*T, n_embd)
    w_fc:   (4*n_embd, n_embd)
    w_proj: (n_embd, 4*n_embd)
    returns (B*T, n_embd)
    """

    class ReLUSquaredMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w_fc, w_proj):
            h = x @ w_fc.t()              # pre-activation z
            relu_h = F.relu(h)           # r = relu(z)
            act = relu_h * relu_h        # a = relu(z)^2
            out = act @ w_proj.t()
            ctx.save_for_backward(x, w_fc, w_proj, h, relu_h)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, w_fc, w_proj, h, relu_h = ctx.saved_tensors
            dtype = grad_output.dtype
            d_act = grad_output @ w_proj.to(dtype)        # dL/da = g @ W_proj
            d_h = 2 * relu_h.to(dtype) * d_act            # d/dz[relu(z)^2] = 2*relu(z)
            act_sq = (relu_h * relu_h).to(dtype)          # a = relu(z)^2
            grad_w_proj = grad_output.t() @ act_sq        # dL/dW_proj = g^T @ a
            grad_w_fc = d_h.t() @ x.to(dtype)             # dL/dW_fc   = d_h^T @ x
            grad_x = d_h @ w_fc.to(dtype)                 # dL/dx      = d_h @ W_fc
            return grad_x, grad_w_fc, grad_w_proj

    return ReLUSquaredMLP.apply(x, w_fc, w_proj)
```
