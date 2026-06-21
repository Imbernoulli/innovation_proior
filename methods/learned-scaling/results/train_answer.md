Depth should be free and it isn't, and that gap is the whole problem. The expressive power of a network grows roughly exponentially in its depth, and a deeper model can always represent everything a shallower one can simply by setting the extra layers to the identity, so stacking layers onto a working model should at worst leave it where it was. In practice the opposite happens: past a few dozen layers the optimizer stalls, and a deeper *plain* network reaches *higher training error* than a shallower one. That is not a generalization gap but an optimization failure — the solution that copies the shallow net and idles the rest is representationally available, yet gradient descent cannot find it. The obstruction lives in how signals and gradients move through depth. Picture a stack of width-preserving maps $x_{i+1} = F[W_i](x_i)$; to first order each layer multiplies an input perturbation by its Jacobian, so if one layer rescales a perturbation by a factor $r$, after $L$ layers it is rescaled by $r^L$, and the backward pass is governed by the same product of Jacobians transposed. Since $r^L$ is brutal in $L$ — exploding for $r$ slightly above $1$, vanishing for $r$ slightly below — the only survivable regime is $r \approx 1$ held at *every* layer.

The sharp version of $r \approx 1$ comes from the signal-propagation picture. Tracking the cosine $x_i \cdot x'_i / (\|x_i\|\,\|x'_i\|)$ of two inputs as they propagate, the network flows to an ordered fixed point at $1$ (all inputs squashed to one direction, gradients vanish) or a chaotic fixed point at $0$ (nearby inputs flung apart, gradients explode); trainability is the edge of chaos between them. Diagnosing it through the input-output Jacobian $J_{io} \equiv \partial x_L / \partial x_0$, the mean squared singular value $\chi \approx 1$ is that edge — but $\chi \approx 1$ on average is *not enough*, because a spectrum that averages to one while spreading from $10^8$ to $10^{-8}$ will amplify some data directions and annihilate the rest. The right condition is the stronger one of *dynamical isometry*: **all** singular values of $J_{io}$ close to $1$, so every input perturbation crosses the network with gain $\approx 1$ and $r^L = 1^L = 1$ exactly. The trouble is that this cannot in general be reached by initialization. A ReLU maps a half-space to zero, producing a structurally zero singular value no matter the weights, so dynamical isometry is impossible for ReLU. The Transformer is worse: LayerNorm, $\mathrm{LayerNorm}(x) = (x - \mathbb{E}[x])/\sqrt{\mathrm{Var}(x)} \cdot \gamma + \beta$, is by construction blind to perturbations that only shift the mean or rescale the variance, contributing $2n$ zero singular values per layer across $n$ tokens; and self-attention $\mathrm{softmax}(QK^\top/\sqrt{d})\cdot V$ in the small-score regime returns a nearly uniform row $\approx 1/n$ that averages all $n$ value vectors into about $d$ surviving directions. Stacked, the deep Transformer's Jacobian has a huge fraction of its singular values decayed to machine precision at init, which is why the field props deep nets up with normalization, careful per-layer init recipes, and learning-rate warm-up.

Lining up the prior residual schemes, every one leaks in the same place. The plain residual $x_{i+1} = \sigma(x_i + F(x_i))$ initializes $F$ at full strength, so at step zero the block is $x + F(x)$ with $F(x)$ not small — the block is *not* the identity at init and its output variance still compounds with depth, roughly a constant factor per layer; the shortcut tames the worst of $r^L$ but does not set $r = 1$. Highway Networks $x_{i+1} = C(x)\,x_i + T(x)\,F(x_i)$ with $T = \sigma(W_T^\top x + b_T)$, $C = 1 - T$ expose the right knob — how much of $F$ enters the stream — but the gate is a full function of $x$ with its own weights, and a finite negative bias only *leans* toward carrying; the gated-ResNet simplification ($W_T = 0$, $b_T = \alpha$, $C = 1-T$) strips it to a scalar but the sigmoid transform/carry split still only approaches $T=0$, $C=1$, never reaches it. The zero-$\gamma$ trick zero-inits a trailing norm layer's scale so the block does start as the identity, but only where such a norm exists and at the cost of zero-initializing a whole channel vector, welded to normalization. FixUp comes top-down — demanding one SGD step change the function by $\Theta(\eta)$ independent of depth — but the cure is an elaborate, architecture-aware bundle (zero-init the last branch layer, rescale branch weights by $L^{-1/(2m-2)}$, add a per-branch scalar multiplier initialized at *one*). Either the block is not the identity at init, or identity-at-init is bought through architecture-specific multi-piece machinery. And warm-up itself is a band-aid: it exists only because the Post-LN Transformer has large parameter gradients near the output at init, so a full learning rate immediately destabilizes it.

What the dynamical-isometry condition actually asks for is staring back: make the block *exactly* the identity at initialization, for *any* $F$, including ReLU and self-attention whose own Jacobians have vanishing singular values — because if the block is the identity, $F$ need not be isometric at all. I propose ReZero (residual with zero initialization). Put a single learnable scalar in front of each residual branch and start it at zero,

$$x_{i+1} = x_i + \alpha_i\, F(x_i), \qquad \alpha_i = 0 \text{ at initialization}.$$

At init every $\alpha_i = 0$, so $x_{i+1} = x_i$ exactly, every block is the identity, $J_{io} = I$, and dynamical isometry holds trivially and exactly — completely indifferent to $F$, because $\alpha = 0$ annihilates $F$'s entire output, vanishing singular values and all. One learnable scalar per layer: not data-dependent like Highway's gate, not a vector like zero-$\gamma$, not a depth-dependent rescaling like FixUp, but the minimal object that makes the block an identity at step zero while leaving $F$ free to wake up later, since $\alpha$ is trainable.

The crucial question is whether this actually *trains*, and fast — the obvious worry being that if $\alpha = 0$ kills $F$'s output it might freeze $F$'s gradient forever. The smallest model that still carries the $r^L$ pathology settles it. Take $L$ single-neuron layers sharing one weight $w$ and one scalar $\alpha$: each layer does $x_{i+1} = (1 + \alpha w) x_i$, so $x_L = (1 + \alpha w)^L x_0$ and $J_{io} = (1 + \alpha w)^L$, which is precisely an $r^L$. Compare the two initializations. With $\alpha = 1$, $w \approx 1$ — the vanilla residual — $J_{io} = 2^L$, and the weight update through the chain rule,

$$w \leftarrow w - \lambda\, L\, \alpha\, x_0\, (1 + \alpha w)^{L-1}\, \partial_x C\big|_{x = x_L},$$

carries a $(1+w)^{L-1} = 2^{L-1}$ factor, forcing a learning rate $\lambda \propto L^{-1}(1+w)^{-(L-1)}$ that is *exponentially small in depth* and balanced on a knife edge. Now set $\alpha = 0$. First, $J_{io} = 1$, the input signal is exactly preserved. Second, that same weight update has an explicit $\alpha$ out front, so at $\alpha = 0$ the gradient on $w$ is *zero* — the deep stack of weights receives no giant ill-conditioned first step, $F$ holds still while everything is at the identity. But the system is not frozen, because $\alpha$'s own gradient is alive: differentiating $x_L = (1+\alpha w)^L x_0$ with respect to $\alpha$ gives

$$\alpha \leftarrow \alpha - \lambda\, L\, w\, x_0\, (1 + \alpha w)^{L-1}\, \partial_x C\big|_{x = x_L},$$

which at $\alpha = 0$ is $\alpha \leftarrow -\lambda L w x_0\, \partial_x C|_{x=x_0}$ — finite, nonzero, and crucially evaluated at $x_L = x_0$, the well-conditioned input itself rather than at some $2^L$-amplified output. So $\alpha$ wakes up first on a clean gradient, grows just enough to keep $1 + \alpha w$ near one, and routes the trajectory *around* the poorly-conditioned $\alpha \approx 1$ ridge that the vanilla residual is forced to sit on; only once $\alpha$ is nonzero does $w$ begin receiving gradient and the layer come online. The remaining explicit depth factor in that first safe update is just $L$, so the learning rate scales polynomially, such as $1/L$, instead of carrying the exponential penalty. This is exactly why the init must be *zero* and not one or merely small: $\alpha = 1$ is the pathology itself, and any nonzero init plants the trajectory off the safe identity point onto a surface where conditioning already depends on $\alpha w$. Zero is the unique value giving exact identity, exact dynamical isometry, and the dead-$w$-gradient-but-live-$\alpha$-gradient structure that lets the network bootstrap gently.

Carried into the Transformer, each layer's two sublayers — multi-head self-attention and the point-wise feed-forward network — each have their contribution multiplied by a learnable residual weight before re-entering the stream, $x_{i+1} = x_i + \alpha_i \cdot \mathrm{sublayer}(x_i)$ with $\alpha_i = 0$ at init. A single shared $\alpha_i$ across both sublayers suffices, since one scalar at zero already zeroes the whole block at init; splitting it would add parameters without changing that behavior. LayerNorm is removed rather than kept — its job of controlling signal scale is now done by the $\alpha$-gated identity, and it was actively contributing $2n$ vanishing singular values per layer — and warm-up falls away because an identity-initialized block has well-behaved gradients at step zero by construction. One practical caution: the $\alpha$'s sit at a highly leveraged point, so a single large step swings a whole layer's contribution; under aggressive schedules they want a gentler, small and steady learning rate. As a sanity check the $|\alpha_i|$ should grow from zero while staying modest (a natural scale at depth $L$ being $O(1/L)$), and the $J_{io}$ singular-value histogram should stay concentrated near one. The result drops into the residual stream as one zero-initialized `resweight` per layer, shared by attention and feed-forward, multiplying each sublayer output before it is added back.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention


class RZTXEncoderLayer(nn.Module):
    """ReZero Transformer encoder layer: each sublayer output is scaled by a
    single per-layer learnable scalar `resweight`, initialized to zero, so the
    layer is the identity at init (exact dynamical isometry). No LayerNorm."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))   # alpha, zero-initialized
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # x_{i+1} = x_i + alpha * self_attn(x_i)
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = src2[0]
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)
        # x_{i+1} = x_i + alpha * FFN(x_i)   (same shared resweight)
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src
```
