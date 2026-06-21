The part of the Transformer block I want to push on is the one I keep treating as fixed furniture: the position-wise feed-forward sublayer. Between attention layers each position's hidden vector $x$ is run through $f(xW_1)W_2$ — project up to a wide hidden dimension, apply a single pointwise nonlinearity, project back down — and that is the whole thing. It is not a small corner of the model: with the usual $4\times$ expansion the two matrices $W_1, W_2$ hold roughly two-thirds of the non-embedding parameters and a large share of the FLOPs. Attention mixes information across positions; this sublayer is the entire per-position transformation, the only place a token's representation gets reshaped on its own, and the single pointwise $f$ is the only nonlinearity in it. So if I want to spend a fixed budget better, this is the obvious place — it is where most of the budget already sits — and the question is whether I can make the layer fit the language-modeling objective better *without* spending more parameters or compute, by changing how the hidden representation is formed rather than by widening it.

The trouble is that every candidate $f$ on the table has the same internal anatomy, and once I see it I cannot unsee it. ReLU is $z\cdot\mathbf{1}[z>0]$, a hard sign gate that multiplies $z$ by a $0/1$ mask and zeros both the output and the gradient on the whole negative half-line. GELU is $z\cdot\Phi(z)$ with $\Phi$ the standard-normal CDF, derived as the expectation of a stochastic $0/1$ mask $m\sim\text{Bernoulli}(\Phi(z))$ whose mean transform is $\Phi(z)\cdot z + (1-\Phi(z))\cdot 0 = z\Phi(z)$. Swish is $z\cdot\sigma(\beta z)$, and at $\beta=1$ it is $z\cdot\sigma(z)$, the logistic-CDF cousin of GELU that also goes by SiLU. Each one is the raw preactivation $z$ multiplied by some gate that is a function of $z$ — the activation is *already* a gating operation, value times gate-of-itself, and this is not my coincidence: the automated activation search that turned up Swish reported exactly this as its central finding, that the winning scalar activations overwhelmingly have the form $b(x, g(x))$, the raw preactivation reused and recombined with a gate of itself (ReLU fits, with $b=\max$ and $g(x)=0$). But in $f(xW_1)W_2$ the value and the gate come from the *same* projection $xW_1$. The layer never gets to ask one question of $x$ to decide *how much* and a different question of $x$ to decide *what*; the gating is real but it cannot be independent, because there is only one learned linear view of $x$ feeding the whole sublayer.

So I propose SwiGLU: untie the gate from the value by giving the layer two up-projections instead of one — a gate projection $W$ and a value projection $V$ — and form each hidden unit as their elementwise product, with the nonlinearity living only on the gate path, then project back down. Writing $\otimes$ for the elementwise product, the layer is

$$\text{FFN}_{\text{SwiGLU}}(x) = \big(\,\text{Swish}_1(xW)\otimes(xV)\,\big)\,W_2,\qquad \text{Swish}_1(z) = z\cdot\sigma(z) = \text{SiLU}(z).$$

This is the gated linear unit lifted out of the gated convolutional language model — where the same shape $(X*W+b)\otimes\sigma(X*V+c)$ appeared as a content projection times a sigmoid gate — and dropped into the FFN in place of $f(xW_1)$. Three things have to be argued: that the product is the right coupling, that the nonlinearity belongs on the gate and not the value, and that Swish is the right gate.

First, why a *product* of two linear views and why that is more than a rearrangement. Drop the activation entirely and take $(xW)\otimes(xV)$: that is a pure bilinear interaction, the multiplicative coupling of two distributed representations that log-bilinear models used, and it makes each hidden unit a degree-two function of $x$ rather than a degree-one preactivation bent by a fixed curve — genuinely more expressive per unit than any single-projection-then-pointwise map. And multiplication specifically is the *safe* way to couple two learned linear maps. The activation search gave a sharp warning here: functions that used division performed poorly because the output explodes when the denominator nears zero, and a learned linear gate $xV$ is never bounded away from zero — it crosses zero constantly. The product of two finite linear maps is always finite; the divisive coupling the search rejected is exactly the one that detonates. So multiplication is not just the GLU shape, it is the coupling that does not blow up.

Second, why carry the value linearly and confine the nonlinearity to the gate. This is a gradient argument, and it is the one I actually trust once a dozen of these layers are stacked. If I had used the LSTM-style gated tanh unit $\tanh(X)\otimes\sigma(X)$, with content through a tanh and gate through a sigmoid, its gradient is

$$\nabla[\tanh(X)\otimes\sigma(X)] = \tanh'(X)\nabla X\otimes\sigma(X) + \sigma'(X)\nabla X\otimes\tanh(X),$$

and *both* paths multiply the upstream gradient by a saturating activation derivative ($0\le\tanh'\le 1$, $0\le\sigma'\le\tfrac14$). There is no derivative-free route through the content, so stacking a dozen of these attenuates the signal a dozen times over. Now keep the content linear, $X\otimes\sigma(X)$:

$$\nabla[X\otimes\sigma(X)] = \nabla X\otimes\sigma(X) + X\otimes\sigma'(X)\nabla X.$$

The first term multiplies the upstream gradient only by the gate *value* $\sigma(X)\in(0,1)$, not by any activation derivative — for the units the gate has opened ($\sigma(X)$ near $1$) the gradient passes essentially undiminished, a clean almost-linear highway back through the layer, a multiplicative skip connection. That is the structural reason to keep the value path linear: it buys two-view gating without paying the per-layer gradient-downscaling tax the both-paths-nonlinear unit pays.

Third, which gate function. The gated-conv unit used a plain sigmoid, but $\sigma(z)\in(0,1)$ can only ever *attenuate* — scale content toward zero, never pass it at more than unit strength, never flip it — which leaves capability on the table when the value path is already linear and unbounded. I want a richer modulation, and I find it back in the very activation family I was choosing between, read through its own $z\cdot\text{gate}(z)$ anatomy. Take $\text{Swish}_1(z)=z\cdot\sigma(z)$ as the gate: it is smooth, it is unbounded above (it grows like $z$ for large positive $z$, so as a gate it can pass content at *greater-than-unit gain*, amplifying where $\sigma$ could not), and it is non-monotonic with a small bump dipping below zero for moderately negative $z$, so as a gate it can softly suppress and even sign-flip the content on that bump rather than only squashing it. Amplify, pass, soften, flip — strictly richer than $\sigma$'s $(0,1)$ on/off, and it is the gate from the family the search kept rediscovering, except now placed on its own projection rather than tied to the value's. The one honest objection to Swish is that its derivative,

$$\text{Swish}_\beta'(z) = \sigma(\beta z) + \beta z\,\sigma(\beta z)\big(1-\sigma(\beta z)\big) = \beta\,\text{Swish}_\beta(z) + \sigma(\beta z)\big(1 - \beta\,\text{Swish}_\beta(z)\big),$$

has magnitude below $1$ for inputs below roughly $1.25$ at $\beta=1$, so it lacks ReLU's clean exact-$1$ gradient flow on that region. But that worry evaporates in this construction, which is the whole point of keeping the value path linear: the gradient highway derived above runs through the gate *value* multiplying the linear content's gradient, not through the gate's derivative at all. So I get clean flow from the value path regardless of the gate's derivative, and I am free to choose the gate purely for the richness of its modulation. The same template instantiates a whole family — sigmoid gives the original GLU, ReLU a hard-gated ReGLU, GELU a GEGLU, nothing-at-all the bilinear product — but $\text{Swish}_1$ is the one I have argued myself into.

The remaining problem is budget, because three matrices now replace two and the entire premise is a matched comparison; bolting on a third projection would confound any quality gain with simply spending more. So the extra projection is paid for by shrinking the hidden width. Let $d$ be $d_\text{model}$ and let the baseline FFN use hidden width $d_\text{ff}$, with $W_1$ of shape $d\times d_\text{ff}$ and $W_2$ of shape $d_\text{ff}\times d$, for $2\,d\,d_\text{ff}$ parameters (biases dropped, as the bias-free FFN convention does). The gated layer has $W$ and $V$ each $d\times d_\text{ff}'$ and $W_2$ of shape $d_\text{ff}'\times d$, for $3\,d\,d_\text{ff}'$ parameters. Matching,

$$3\,d\,d_\text{ff}' = 2\,d\,d_\text{ff}\quad\Rightarrow\quad d_\text{ff}' = \tfrac{2}{3}\,d_\text{ff}.$$

FLOPs scale the same way — three $d\times d_\text{ff}'$ matmuls equal two $d\times d_\text{ff}$ matmuls — so matching parameters matches compute. With the standard $4\times$ expansion $d_\text{ff}=4d$ this gives $d_\text{ff}' = \tfrac{2}{3}\cdot 4d = \tfrac{8}{3}d \approx 2.667\,d$, and the counts match exactly: baseline $2\,d\cdot 4d = 8d^2$, gated $3\,d\cdot\tfrac{8}{3}d = 8d^2$. At $d=768$, $d_\text{ff}=3072\to d_\text{ff}'=2048$. Biases are omitted on all three matrices, both to keep the accounting clean and because the FFN baseline already drops them; if a host implementation rounds $d_\text{ff}'$ to a hardware-friendly multiple, that is an explicit engineering choice whose small budget delta should be recorded separately from the $2/3$ formula. The forward pass is then a true drop-in: input $x$ is $(\text{batch},\text{length},d)$, the gate projection takes it to $(\text{batch},\text{length},d_\text{ff}')$ and applies $\text{Swish}_1$, the value projection takes it to the same shape with no activation, the two are multiplied elementwise, dropout is applied to that hidden product when enabled, and the down-projection returns it to $(\text{batch},\text{length},d)$ — touching nothing in attention, normalization, the data, or the optimizer. $\text{Swish}_1$ is just $x\cdot\sigma(x)$, the SiLU primitive every framework ships, so the gate adds no new parameters and is a single call. I have no mechanistic proof this lowers loss, and I will not invent one; what I have are converging design-time reasons — a multiplicative two-view interaction with the safe coupling, a linear value path giving a clean gradient highway, and a richer-than-sigmoid gate whose derivative is made irrelevant by that linear path — to expect it to help at strictly matched budget, validated against the baseline FFN on held-out language-modeling perplexity.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: ( Swish_1(xW) ⊗ (xV) ) W2.

    Pass d_ff = (2/3) * d_ff_base; for T5-base, d_ff_base=3072 -> d_ff=2048.
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.wi_0 = nn.Linear(d_model, d_ff, bias=False)  # gate projection W
        self.wi_1 = nn.Linear(d_model, d_ff, bias=False)  # value projection V
        self.dropout = nn.Dropout(dropout)
        self.wo = nn.Linear(d_ff, d_model, bias=False)    # down projection W2

    def forward(self, x):                                  # (B, T, d_model) -> (B, T, d_model)
        # F.silu(z) == z * sigmoid(z) == Swish_1(z).
        h = F.silu(self.wi_0(x)) * self.wi_1(x)
        return self.wo(self.dropout(h))
```
