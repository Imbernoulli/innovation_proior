Inside almost every multi-scale or multi-source model there is one line I write a dozen times without ever questioning it: I have $V$ feature maps that came from different places — different pyramid levels, different resolutions, in a weather model different physical variables each tokenized on its own — and once I have resized them to a common shape I collapse the stack $(I_1, \dots, I_V)$ at every spatial location into a single map $O$. What I write is $O = I_1 + I_2 + \cdots + I_V$, or I divide by $V$ and call it a mean. That same primitive is everywhere: FPN merges its lateral and upsampled top-down maps by element-wise addition, $P^{\text{out}}_l = \mathrm{Conv}(P^{\text{in}}_l + \mathrm{Resize}(P^{\text{out}}_{l+1}))$; PANet fuses its pooled per-level grids by element-wise max or sum; NAS-FPN, whatever topology the search hands it, still combines the inputs arriving at a node by summing; and in variable aggregation the parameter-free baseline is exactly the uniform mean over the $V$ variables. Every one of these gives each source the weight one, and nobody checked whether that is right. The trouble is that the sources are heterogeneous by construction — different resolutions, different semantics, literally different physical fields — so there is no reason all $V$ deserve equal say at a given location. This is not abstract: when PANet instrumented which pyramid level each pooled feature actually came from, it found that the usefulness of a source is only weakly tied to the level it nominally lives on — for small objects assigned to the finest level, on the order of 70% of the helpful features came from other levels, and for large objects assigned to the coarsest level more than half came from lower levels. Sources contribute unequally and which one carries the load shifts case to case, yet the equal-weight combine gives the network no handle to act on that. The expressive cure, attention, captures source- and even location-dependent mixing but drags in query/key/value projections scaling with the channel dimension plus a per-location attention computation — a real tax on top of an already-large backbone. So the landscape offers "free but rigidly equal" at one end and "expressive but expensive" at the other, with nothing in between, and any middle ground also has to keep the fused output on a sane scale, since the blocks that follow expect features in a predictable numerical range.

I propose the learned weighted sum: fuse the sources by a weighted combination $O = \sum_i w_i I_i$ in which the weights are learned, but constrained so that only the relative split between sources is free, never the overall scale. The first design choice is that one scalar weight per source is enough. The weight $w_i$ could be a scalar, a length-$D$ vector (one per channel), or a full per-pixel tensor; the per-channel and per-pixel forms are strictly more expressive, but the quantity I actually want to decide — "how much does source $i$ matter overall" — is naturally one number per source. With $V$ sources that is $V$ parameters total, utterly negligible next to the convolutions and projections around it, so I take the scalar form as the default and keep the per-channel vector (softmax taken over the source axis independently per channel) in reserve for when channel-level discrimination is genuinely needed. The second and load-bearing choice comes from watching the naive version break. If I let $w_i$ be free real numbers, the map from $(w_1, \dots, w_V)$ to $O$ is linear and homogeneous of degree one in the weights: scale every $w_i$ by $c$ and $O$ scales by $c$. Nothing pins the overall magnitude, so an entire ray of settings produces outputs differing only by a global gain, and the optimizer drifts along it. That is bad twice over: the downstream network is tuned to features at a particular scale, so if $\sum_i w_i$ wanders to ten or a tenth I have silently multiplied the fused representation by ten or a tenth and knocked the rest of the net off its operating point; and there is no ceiling at all, so an unbounded multiplicative gain in the middle of a deep network is exactly what makes training go unstable. The failure points at the cure. I never cared about the absolute size of the weights, only the relative contribution — a ratio, not a magnitude — so I quotient out the nuisance scale by forcing the weights onto the simplex $\{w_i \ge 0,\ \sum_i w_i = 1\}$. Then

$$O = \sum_{i=1}^{V} w_i\, I_i, \qquad w_i \ge 0, \quad \sum_{i=1}^{V} w_i = 1$$

is a convex combination of the inputs: it lives in their convex hull, so it sits on the same scale as a single input, it cannot introduce an arbitrary gain, and when all the $w_i$ are equal it collapses back to the plain mean. The equal-weight average I started from is just the uniform point of this family, and now the network may pick any other point of it.

What remains is purely mechanical: parameterize a point on the simplex with free, unconstrained parameters so that plain SGD or Adam trains it with no projection step and no constrained optimizer. I do not want to carry $w_i \ge 0,\ \sum_i w_i = 1$ as hard constraints and project after every step; I want raw parameters $a \in \mathbb{R}^V$ living anywhere, and a fixed differentiable map onto the simplex. Nonnegativity from an arbitrary real comes from the exponential, $e^{a_i} > 0$ always; sum-to-one comes from dividing by the total. Together that is softmax,

$$w_i = \frac{e^{a_i}}{\sum_{j=1}^{V} e^{a_j}},$$

which does exactly and only what I asked. Every $w_i$ is strictly positive and they sum to one by construction, so the output is always a valid distribution over the $V$ sources — a satisfying way to read it, since $w_i$ is the network's estimate of how much source $i$ should carry. The raw $a_i$ are unconstrained, so I register them as ordinary parameters and let any optimizer move them freely; softmax does the projection for me, smoothly, every forward pass, and is differentiable everywhere so backprop through it is automatic. The gradient on $a_i$ couples all the sources, because the denominator depends on all of them, which is correct — pushing one source up necessarily pushes the relative share of the others down, since they compete for a budget that sums to one. The constraint is baked into the functional form rather than bolted on as a penalty, which is exactly why it composes cleanly with the rest of the network. The degenerate cases confirm the design: if all $a_i$ are equal — in particular if I initialize them all to the same value — then $w_i = e^0/(V e^0) = 1/V$, so at initialization the softmax-weighted sum is precisely the plain mean. That is a genuinely good place to start: I begin from the safe equal-contribution prior, the very baseline I am trying to improve, and training perturbs the $a_i$ off uniform only insofar as the data rewards it, with no cold-start in some wild corner of the simplex. Because softmax is shift-invariant, $e^{a_i + c}/\sum_j e^{a_j + c} = e^{a_i}/\sum_j e^{a_j}$, any constant initialization (zeros or ones) gives the same uniform start. Against the heavyweight alternative this is the deliberate minimal answer: cross-attention recomputes the mixing per location and carries projection matrices and an attention computation, whereas my $V$ scalars are a single global mixing distribution shared across every location and example — if the global "how much does each source matter" split is most of the signal, as the PANet diagnostic suggests, I get most of the benefit at essentially none of the cost.

There is one cheaper sibling worth naming, fast normalized fusion, for when the softmax reduction's GPU latency actually bites. Replace the exponential with a ReLU and normalize by the sum, $w_i = \mathrm{relu}(a_i)/(\varepsilon + \sum_j \mathrm{relu}(a_j))$ with $\varepsilon = 10^{-4}$ guarding against dividing by zero. The normalized weights still lie in $[0,1]$ and sum to one up to $\varepsilon$, and in practice this learns very nearly the same per-source split as softmax. It trades softmax's strict smoothness — a ReLU can pin a weight at exactly zero, where it gets no gradient and is stuck — for speed, and because all-zero raw weights would leave the fusion dead, this variant must be initialized at a positive constant such as ones rather than at zero. Either way the structural idea is identical: nonnegative, normalized, learned per-source weights. The state is one learnable length-$V$ vector; the forward pass is a softmax over the source axis, a broadcast multiply against the $[B, V, L, D]$ stack, and a sum over the source axis collapsing $V \to 1$ to leave $[B, L, D]$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariableAggregator(nn.Module):
    """Learned, softmax-normalized weighted sum over V source feature maps.

    Input  x: [B, V, L, D]   (V source maps per spatial location)
    Output:   [B, L, D]      (one fused map per location)
    """

    def __init__(self, embed_dim, num_heads, num_vars):
        super().__init__()
        self.embed_dim = embed_dim          # D (unused: weights are per-source scalars)
        self.num_heads = num_heads          # unused: no attention
        self.num_vars = num_vars            # V
        # One raw scalar weight per source; zeros -> uniform 1/V (plain mean) at init.
        self.var_weights = nn.Parameter(torch.zeros(num_vars), requires_grad=True)

    def forward(self, x):
        # x: [B, V, L, D]
        w = F.softmax(self.var_weights, dim=0)      # [V] -> simplex (nonneg, sums to 1)
        w = w.view(1, self.num_vars, 1, 1)          # [1, V, 1, 1] broadcast over B, L, D
        out = (x * w).sum(dim=1)                     # convex combination over sources -> [B, L, D]
        return out
```

Fast-normalized variant (requires changing the parameter initialization to ones):

```python
        # In __init__, use this instead of the zero init above:
        self.var_weights = nn.Parameter(torch.ones(num_vars), requires_grad=True)

        # In forward:
        w = F.relu(self.var_weights)
        w = w / (w.sum(dim=0) + 1e-4)               # ReLU + normalize; no exp, no softmax reduction
        w = w.view(1, self.num_vars, 1, 1)
```
