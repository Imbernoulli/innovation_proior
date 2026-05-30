OK, let me think about what I actually want. I want a density estimator for general continuous data
— UCI tables, natural-image patches, digits — and I want the density itself, exact and cheap, for
any datapoint I'm handed, not just for points I generated myself. That last clause matters: the
whole reason I'd reach for a density estimator over a sample-only generator is to *score* external
data — to build likelihood-free inference, learned priors, importance proposals. So the bar is: for
any x, return the exact p(x) in one fast pass on a GPU, and be flexible enough to capture complex,
multimodal structure.

What do I have that's both flexible and tractable? Two families. Autoregressive models factor the
joint by the chain rule, p(x) = ∏_i p(x_i | x_{1:i-1}), and learn each one-dimensional conditional;
the log-likelihood is exact and just a sum of per-dimension log-conditionals. Normalizing flows write
p(x) = π_u(f^{-1}(x)) · |det(∂f^{-1}/∂x)| for an invertible f that pushes a simple base density π_u
(a standard Gaussian, say) onto the data; this is tractable exactly when f is easy to invert and its
Jacobian determinant is easy, and — the property I'll lean on hard — both of those survive
composition, so I can deepen a flow just by stacking copies of f.

Let me look harder at the autoregressive family, because it gives the exact likelihood I want for
free. The cleanest tractable version uses masking: take a fully-connected net with D inputs and D
outputs and drop connections so output i sees only inputs 1, …, i−1. Then output i computes the
parameters of the i-th conditional p(x_i | x_{1:i-1}), the autoregressive property holds by
construction, and — crucially — I get *all* the conditionals in one parallel forward pass instead of
the D sequential hidden-state updates a recurrent autoregressive model would need. That's MADE
(Germain et al. 2015). The way it drops connections is worth keeping in mind because I'll reuse it:
give each input and hidden unit an integer degree, an input's degree being its index in the chosen
order and the D outputs taking degrees 0 through D−1, and allow a connection only from a lower-or-equal
degree to a higher one. For output i to depend on all inputs of degree below i with no accidental
extra independences, every hidden layer has to contain every degree; then the mask between two
adjacent layers is just M[j, k] = 1{ degree_next[j] ≥ degree_prev[k] }.

But a single masked autoregressive model with simple conditionals has a ceiling. Suppose I make each
conditional a single Gaussian. Then a whole class of densities is out of reach — anything where the
conditional of x_i given the past is multimodal, the model just can't represent. And there's the
order problem on top: with single-Gaussian conditionals one variable ordering might fit a density
perfectly while another can't fit it at all, and I have no principled way to pick among the
factorially many orders. There's a sharp little diagnostic for this that I'll hold onto. If my model
is x = f(u) for some external randomness u with a standard-normal target, then for a good fit, when I
run the *data* back through f^{-1} to recover the u's that "would have generated" each training point,
those recovered u's should look like independent standard normals. If I scatter-plot them and they
come out visibly non-Gaussian — skewed, curved, clumped — that's direct evidence the model is a bad
fit. So the failure is *visible in the residual randomness*. That suggests where to push: if the u's
the model produces aren't actually distributed like the base density, then maybe I should *model the
density of those u's* with another model, instead of just declaring them standard normal and eating
the error.

To make that move precise I need to take the "x = f(u)" picture literally rather than metaphorically.
Here's the realization (Kingma et al. 2016 pointed this out): an autoregressive model, *viewed as a
generator*, already is such a transform. With single-Gaussian conditionals,
p(x_i | x_{1:i-1}) = N(x_i; μ_i, (exp α_i)²) where μ_i = f_{μ_i}(x_{1:i-1}) and α_i = f_{α_i}(x_{1:i-1})
are the conditional's mean and log-standard-deviation, sampling proceeds by drawing u_i ~ N(0,1) and
setting

  x_i = u_i · exp(α_i) + μ_i.

So the vector of internal random numbers u, the things a sampler pulls from randn(), maps to data x by
a function x = f(u) with u ~ N(0, I). Is f invertible, and is its Jacobian tractable? Given a datapoint
x, I can recover the u that would have produced it, one coordinate at a time, because μ_i and α_i depend
only on x_{1:i-1}, which I already have:

  u_i = (x_i − μ_i) · exp(−α_i),   μ_i = f_{μ_i}(x_{1:i-1}),   α_i = f_{α_i}(x_{1:i-1}).

And the Jacobian of this inverse map is triangular by the autoregressive structure: ∂u_i/∂x_j is zero
for j > i (u_i doesn't see later coordinates), so the matrix is lower-triangular and its determinant is
the product of the diagonal. The diagonal entry is ∂u_i/∂x_i = exp(−α_i) — only the explicit (x_i − μ_i)
factor contributes on the diagonal, since μ_i and α_i are functions of *earlier* coordinates and don't
touch the diagonal — so

  |det(∂f^{-1}/∂x)| = exp(−Σ_i α_i),   α_i = f_{α_i}(x_{1:i-1}).

Substitute the inverse map and this determinant into the change-of-variables formula and I have the
exact density of any x. So an autoregressive model with Gaussian conditionals *is* a normalizing flow,
no metaphor: f maps base randomness to data, f^{-1} maps data to base randomness in one masked pass,
and the log-det is just −Σ α_i.

Now I can act on the diagnostic. If the u's that f^{-1} produces from the data aren't standard normal —
which is exactly the symptom of a bad single-Gaussian-conditional fit — then instead of forcing them to
be standard normal, I'll *model their density with another autoregressive flow of the same type*, and
recurse. Stack flows M_1, M_2, …, M_K: M_2 models the random numbers u_1 that M_1 emits, M_3 models the
random numbers u_2 that M_2 emits, and so on, with only the final u_K declared standard normal. Because
each layer is an invertible, tractable-Jacobian flow, the stack is again an invertible, tractable-Jacobian
flow — the log-det of the whole is the sum of the layers' log-dets, and the density of x is the base
density at the fully-transformed code plus that sum. And the stack is genuinely more expressive than any
one layer: each individual MADE has unimodal Gaussian conditionals, but composing several of them lets the
*overall* model express multimodal conditionals, because a smooth invertible reshaping of a Gaussian can
be multimodal in the original coordinates. I implement each layer's {f_{μ_i}, f_{α_i}} with a MADE that
has Gaussian outputs — it emits μ_i and α_i for all i in one masked pass — so the entire x → u direction,
and hence the density, is one parallel pass per layer with no sequential recursion. That's the model:
stacked masked autoregressive flows. The conditioners read the data (and, deeper in the stack, the
previous layer's random numbers), which is the whole point — it's what makes density evaluation a single
pass.

I should pin down a choice I slid past: why single-Gaussian conditionals per layer rather than mixtures?
Mixtures would make each conditional individually multimodal, but then the per-coordinate map u_i ↦ x_i
is no longer a clean invertible affine recursion — a mixture-CDF inverse is messier and the tidy
triangular-Jacobian-with-diagonal-exp(α_i) structure I just used would break. I'd rather keep each layer
a simple invertible affine recursion (so the flow machinery stays exact and cheap) and buy flexibility by
*stacking*, which composition hands me for free. If I do want the universal-approximation guarantee that a
mixture base would give, I can always put a mixture-of-Gaussians MADE as the *base density* under the
stack and train them jointly — but the workhorse layer stays single-Gaussian.

There's a design fork hiding in the conditioners that I should stare at, because it's the difference
between two completely different models. In my layer, μ_i and α_i are functions of the previous *data*
variables x_{1:i-1}. What if I instead made them functions of the previous *random numbers* u_{1:i-1}?
The layer recursion looks almost identical, x_i = u_i·exp(α_i) + μ_i, but now μ_i = f_{μ_i}(u_{1:i-1}),
α_i = f_{α_i}(u_{1:i-1}). That's Inverse Autoregressive Flow (Kingma et al. 2016). The single-character
swap — condition on x_{1:i-1} versus u_{1:i-1} — flips the computational trade-off entirely, and it's
worth tracing exactly why, because it tells me which model is right for *my* task.

Take my version, conditioners reading x. To compute the density of an external x, I need the u's, and
u_i = (x_i − μ_i)exp(−α_i) with μ_i, α_i depending on x_{1:i-1} — but the whole of x is given, so a single
masked pass produces every μ_i, α_i and hence every u_i at once. Density of any external point: one pass.
But sampling is sequential: to generate x_i I need μ_i, α_i, which need x_{1:i-1}, which I haven't
generated yet — so I must produce x_1, then x_2, …, D passes in order. Now take the IAF version,
conditioners reading u. Sampling: draw all u, and since μ_i, α_i depend only on u_{1:i-1} (all available
up front), one masked pass gives every x_i — sampling and scoring-your-own-samples in one pass. But to
score an *external* x, I'd have to recover the u's, and u_i now depends on u_{1:i-1} through the
conditioners, so I must solve for u_1, then u_2, …, D sequential passes. So the two are mirror images:
conditioning on x gives one-pass density and D-pass sampling; conditioning on u gives one-pass sampling
and D-pass external-density. My task is density estimation of arbitrary data, so I condition on x — one
pass for the thing I do constantly. IAF conditions on u because it was built as a recognition network for
variational inference, where it only ever scores its *own* samples. Same recursion, opposite specializations.

That mirror symmetry is suspicious enough that I want to know whether the two are equivalent at a deeper
level, not just dual in cost. Let me see what training my model by maximum likelihood is actually doing in
IAF's language. Let π_x be the data density I'm fitting, π_u the base density, f my u → x transform, so my
model density on x-space is p_x(x) = π_u(f^{-1}(x)) |det(∂f^{-1}/∂x)|. The inverse map f^{-1}: x → u, read
the other way, describes an implicit IAF whose base density is π_x; it induces a density on u-space,
p_u(u) = π_x(f(u)) |det(∂f/∂u)|. Maximizing my total log-likelihood Σ_n log p_x(x_n) is, in the limit,
minimizing the KL from the data to my model, KL(π_x ‖ p_x). Write it out and change variables x ↦ u:

  KL(π_x ‖ p_x) = E_{π_x}[ log π_x(x) − log π_u(f^{-1}(x)) − log|det(∂f^{-1}/∂x)| ].

Push the expectation through x = f(u) so it's taken under p_u, and the change-of-variables turns
−log|det(∂f^{-1}/∂x)| into +log|det(∂f/∂u)| inside:

  = E_{p_u}[ log π_x(f(u)) − log π_u(u) + log|det(∂f/∂u)| ]
  = E_{p_u}[ log p_u(u) − log π_u(u) ]
  = KL(p_u ‖ π_u),

using the definition of p_u in the middle line. So maximizing my likelihood — minimizing KL(π_x ‖ p_x) —
is *exactly* minimizing KL(p_u ‖ π_u), which is the objective you minimize when you train an IAF (with base
π_x, transform f^{-1}) as a variational recognition network whose target posterior is π_u. Training my flow
as a density estimator of π_x is the same computation as variationally fitting an implicit IAF to my base
density. The duality isn't just a cost trade-off; the two optimization problems are the same problem viewed
from the two ends of f. That's the strongest evidence I have that conditioning-on-x and conditioning-on-u
are the right pair to think about together.

One more relative to place, because it sharpens what flexibility I'm actually buying. Real NVP stacks
coupling layers: split at index d, copy x_{1:d} = u_{1:d}, and affinely transform the rest,
x_{d+1:D} = u_{d+1:D} ⊙ exp(α) + μ with α, μ functions of only u_{1:d} (NICE is the α = 0,
volume-preserving case). Is that a different beast or a special case of mine? Take my MAF recursion and
set μ_i = α_i = 0 for i ≤ d (so the first d coordinates are copied) and let μ_i, α_i for i > d depend only
on x_{1:d} (not on the intervening x_{d+1:i-1}). That *is* the coupling layer. So a coupling layer is a
restricted MAF layer — and equally a restricted IAF layer — in which a fixed block is copied and the rest
is transformed as a function of only that block. My layer is strictly more flexible: every coordinate is
scaled and shifted as a function of *all* previous coordinates, not just of a frozen prefix. The price is
exactly the sampling cost I already traced: Real NVP copies half and transforms half as a function of the
copied half, so it can both sample and score in one pass, whereas I pay D passes to sample. For density
estimation, where I score constantly and sample rarely, that's the right trade.

Now two practical pieces before code. First, the order. A single autoregressive layer is order-sensitive,
and I don't know the best order — but I'm stacking layers, so I don't have to commit to one. I'll use the
dataset's natural order for the first layer (the one that directly touches the data) and *reverse* the
order for each successive layer, so dependencies the first order handles poorly get a second chance under
the opposite order deeper in the stack. Second, depth. Stacking many of these affine flow layers makes a
deep composition, and deep compositions of unnormalized affine maps drift in scale and get hard to train.
I want something to renormalize activations *between* layers — but it has to be a legal flow layer, meaning
invertible with a tractable Jacobian. Batch normalization is elementwise affine, so it qualifies. Let x be
the side near the data and u the side near the base; a batchnorm flow layer is

  x = (u − β) ⊙ exp(−γ) ⊙ (v + ε)^{1/2} + m,

with inverse

  u = (x − m) ⊙ (v + ε)^{−1/2} ⊙ exp(γ) + β,

where m, v are the running (minibatch at train, full-train at test) mean and variance and β, γ are learned.
I exponentiate γ — unlike textbook batchnorm, which uses a raw scale — for two reasons: it forces the scale
positive (keeping the map invertible), and it makes the log-determinant fall out cleanly. The inverse map
multiplies coordinate i by (v_i + ε)^{−1/2}·exp(γ_i), so

  |det(∂f^{-1}/∂x)| = exp( Σ_i [ γ_i − ½ log(v_i + ε) ] ),

which I just add to the running log-det sum like any other layer. Slotting one of these between every two
autoregressive layers cuts training time and stabilizes the deep stack.

Let me write it. The conditioner is a Gaussian MADE — masked linears with degree-based masks, doubled
outputs for (μ, α):

```python
import numpy as np
from numpy.random import permutation, randint
import torch
import torch.nn as nn
from torch.nn import functional as F


class MaskedLinear(nn.Linear):
    """Linear with a fixed binary mask on the weight: y = x @ (mask * W.T) + b."""
    def __init__(self, n_in, n_out, bias=True):
        super().__init__(n_in, n_out, bias)
        self.mask = None
    def set_mask(self, mask):
        self.mask = mask
    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """One pass outputs (mu, alpha) for every coordinate, obeying the autoregressive property."""
    def __init__(self, n_in, hidden_dims, random_order=False, seed=None):
        super().__init__()
        np.random.seed(seed)
        self.n_in, self.n_out = n_in, 2 * n_in          # gaussian: mu and alpha per coordinate
        self.hidden_dims = hidden_dims
        dims = [n_in, *hidden_dims, self.n_out]
        layers = []
        for i in range(len(dims) - 2):
            layers += [MaskedLinear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [MaskedLinear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)
        self._make_masks(random_order)

    def _make_masks(self, random_order):
        L, D = len(self.hidden_dims), self.n_in
        deg = {0: permutation(D) if random_order else np.arange(D)}   # input degrees = order index
        for l in range(L):                                            # every hidden layer: degrees in [min_prev, D-1]
            deg[l + 1] = randint(low=deg[l].min(), high=D - 1, size=self.hidden_dims[l])
        deg[L + 1] = deg[0]                                           # output degrees match input order
        masks = []
        for i in range(len(deg) - 1):
            prev, nxt = deg[i], deg[i + 1]
            M = (nxt[:, None] >= prev[None, :]).astype(int)           # connect only lower-or-equal -> higher degree
            masks.append(torch.tensor(M, dtype=torch.float32))
        masks[-1] = torch.cat((masks[-1], masks[-1]), dim=0)          # duplicate for (mu, alpha) outputs
        it = iter(masks)
        for m in self.net.modules():
            if isinstance(m, MaskedLinear):
                m.set_mask(next(it))

    def forward(self, x):
        return self.net(x.float())
```

each flow layer is the affine autoregressive recursion, with the reverse-order flip and the −Σα log-det:

```python
class MAFLayer(nn.Module):
    def __init__(self, dim, hidden_dims, reverse):
        super().__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims)
        self.reverse = reverse

    def forward(self, x):                              # x -> u, one pass (density direction)
        mu, logp = torch.chunk(self.made(x), 2, dim=1) # logp parameterizes -2*alpha
        u = (x - mu) * torch.exp(0.5 * logp)           # u_i = (x_i - mu_i) exp(-alpha_i)
        u = u.flip(dims=(1,)) if self.reverse else u   # reverse the order for the next layer
        log_det = 0.5 * torch.sum(logp, dim=1)         # log|det| = -sum_i alpha_i
        return u, log_det

    def backward(self, u):                             # u -> x, D sequential steps (sampling)
        u = u.flip(dims=(1,)) if self.reverse else u
        x = torch.zeros_like(u)
        for i in range(self.dim):
            mu, logp = torch.chunk(self.made(x), 2, dim=1)
            x[:, i] = mu[:, i] + u[:, i] * torch.exp(torch.clamp(-0.5 * logp[:, i], max=10))
        return x, None
```

the invertible batchnorm flow layer with the exponentiated scale:

```python
class BatchNormLayer(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(dim))     # shift
        self.gamma = nn.Parameter(torch.zeros(dim))    # exponentiated scale
        self.eps = eps

    def forward(self, x):                              # x -> u
        m = x.mean(0); v = x.var(0)                    # minibatch stats (full-train at test)
        u = (x - m) / torch.sqrt(v + self.eps) * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma - 0.5 * torch.log(v + self.eps))
        return u, log_det
```

and the stack, summing log-dets and adding the standard-Gaussian base log-density:

```python
import math

class MAF(nn.Module):
    def __init__(self, dim, n_layers, hidden_dims, use_reverse=True):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        for _ in range(n_layers):                      # alternate order; renormalize between layers
            self.layers.append(MAFLayer(dim, hidden_dims, reverse=use_reverse))
            self.layers.append(BatchNormLayer(dim))

    def forward(self, x):
        log_det_sum = torch.zeros(x.shape[0])
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_sum = log_det_sum + log_det
        return x, log_det_sum                          # x is now u (base space)

    def log_prob(self, x):
        u, log_det = self.forward(x)
        base = -0.5 * (u ** 2 + math.log(2 * math.pi)).sum(dim=1)   # standard-Gaussian base
        return base + log_det
```

trained by maximizing this exact log-likelihood with Adam (a smaller step for the deep stack), a little
weight decay, and early stopping:

```python
optimizer = torch.optim.Adam(maf.parameters(), lr=1e-4, weight_decay=1e-6)
for x in dataloader:
    loss = -maf.log_prob(x).mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

So the causal chain: I wanted exact, one-pass density of arbitrary data, which pointed me at
autoregressive models for the exact likelihood and at masking (MADE) to get that likelihood in one
parallel pass; a single masked model with single-Gaussian conditionals is too rigid and order-sensitive,
and the tell is that the random numbers it implies for the data don't come out standard normal; reading
"x = f(u)" literally shows an autoregressive model already *is* a normalizing flow with a triangular
Jacobian and log-det −Σα, so I can model those non-Gaussian random numbers with another such flow and
stack, gaining multimodality from composition while keeping each layer a cheap invertible affine recursion;
conditioning the layer on the data x (rather than on the random numbers u, which would give IAF) is what
makes density evaluation one pass — and maximizing my likelihood turns out to be exactly the variational
objective an implicit IAF would minimize, while the coupling layer of Real NVP is just my layer with a
frozen prefix; reversing the order between layers and slotting an invertible batchnorm between them keeps
the deep stack expressive and trainable.
