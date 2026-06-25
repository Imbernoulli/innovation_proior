OK, let me think about why my variational autoencoders keep leaving performance on the table. I'm
fitting an inference model q(z|x) to a generative model p(x, z) by maximizing the variational lower
bound, and I can write that bound two ways. As an expectation,

  log p(x) ≥ E_{q(z|x)}[ log p(x, z) − log q(z|x) ] = L(x; θ),

and, rearranged, L(x; θ) = log p(x) − KL(q(z|x) ‖ p(z|x)). The second form is the one that bites: the
gap between my bound and the true log-likelihood is exactly the KL from my approximate posterior to the
true posterior. If q can't represent the shape of p(z|x), I pay for it directly in a loose bound. And
what shape do I let q take? A diagonal Gaussian, q(z|x) = N(μ(x), σ²(x)), reparameterized as
z = μ(x) + σ(x) ⊙ ε with ε ~ N(0, I). True posteriors are essentially never factorized — the latent
dimensions are correlated — so a diagonal Gaussian is structurally wrong, and the bound is loose by
precisely the amount the posterior fails to factorize.

So why am I stuck with diagonal Gaussians? Because of what I need from q at every single optimization
step, for every datapoint in the minibatch. I need to (1) evaluate and differentiate q's density, and
(2) draw a sample from q — both, both cheap. And if z is high-dimensional and I'm on a GPU, I need those
operations to be *parallel across the dimensions of z*, not a loop. The diagonal Gaussian is the lazy
answer that satisfies all three. The question is whether I can have a q that is far more expressive and
still cheap-to-score, cheap-to-sample, and parallel — and that keeps working when z is not a few hundred
dimensions but thousands of spatially-organized ones.

The normalizing-flow idea is the natural place to push. Start from a simple z_0 ~ q(z_0|x) and refine it
through a chain of invertible maps z_t = f_t(z_{t-1}, x); as long as each step's Jacobian determinant is
computable, the density of the last iterate telescopes,

  log q(z_T|x) = log q(z_0|x) − Σ_{t=1}^{T} log |det(∂z_t/∂z_{t-1})|.

That's exactly the machinery I want — refine a Gaussian into something flexible while tracking its
density. But the flows on offer don't scale. The planar flow f_t(z) = z + u·h(wᵀz + b) updates z through
a single scalar bottleneck wᵀz + b: one direction per step. To reshape a high-dimensional correlated
posterior I'd need a chain as long as the dimensionality is wide, which is hopeless for the latent space
of an image model. These planar and radial flows are demonstrably fine for a few hundred dimensions and
not for thousands. So I need a flow step that touches *all* the dimensions of z at once and still has a
log-determinant I can compute without an O(D³) determinant. The cheap log-determinants I know of come
from special Jacobian structure: diagonal (too weak — that's just the diagonal Gaussian again), or
triangular (det = product of the diagonal, computable in O(D)). Triangular it is, then — but where do I
find a flexible, learned map whose Jacobian is triangular over thousands of dimensions?

Where do I already have flexible, triangular-Jacobian functions over high-dimensional vectors? The
autoregressive density estimators — MADE, PixelCNN, WaveNet. Take a vector y with a chosen order and a
network that outputs a mean and standard deviation for each coordinate, [μ(y), σ(y)], with the
autoregressive property: ∂[μ_i, σ_i]/∂y_j = [0, 0] for j ≥ i, i.e. coordinate i's parameters depend only
on the earlier coordinates. Sampling from this model is the recursion

  y_0 = μ_0 + σ_0·ε_0,    y_i = μ_i(y_{1:i-1}) + σ_i(y_{1:i-1})·ε_i,    ε ~ N(0, I).

This is flexible and high-dimensional — exactly the kind of map I want — but it's useless as a posterior
to sample from, because the recursion is *sequential*: to get y_i I need y_{1:i-1} first, so generating
a D-dimensional sample costs D sequential steps. That's the opposite of the parallelism I need. So I'd
write off autoregressive functions for sampling-based inference... except let me look at the *inverse*
of that map before I give up.

Given a full vector y, recover the noise that produced it: from y_i = μ_i + σ_i·ε_i,

  ε_i = (y_i − μ_i(y_{1:i-1})) / σ_i(y_{1:i-1)).

Stare at this for a second. Every μ_i and σ_i is a function of y_{1:i-1}, and the *entire* y is given —
so I can compute all the μ_i, σ_i in one pass, and then every ε_i in parallel, because ε_i depends only
on y, not on the other ε's. The forward (sampling) direction was sequential precisely because each y_i
fed the next conditioner; the inverse direction has all of y in hand from the start, so it parallelizes
completely. One pass, all dimensions at once. That's the first half of what I need.

The second half is the Jacobian. The map y ↦ ε is autoregressive, so ∂ε_i/∂y_j = 0 for j > i — the
Jacobian dε/dy is lower-triangular. Its diagonal is ∂ε_i/∂y_i, and since μ_i and σ_i depend only on
*earlier* coordinates, the only y_i-dependence on the diagonal is through the explicit division, giving
∂ε_i/∂y_i = 1/σ_i(y). The determinant of a triangular matrix is the product of its diagonal, so

  log |det(dε/dy)| = Σ_i −log σ_i(y),

a sum of logs of the per-coordinate scales — trivially cheap, and over all dimensions at once. That is
the structure I was looking for: flexible, parallel, and a log-determinant in O(D). The forward sampling
direction was sequential and useless; its inverse is parallel and tractable. I'll chain these inverse
autoregressive steps into a normalizing flow.

Now make it concrete as a posterior. The encoder reads x and produces the parameters of an initial,
simple Gaussian — μ_0, σ_0 — and one extra vector h, a context I'll feed into every refinement step so
each step can condition on the datapoint without re-running the encoder. Initialize the chain by
reparameterizing the initial Gaussian,

  z_0 = μ_0 + σ_0 ⊙ ε,    ε ~ N(0, I),

and then apply T refinement steps, each a *different* autoregressive network taking z_{t-1} and h and
outputting μ_t, σ_t:

  z_t = μ_t + σ_t ⊙ z_{t-1}.

Each step is autoregressive with respect to z_{t-1}, so ∂z_t/∂z_{t-1} is triangular with σ_t on the
diagonal and determinant ∏_i σ_{t,i}. (The Jacobian with respect to h is unconstrained, but that's fine
— h is just extra context, it doesn't enter the z-to-z determinant.) Now assemble the density. The
initial step gives log q(z_0|x) = log N(ε; 0, I) − Σ_i log σ_{0,i}, because z_0 = μ_0 + σ_0 ⊙ ε rescales
the noise; writing out the standard Gaussian, that's −Σ_i (½ ε_i² + ½ log 2π + log σ_{0,i}). Each following
step contributes −Σ_i log σ_{t,i} to the telescoped density. Summing over the chain,

  log q(z_T|x) = − Σ_i ( ½ ε_i² + ½ log 2π + Σ_{t=0}^{T} log σ_{t,i} ).

I want to be sure this formula is right before I build anything on it, because the whole telescoping
argument rests on each step's Jacobian really being triangular with σ_t on the diagonal. The clean test
is the change-of-variables identity itself: the map ε ↦ z_T is deterministic once I fix the encoder
outputs, so I can compute log q(z_T) two ways and demand they agree — once by my formula above, and once
by the brute-force log N(ε; 0, I) − log|det(∂z_T/∂ε)| with the full Jacobian from autodiff. I make this
concrete with D = 4, T = 3 inverse-autoregressive steps and random net weights, a case small enough that
a dense 4×4 Jacobian and its exact log-determinant are available. Running both: my analytic log q comes
out to 1.01707 and the autodiff change-of-variables value to 1.01708 — they agree to about 1e-5, which is
just float noise. So the telescoped formula is the genuine density, not an approximation.

While I have the Jacobian in hand I check the structural claim directly, since that's what the whole idea
hangs on. The Jacobian of a single step on D = 4 prints as

  [[0.767  0     0     0   ]
   [0.031  0.941 0     0   ]
   [-0.060 0.090 0.851 0   ]
   [0.012 -0.002 0.017 0.996]],

i.e. exactly lower-triangular (every entry above the diagonal is 0 to machine precision), and its diagonal
[0.767, 0.941, 0.851, 0.996] equals the σ_t the net emitted for that step, coordinate for coordinate. That
is precisely the triangular-with-σ-on-the-diagonal structure the −Σ log σ log-determinant relies on. Good:
the density is exact and the structure is what I assumed.

That tells me the bookkeeping is right; now I want to know what the *simplest* instance actually
represents, to be sure I'm gaining real posterior flexibility and not just reshuffling a diagonal Gaussian.
Take one linear autoregressive step. A full-covariance Gaussian N(m, C) can be written coordinate by
coordinate:

  y_i = μ_i(y_{1:i-1}) + σ_i(y_{1:i-1})·ε_i,

with μ_i(y_{1:i-1}) = m_i + C[i, 1:i-1]·C[1:i-1, 1:i-1]^{-1}(y_{1:i-1} − m_{1:i-1}) and
σ_i²(y_{1:i-1}) = C[i, i] − C[i, 1:i-1]·C[1:i-1, 1:i-1]^{-1}·C[1:i-1, i]. Inverting that Gaussian
autoregression is linear: ε = (y − μ(y))/σ(y) = L(y − m), with L a lower-triangular whitening matrix,
the inverse Cholesky factor of C under this ordering. The flow direction I actually sample in goes the
other way: start from a factorized Gaussian y = μ(x) + σ(x) ⊙ ε and apply a lower-triangular linear map
to get z. If I restrict that triangular map to have ones on the diagonal, its determinant is one, so
q(z|x) = q(y|x) and no extra log-determinant appears; the diagonal scales in the starting Gaussian plus
the unit-diagonal triangular map should give an LDLᵀ factorization that reaches *any* full covariance
matrix.

I should check that last reach-any claim rather than wave at it, because if the unit-diagonal restriction
secretly loses covariances, the linear case would be weaker than a full Gaussian and the whole premise
would be shaky. So take an arbitrary 3×3 SPD target C = AAᵀ + 0.1·I with A random. Its Cholesky factor is
G with C = GGᵀ; dividing each column of G by its own diagonal entry, L = G / diag(G), gives a
unit-diagonal lower-triangular L, and D = diag(G)². I check three things numerically: diag(L) = [1, 1, 1]
with the strict upper triangle 0 (so L is genuinely unit-lower-triangular), det(L) = 1.0000 (so the map is
volume-preserving, no stray log-determinant), and max|C − L·diag(D)·Lᵀ| = 2.4e-7 — the reconstruction lands
back on the original C to float precision. So the unit-diagonal triangular map times a diagonal Gaussian
does reach an arbitrary full covariance, with determinant exactly one. The linear bottom of the
construction is therefore not a toy: it is exactly a full-covariance Gaussian posterior, and the nonlinear
autoregressive nets generalize beyond Gaussian from there.

Now a wall I'll hit in practice. The step z_t = μ_t + σ_t ⊙ z_{t-1} multiplies by a learned σ_t every
layer, and an unconstrained σ_t can blow up or vanish across a deep chain — multiplying T learned scales
together is numerically treacherous, and the chain has no natural starting point. I want a step that's
bounded and that begins benign. Let the autoregressive net output two *unconstrained* vectors m_t and s_t,
squash the scale through a sigmoid, σ_t = sigmoid(s_t) ∈ (0, 1), and write the update as a gated blend:

  z_t = σ_t ⊙ z_{t-1} + (1 − σ_t) ⊙ m_t.

This is just the general step with μ_t = (1 − σ_t) ⊙ m_t, so the density formula above is unchanged — the
diagonal of the Jacobian is still σ_t and the log-det is still −Σ log σ_t. But now σ_t is bounded in (0, 1),
so the per-step scaling can't explode; it's an LSTM-style convex combination of "keep z_{t-1}" and "write
the new value m_t." And it hands me a free initialization: if I bias s_t so that σ_t starts near 1 — push
s_t up by a small positive constant before the sigmoid, the LSTM forget-gate-bias trick — then z_t ≈ z_{t-1}
at the start and the whole flow begins as the identity, so optimization starts from a clean factorized
Gaussian and only has to *learn the deviations* from it rather than fight an arbitrary initial warp. The
gating is for numerical stability; the bias init is for a benign starting point.

One more cheap improvement. Each autoregressive step has a fixed variable order, and a fixed order means
the late coordinates can condition on everything while the early ones condition on little. If I *reverse*
the order of the variables between steps, the coordinates that were "early" (and barely conditioned) in one
step become "late" (richly conditioned) in the next, so the chain mixes dependencies in both directions. A
reversal is just a permutation — volume-preserving, determinant ±1 in magnitude — so it leaves the density
formula exactly as it is. Free flexibility.

Now, where does the flexibility actually buy me a tighter bound, and is improving q the only lever? Recall
L = log p(x) − KL(q ‖ p(z|x)): the gap depends on how well q matches the *true* posterior, and the true
posterior is itself a function of the generative model p. So I can also make the bound easier to close by
making the true posterior a shape my IAF can match. Concretely, in a deep latent-variable model with
latents z_1, …, z_L, let the *prior* be autoregressive across the layers,

  p(z_{1:L}) = p(z_L) ∏_{l<L} p(z_l | z_{l+1:L}),

rather than factorized. A more flexible prior makes the true posterior more flexible too, and an IAF
posterior is well-equipped to match it — so I get a tighter bound without making the generative model
itself any less flexible. The IAF on the q side and an autoregressive prior on the p side pull in the same
direction: both increase flexibility where the KL gap lives.

The ordering of inference in that deep stack matters. If the generative model samples top-down,
z_L first and then z_{L-1}, ..., z_1, a purely bottom-up inference model samples in the opposite order:
q(z_1|x)q(z_2|z_1,x)... That is easy for recognition, but each posterior layer misses the top-down
state that defines its prior. I can keep the cheap bottom-up evidence by running a deterministic
bottom-up pass first, storing h_l^(q), and then sample the stochastic variables top-down in the same
order as the generative model, conditioning q(z_l|.) on both h_l^(q) and the current top-down state
h_l^(p). That gives each local posterior the evidence from x and the prior context it is being compared
against. The IAF context h is exactly the place to feed those deterministic activations.

One more practical wall appears when there are many stochastic layers. Early in training, the
reconstruction term is weak, and the model can fall into the quiet state q(z|x) ≈ p(z): the KL terms are
near zero, the stochastic layers carry no information, and the encoder receives a low-signal gradient for
escaping. Annealing the KL weight upward is one way to avoid punishing latent information too early,
but then training depends on a schedule. A cleaner objective is to stop rewarding the model for using
less than a small amount of information in each group of latents. Split the latents into K groups j, and
on a minibatch M replace the usual reconstruction-minus-KL objective by

  L_λ = E_{x∈M} E_{q(z|x)}[log p(x|z)] − Σ_{j=1}^{K} max(λ, E_{x∈M}[KL(q(z_j|x) ‖ p(z_j))]).

If a group's average KL is below λ, the penalty is the constant λ, so the gradient no longer pushes that
group toward zero information. Once it uses more than λ nats, the term becomes the ordinary KL again.
So the objective is still the ELBO above the threshold, but it removes the local attraction of the
zero-information solution below the threshold.

The flow-prior trade is really a coordinate choice. Suppose the autoregressive transform whitens a
structured latent y into z = (y − μ(y))/σ(y). If I call y the model's latent variable, then the prior over y
is autoregressive and a simple posterior over the whitened z corresponds, through the inverse change of
variables, to an IAF posterior over y. If instead I call z the model's latent variable, the prior can be
factorized and the posterior carries the autoregressive flow. Those are two coordinate descriptions of the
same triangular change of variables; moving flexibility between the prior and posterior changes where the
computation sits, not the underlying reason the KL gap gets easier to close.

Let me write the core step, the numerically-stable gated version with the forget-gate-bias init. The
autoregressive network is a masked net that takes the current z and the context h and outputs the two
unconstrained vectors:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        self.register_buffer("mask", mask.float())

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class AutoregressiveNN(nn.Module):
    """Masked net: output i depends only on z_{1:i-1}; h can affect every output."""
    def __init__(self, dim, hidden, context_dim):
        super().__init__()
        input_degrees = torch.arange(1, dim + 1)
        if dim == 1:
            hidden_degrees = torch.zeros(hidden, dtype=torch.long)
        else:
            hidden_degrees = torch.arange(hidden) % (dim - 1) + 1
        output_degrees = torch.cat([input_degrees, input_degrees])
        mask_in = (hidden_degrees[:, None] >= input_degrees[None, :]).float()
        mask_out = (output_degrees[:, None] > hidden_degrees[None, :]).float()
        self.input = MaskedLinear(dim, hidden, mask_in)
        self.output = MaskedLinear(hidden, 2 * dim, mask_out)
        self.context = nn.Linear(context_dim, 2 * dim)

    def forward(self, z, h):
        hidden = F.elu(self.input(z))
        m, s = torch.chunk(self.output(hidden) + self.context(h), 2, dim=1)
        return m, s


class Encoder(nn.Module):
    def __init__(self, x_dim, z_dim, context_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2 * z_dim + context_dim),
        )
        self.z_dim = z_dim
        self.context_dim = context_dim

    def forward(self, x):
        params = self.net(x.reshape(x.size(0), -1))
        mu0 = params[:, :self.z_dim]
        log_sigma0 = params[:, self.z_dim:2 * self.z_dim].clamp(min=-7.0, max=7.0)
        h = params[:, 2 * self.z_dim:]
        return mu0, log_sigma0, h


class RefiningStep(nn.Module):
    """One refinement: z <- sigma*z + (1-sigma)*m; add -sum(log sigma) to log q."""
    def __init__(self, dim, hidden, context_dim, forget_bias=1.5):
        super().__init__()
        self.net = AutoregressiveNN(dim, hidden, context_dim)
        self.forget_bias = forget_bias

    def forward(self, z, h):
        m, s = self.net(z, h)
        sigma = torch.sigmoid(s + self.forget_bias)         # near identity at initialization
        z = sigma * z + (1.0 - sigma) * m                   # bounded LSTM-style blend
        log_density_delta = -torch.log(sigma.clamp_min(1e-8)).sum(dim=1)
        return z, log_density_delta
```

the encoder emits the initial Gaussian's parameters and the context, and the posterior reparameterizes,
then refines, accumulating the exact log-density of the chain:

```python
class FlexiblePosterior(nn.Module):
    def __init__(self, x_dim, z_dim, context_dim, n_steps, hidden, reverse_between_steps=True):
        super().__init__()
        self.encoder = Encoder(x_dim, z_dim, context_dim, hidden)
        self.steps = nn.ModuleList(
            [RefiningStep(z_dim, hidden, context_dim) for _ in range(n_steps)]
        )
        self.reverse_between_steps = reverse_between_steps

    def sample_and_log_prob(self, x):
        mu0, log_sigma0, h = self.encoder(x)
        eps = torch.randn_like(mu0)
        z = mu0 + torch.exp(log_sigma0) * eps               # z_0 = mu_0 + sigma_0 * eps
        # log q(z_0|x) = -sum_i ( 1/2 eps_i^2 + 1/2 log 2pi + log sigma_{0,i} )
        log_q = -(0.5 * eps ** 2 + 0.5 * math.log(2 * math.pi) + log_sigma0).sum(dim=1)
        for i, step in enumerate(self.steps):               # reverse variable order between steps
            z, log_density_delta = step(z, h)
            log_q = log_q + log_density_delta
            if self.reverse_between_steps and i + 1 < len(self.steps):
                z = z.flip(dims=(1,))                       # permutation has |det| = 1
        return z, log_q
```

and the bound is the usual reparameterized ELBO, now with this flexible log q:

```python
def elbo(decoder, prior, x, z, log_q):
    log_px_z = decoder.log_prob(x, z)                       # log p(x|z)
    log_pz = prior.log_prob(z)                              # log p(z); may be an autoregressive prior
    return (log_px_z + log_pz - log_q).mean()               # E[ log p(x,z) - log q(z|x) ]

def make_optimizer(params):
    return torch.optim.Adam(params)
```

So the causal chain: the VAE bound is loose by KL(q ‖ true posterior), and a diagonal-Gaussian q can't
match a correlated posterior, so I wanted a much more flexible q that is still cheap-to-score,
cheap-to-sample, and parallel in high dimensions; planar flows squeeze everything through a scalar and
don't scale; autoregressive networks are flexible and high-dimensional but their *sampling* is sequential
— yet their *inverse* ε = (y − μ)/σ is parallel in one pass and has a triangular Jacobian with log-det
−Σ log σ, exactly the flow step I needed; chaining these inverse-autoregressive steps over a
reparameterized Gaussian gives an exact, cheap posterior density, whose simplest linear instance gives
a full-covariance Gaussian through a unit-diagonal triangular map; a sigmoid-gated LSTM-style update with a
forget-gate-bias init keeps the deep chain numerically stable and starting from the identity; reversing
the order between steps mixes dependencies for free; bidirectional inference gives each stochastic layer
both bottom-up evidence and top-down prior context; free bits remove the zero-information attraction in
deep stacks; and the autoregressive-prior view is the same triangular change of variables seen from the
other coordinate system.
