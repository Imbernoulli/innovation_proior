OK, let me think this through from scratch. I want to model the density of complex,
high-dimensional continuous data — images, faces — with no labels, and I want the thing nobody
currently has all at once. I want to *evaluate* the exact likelihood of a data point, I want to
*sample* easily and without bias, I want to *infer* a latent code for any image exactly, and I
want that latent code to actually mean something. Every approach I know hands me a subset of
these and charges me the rest.

Let me say what I mean by "the latent code means something," because that's the seed of
everything. The deep-learning bet is that a good representation captures the data's factors of
variation. So here's a sharp way to operationalize "good": a good representation is one in which
the data distribution is *easy to model*. The easiest distribution to model is one with
independent coordinates — a factorial distribution, p_H(h) = ∏_d p_{H_d}(h_d). So I'll ask the
learner to find a transformation h = f(x) that maps the tangled data into a space where the
transformed distribution factorizes. If I can pull that off, then modeling p_X collapses to
fitting one transformation into a fixed, simple, factorial prior. That's the whole program.

Now, which existing machinery gets me there? If I go undirected — a deep Boltzmann machine — the
energy is easy to write, but the partition function is intractable, so training leans on MCMC,
sampling leans on MCMC, evaluation leans on annealed importance sampling, and the chains mix
slowly exactly when the distribution has sharp modes, which images do. Exact anything is off the
table, and the AIS estimate of likelihood can even be optimistic. If I go directed with a
variational autoencoder, ancestral sampling is clean and inference is amortized and fast, but the
encoder q(h|x) is *stochastic* and only a variational *approximation* to the true posterior, so I
inject noise into the autoencoder loop; I'm maximizing a *lower bound*, not the likelihood, and a
loose bound can leave unstructured noise smeared through the generations; and the imperfect
decoder p(x|h) forces a reconstruction term that has me modeling low-level noise at the visible
layer. If I go fully autoregressive, p(x) = ∏_i p(x_i | x_{<i}), I get an *exact* likelihood and
real flexibility — wonderful — but to draw one image I sample D pixels one after another in a
fixed order, which is sequential, non-parallelizable, painful at image scale, and leaves me with
no latent representation at all and a nagging dependence on the ordering I picked. And if I go
adversarial, I get sharp samples and I sidestep both inference and the likelihood, but that's the
problem: no likelihood means no density evaluation and no way to measure diversity, and there's
no encoder from x back to the latent.

Stare at that list. The exact-likelihood families are either intractable to train or sequential
to sample; the easy-to-sample families give up either the exact likelihood or the exact encoder.
I want the maximum-likelihood principle — for an exact objective and an exact, *deterministic*
encoder — without the partition function and without the sequential bottleneck. Is there any
corner of probability where exact maximum likelihood costs me nothing extra and still hands me a
generator I can invert?

There is one. Suppose my encoder f, the thing carrying x to its latent h, is not just some net
but a *bijection* — invertible, same dimension on both sides, with inverse f^{-1} carrying h back
to x. Then I don't need a discriminator to dodge the likelihood and I don't need an approximate
posterior to do inference, because inference is literally h = f(x), exact and direct, and
sampling is draw h ~ p_H, return x = f^{-1}(h). What's the density it induces? Push an
infinitesimal volume dx through f; it lands on a volume |det(∂f/∂x)| dx in h-space. Probability
mass is conserved: p_X(x) dx = p_H(h) dh, so

  p_X(x) = p_H(f(x)) · |det(∂f(x)/∂x)|,

and taking logs,

  log p_X(x) = log p_H(f(x)) + log |det(∂f(x)/∂x)|.

With the factorial prior I wanted, the first term splits into a sum, and I get the whole training
criterion:

  log p_X(x) = Σ_d log p_{H_d}(f_d(x)) + log |det(∂f(x)/∂x)|.

Pick a simple fixed prior p_H — say independent logistic or Gaussian coordinates — and I can
compute the exact log-likelihood of any x, train by maximizing it directly, sample by inverting,
and infer by f. All four wishes from one identity.

I should pause on that second term, because at first glance it looks like bookkeeping and it's
actually the load-bearing piece. Think about what an invertible *preprocessing* could do to a
naive likelihood. If I just contract the data toward a point, I pile probability mass into a tiny
region and the density there shoots up — I can inflate likelihood arbitrarily by shrinking,
which is meaningless. The determinant term is exactly the correction that kills that cheat: when
f contracts volume, |det ∂f/∂x| is small and log of it is negative, paying back precisely the
likelihood I tried to steal. So the det term *penalizes contraction and rewards expansion at the
data points* — it pushes the model to spend representational volume on the regions that actually
carry data. That's the right inductive pressure, not an accident. And there's a clean way to see
the kinship with the variational view: if f is a deterministic, exactly invertible encoder, then
f and f^{-1} are a *perfect* autoencoder pair, so a reconstruction term would be a constant I can
drop. What's left of a variational criterion is the prior term — log p_H(f(x)), pushing the code
to be likely under the prior, the analogue of the KL — and the log-det term, which measures local
volume change, the analogue of the entropy. The VAE pays for an imperfect, noisy encoder with a
reconstruction term and an injected-noise decoder; the bijective route pays nothing for either
because the encoder is exact.

So why isn't everyone already training densities this way? Because of that determinant. For an
arbitrary f on R^D, ∂f/∂x is a D×D Jacobian, and its determinant is an O(D^3) computation — and a
numerically badly-conditioned one — that I'd have to redo every gradient step, on top of needing
f itself to be invertible. For an image with thousands of dimensions that's hopeless per step and
fragile besides. The clean exact objective is being strangled by one term. The entire problem
reduces to a design question: can I build an f that is genuinely expressive, trivially
invertible, *and* whose Jacobian determinant I can read off cheaply?

When is a determinant cheap? The determinant of a triangular matrix is just the product of its
diagonal entries — O(D), no factorization, nothing to condition badly. So if I could force the
Jacobian of f to be triangular *by construction*, the determinant collapses to a sum of logs of
diagonal terms and the obstruction evaporates. And this isn't a trick sitting off to the side:
it's exactly *why* autoregressive models are tractable in the first place — conditioning each
coordinate on the earlier ones makes the implied map's Jacobian strictly triangular. The trouble
is that autoregressive triangularity comes bundled with sequential sampling, which is the very
thing I'm trying to escape. I want the triangular Jacobian *without* the sequential inverse. And
composition is on my side: if I write f = f_L ∘ … ∘ f_1, the forward pass composes the layers, the
inverse composes the layer-inverses in reverse order, and the Jacobian determinant is the product
of the layers' determinants. So I only need to design a tractable *elementary* bijection and stack
it.

One obvious way to get a triangular Jacobian is to literally build a net out of triangular weight
matrices with invertible activations. But that strangles the architecture — once the weight
matrices must be triangular, the only design freedom I have left is depth and the choice of
nonlinearity, and I can't pour in arbitrary capacity. I want triangular *Jacobian* without
triangular *weights*. Let me get there from the structure of the map instead.

Split the D coordinates of x into two blocks via a partition I_1, I_2 with d = |I_1|. Leave the
first block completely untouched, and transform the second block using only a function of the
first:

  y_{I_1} = x_{I_1},
  y_{I_2} = g( x_{I_2} ; m(x_{I_1}) ),

where g is some "coupling law" — a map that is invertible in its first argument once its second
argument is fixed — and m is whatever function I like. Why does this single move help on every
axis at once? Look at the Jacobian. The top-left block ∂y_{I_1}/∂x_{I_1} is the identity, because
that half is copied straight through. The top-right block ∂y_{I_1}/∂x_{I_2} is zero, because the
copied half doesn't depend on the second half at all. So the Jacobian is automatically
block-lower-triangular,

  ∂y/∂x = [[ I_d , 0 ],
           [ ∂y_{I_2}/∂x_{I_1} , ∂y_{I_2}/∂x_{I_2} ]],

and its determinant is the product of the diagonal blocks, det = det(∂y_{I_2}/∂x_{I_2}). The messy
off-diagonal block — the one stuffed with the derivatives of m — *never enters the determinant*.
That's the crucial consequence: m can be arbitrarily complicated, a deep ReLU MLP with as much
capacity as I want, and it still contributes nothing to the determinant. And invertibility is
free in one parallel pass: given y, the first block already hands me x_{I_1} = y_{I_1}, which is
exactly the argument m was conditioned on, so I can recover x_{I_2} = g^{-1}(y_{I_2}; m(y_{I_1}))
as long as g is invertible in its first slot — and I *never have to invert m itself*. Expressive
conditioner, trivial inverse, triangular Jacobian. That's the coupling layer.

Now I need to choose the coupling law g. The simplest invertible-in-first-argument map is pure
addition: g(a; b) = a + b, so b = m(x_{I_1}) just shifts the second block,

  y_{I_2} = x_{I_2} + m(x_{I_1}),   inverse   x_{I_2} = y_{I_2} − m(y_{I_1}).

The inverse is a subtraction — as cheap as the forward pass — and the bottom-right Jacobian block
∂y_{I_2}/∂x_{I_2} is the identity, because a pure shift has unit slope in its own argument. So the
determinant of the *whole* additive coupling is exactly 1, and its log-det contribution is zero.
That's gorgeous for tractability: the determinant term in my criterion just vanishes for these
layers, and m can be as deep as I please.

Could I do something richer than addition and still keep the inverse trivial? Sure — a
multiplicative law g(a; b) = a ⊙ b with b ≠ 0, or an affine law g(a; b) = a ⊙ b_1 + b_2 with
b_1 ≠ 0 if m outputs two vectors. Those would give a non-unit bottom-right block and so a
non-trivial determinant. I'll hold that thought, but I'm going to *choose* additive for now, and
the reason is numerical: if m is a rectified (ReLU) network, the additive coupling is piecewise
*linear*, which is stable to optimize, whereas multiplying by a learned, possibly tiny or huge
factor invites blow-ups and vanishing. Stability wins here; I take the additive law and accept
unit determinant for the moment.

One coupling layer leaves a whole block — I_1 — completely unchanged. That can't be the end. I
have to compose layers and *alternate which block gets frozen*, so that whatever was held fixed
in one layer is the thing being transformed in the next. How many do I need? After one layer, I_1
is untouched. After a second layer with the roles swapped, I_2 (now the frozen one) gets a
function of the already-modified I_1 — but I should check carefully whether every coordinate can
genuinely influence every other. Walk the Jacobian of the composition: with two alternating
layers, the dependency still hasn't closed the loop in both directions for all coordinates. It
takes *at least three* coupling layers before all dimensions can influence one another through the
composed map. I'll use a few more than the minimum to be safe — four — so the stack is genuinely
expressive in both partitions.

Now I hit the wall I parked earlier. Every additive coupling has unit Jacobian determinant, so a
composition of them *also* has unit determinant — the whole map is **volume-preserving**. It can
bend, fold, and shuffle probability mass around space, but it can never locally compress or
expand it. And density estimation is fundamentally about volume change: I need to *contract*
volume where the data piles up (so the density there is high) and *expand* it where data is
sparse. A volume-preserving f simply cannot do that. I've built an expressive, invertible,
cheap-determinant map whose determinant is stuck at 1 — which means the one term I fought to make
tractable is now identically the most boring value it could take, and it does no modeling work at
all.

What's the minimal change that keeps the gift and lifts the constraint? The unit determinant came
entirely from the coupling stack. I don't want to spoil the additive layers' stability. So put
the volume change somewhere else: bolt a single diagonal scaling as the *top* layer, mapping
h_i → S_ii · h_i. A pure diagonal map is trivially invertible (divide) and its Jacobian is
diagonal, so its log-determinant is just Σ_i log|S_ii| — still O(D), still clean. To keep S_ii
strictly positive and make the log-det especially simple, parametrize it exponentially: write the
scaling as h = exp(s) ⊙ h^{(top)} with a free vector s, so S_ii = exp(s_i) and log|S_ii| = s_i.
Now the criterion, summing the per-layer log-dets (zero for every coupling) plus this last one, is

  log p_X(x) = Σ_i [ log p_{H_i}(f_i(x)) + log|S_ii| ]  =  Σ_i log p_{H_i}(f_i(x)) + Σ_i s_i,

and the scaling restores exactly the volume freedom the couplings lacked. It's worth noticing what
this diagonal does semantically. The two terms it sits between pull against each other: the prior
term wants the code small (high prior density near the origin), which pushes the scales down; but
the log-det term Σ_i s_i grows as the scales grow, and as s_i → −∞ (S_ii → 0) it goes to −∞, so it
*forbids* any scale collapsing to zero. The equilibrium it settles into tells me how much
variation the model decided to keep on each latent coordinate — σ_d = S_dd^{-1} reads like the
scale of independent component d, the nonlinear analogue of a PCA eigenspectrum. A coordinate with
S_ii large (σ_d small) is one the model chose to suppress; in the limit S_ii → ∞ it has effectively
removed a dimension, which is the model discovering a lower-dimensional manifold inside the data.
The expensive couplings do the bending; one cheap diagonal does the volume accounting and the
manifold selection.

I still owe the prior a choice. It has to be factorial, and I'll take it from a standard family.
Gaussian per coordinate gives log p_{H_d}(h) = −½(h² + log 2π). Logistic per coordinate gives
log p_{H_d}(h) = −log(1 + e^{h}) − log(1 + e^{−h}) = −softplus(h) − softplus(−h). Which is better
to train against? The score — the gradient of the log-prior — is what flows back into f, and for
the Gaussian that score is −h, which is *unbounded*: a code that lands far out gets a huge gradient
that can yank the transform around. For the logistic the score is the difference of two sigmoids,
σ(−h) − σ(h), which lives in (−1, 1) and saturates gently. A bounded, better-behaved gradient is
easier to optimize, so I'll generally reach for the logistic prior.

Before I trust the whole edifice, let me make sure the framework is actually expressive enough to
recover things I know it should contain, because that's the real test of whether the couplings are
a serious model class or a toy. Two checks.

First, plain whitening. Can this framework learn a Gaussian? Take a single affine map z = Lx + b
with L lower triangular and b a bias, scored against a standard Gaussian prior. The Jacobian is L,
which is triangular, so its log-determinant is Σ log|L_ii| — tractable, same machinery. Maximizing
the change-of-variables likelihood of this affine-into-Gaussian model is exactly fitting a
Gaussian by maximum likelihood, with L playing the role of the Cholesky factor of the inverse
covariance. So the framework subsumes learned whitening as the linear special case — good, the
machinery is consistent, and I can even use such an affine NICE model as an approximate whitening
front-end before the nonlinear couplings.

Second, and more striking: where does the variational autoencoder sit relative to this? Let me try
to *write a VAE as one of these models* and see what falls out. The reparameterized VAE has a
recognition net z = g_φ(ε | x) with ε ~ N(0, I), a generator, and a Gaussian conditional p(x | z)
with scale σ. Define the standardized residual ξ = (x − f_θ(z)) / σ, and put a standard Gaussian
prior jointly on h = (z, ξ). The map from (ε, ξ) to the observed pair is invertible: z is recovered
from ε and x by the recognition net, and x = f_θ(z) + σξ recovers x. This is a two-coupling-layer
NICE model on the pair (x, ε) — one coupling realizes z from ε given x, the other realizes x from
ξ given z — with the σ-scaling supplying the only nontrivial log-determinant, −D_X log σ. Write its
change-of-variables log-density on (x, ε):

  log p_{(x,ε)}(x, ε) = log p_H(h) − D_X log σ + log |det ∂g_φ/∂ε(ε; x)|.

Now subtract log p_ε(ε) from both sides. The change-of-variables for the recognition net relates
its prior-pushforward to q, so that combination collapses the g_φ Jacobian term together with
log p_ε into −log q_{Z|X}(z):

  log p_{(x,ε)}(x, ε) − log p_ε(ε) = log p_H(h) − D_X log σ − log q_{Z|X}(z).

Split the standard-Gaussian h-prior into its two independent pieces, log p_H(h) = log p_ξ(ξ) +
log p_Z(z), and group:

  = log p_ξ(ξ) − D_X log σ + log p_Z(z) − log q_{Z|X}(z).

But log p_ξ(ξ) − D_X log σ is precisely log p_{X|Z}(x | z): a Gaussian conditional on x with mean
f_θ(z) and scale σ has log-density equal to the standard-Gaussian log-density of the standardized
residual ξ minus the D_X log σ from the change of variables x = f_θ(z) + σξ. So

  = log p_{X|Z}(x | z) + log p_Z(z) − log q_{Z|X}(z),

which is exactly the Monte-Carlo estimate of the SGVB / ELBO objective. The VAE *is* a NICE model
on the augmented pair (x, ε) with two affine couplings and a Gaussian prior — it's just maximizing
the joint likelihood of (x, ε). That the variational machinery drops out of the change-of-variables
identity this cleanly is the strongest evidence I have that the coupling framework is the right
general object, with the VAE as a special, noise-augmented case.

Now the practical preprocessing question, because it bites continuous-likelihood models hard.
Pixels are discrete — 256 levels per channel — but I'm fitting a *continuous* density. A continuous
density evaluated on a finite set of discrete points can place an arbitrarily tall, thin spike on
each one and send the likelihood to +∞; the number becomes meaningless and the training degenerate.
The fix is to dequantize: add uniform noise the width of one quantization bin and rescale into a
bounded box, so the discrete grid becomes continuous data and the expected log-likelihood gets a
proper finite upper bound. Concretely, add uniform noise of 1/256 and rescale to [0, 1]^D for the
grayscale and house-number sets, and for CIFAR-10 add 1/128 and rescale to [−1, 1]^D. When I want
the model-agnostic bits-per-dimension number for comparison, I undo the rescaling bookkeeping by
adding back D·log(256) to the negative log-likelihood and dividing by D·log 2.

Let me now lay out the whole architecture I'll actually train. Partition the pixels into odd
(I_1) and even (I_2) coordinates — a simple checkerboard-in-1D split that mixes spatially adjacent
information. Stack four additive coupling layers, alternating which parity is transformed at each
layer:

  h^{(1)}_{I_1} = x_{I_1},            h^{(1)}_{I_2} = x_{I_2} + m^{(1)}(x_{I_1})
  h^{(2)}_{I_2} = h^{(1)}_{I_2},      h^{(2)}_{I_1} = h^{(1)}_{I_1} + m^{(2)}(h^{(1)}_{I_2})
  h^{(3)}_{I_1} = h^{(2)}_{I_1},      h^{(3)}_{I_2} = h^{(2)}_{I_2} + m^{(3)}(h^{(2)}_{I_1})
  h^{(4)}_{I_2} = h^{(3)}_{I_2},      h^{(4)}_{I_1} = h^{(3)}_{I_1} + m^{(4)}(h^{(3)}_{I_2})

and finish with the diagonal scaling h = exp(s) ⊙ h^{(4)}. Each m^{(k)} is a deep rectified MLP
with a linear output layer — five hidden layers of 1000 units for the grayscale digits, four of
5000 for faces, four of 2000 for the house-numbers and natural-image sets. Logistic prior for the
sets where it trains best, Gaussian for faces. Optimize the exact log-likelihood with Adam at
learning rate 10^{-3}.

Let me turn this into code. The coupling layer reshapes the flat vector into pairs and routes by
parity, runs the conditioner MLP on the frozen half, and adds (or, in reverse, subtracts) the
shift on the transformed half — no log-determinant because additive couplings contribute zero:

```python
import torch
import torch.nn as nn

class Coupling(nn.Module):
    """One additive coupling layer: freeze one parity, shift the other by an MLP of it.
    y_on = x_on + m(x_off); inverse subtracts. Jacobian is unit-diagonal -> det = 1."""
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        super().__init__()
        self.mask_config = mask_config            # which parity is the frozen conditioner half
        self.in_block = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
            for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)   # linear output head

    def forward(self, x, reverse=False):
        B, W = x.size()
        x = x.reshape(B, W // 2, 2)
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]
        # the conditioner m never has to be inverted, so it can be arbitrarily deep
        out = self.in_block(off)
        for block in self.mid_block:
            out = block(out)
        shift = self.out_block(out)
        on = on - shift if reverse else on + shift     # additive law; inverse is subtraction
        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape(B, W)


class Scaling(nn.Module):
    """The diagonal volume-change the couplings can't do. h = exp(s) * x; log|det| = sum(s)."""
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, dim))     # s; exp(0)=1 at init

    def forward(self, x, reverse=False):
        log_det_J = torch.sum(self.scale)                  # sum_i s_i
        x = x * torch.exp(-self.scale) if reverse else x * torch.exp(self.scale)
        return x, log_det_J


class NICE(nn.Module):
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config):
        super().__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim
        self.coupling = nn.ModuleList([                    # alternate the frozen parity each layer
            Coupling(in_out_dim, mid_dim, hidden, (mask_config + i) % 2)
            for i in range(coupling)])
        self.scaling = Scaling(in_out_dim)

    def f(self, x):                                        # encoder: data -> latent
        for layer in self.coupling:
            x = layer(x)
        return self.scaling(x)                             # returns (z, log_det_J)

    def g(self, z):                                        # decoder: latent -> data (exact inverse)
        x, _ = self.scaling(z, reverse=True)
        for layer in reversed(self.coupling):
            x = layer(x, reverse=True)
        return x

    def log_prob(self, x):                                 # the exact NICE criterion
        z, log_det_J = self.f(x)
        return torch.sum(self.prior.log_prob(z), dim=1) + log_det_J

    def sample(self, n):                                   # h ~ prior, x = f^{-1}(h)
        z = self.prior.sample((n, self.in_out_dim))
        return self.g(z)

    def forward(self, x):
        return self.log_prob(x)
```

and the factorial logistic prior, the one with the bounded score:

```python
import torch
import torch.nn.functional as F

class StandardLogistic(torch.distributions.Distribution):
    def log_prob(self, z):
        # log p(z) = -softplus(z) - softplus(-z), per coordinate
        return -(F.softplus(z) + F.softplus(-z))

    def sample(self, size):
        u = torch.rand(size)                  # inverse-CDF sampling of the logistic
        return torch.log(u) - torch.log1p(-u)
```

trained against the exact likelihood with Adam:

```python
optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, betas=(0.9, 0.01), eps=1e-4)

for x in dataloader:                          # x already dequantized into a continuous box
    loss = -flow.log_prob(x).mean()           # maximize exact log-likelihood
    optimizer.zero_grad(); loss.backward(); optimizer.step()

# comparable bits/dim undoes the dequantization rescaling:
# bpd = (loss + math.log(256.) * in_out_dim) / (in_out_dim * math.log(2.))
```

So the causal chain, start to finish: I wanted exact likelihood, exact inference, and easy unbiased
sampling together, which forced me onto the change-of-variables identity with a bijective encoder;
that identity was strangled by an O(D³) Jacobian determinant; triangular Jacobians make the
determinant O(D), and a coupling layer — freeze half, shift the other half by an arbitrary network
of the frozen half — gives a triangular Jacobian with a trivial inverse and lets the conditioner be
as deep as I like; the cheapest coupling law, additive, is numerically stable but volume-preserving,
so I added one exponentiated diagonal scaling on top to restore volume change and, with it, an
automatic manifold-and-importance accounting over the latent coordinates; a factorial logistic prior
with its bounded score makes the whole exact objective easy to optimize; and dequantization makes the
continuous-density numbers well posed. The same identity, run on an augmented pair, reproduces the
variational autoencoder as a special case, which is what convinces me the coupling framework is the
right general object rather than a one-off.
