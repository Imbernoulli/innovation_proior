## Research question

Fix two integers `k << n`. Draw a uniform binary vector `x` (write it as `x in {±1}^n`, with
`{0,1}` and XOR an equivalent encoding) and label it by the parity of a hidden size-`k` subset
`S` of the coordinates: `y = chi_S(x) = prod_{i in S} x_i` (equivalently `y = (sum_{i in S} x_i)
mod 2`). The learner sees `(x, y)` pairs and must recover an accurate classifier — in effect,
identify `S`.

This problem sits on a sharp computational/statistical divide. *Statistically* it is trivial:
to single out one of the `C(n,k)` candidate subsets needs only `log C(n,k) = Theta(k log n)`
labelled examples. *Computationally* it is believed hard: a large class of algorithms provably
needs `n^{Omega(k)}` work, far more than the handful of samples that information-theoretically
determine the answer. The gap between these two is exactly what makes sparse parity a clean
stress test for *optimization* rather than statistics.

The pressing question is not a new algorithm hand-built for parity (Gaussian elimination already
solves the noiseless case in `O(n^3)`). It is a *diagnostic* question about ordinary deep
learning: when you train a generic neural network on this task with off-the-shelf initialization
and an off-the-shelf optimizer — no sparsity prior baked into the architecture, no special
trick — does it work, how fast, and *by what mechanism*? A satisfactory answer has to confront a
specific puzzle: the training loss and accuracy curves on this task sit flat at chance for a long
time and then jump discontinuously to a solved state. Is the long flat phase a blind search that
happens to get lucky, or is something measurable improving underneath it the whole time? And how
can either be true given a well-known argument that the gradient on such problems carries
essentially no information about which parity is the target?

## Background

**Orthogonality of parities.** The parity functions are mutually orthonormal under the
correlation inner product: for any `S, S' subseteq [n]`,
`E_{x~Unif}[ chi_S(x) chi_{S'}(x) ] = 1` if `S' = S` and `0` otherwise. Equivalently
`E_{(x,y)~D_S}[ chi_{S'}(x) y ] = 1[S'=S]`. The `{chi_S}` therefore form an orthonormal basis of
all functions `f: {±1}^n -> R`, with the Fourier expansion `f = sum_S fhat(S) chi_S`,
`fhat(S) = E_x[ f(x) chi_S(x) ]`. The brutal consequence for a learner who guesses a subset `S'`:
its correlation with the labels is exactly zero unless `S'` is *exactly* `S`. There is no partial
credit — overlapping `k-1` of the right indices reads the same (zero correlation) as overlapping
none. A learner that only ever sees correlations gets no "warmer/colder" signal and is forced
into exhaustive search over the `C(n,k)` subsets.

**The statistical-query (SQ) lower bound (Kearns 1998).** The SQ model formalizes "learning
through aggregate statistics rather than individual examples": the learner submits a query
`q: {±1}^n -> [-1,1]` and an oracle returns any value within tolerance `tau` of
`E_{D}[q]`. Almost every noise-robust, gradient-style learner is an SQ algorithm. Parity
orthogonality implies that any single query has non-trivial correlation (above `tau`) with only a
`1/tau^2` fraction of the parities, so to single out the right one an SQ algorithm needs
`T / tau^2 >= Omega(n^k)`: queries times precision-squared cannot beat exhaustive search. The
noiseless problem escapes the SQ model via Gaussian elimination, but learning *noisy* sparse
parity even at vanishing noise is conjectured to need `n^{Omega(k)}` time (Alekhnovich 2003), a
hardness that underwrites several cryptosystems. Gradient descent with a constant gradient-noise
level is essentially an SQ algorithm (Abbe & Sandon 2020), so it inherits this floor: `n^{O(k)}`
iterations is the *best one could hope for* from a gradient method, and matching it would be
striking.

**Gradient concentration: "no signal" (Shalev-Shwartz, Shamir & Shammah 2017).** This line gives
the pessimistic reading. For the objective `F_h(w) = E_x[ loss(p_w(x), h(x)) ]` of learning a
target `h` drawn from a family `H`, define the gradient's *signal about the target* as the
variance over the family,
`Var(H, F, w) = E_h || grad F_h(w) - E_{h'} grad F_{h'}(w) ||^2`. Their Theorem 1: if the
members of `H` are pairwise orthogonal with `E[h^2] <= 1`, the predictor has
`E||d p_w/dw||^2 <= G(w)^2`, and the loss is the square loss or a 1-Lipschitz classification loss,
then `Var(H, F, w) <= G(w)^2 / |H|`, *independent of the network architecture or width*. Applied
to the family of *all* `2^d` parities on `d` bits, this gives `Var <= G(w)^2 / 2^d` —
exponentially small. By Chebyshev, the gradient at any fixed point is essentially the same no
matter which parity is the target; the reported experiments confirm that learning a random
length-`d` parity stalls around `d = 30`. The stated conclusion is that no gradient-based method
should succeed on random parities. The bound's strength is its target family: it is taken over
*all* parities, a family of every degree, so the gradient genuinely averages to a target-blind
constant.

**Fourier spectrum of majority (O'Donnell 2014; Titsworth 1962).** For random-sign or all-ones
weights `w in {±1}^n` and `|b| < 1`, the threshold `1[w·x + b > 0]` equals
`(1 + Maj(w ⊙ x))/2`, where `Maj(x) = sgn(sum_i x_i)` (with the usual odd-`n`/tie-convention
caveat). Its Fourier coefficients are therefore, up to sign and a factor `1/2`, those of
majority. Majority's spectrum is known in closed form: writing `xi_k := \widehat{Maj}([k])`, one has
`xi_k = 0` for even `k`,
and for odd `k`,
`xi_k = (-1)^{(k-1)/2} [ C((n-1)/2, (k-1)/2) / C(n-1, k-1) ] · 2^{-(n-1)} C(n-1, (n-1)/2)`,
with `|xi_k|` decaying like `n^{-(k-1)/2}`. The coefficients at adjacent degrees obey
`|xi_{k-1}| = ((n-k)/(k-1)) |xi_{k+1}|`, so a low-degree coefficient strictly dominates the
neighboring higher-degree one.

**The lazy / neural-tangent-kernel regime (Jacot, Gabriel & Hongler 2018).** When a wide network's
weights barely move from initialization, training is well approximated by a linear model in fixed
features — kernel regression with the neural tangent kernel — and is convex. In this regime the
features are frozen at init; nothing is "learned" beyond the readout layer. A counting/span
argument bounds what such fixed features can represent with margin: for any `D`-dimensional
embedding `Psi` with `sup_x ||Psi(x)|| <= 1` and a norm budget `R`, if `D R^2 < eps^2 C(n,k)` then
some size-`k` subset `S` has `inf_{||w||<=R} E[loss(Psi(x)·w, y)] > 1 - eps`. To express all
parities with margin a fixed-feature model needs `D = Omega(n^k)`.

**Grokking on algorithmic data (Power et al. 2022).** On small modular-arithmetic datasets,
training accuracy saturates early while validation accuracy stays at chance for a long time and
then *suddenly* generalizes; weight decay accelerates this delayed transition, and smaller
datasets need more optimization before it occurs. This is the multi-pass, finite-data
counterpart of a long flat phase followed by an abrupt jump.

**Diagnostic observations about the training dynamics (black-box, no internal probing).** A set
of measurements about ordinary networks on this task, knowable by watching loss/accuracy and
timing convergence:
- Convergence-time histograms over very many random trials have a heavy upper tail but
  essentially *no* mass near `t = 0`. A memoryless random search over subsets would have
  convergence time distributed as `Geom(1/C(n,k))`, whose mass is *highest* at `t = 0` and
  decreases monotonically — the opposite shape.
- Convergence time *adapts to* the sparsity `k`, scaling like `n^{O(k)}` on small instances,
  rather than the `2^{Omega(n)}` that a search over parameters or subsets would give.
- Loss curves and convergence times are concentrated *conditioned on the random initialization*,
  and only weakly dependent on the stochasticity of the batches: the trajectory is largely
  determined by where training starts.
- For larger `n` the power-law exponent worsens (an "elbow" in the scaling curve).
- Increasing model width gives sharply diminishing returns, not the `1/r` speedup that running
  `r` parallel searches would produce.

## Baselines

These are the prior positions a diagnostic account of parity learning sits against.

**Exhaustive / random search over subsets.** The brute-force baseline: enumerate (or randomly
probe) the `C(n,k)` subsets and test each. Forced by orthogonality on any correlation-only
learner; matches the SQ floor `n^{Omega(k)}` but with the *memoryless* signature above — earliest
successes most likely, convergence time independent of the algorithm's state, perfect parallel
speedup with more copies. **Limitation:** its predicted statistics (mode-at-zero histogram, no
`k`-adaptivity beyond the subset count, linear parallel speedup) are flatly contradicted by the
observed black-box dynamics; as a description of what a trained network is doing it does not fit.

**Stochastic gradient Langevin dynamics ("stumbling in the dark").** A refinement of the search
reading: SGD with its noise behaves like a diffusion bouncing around the loss landscape until it
falls into the basin of the solution, with no useful drift in between. It predicts a
`2^{Omega(n)}`-ish convergence time and is consistent with the loss looking flat. **Limitation:**
it predicts the same memoryless, init-insensitive, parallelizable behavior as exhaustive search,
and likewise clashes with the measured concentration-conditioned-on-init and `n^{O(k)}` scaling.
It also offers no quantity that improves during the plateau.

**The gradient-concentration verdict (Shalev-Shwartz et al. 2017).** As above: over the family of
all `2^d` parities the gradient variance is `<= G^2/2^d`, so "the gradient carries no signal" and
gradient methods should fail. **Limitation / where it stalls:** the bound is taken over the dense
family of *all* parities, of every degree. The sparse problem fixes `|S| = k`, so the relevant
family has only `C(n,k) ≈ n^k` members; the very same theorem then reads `Var <~ G^2 / n^k`,
which is small only *polynomially* in `n^k`, not exponentially. The pessimistic conclusion is an
artifact of measuring against the wrong (much larger) family; what the bound actually leaves open
is whether — and through what concrete scalar quantity — that polynomial-sized residual signal
can be picked up by a gradient method, and what it would look like over the course of training.

**Fixed-feature / NTK learning of parity (Jacot et al. 2018 and its parity analyses).** Treat the
network as a fixed kernel and fit the readout convexly. Clean and convex; gives good *sample*
complexity for `k = 2` parity. **Limitation:** by the counting bound above, fitting all size-`k`
parities with margin in this regime requires `D = Omega(n^k)` features, i.e. an enormously wide
network. It cannot account for success at modest or tiny width, where the weights demonstrably
must move away from their initialization — so it describes a different regime than the one in
question.

**Statistical-mechanics analyses of the parity machine.** A long line in the statistical physics
of learning studies the "parity machine" — the sign of a product of `k` linear units — in the
thermodynamic limit, characterizing equilibrium learning curves and, in some treatments, plateaus
in the generalization error along the training trajectory. **Limitation:** these analyses
typically target an idealized infinite-size limit or a Gibbs-equilibrium endpoint, and have not
pinned down that `k`-sparse parity is learnable by gradient descent in a number of iterations
near the known lower bound, nor isolated the trajectory-level quantity that drives the
plateau-then-jump in this specific problem.

## Evaluation settings

The natural yardsticks for a diagnostic study of this problem, all definable before any mechanism
is proposed:

- **Task.** `(n,k)`-sparse parity for small `k` (e.g. `k in {2,3,4}`) over a range of `n`; the
  hidden subset `S` drawn uniformly from `C(n,k)`. By permutation symmetry one may fix `S = [k]`
  without loss of generality.
- **Online (single-pass) protocol.** At each step draw a *fresh* i.i.d. batch of `B` examples
  from `D_S` and take one gradient step; this couples iterations to independent samples and
  removes overfitting as a confound. A separate finite-sample (multi-pass) protocol — reuse a
  fixed dataset for many epochs — is the natural setting for studying delayed generalization.
- **Architectures.** A fixed two-layer MLP `f(x; W,b,u) = u^T sigma(Wx + b)` of width `r` is the
  default; `sigma` is ReLU or a degree-`k` polynomial. Single-neuron variants (special
  activations), the parity machine / its real-output analogue, and a Transformer are alternative
  vehicles.
- **Optimizers and losses.** Plain SGD (and the Adam family for some architectures), with hinge,
  square, cross-entropy, or — for the cleanest analysis — the correlation loss
  `ell(y, yhat) = -y·yhat`. Standard initializations: uniform (Glorot/Xavier scale), Gaussian
  (Kaiming scale), or random-sign.
- **Metrics.** Held-out classification accuracy / loss vs. iterations, the convergence time `t_c`
  (first step reaching a target accuracy), and its scaling with `n` and `k`; convergence-time
  distributions over many random seeds; and dependence of `t_c` on resources (samples, width,
  batch size). Population quantities are estimated on a large fixed validation batch.

## Code framework

The substrate is the standard supervised-training harness, already in place. A fixed two-layer
MLP is built once; an initialization routine sets its parameters from a data-independent random
scheme; a batch routine produces fresh `(x, y)` examples for the hidden secret; an
optimizer-configuration routine returns the hyperparameters; and a fixed driver runs minibatch
training and reports held-out accuracy. The open slots are exactly the data-independent choices a
study would vary, plus one generic state-reporting hook.

```python
import torch
from torch import nn


def build_model(n_features: int, width: int = 512) -> nn.Sequential:
    """Fixed two-layer MLP: Linear -> ReLU -> Linear -> Sigmoid. Width is fixed."""
    return nn.Sequential(
        nn.Linear(n_features, width),
        nn.ReLU(),
        nn.Linear(width, 1),
        nn.Sigmoid(),
    )


def parity_labels(x: torch.Tensor, secret: tuple[int, ...]) -> torch.Tensor:
    """Label = parity (XOR / sum-mod-2) of the bits at the hidden indices."""
    idx = torch.tensor(secret, dtype=torch.long, device=x.device)
    return x.index_select(1, idx).sum(dim=1).remainder(2).to(torch.float32)


def init_model(model: nn.Sequential) -> None:
    """Set parameters from a data-INDEPENDENT random scheme (must not see `secret`)."""
    # TODO: the initialization scheme to use here.
    pass


def make_online_batch(secret, n_features: int, batch_size: int, generator):
    """Produce one training batch (x in {0,1}^n, y = parity_labels(x, secret))."""
    # TODO: the batch-sampling rule to use here.
    pass


def get_optimizer_config() -> dict:
    """Return the optimizer hyperparameters for the fixed training loop."""
    # TODO: the optimizer hyperparameters to use.
    pass


def make_test_set(secret, n_features: int, test_size: int, generator):
    """Produce held-out examples for the fixed evaluation routine."""
    # TODO: the held-out sampling rule to use here.
    pass


def state_probe(model: nn.Sequential, init_state) -> float:
    """Optional scalar read from the training state for logging."""
    # TODO: the state statistic to report.
    pass


def evaluate_accuracy(model: nn.Module, test_x, test_y, device) -> float:
    model.eval()
    with torch.no_grad():
        preds = model(test_x.to(device)).view(-1)
        return ((preds >= 0.5) == (test_y.to(device) >= 0.5)).float().mean().item()


def train_parity(secret, n_features, width=512, batch_size=128, steps=100_000,
                 test_size=16_384, seed=0, device="cpu"):
    """Fixed driver: build, init, optimize on fresh minibatches, evaluate."""
    model = build_model(n_features, width).to(device)
    init_model(model)
    init_state = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt_cfg = get_optimizer_config()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"],
        betas=(opt_cfg["beta1"], opt_cfg["beta2"]),
        eps=opt_cfg["eps"],
        weight_decay=opt_cfg["wd"],
    )
    criterion = nn.BCELoss()
    train_gen = torch.Generator().manual_seed(seed)
    test_gen = torch.Generator().manual_seed(seed + 1)
    test_x, test_y = make_test_set(secret, n_features, test_size, test_gen)

    for step in range(1, steps + 1):
        xb, yb = make_online_batch(secret, n_features, batch_size, train_gen)
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb).view(-1), yb)
        loss.backward()
        optimizer.step()
        if step % 250 == 0:
            _ = state_probe(model, init_state)

    return evaluate_accuracy(model, test_x, test_y, device)
```

The data pipeline (`parity_labels`, the fresh-batch loop), the `AdamW` optimizer, the `BCELoss`,
and the fixed `build_model` already exist. The open slots are `init_model`, `make_online_batch`,
`make_test_set`, `get_optimizer_config`, and `state_probe`.
