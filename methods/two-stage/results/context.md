# Context: feature learning in two-layer networks on multi-index targets

## Research question

Take Gaussian inputs `z ~ N(0, I_d)` in high dimension and a target that depends on the
input only through a low-dimensional projection:

```
y = f*(z) = g*( <w1*, z>, ..., <wr*, z> ),
```

where `w1*, ..., wr*` are orthonormal "teacher" directions spanning an `r`-dimensional
subspace `V* = span(w1*, ..., wr*)` with `r << d`. This is the *multi-index model*: all of
the structure lives in `V*`, the inputs themselves are isotropic. A two-layer network

```
f_hat(z; W, a) = (1/sqrt(p)) sum_i a_i * sigma( <w_i, z> )
```

with first-layer weights `W in R^{p x d}` and second-layer weights `a in R^p` is the
simplest universal approximator that could, in principle, *adapt* to this structure — its
first-layer rows could rotate to line up with `V*`, after which the readout only has to fit
an `r`-dimensional function. Kernel/random-feature methods cannot do this: with a fixed
first layer they do not adapt to which directions matter, and pay the high-dimensional
price for it.

The precise problem: **how, and how fast (in samples), do the first-layer rows align with
`V*` under gradient descent?** Concretely, can we get the alignment
`||Pi* w_i|| / ||w_i||` (the fraction of a neuron's weight inside `V*`, `Pi*` the
orthogonal projector onto `V*`) to become order one, with a sample budget that beats the
naive rate — and with dynamics tractable enough to analyze? A solution has to (i) make the
first layer rotate into `V*`, (ii) then turn those rotated features into an actual
predictor of `f*`, and (iii) do both with a sample/iteration cost that is provably better
than the kernel and one-pass-SGD baselines below.

## Background

**Hermite analysis of Gaussian targets.** Under `z ~ N(0, I_d)` the natural orthonormal
basis for `L^2(R^d, gamma)` is the Hermite tensors `H_k(z)`. Any square-integrable `f`
expands as `f(z) = sum_k <C_k(f), H_k(z)>`, with `C_k(f)` an order-`k` tensor, and the
inner product factorizes as `<f, g>_gamma = sum_k <C_k(f), C_k(g)>`. For a low-rank
target `f*(z) = g*(W* z)` with `W* W*^T = I_r`, the coefficients satisfy
`C_k(f*) = C_k(g*) . (W*, ..., W*)` (multilinear contraction), so **every singular vector
of every `C_k(f*)` lies in `V*`** — the Hermite tensors of the target *are* the directions
worth learning. The single most important scalar is the index of the first nonzero
coefficient:

- **Information exponent / leap index `ell`.** For a single-index target this is the
  *information exponent* (Ben Arous, Gheissari, Jagannath 2021): the smallest `k` with
  `C_k(f*) != 0`. Its multi-index generalization is the *leap index*
  `ell = min{ k > 0 : <f*, H_k>_gamma != 0 }`. It controls how visible the relevant
  directions are to a gradient: a degree-`ell` target hides its directions behind `ell-1`
  orders of "flatness," and that is exactly what makes them hard to find.

**Stein's lemma** is the workhorse identity: for `z ~ N(0,I)`,
`E[z h(z)] = E[grad_z h(z)]`. Applied to a first-layer gradient it converts the raw
correlation `E[z sigma'(<w,z>) f*(z)]` into Hermite-tensor contractions of `f*` against
powers of `w`, which is how one reads off *which* Hermite component a single gradient step
can see.

**Two regimes of a two-layer network.** Fixing `W` at its random initialization and only
training `a` gives a kernel model with feature map `z -> sigma(W z)` — the *conjugate
kernel* (CK), and the *random-features* (RF) model when `W` is random. The opposite — a
large enough step on `W` — is the *feature-learning* regime. The contrast between them is
the whole story:

- **The degree-`k` barrier of random features (Mei, Misiakiewicz, Montanari).** With a
  *fixed* first layer, an RF/kernel model in the proportional regime can only learn a
  degree-`k` polynomial approximation of the target once `n, p = Omega(d^{k + delta})`. So
  to capture a degree-`ell` piece of `f*` a kernel needs `n = Theta(d^ell)` samples, and
  it *never* picks up which low-dimensional directions matter — it pays the full ambient
  dimension.

- **The "early phase" empirical finding.** Practically, the non-kernel behaviour of
  network training shows up in the *first few* gradient steps, and is most pronounced under
  large learning rates; the representation moves fastest at the start. This motivates
  asking what a *single* (or a few) large gradient step(s) on the first layer already buy
  you over the fixed kernel.

**Scaling / parametrization.** Writing the network with a `1/sqrt(p)` prefactor (so the
second layer is divided by an extra `sqrt(p)` relative to the kernel/NTK scaling) is the
*mean-field*-style parametrization under which the first-layer neurons actually move and
align with the target during training, rather than staying frozen near initialization as
in the lazy/NTK scaling.

## Baselines

**One-pass SGD on a randomly initialized network (information-exponent rate).** Run plain
stochastic gradient descent, one (or few) samples at a time, jointly or on the first
layer, from a random start. Ben Arous, Gheissari & Jagannath (2021) show via a
summary-statistics / overlap analysis that learning a single-index target with information
exponent `ell` requires `n = tau = Theta(d^{ell-1})` samples and iterations (`d log d` for
`ell = 2`, `d^{ell-1}` for `ell > 2`). The mechanism: at a random start the overlap of the
weights with the unknown direction is `O(1/sqrt(d))`, so the gradient's correlation with it
is tiny and the walk away from the initial saddle is slow; escaping takes polynomially many
steps. **Gap:** the rate degrades steeply with `ell`, the saddle escape is sequential and
slow, and the joint dynamics of `W` and `a` are coupled and hard to analyze cleanly.

**Random-features / conjugate-kernel ridge regression (Mei et al.; Ba et al.'s lazy
regime).** Freeze the first layer at initialization `W^0`, build features `sigma(W^0 z)`,
and fit the second layer by ridge regression. This is convex and has a closed form, and its
generalization in the proportional limit is by now precisely characterized via Gaussian
equivalence. **Gap:** with `W^0` fixed the model is *non-adaptive* — it learns at most a
degree-`min(kappa_1, kappa_2)` polynomial (`p = O(d^{kappa_1})`, `n = O(d^{kappa_2})`) and
gets no benefit from the low-dimensional structure of `f*`; it pays `n = Theta(d^k)` to
reach a degree-`k` piece and never discovers `V*`. Even taking one *small* (`eta = Theta(1)`)
gradient step on `W` first leaves the model in this same "linear/lazy" regime — the trained
features are equivalent to a noisy linear model and cannot beat the best linear predictor on
the input.

**Multi-direction recovery with a large batch and label preprocessing (Damian, Lee,
Soltanolkotabi 2022).** For multi-index polynomials `f* = g*(U z)` they take *one* gradient
step on the first layer with a large batch `n = O(d^2)`, after first *preprocessing the
labels* to subtract a plug-in estimate of the low-degree Hermite components
(`y_nu <- y_nu - sum_{m<k} <c_hat_m, H_m(z_nu)>`, `c_hat_m = (1/n) sum_nu y_nu H_m(z_nu)`),
then fit the second layer. Removing the dominant low-degree component lets the single
gradient step "see" several directions at once and specialize to them. **Gap:** it is tied
to a single step and to `n = O(d^2)`; whether that batch size is *necessary*, and what a few
steps with the much smaller `n = O(d)` could do, is left open — and the preprocessing is an
extra estimation stage that is only reliable for `n = omega(d polylog d)`.

**Staircase / leap analyses on Boolean and one-pass-SGD settings (Abbe, Boix-Adsera,
Misiakiewicz 2021/2022/2023).** A "staircase" property of the target — each relevant
coordinate appearing in a monomial together with already-learned ones — characterizes which
sparse Boolean functions one-pass SGD learns with `O(d)` samples, later partially extended
to isotropic Gaussian data with batch-one SGD (requiring coordinate-wise projections to
escape saddles). **Gap:** the sharp results are Boolean and/or basis-dependent and rely on
one-pass batch-one SGD with vanishing per-step correlation, so saddle escape is slow and the
analysis needs explicit projections; a basis-independent, large-batch, Gaussian-data
treatment that also turns recovered features into a generalization bound is missing.

## Evaluation settings

Pre-method yardsticks, all available at the time:

- **Data model.** Isotropic Gaussian inputs `z ~ N(0, I_d)`; an orthonormal teacher
  `w1*, ..., wr*` drawn from the Haar measure; a fixed polynomial / Hermite link `g*` whose
  lowest nonzero Hermite degree sets the leap index `ell`. Standard student activations:
  ReLU, or a chosen Hermite polynomial.
- **Comparisons.** Against (a) random-features / CK ridge at the same `(n, p)` (the
  degree-`k` kernel barrier), and (b) one-pass SGD at its information-exponent rate. The
  proportional/linear-width regime `n, d, p -> infinity` with `n/d`, `p/d` fixed is the
  setting in which kernel performance is precisely known.
- **Metrics.** Population/prediction risk `E[(f_hat(z) - f*(z))^2]` on a fresh test set;
  and *feature learning* directly: how much of each first-layer row lies in `V*`, e.g. the
  alignment ratio `||Pi* w_i||^2 / ||w_i||^2` or the subspace recovery error
  `||P_Uhat - P_U*||_F` between the learned and true subspaces. Number of gradient steps
  (iteration complexity) and batch size (sample complexity) are reported as the resource
  axes.

(Settings only — no method outcomes.)

## Code framework

The existing substrate: a two-layer network with a `1/sqrt(p)` prefactor, a chosen
elementwise activation `sigma` and its derivative, a Gaussian data sampler for the teacher,
and a closed-form ridge solver. Stage-1 first-layer training and the way the two layers are
combined into a predictor are the empty slots.

```python
import numpy as np

# --- existing primitives ---------------------------------------------------

def sigma(x):            # chosen activation (e.g. relu), already available
    return np.maximum(x, 0.0)

def sigma_prime(x):      # its derivative
    return (x > 0).astype(float)

def net(Z, W, a, p):     # two-layer network with 1/sqrt(p) prefactor
    return (1.0 / np.sqrt(p)) * sigma(Z @ W.T) @ a

def sample_data(n, d, teacher, link):
    """Fresh Gaussian batch and its multi-index labels."""
    Z = np.random.randn(n, d)
    Y = link(Z @ teacher.T)           # y = g*(U z)
    return Z, Y

def ridge_estimator(X, y, lam):
    """Closed-form ridge readout; picks the cheaper normal-equation branch."""
    m, p = X.shape
    if m >= p:
        return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)
    else:
        return X.T @ np.linalg.solve(X @ X.T + lam * np.eye(m), y)


# --- slots the method will fill in -----------------------------------------

def init_first_layer(p, d):
    # random first-layer rows; second layer fixed separately
    # TODO: the initialization the procedure will use
    pass

def train_first_layer(W, a, teacher, link, p, d, eta, lam, n, T):
    """Stage 1: how the first layer is driven from the data.

    The rule that updates W from fresh batches (and any control of the row
    norms) is exactly what we have to design.
    """
    # TODO: the first-layer training procedure we will design
    pass

def fit_predictor(W, a, teacher, link, p, d, lam, n):
    """Stage 2: how the trained features are turned into a predictor of f*."""
    # TODO: the readout we will design
    pass
```

The method fills `init_first_layer`, `train_first_layer`, and `fit_predictor`; the final
code reuses `net`, `sample_data`, and `ridge_estimator` unchanged.
