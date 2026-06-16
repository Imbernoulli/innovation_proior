# SVRG — Stochastic Variance Reduced Gradient

SVRG minimizes a finite sum P(w) = (1/n) Σ_i ψ_i(w) at SGD's per-step cost but with gradient
descent's linear rate and a constant step size, by reducing the variance of the stochastic
gradient with a periodically refreshed full-gradient snapshot used as a control variate. It keeps
only O(d) extra memory — one snapshot vector and one average-gradient vector — so unlike
stored-table methods it applies to structured-prediction and neural-network objectives.

## Problem

Minimize P(w) = (1/n) Σ_{i=1}^n ψ_i(w) with each ψ_i convex and L-smooth and P γ-strongly convex
(L ≥ γ > 0; condition number κ = L/γ). Full gradient descent is linear but costs n gradients per
step. SGD costs one gradient per step but is only sublinear O(1/t): the per-example gradients do
not vanish at the optimum (∇ψ_i(w*) ≠ 0 while ∇P(w*) = 0), so with σ² = (1/n) Σ_i ‖∇ψ_i(w*)‖² > 0
a constant step η leaves a squared-distance floor of order η σ²/γ around w*; killing it forces
η_t → 0, which is what drags the rate to O(1/t). That O(1/t) is optimal for an oracle restricted to
unbiased gradient measurements, but the finite, fixed sum supplies more structure than that oracle,
which is the loophole SVRG exploits.

## Key idea — control variate with a periodic full-gradient snapshot

Estimate ∇P(w) by the classical control-variate estimator E[X] ≈ X − Y + E[Y]: take X = ∇ψ_i(w),
the same-index reference Y = ∇ψ_i(w̃) at a snapshot point w̃, and its exact mean
E_i[Y] = (1/n) Σ_i ∇ψ_i(w̃) = ∇P(w̃) =: μ̃ (one full pass). The update direction is

  v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃.

- **Unbiased:** E_i[v] = ∇P(w) − ∇P(w̃) + μ̃ = ∇P(w), for any reference w̃.
- **Vanishing variance:** the matched index i makes ∇ψ_i(w) − ∇ψ_i(w̃) → 0 and μ̃ = ∇P(w̃) → 0 as
  w, w̃ → w*, so v → 0. The per-example offset ∇ψ_i(w*) that floored SGD now appears in both terms
  at the same i and cancels — so a constant step survives, with no decay.

The exact μ̃ is the only costly piece, so it is amortized in epochs: snapshot w̃, compute μ̃ once,
run m cheap inner steps against the held (w̃, μ̃), then refresh. Only w̃ and μ̃ are stored — O(d), no
per-example gradient table (the cost SAG and SDCA pay, and the reason they don't port to neural
nets).

Equivalent view: v is plain SGD on a re-centered representation. With ψ̃_i(w) = ψ_i(w) −
(∇ψ_i(w̃) − μ̃)^T w one has ∇ψ̃_i(w) = v and (1/n) Σ_i ψ̃_i = P (the affine shifts average to zero),
so SVRG is SGD on the same objective re-centered to lower per-sample variance.

## Algorithm (Procedure SVRG)

```
Parameters: inner length m, step size η  (constant, η < 1/(2L))
Initialize  w̃_0
for s = 1, 2, ... :
    w̃ = w̃_{s-1}
    μ̃ = (1/n) Σ_{i=1}^n ∇ψ_i(w̃)               # one full pass
    w_0 = w̃
    for t = 1, ..., m :
        pick i_t ∈ {1, ..., n} uniformly
        w_t = w_{t-1} − η ( ∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃ )
    option I  (practice): w̃_s = w_m            # last iterate (or average of inner iterates)
    option II (analysis): w̃_s = w_t,  t ~ Uniform{0, ..., m-1}
```

Cost per epoch: n (for μ̃) + 2m (two example-gradients per inner step). Choose m = O(n) to amortize
the snapshot pass (e.g. m = 2n convex, m = 5n nonconvex). For linear models ∇ψ_i(w) =
φ_i'(w^T x_i) x_i, the snapshot gradient compresses to one stored scalar per example and need not be
recomputed (matching SAG/SDCA memory), dropping the inner cost.

## Linear-convergence theorem (option II)

**Theorem.** Assume every ψ_i is convex and L-smooth, P is γ-strongly convex with γ > 0, and
w* = argmin P. If m is large enough that

  α = 1/(γ η (1 − 2Lη) m) + 2Lη/(1 − 2Lη) < 1   (requires η < 1/(2L)),

then SVRG with option II converges geometrically in expectation:

  E[P(w̃_s) − P(w*)] ≤ α^s [P(w̃_0) − P(w*)].

At conditioning κ = L/γ = n, taking η = 0.1/L and a sufficiently large constant multiple of n
(for example m = 50n gives α = 0.5) gives O(n ln(1/ε)) gradient evaluations to reach accuracy ε
— versus n² ln(1/ε) for batch gradient descent — matching SAG/SDCA but with only O(d) memory.

### Proof

**Lemma (smoothness ⇒ gradient gap bounded by suboptimality).** Fix i and set
g_i(w) = ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*). Then ∇g_i(w*) = 0, so w* minimizes the convex
L-smooth g_i and g_i(w*) = 0. Minimizing the smoothness upper bound of one gradient step,

  0 = g_i(w*) ≤ min_{η'} [ g_i(w) − η' ‖∇g_i(w)‖² + 0.5 L η'² ‖∇g_i(w)‖² ] = g_i(w) − (1/2L)‖∇g_i(w)‖²,

so ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L [ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*)]. Averaging over i and using
∇P(w*) = 0 (the linear terms vanish),

  (1/n) Σ_i ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L [P(w) − P(w*)].    (8)

**Second-moment bound.** With v_t = ∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃, insert ∇ψ_{i_t}(w*) and
split v_t = a + b, a = ∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w*), b = ∇ψ_{i_t}(w*) − ∇ψ_{i_t}(w̃) + μ̃.
Using ‖a + b‖² ≤ 2‖a‖² + 2‖b‖², then ‖b‖² = ‖ξ − Eξ‖² and
E‖ξ − Eξ‖² = E‖ξ‖² − ‖Eξ‖² ≤ E‖ξ‖² on
ξ = ∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*) (whose mean is ∇P(w̃) = μ̃), and (8) at w_{t-1} and at w̃:

  E‖v_t‖² ≤ 2 E‖∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w*)‖² + 2 E‖∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)‖²
         ≤ 4L [P(w_{t-1}) − P(w*)] + 4L [P(w̃) − P(w*)].    (★)

This is the variance reduction made rigorous: as w_{t-1}, w̃ → w*, the right side → 0.

**One-step contraction.** E_{i_t}[v_t] = ∇P(w_{t-1}); with convexity
−(w_{t-1} − w*)^T ∇P(w_{t-1}) ≤ P(w*) − P(w_{t-1}) and (★),

  E‖w_t − w*‖² ≤ ‖w_{t-1} − w*‖² − 2η(1 − 2Lη)[P(w_{t-1}) − P(w*)] + 4Lη²[P(w̃) − P(w*)].

The middle coefficient is a genuine decrease iff η < 1/(2L) — a constant ceiling.

**Telescope + strong convexity.** Fix w̃ = w̃_{s-1}, w_0 = w̃; sum over t = 1..m (the ‖·−w*‖² terms
telescope), take full expectation, drop E‖w_m − w*‖² ≥ 0, use option II so
(1/m) Σ_t E[P(w_{t-1}) − P(w*)] = E[P(w̃_s) − P(w*)], and ‖w̃ − w*‖² ≤ (2/γ)[P(w̃) − P(w*)]:

  2η(1 − 2Lη) m · E[P(w̃_s) − P(w*)] ≤ (2/γ + 4Lmη²) · E[P(w̃) − P(w*)].

Dividing gives E[P(w̃_s) − P(w*)] ≤ α · E[P(w̃_{s-1}) − P(w*)]; iterating over s yields the bound. ∎

α's first term → 0 as m → ∞ and its second term is small for small η, so α < 1 is attainable for
η a constant fraction of 1/L and m = O(κ), which becomes O(n) in the κ = n comparison above. For
smooth convex P without strong convexity the same machinery gives O(1/T); for a nonconvex model
warm-started near a locally strongly convex minimum the theorem applies locally for local geometric
convergence with a constant step.

## Relation to prior methods (one mechanism, three costumes)

- **SAG** (Le Roux, Schmidt & Bach 2012): stores a per-example gradient table and steps along its
  average. Its stored gradients are a (biased, table-based) control variate; SVRG replaces the table
  with a single shared snapshot + its mean → O(d) instead of O(n).
- **SDCA** (Shalev-Shwartz & Zhang 2012): with α_i* = −(1/λn)∇φ_i(w*) and w = Σ_i α_i, its update
  direction ∇φ_i(w) + λn α_i → 0 as (w, α) → (w*, α*), so (1/n) Σ_i (∇φ_i(w) + λn α_i)² → 0 — SDCA's
  variance also vanishes at the optimum. Same mechanism; stored duals play the snapshot's role.

## Working code

Filling the update-direction slot of the finite-sum harness; the only state is the snapshot w̃ and
the average gradient μ̃ (both O(d)).

```python
import torch


class FiniteSumOptimizer:
    """Stochastic Variance Reduced Gradient for P(w) = (1/n) Σ_i ψ_i(w).

    Each epoch: snapshot w̃, pay one full pass for μ̃ = ∇P(w̃), then run m cheap inner steps
    along the control-variate direction
        v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃     (E_i[v] = ∇P(w); variance → 0 as w, w̃ → w*),
    so the step size stays CONSTANT and the rate is linear. Only w̃ and μ̃ are stored (O(d)).
    """

    def __init__(self, problem, lr, inner_steps):
        self.problem = problem            # grad_batch(w_state, idx), full_grad(), loss_batch(idx)
        self.params = problem.params      # the live parameters w
        self.lr = lr                      # constant η < 1/(2L)
        self.inner_steps = inner_steps    # m random inner updates per epoch
        self.snapshot = None              # w̃
        self.mu = None                    # μ̃ = ∇P(w̃)

    def _clone_params(self):
        return [p.data.clone() for p in self.params]

    def train_one_epoch(self):
        # snapshot: hold w̃ and compute the exact mean μ̃ = ∇P(w̃) (one full pass)
        self.snapshot = self._clone_params()
        self.mu = self.problem.full_grad()

        n, b = self.problem.n, self.problem.batch_size
        total_loss, n_batches = 0.0, 0

        for _ in range(self.inner_steps):
            idx = torch.randint(n, (b,))

            grad_cur = self.problem.grad_batch(self.params, idx)     # ∇ψ_i(w_{t-1})
            grad_snap = self.problem.grad_batch(self.snapshot, idx)  # ∇ψ_i(w̃), same indices

            # v = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃   (unbiased for ∇P(w))
            with torch.no_grad():
                for p, gc, gs, mu in zip(self.params, grad_cur, grad_snap, self.mu):
                    p.data.add_(gc - gs + mu, alpha=-self.lr)        # constant-step in-place update

            total_loss += float(self.problem.loss_batch(idx))
            n_batches += 1

        # refresh reference from the last inner iterate (option I)
        return {"avg_loss": total_loss / max(n_batches, 1), "full_grad_count": 1}
```

## Why it works, in one line

SGD's rate is capped by a variance floor (per-example gradients don't vanish at w*), not bias; a
matched-index control variate with a periodically refreshed full-gradient snapshot keeps the update
unbiased while driving its variance to zero as the iterate approaches the optimum, restoring linear
convergence with a constant step and O(d) memory.
