# PAC-Bayes-λ bound, distilled

The PAC-Bayes-λ bound (a Catoni-style trade-off bound, the "flamb" objective) is a
relaxation of the PAC-Bayes-kl inequality that introduces a trade-off parameter `λ`, making
the right-hand side **convex in the posterior `ρ` for fixed `λ`** and **convex in `λ` for
fixed `ρ`** — and holding **uniformly over all `λ ∈ (0,2)`** simultaneously. This lets a
stochastic predictor's posterior be trained by directly minimizing a generalization
certificate (no cross-validation of the empirical-risk / complexity trade-off), with the
final certificate read off by the tighter PAC-Bayes-kl inversion.

## Problem it solves

Learn a randomized classifier (a distribution `ρ` over hypotheses) by minimizing its
PAC-Bayes generalization bound over `ρ`. The tightest standard bound, PAC-Bayes-kl, is not
convex in `ρ`, so it cannot be used as a training objective; the usual fallback — a linear
trade-off `E_ρ[L̂] + β·KL/n` with `β` cross-validated — is expensive and can mislead. The goal
is a bound that is nearly as tight, convex in `ρ`, and whose trade-off is set from the data.

## Key idea

Start from PAC-Bayes-kl, `kl(E_ρ[L̂] ‖ E_ρ[L]) ≤ (KL(ρ‖π) + ln(2√n/δ))/n`. Relax the binary
KL by the refined Pinsker inequality `kl(p‖q) ≥ (q−p)²/(2q)` (for `p<q`):
```
E_ρ[L] − E_ρ[L̂] ≤ √( 2·E_ρ[L]·(KL(ρ‖π) + ln(2√n/δ))/n ).
```
The unknown `E_ρ[L]` is under the root. Linearize the root with the deterministic identity
`√(xy) ≤ ½(λx + y/λ)`, valid for *all* `λ > 0` (so the resulting bound holds for all `λ`,
unlike Catoni's fixed-`λ` bound). With `x = E_ρ[L]`, `y = 2(KL + ln(2√n/δ))/n`:
```
(1 − λ/2) E_ρ[L] ≤ E_ρ[L̂] + (KL(ρ‖π) + ln(2√n/δ))/(λn),
```
and dividing by `1 − λ/2 > 0` (i.e. `λ < 2`) gives the bound.

## The bound

For any prior `π` independent of `S`, any `δ ∈ (0,1)`, with probability `≥ 1−δ` over `S`,
simultaneously for all `ρ` and all `λ ∈ (0,2)`:
```
E_ρ[L(h)] ≤  E_ρ[L̂(h,S)]        +        KL(ρ‖π) + ln(2√n/δ)
            ─────────────                ───────────────────────
              1 − λ/2                      λ (1 − λ/2) n
```

- **Convex in `ρ` (fixed `λ`):** `E_ρ[L̂]` is linear, `KL(ρ‖π)` convex. Minimizer is the
  **Gibbs posterior** `ρ_λ(h) = π(h) e^{−λ n L̂(h,S)} / E_π[e^{−λ n L̂(h',S)}]`.
- **Convex in `λ` (fixed `ρ`):** form `c₁/(1−λ/2) + c₂/(λ(1−λ/2))`. Minimizer
  `λ = 2 / ( √( 2n E_ρ[L̂]/(KL + ln(2√n/δ)) + 1 ) + 1 )`, always `< 1`, and `≥ 1/√n` for
  `n ≥ 4`.
- **Alternating minimization:** apply the two closed-form updates in turn; each decreases the
  bound monotonically. (Not jointly convex, so this gives a local minimum a priori.)

## Global convergence (1-D reduction + quasiconvexity)

Substituting `ρ_λ` (using `KL(ρ_λ‖π) = −nλ E_{ρ_λ}[L̂] − ln E_π[e^{−nλL̂}]`) collapses the
bound to a single-variable function:
```
F(λ) = ( −ln E_π[e^{−nλ L̂(h,S)}] + ln(2√n/δ) ) / ( nλ(1 − λ/2) ) = f(λ) g(λ),
```
with `f(λ) = −(1/n)ln E_π[e^{−nλL̂}] + ln(2√n/δ)/n` and `g(λ) = 1/(λ(1−λ/2))`.

Derivatives: `f'(λ) = E_{ρ_λ}[L̂] ≥ 0`, `f''(λ) = −n Var_{ρ_λ}[L̂] ≤ 0` (so `f` is concave);
`g'(λ) = (λ−1)/(λ²(1−λ/2)²)`, `g''(λ) = (3(λ−1)²+1)/(2λ³(1−λ/2)³) > 0` (so `g` is convex).
Stationary points satisfy `(KL(ρ_λ‖π) + ln(2√n/δ))/n = λ² E_{ρ_λ}[L̂]/(2(1−λ))`, forcing
`λ ≥ √(ln(2√n/δ)/n)` (for `n ≥ 7`). At a stationary point
```
F''(λ) = (1/(λ(1−λ/2))) · ( E_{ρ_λ}[L̂]/(1−λ) − n Var_{ρ_λ}[L̂] ),
```
positive iff `E_{ρ_λ}[L̂] > (1−λ) n Var_{ρ_λ}[L̂]`  (equiv. `2KL(ρ_λ‖π) + ln(4n/δ²) >
λ²n² Var_{ρ_λ}[L̂]`). If either condition holds for all `λ ∈ [√(ln(2√n/δ)/n), 1]`, every
stationary point is a local min ⇒ `F` is **strongly quasiconvex** ⇒ alternating minimization
reaches the **global** minimum.

**Finite-`H` sufficient condition.** With uniform prior, set
`x_h = L̂(h,S) − min_h L̂(h,S)`, `a = √(ln(4n/δ²))/(n√3)`, `b = ln(3mn²)/√(n ln(2√n/δ))`,
`K = (e²/12)ln(4n/δ²)`. If at most `K` hypotheses have `x_h ∈ (a,b)` ("mediocre"), then
`Var_{ρ_λ}[L̂] ≤ ln(4n/δ²)/(λ²n²)` for all `λ` in the window, so `F` is strongly quasiconvex.
(Proof bounds `E_{ρ_λ}[x²]` over good/mediocre/bad pieces, using `x²e^{−nλx} ≤ 4/(e²n²λ²)`
and the monotone tail; the interval cutoffs `a,b` can be retuned via fractions `α,β`.)

## Constructed hypothesis space (when `H` is infinite / partition function intractable)

Train `m` weak learners, each on `r` subsampled points, validate each on the remaining `n−r`.
Everything carries over with `n → n−r` and `L̂ → L̂_val`, since validation errors are `(n−r)`
i.i.d. with mean `L(h)`, so `E_S[e^{(n−r)kl(L̂_val,L)}] ≤ 2√(n−r)`.

## Engineering instantiation (stochastic NN, PAC-Bayes-with-Backprop)

- Posterior `ρ` = diagonal Gaussian per weight (reparametrized `σ = log(1+exp(ρ))`),
  analytic `KL(ρ‖π)` to a Gaussian prior.
- Loss bounded into `[0,1]`: clamp log-probabilities at `ln(pmin)` so NLL `≤ ln(1/pmin)`,
  rescale by `1/ln(1/pmin)`.
- Alternating minimization by SGD: one step descending the bound in the posterior params,
  then a **separate** step descending it in `λ` (its own optimizer; otherwise `λ` stays at
  init). The raw `λ` parameter is sigmoid-scaled into `[1/√n, 1]`; the PBB code initializes
  the raw value high, so the scaled value starts just below `1`.
- Data-dependent prior: split the data, ERM-train a deterministic net on one half to set the
  prior mean, minimize/evaluate the bound on the other half — shrinks `KL`.
- Final certificate via **PAC-Bayes-kl inversion** on the learned `ρ` (tighter than the `λ`
  bound used for training): `risk_cert = inv_kl(emp_risk, (KL + ln(2√n/δ))/n)`.

## Working code

Fills the `compute_bound` / `train_step` / `compute_risk_certificate` slots of the harness
(helpers `get_total_kl`, `inv_kl(q,c)`, `compute_01_risk`, and `model(x, sample=…)` already
exist):

```python
import math
import torch
import torch.nn.functional as F


class Lambda_var(torch.nn.Module):
    """Sigmoid-scaled lambda variable used by the PBB flamb objective."""

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = torch.nn.Parameter(torch.tensor([lamb], dtype=torch.float32))
        self.min = 1.0 / math.sqrt(n)

    @property
    def lamb_scaled(self):
        return torch.sigmoid(self.lamb) * (1.0 - self.min) + self.min


class BoundOptimizer:
    """PAC-Bayes-lambda bound. Convex in the posterior for fixed lambda and in
    lambda for fixed posterior; minimized by alternating descent. The lambda
    parameter is sigmoid-scaled into [1/sqrt(n), 1], matching PBB flamb."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-4, initial_lamb=6.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        self.initial_lamb = initial_lamb
        self.lambda_var = None
        self._posterior_optimizer = None
        self._lambda_optimizer = None

    def _ensure_state(self, model, n, device):
        if self._posterior_optimizer is None:
            self._posterior_optimizer = torch.optim.SGD(
                model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        if self.lambda_var is None:
            self.lambda_var = Lambda_var(self.initial_lamb, n).to(device)
            self._lambda_optimizer = torch.optim.SGD(
                self.lambda_var.parameters(), lr=self.learning_rate,
                momentum=self.momentum)

    def _kl(self, model):
        return model.compute_kl() if hasattr(model, "compute_kl") else get_total_kl(model)

    def _bounded_loss_and_error(self, model, data, target):
        log_probs = model(data, sample=True, clamping=True, pmin=self.pmin)
        loss_ce = F.nll_loss(log_probs, target) / math.log(1.0 / self.pmin)
        pred = log_probs.max(1, keepdim=True)[1]
        error = 1.0 - pred.eq(target.view_as(pred)).sum().item() / target.size(0)
        return loss_ce, error

    def compute_bound(self, empirical_risk, kl, n, delta):
        # L_hat/(1 - lam/2) + (KL + ln(2 sqrt(n)/delta)) / (n lam (1 - lam/2))
        lam = self.lambda_var.lamb_scaled
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (
            n * lam * (1.0 - lam / 2.0))
        return empirical_risk / (1.0 - lam / 2.0) + kl_term

    def train_step(self, model, data, target, device, n_bound, delta):
        self._ensure_state(model, n_bound, device)
        model.train()
        self.lambda_var.train()
        data, target = data.to(device), target.to(device)

        self._posterior_optimizer.zero_grad()
        kl = self._kl(model)
        loss_ce, err = self._bounded_loss_and_error(model, data, target)
        bound = self.compute_bound(loss_ce, kl, n_bound, delta)
        bound.backward()
        self._posterior_optimizer.step()

        self._lambda_optimizer.zero_grad()
        kl_l = self._kl(model)
        loss_ce_l, err_l = self._bounded_loss_and_error(model, data, target)
        lam_bound = self.compute_bound(loss_ce_l, kl_l, n_bound, delta)
        lam_bound.backward()
        self._lambda_optimizer.step()

        return {
            "train_bound": bound.item(),
            "lambda_bound": lam_bound.item(),
            "kl_per_sample": (kl / n_bound).item(),
            "bounded_nll": loss_ce.item(),
            "train_error": err,
            "lambda": self.lambda_var.lamb_scaled.item(),
        }

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 delta_test=0.01, mc_samples=1000):
        self._ensure_state(model, len(bound_loader.dataset), device)
        model.eval()
        n_bound = len(bound_loader.dataset)

        total_ce, total_01, batches = 0.0, 0.0, 0
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                ce_mc, err_mc = 0.0, 0.0
                for _ in range(mc_samples):
                    loss_ce, err = self._bounded_loss_and_error(model, data, target)
                    ce_mc += loss_ce.item()
                    err_mc += err
                total_ce += ce_mc / mc_samples
                total_01 += err_mc / mc_samples
                batches += 1

        raw_ce = total_ce / batches
        raw_01 = total_01 / batches
        empirical_ce = inv_kl(raw_ce, math.log(2.0 / delta_test) / mc_samples)
        empirical_01 = inv_kl(raw_01, math.log(2.0 / delta_test) / mc_samples)

        kl = self._kl(model).item()
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta_test)) / n_bound
        risk_ce = inv_kl(empirical_ce, c)
        risk_01 = inv_kl(empirical_01, c)

        ce_bound = self.compute_bound(
            torch.tensor(empirical_ce, device=device),
            torch.tensor(kl, device=device),
            n_bound,
            delta,
        ).item()

        metrics = {
            "raw_mc_ce": raw_ce,
            "raw_mc_01": raw_01,
            "empirical_ce": empirical_ce,
            "empirical_01": empirical_01,
            "kl_divergence": kl,
            "risk_ce": risk_ce,
            "ce_bound": ce_bound,
            "lambda": self.lambda_var.lamb_scaled.item(),
        }
        return risk_01, metrics
```

## Relation to prior bounds

- **PAC-Bayes-kl** (Seeger 2002; Maurer 2004): the tightest standard bound, but non-convex in
  `ρ`. PAC-Bayes-λ is a convex relaxation of it, used here as the training objective; the
  learned posterior is certified afterward by PAC-Bayes-kl inversion.
- **PAC-Bayes-classic / McAllester**: `E_ρ[L̂] + √((KL+ln(2√n/δ))/(2n))`, from Pinsker
  `kl ≥ 2(p−q)²`. Loose at low risk (additive `√`-penalty doesn't shrink).
- **PAC-Bayes-quadratic**: `(√(E_ρ[L̂]+κ)+√κ)²` from refined Pinsker solved as a quadratic in
  `√(E_ρ[L])`; tight at low risk but no separate tunable trade-off.
- **Catoni's parametrized bound** (Catoni 2007): also convex in `ρ` with the Gibbs posterior
  optimum, but holds only for a *fixed* `λ` → tuning `λ` needs a union bound over a grid.
  PAC-Bayes-λ holds for **all** `λ ∈ (0,2)`, so `λ` is optimized continuously for free.
