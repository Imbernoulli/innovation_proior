The problem is to learn a randomized classifier, a distribution ρ over hypotheses, by directly minimizing a PAC-Bayesian generalization certificate. PAC-Bayes bounds give the tightest certificates: with probability at least 1−δ over the sample S, the true risk E_ρ[L(h)] is controlled by the empirical risk E_ρ[L̂(h,S)] and the KL divergence KL(ρ‖π) to a data-independent prior π. The tightest standard bound is the PAC-Bayes-kl inequality, kl(E_ρ[L̂] ‖ E_ρ[L]) ≤ (KL(ρ‖π) + ln(2√n/δ))/n, which is inverted numerically to obtain a certificate. The trouble is that this certificate is not convex in ρ, so it cannot be used directly as a training objective. Practitioners therefore fall back on a linear surrogate E_ρ[L̂] + β·KL(ρ‖π)/n and choose β by cross-validation, which is expensive and can be misleading.

Existing convex relaxations do not fix the problem cleanly. The classic Pinsker relaxation gives an explicit additive bound, but the square-root penalty does not shrink with the empirical risk, so it is loose when the posterior is good. The refined Pinsker relaxation keeps low-risk tightness, but the unknown true risk remains trapped under a square root, making the bound awkward to optimize and lacking a freely tunable trade-off. Catoni's parametrized bound restores convexity in ρ and admits a closed-form Gibbs posterior minimizer, but it holds only for a single fixed trade-off parameter λ chosen before seeing the data. Tuning λ from the sample requires a union bound over a grid, which costs an extra logarithmic factor and yields only a discrete trade-off. What is needed is a bound that is convex in ρ, nearly as tight as PAC-Bayes-kl, and valid simultaneously for all values of the trade-off parameter so that the parameter can be optimized on the data without a union bound.

The method I propose is the PAC-Bayes-λ bound, also known as the flamb objective in the PAC-Bayes with Backpropagation framework. It starts from the refined Pinsker inequality kl(p‖q) ≥ (q−p)²/(2q) applied to PAC-Bayes-kl, which gives

E_ρ[L] − E_ρ[L̂] ≤ √( 2·E_ρ[L]·(KL(ρ‖π) + ln(2√n/δ))/n ).

The right-hand side has the form √(xy). The key step is to linearize the square root with the deterministic identity √(xy) ≤ ½(λx + y/λ), which holds for every λ > 0 at once. Setting x = E_ρ[L] and y = 2(KL(ρ‖π) + ln(2√n/δ))/n, rearranging, and dividing by 1 − λ/2 gives the PAC-Bayes-λ bound:

E_ρ[L] ≤ E_ρ[L̂]/(1 − λ/2) + (KL(ρ‖π) + ln(2√n/δ))/(λ(1 − λ/2)n),

valid with probability at least 1−δ simultaneously for all ρ and all λ ∈ (0,2). Because the linearizing identity holds uniformly, no union bound is paid for λ.

For fixed λ the bound is convex in ρ: the empirical risk is linear in ρ and the KL term is convex, so the minimizer is the Gibbs posterior ρ_λ(h) ∝ π(h) e^{−λ n L̂(h,S)}. For fixed ρ the bound is convex in λ, and its minimizer has the closed form λ = 2 / (√(2n E_ρ[L̂]/(KL + ln(2√n/δ)) + 1) + 1), which always lies below 1 and typically above 1/√n. Thus the training procedure is alternating minimization: fix λ and update ρ toward the Gibbs posterior, then fix ρ and update λ to its closed-form optimum, repeat. Each step decreases the bound monotonically.

The method is not jointly convex in (ρ,λ), but reducing to λ alone by substituting the Gibbs posterior gives a one-dimensional function F(λ) whose quasiconvexity can be checked. Its second derivative at stationary points is positive whenever E_{ρ_λ}[L̂] > (1−λ)n Var_{ρ_λ}[L̂], so under modest posterior-variance conditions alternating minimization converges to the global minimum. In practice the optimization is implemented by gradient descent: λ is represented as a learnable scalar sigmoid-scaled into [1/√n, 1], and each minibatch takes one SGD step on the posterior followed by one SGD step on λ. The loss must be bounded into [0,1] by clamping log-probabilities at ln(pmin) and rescaling the NLL by 1/ln(1/pmin). The final reported certificate is obtained by inverting the tighter PAC-Bayes-kl inequality on the learned posterior.

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
    """PAC-Bayes-lambda bound. Convex in the posterior for fixed lambda and
    convex in lambda for fixed posterior, so it is minimized by alternating
    descent over the posterior and over lambda. The lambda parameter is
    sigmoid-scaled into [1/sqrt(n), 1], matching the PBB flamb implementation."""

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
        # PAC-Bayes-lambda: L_hat/(1 - lam/2) + (KL + ln(2 sqrt(n)/delta)) / (n lam (1 - lam/2))
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
        # Certify with PAC-Bayes-kl inversion (tightest) on the learned rho,
        # not the relaxed lambda bound used for training.
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
