## Catoni / lambda bound (flamb)

**Problem.** The default scaffold trains against the McAllester additive bound
`L̂ + √((KL+log(2√n/δ))/(2n))`, whose complexity enters as a square root — its gradient w.r.t. KL
*shrinks* as KL grows, so it only weakly penalizes a posterior drifting from the prior. I want a bound
whose complexity term is linear in KL near the operating point, still convex in `Q` and differentiable.

**Key idea.** Relax the parent PAC-Bayes-kl inequality through Catoni's localized trade-off rather than
through Pinsker:
```
L(Q) <= L̂(Q)/(1 - λ/2) + (KL(Q‖P) + log(2√n/δ)) / (n·λ·(1 - λ/2)),   λ ∈ (0,2).
```
For fixed `λ` this is convex in `Q` (linear `L̂` + convex `KL`), and the complexity term is *linear*
in KL. The trade-off `λ` is tuned to the data by carrying it as a learnable scalar with its own
optimizer.

**Why it is the weakest rung here.** The bound is valid and non-vacuous, but in this implementation `λ`
is a free SGD scalar merely clamped into `(0.01, 1.99)` — not pinned to its analytic optimum. It can
co-adapt with the posterior into a small-`λ`, large-KL corner: as `λ→0` the empirical weighting
`L̂/(1−λ/2)→L̂` (cheapest) while the complexity prefactor `1/(nλ(1−λ/2))` explodes, so the optimizer is
tempted to let KL run. The NLL surrogate is also fed *unrescaled* (no `1/log(1/pmin)`), over-weighting
the empirical side against KL. Result: the KL balloons and the inverted certificate inflates.

**Why it differs from the generic PAC-Bayes-λ method.** The canonical flamb objective sigmoid-scales
`λ` into `[1/√n, 1]` and alternates two closed-form-motivated SGD steps with a rescaled bounded NLL;
this task's edit surface instead uses a *raw* learnable `λ` hard-clamped to `(0.01, 1.99)`, a detached
`λ`-update step, plain unrescaled NLL, and a single (uncorrected) `inv_kl` certificate.

**Hyperparameters.** `prior_sigma=0.03`, `pmin=1e-5`, `learning_rate=0.001`, `momentum=0.95`,
`initial_lambda=0.5`, `lambda_lr=0.01`, `delta=0.025`, `mc_samples=1000`. `λ` clamped to `(0.01, 1.99)`.

```python
class BoundOptimizer:
    """Catoni/Lambda PAC-Bayes bound (flamb).

    Bound: emp_risk / (1 - lam/2) + (KL + log(2*sqrt(n)/delta)) / (n*lam*(1 - lam/2))
    Lambda is a learnable parameter optimized jointly with the posterior.
    Tighter than McAllester when lambda is well-tuned.
    """

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5, initial_lambda=0.5, lambda_lr=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        # Lambda parameter for the Catoni bound (learnable)
        self._lambda_param = torch.tensor(initial_lambda, requires_grad=True)
        self.lambda_lr = lambda_lr
        self._lambda_optimizer = None

    def _get_lambda(self):
        """Get clamped lambda value in (0, 2)."""
        return torch.clamp(self._lambda_param, min=0.01, max=1.99)

    def _ensure_lambda_optimizer(self):
        if self._lambda_optimizer is None:
            self._lambda_optimizer = torch.optim.SGD(
                [self._lambda_param], lr=self.lambda_lr
            )

    def compute_bound(self, empirical_risk, kl, n, delta):
        """Catoni/Lambda bound."""
        lam = self._get_lambda()
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (
            n * lam * (1.0 - lam / 2.0)
        )
        bound = empirical_risk / (1.0 - lam / 2.0) + kl_term
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        """Training objective: Catoni/lambda bound with joint lambda optimization."""
        # Ensure lambda is on correct device
        if self._lambda_param.device != device:
            self._lambda_param = self._lambda_param.to(device).detach().requires_grad_(True)
            self._lambda_optimizer = None
        self._ensure_lambda_optimizer()

        output = model(data, sample=True)
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        nll = F.nll_loss(log_probs, target)

        kl = get_total_kl(model)

        # Update lambda on a detached copy — the outer loop's optimizer.step()
        # only knows about posterior params, so lambda would stay frozen at
        # init without this explicit step. Before the fix, lambda=1.0 caused
        # the Catoni bound to double the KL contribution (1-lam/2=0.5), which
        # forced KL to grow to ~10x McAllester's value.
        self._lambda_optimizer.zero_grad()
        lam = self._get_lambda()
        lam_bound = nll.detach() / (1.0 - lam / 2.0) + (
            kl.detach() + math.log(2.0 * math.sqrt(n_bound) / delta)
        ) / (n_bound * lam * (1.0 - lam / 2.0))
        lam_bound.backward()
        self._lambda_optimizer.step()

        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        """Evaluate Catoni risk certificate with PAC-Bayes-kl inversion."""
        model.eval()
        n_bound = len(bound_loader.dataset)

        # 1. Empirical 0-1 risk via MC sampling
        emp_risk_01 = compute_01_risk(model, bound_loader, device,
                                      mc_samples=mc_samples)

        # 2. NLL-based empirical risk
        total_nll = 0.0
        total_samples = 0
        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, sample=True)
                log_probs = F.log_softmax(output, dim=1)
                log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
                nll = F.nll_loss(log_probs, target, reduction="sum")
                total_nll += nll.item()
                total_samples += target.size(0)
        emp_nll = total_nll / total_samples

        # 3. KL divergence
        with torch.no_grad():
            dummy_data = next(iter(bound_loader))[0][:1].to(device)
            model(dummy_data, sample=True)
            kl = get_total_kl(model).item()

        # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # 5. CE bound using Catoni formula
        emp_nll_t = torch.tensor(emp_nll)
        kl_t = torch.tensor(kl)
        ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()

        metrics = {
            "empirical_01_risk": emp_risk_01,
            "empirical_nll": emp_nll,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
            "lambda": self._get_lambda().item(),
        }

        return risk_cert_01, metrics
```
