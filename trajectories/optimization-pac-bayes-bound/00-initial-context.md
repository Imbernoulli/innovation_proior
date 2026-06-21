## Research question

I need a single number, computed from the training data, that is a true high-probability upper bound on the unseen-data error of an overparameterized neural network. The object being designed is the **PAC-Bayes bound itself** — the functional that maps empirical risk and `KL(Q‖P)` to a risk certificate, the training objective derived from it, and the final certificate evaluation. Everything else — stochastic layers, the data-dependent prior split, the outer SGD loop — is fixed.

## Prior art / Background / Baselines

- **McAllester's PAC-Bayesian theorem.** Replaces the union-bound penalty `log|H|` with `KL(Q‖P)`, measuring how far the learner moves from a data-free prior rather than the size of the hypothesis class. The bound takes an additive square-root form: `L(Q) ≤ L̂(Q) + √((KL(Q‖P) + log(2√n/δ))/(2n))`.
- **Seeger / Langford-Caruana PAC-Bayes-kl.** Controls the binary KL between empirical and true risk, giving a bound implicit in the true risk. It can be inverted numerically (binary-KL inversion) to report a risk certificate.
- **Maurer's constant refinement.** Sharpens the exponential-moment constant, halving the sample-size dependence inside the logarithm of the McAllester-style bound.
- **Catoni's localization bound.** A fixed-`λ` trade-off that is convex in `Q`, with a Gibbs-posterior optimum: `L(Q) ≤ L̂(Q)/(1−λ/2) + (KL(Q‖P) + log(2√n/δ))/(n·λ·(1−λ/2))` for `λ ∈ (0,2)`.
- **PAC-Bayes with backprop and data-dependent priors.** Parameterizes `Q` as diagonal Gaussians over weights with an analytic KL, trains a deterministic prior on one half of the data, and minimizes a PAC-Bayes bound by SGD on the disjoint half.

## Fixed substrate / Code framework

The pipeline is frozen. `ProbLinear` and `ProbConv2d` layers carry Gaussian weights `N(μ, σ²)` with `σ = log(1+e^ρ)` and an analytic per-layer KL to a data-dependent Gaussian prior; `get_total_kl(model)` sums them. The loop: (i) ERM-train a deterministic model on `prior_frac=0.5` of the data and copy its weights into both posterior and prior means via `transfer_weights_to_stochastic`; (ii) minimize the bound on the disjoint bound split with SGD (`lr=0.001`, `momentum=0.95`, `prior_sigma=0.03`, `pmin=1e-5`). Two architectures are exercised: a `784-600-600-600-10` FCN and a 2-conv-2-fc CNN. Helpers provided: `model(x, sample=True/False)` (stochastic draw vs posterior mean), `inv_kl(q, c)` (binary-KL inversion by bisection), and `compute_01_risk(model, loader, device, mc_samples)` (Monte-Carlo estimate of the 0-1 risk).

## Editable interface

Exactly one region is editable — the `BoundOptimizer` class in `PBB/custom_pac_bayes.py` (lines 460–604). It must implement:

- `compute_bound(empirical_risk, kl, n, delta) -> bound` — the PAC-Bayes functional (a tensor, so it is differentiable when `empirical_risk`/`kl` carry gradients).
- `train_step(model, data, target, device, n_bound, delta) -> loss` — the training objective: stochastic forward pass, surrogate empirical risk, KL, bound formula, scalar loss for SGD.
- `compute_risk_certificate(model, bound_loader, device, delta, mc_samples) -> (risk_cert_01, metrics)` — the final certificate: MC-estimate the empirical 0-1 risk, read off the KL, PAC-Bayes-kl-invert to the reported `risk_certificate` (primary metric, **lower is better**), plus `test_error`, `kl_divergence`, `ce_bound`, `empirical_01_risk`.

The starting point is the scaffold default: the **McAllester/Maurer additive square-root bound**, with an unrescaled NLL surrogate and a single (uncorrected) `inv_kl` certificate. Replace exactly this class and nothing else.

```python
# EDITABLE region of PBB/custom_pac_bayes.py (lines 460-604) — default fill
class BoundOptimizer:
    """PAC-Bayes bound computation and posterior optimization.

    Default: McAllester/Maurer bound (fclassic).
      B(Q,S) = empirical_risk + sqrt((KL(Q||P) + log(2*sqrt(n)/delta)) / (2n))
    The goal is the tightest (lowest) 0-1 risk certificate.
    """

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.1,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin

    def compute_bound(self, empirical_risk, kl, n, delta):
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        bound = empirical_risk + torch.sqrt(kl_term)
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        output = model(data, sample=True)
        # Bounded cross-entropy as surrogate for 0-1 loss
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        nll = F.nll_loss(log_probs, target)
        kl = get_total_kl(model)
        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        model.eval()
        n_bound = len(bound_loader.dataset)

        # 1. Empirical 0-1 risk via MC sampling
        emp_risk_01 = compute_01_risk(model, bound_loader, device,
                                      mc_samples=mc_samples)

        # 2. NLL-based empirical risk for the CE bound
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

        # 3. KL from a single forward pass
        with torch.no_grad():
            dummy_data = next(iter(bound_loader))[0][:1].to(device)
            model(dummy_data, sample=True)
            kl = get_total_kl(model).item()

        # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # 5. Direct CE bound
        emp_nll_t = torch.tensor(emp_nll)
        kl_t = torch.tensor(kl)
        ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()

        metrics = {
            "empirical_01_risk": emp_risk_01,
            "empirical_nll": emp_nll,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
        }
        return risk_cert_01, metrics
```

## Evaluation settings

Three settings over seeds {42, 123, 456}: **MNIST-FCN** (`784-600-600-600-10` fully connected), **MNIST-CNN** (2 conv + 2 fc), and **FashionMNIST-CNN** (same CNN, harder dataset). The primary metric is `risk_certificate` (the 0-1 loss PAC-Bayes bound), **lower is better**. Also recorded: `test_error`, `kl_divergence`, `ce_bound`, and `empirical_01_risk`. Training uses a 50/50 data-dependent prior split.
