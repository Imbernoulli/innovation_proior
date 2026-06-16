## Research question

I have a neural network with far more parameters than training examples, and on MNIST it generalizes
to a few percent test error — but every capacity-based guarantee I can write for it (VC dimension,
Rademacher complexity) is vacuous, returning "error ≤ 1." I want a single number, computed from the
training data, that is a true high-probability upper bound on the unseen-data error and is small
enough to be worth printing. The one thing being designed is the **PAC-Bayes bound itself** — the
functional that turns an empirical risk and a posterior/prior KL into a risk certificate, the training
objective that drives the posterior toward a small certificate, and the final certificate evaluation.
Everything else — the stochastic layers, the data-dependent prior split, the outer SGD loop — is
fixed.

## Prior art before the first rung (PAC-Bayes lineage)

The certificate machinery the first rung reacts to is the resolution of a line of PAC-Bayesian
results. These are the ancestors that precede the ladder; the fixed substrate below is what they
converged to.

- **McAllester's PAC-Bayesian theorems (McAllester, *Machine Learning* 1999).** Replace the
  union-bound penalty `log|H|` — vacuous for an infinite hypothesis class — with `KL(Q‖P)`, the
  divergence of a data-dependent posterior `Q` from a data-free prior `P`. The complexity now measures
  how far the learner moved, not how big the menu was. Gap: the original constants are loose, and the
  bound it gives is the additive square-root form, which does not shrink as empirical risk shrinks.
- **Seeger / Langford-Caruana PAC-Bayes-kl (Seeger, 2002; Langford & Caruana, 2002).** Control the
  *binary KL* between empirical and true risk, `kl(L̂(Q)‖L(Q)) ≤ (KL(Q‖P)+log(·/δ))/n`. Provably the
  tightest of the standard bounds, sharp at low risk. Gap: it is implicit in the true risk and
  non-convex in `Q`, so it can be *inverted* to report a number but cannot be differentiated as a
  training objective.
- **Maurer's note (Maurer, arXiv:cs/0411099, 2004).** Sharpens the exponential-moment constant to
  `E_S[e^{n·kl(L̂‖L)}] ≤ 2√n` for `n ≥ 8`, halving the sample-size dependence inside the log from
  `log(2n)` to `log(2√n)`. Gap: it tightens the constant but leaves the bound's *shape* unchanged.
- **Catoni's localization (Catoni, IMS Lecture Notes 56, 2007).** A fixed-`λ` trade-off bound, convex
  in `Q`, whose `Q`-optimum is the Gibbs posterior `∝ P·e^{−λnL̂}`. Gap: it holds for a *single* `λ`
  chosen before the data, so tuning `λ` to the sample costs a union bound over a grid.
- **PAC-Bayes-with-Backprop / data-dependent priors (Dziugaite & Roy, UAI 2017, arXiv:1703.11008;
  Pérez-Ortiz, Rivasplata, Shawe-Taylor, Szepesvári, JMLR 2021, arXiv:2007.12911).** The substrate
  this task is built on: parameterize `Q` as diagonal Gaussians over weights with an analytic KL,
  train a deterministic prior on one half of the data and evaluate the bound on the disjoint half, and
  minimize the bound by SGD. Gap (the lever this ladder turns): with the substrate fixed, the
  *certificate's tightness is entirely decided by the bound functional and the prior/posterior
  coupling it induces* — the same posterior can produce a certificate of 0.12 or of 0.03 depending on
  which bound it was trained against.

## The fixed substrate

A PAC-Bayes-with-Backprop pipeline is frozen and must not be touched. Stochastic layers `ProbLinear`
and `ProbConv2d` carry Gaussian weights `N(μ, σ²)` with `σ = log(1+e^ρ)` and an analytic per-layer KL
to a data-dependent Gaussian prior; `get_total_kl(model)` sums them. The loop (i) ERM-trains a
deterministic model on a `prior_frac=0.5` split and copies its weights into both the posterior means
and the prior means via `transfer_weights_to_stochastic`, then (ii) minimizes the bound on the
disjoint bound split with SGD (`lr=0.001`, `momentum=0.95`, `prior_sigma=0.03`, `pmin=1e-5`). Three
architectures are exercised: a `784-600-600-600-10` FCN and a 2-conv-2-fc CNN. Helpers provided:
`model(x, sample=True/False)` (stochastic draw vs posterior mean), `inv_kl(q, c)` (the binary-KL
inversion: largest `p ≥ q` with `kl(q‖p) ≤ c`, by bisection), and `compute_01_risk(model, loader,
device, mc_samples)` (Monte-Carlo estimate of the 0-1 risk).

## The editable interface

Exactly one region is editable — the `BoundOptimizer` class in `PBB/custom_pac_bayes.py` (lines
460–604). Every method on the ladder is a fill of this same three-method contract:

- `compute_bound(empirical_risk, kl, n, delta) -> bound` — the PAC-Bayes functional (a tensor, so it
  is differentiable when `empirical_risk`/`kl` carry gradients).
- `train_step(model, data, target, device, n_bound, delta) -> loss` — the training objective: a
  stochastic forward pass, a surrogate empirical risk, the KL, fed through a bound formula, returned
  as the scalar SGD descends.
- `compute_risk_certificate(model, bound_loader, device, delta, mc_samples) -> (risk_cert_01, metrics)`
  — the final certificate: MC-estimate the empirical 0-1 risk, read off the KL, and PAC-Bayes-kl-invert
  to the reported `risk_certificate` (primary metric, **lower is better**), plus `test_error`,
  `kl_divergence`, `ce_bound`, `empirical_01_risk`.

The starting point is the scaffold default: the **McAllester/Maurer additive square-root bound**, with
an unrescaled NLL surrogate and a single (uncorrected) `inv_kl` certificate. Each method on the ladder
replaces exactly this class and nothing else.

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

Three settings spanning architecture and dataset difficulty — **MNIST-FCN** (`784-600-600-600-10`
fully connected), **MNIST-CNN** (2 conv + 2 fc), and **FashionMNIST-CNN** (same CNN, harder dataset)
— each over three seeds {42, 123, 456}. The primary metric is `risk_certificate` (the 0-1 loss
PAC-Bayes bound), **lower is better**. Also recorded: `test_error` (posterior-mean error),
`kl_divergence` (the `KL(Q‖P)` driving the certificate), `ce_bound` (the cross-entropy-style bound),
and `empirical_01_risk` (the MC empirical 0-1 risk on the bound set). Training uses the data-dependent
prior split: 50% trains the deterministic prior, the disjoint 50% trains and evaluates the bound.
