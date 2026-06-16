# McAllester-style classic PAC-Bayes bound, with Maurer's constant

The artifact here is the `fclassic` / classic PAC-Bayes square-root certificate used in
PAC-Bayes-with-Backprop. Historically, McAllester (1999) introduced PAC-Bayesian mixture
bounds with looser constants; the modern code path uses the Seeger/Maurer PAC-Bayes-kl
form and then applies the standard Pinsker relaxation to obtain the McAllester-style
additive bound:

```text
E_{h~Q}[R(h)] <= E_{h~Q}[r(h)]
                 + sqrt(( KL(Q || P) + log(2*sqrt(n)/delta) ) / (2n)).
```

Here `P` is fixed without seeing the bound-evaluation sample, `Q` may be chosen after seeing
that sample, `r` is empirical risk on the bound set, `R` is true risk, and the loss is in
`[0,1]`. The PAC-Bayes-kl certificate without the inner Monte Carlo correction has
confidence `1 - delta` over the bound sample, uniformly for all posteriors `Q`.

## Core theorem

The sharper parent certificate is the PAC-Bayes-kl bound:

```text
kl(E_Q[r] || E_Q[R])
    <= ( KL(Q || P) + log(2*sqrt(n)/delta) ) / n.
```

Maurer's exponential-moment lemma gives the `2*sqrt(n)` term:

```text
E_S exp(n * kl(r(h) || R(h))) <= 2*sqrt(n).
```

Pinsker's inequality `kl(p||q) >= 2(p-q)^2` turns the implicit kl-ball into the additive
square-root training objective above. For reporting, do not use the Pinsker relaxation:
invert the binary KL directly.

## Algorithm

- Train a data-dependent reference `P` on one split, then evaluate the bound only on the
  disjoint bound split.
- Parameterize `Q` as independent Gaussian weights and compute analytic `KL(Q || P)`.
- Train with bounded cross-entropy: clamp true-label probabilities at `pmin`, then divide
  NLL by `log(1/pmin)` so the loss lies in `[0,1]`.
- Report the 0-1 Gibbs risk certificate by nested inversion:
  first correct the Monte Carlo estimate of `E_Q[r]`, then apply the outer PAC-Bayes-kl
  inversion. If both `delta` and `delta_mc` are used, the final certificate is valid with
  confidence `1 - delta - delta_mc` (union bound).

## Working code

```python
import math
import torch
import torch.nn.functional as F


class BoundOptimizer:
    """McAllester/Maurer classic PAC-Bayes objective.

    Trains with the Pinsker square-root relaxation and reports the tighter
    PAC-Bayes-kl certificate with a Monte Carlo correction.
    """

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin

    def compute_bound(self, empirical_risk, kl, n, delta):
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        return empirical_risk + torch.sqrt(kl_term)

    def train_step(self, model, data, target, device, n_bound, delta):
        output = model(data, sample=True)
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        bounded_nll = F.nll_loss(log_probs, target) / math.log(1.0 / self.pmin)
        kl = get_total_kl(model)
        return self.compute_bound(bounded_nll, kl, n_bound, delta)

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        model.eval()
        n_bound = len(bound_loader.dataset)
        delta_mc = 0.01

        total_01 = 0.0
        total_nll = 0.0
        total_samples = 0

        with torch.no_grad():
            for data, target in bound_loader:
                data, target = data.to(device), target.to(device)
                batch_size = target.size(0)

                for _ in range(mc_samples):
                    output = model(data, sample=True)
                    pred = output.argmax(dim=1)
                    total_01 += (pred != target).sum().item()

                    log_probs = F.log_softmax(output, dim=1)
                    log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
                    batch_nll = F.nll_loss(log_probs, target, reduction="sum")
                    total_nll += (batch_nll / math.log(1.0 / self.pmin)).item()

                total_samples += batch_size

            dummy = next(iter(bound_loader))[0][:1].to(device)
            model(dummy, sample=True)
            kl = get_total_kl(model).item()

        emp_01_mc = total_01 / (total_samples * mc_samples)
        emp_nll_mc = total_nll / (total_samples * mc_samples)

        mc_radius = math.log(2.0 / delta_mc) / mc_samples
        emp_01 = inv_kl(emp_01_mc, mc_radius)
        emp_nll = inv_kl(emp_nll_mc, mc_radius)

        outer_radius = (
            kl + math.log(2.0 * math.sqrt(n_bound) / delta)
        ) / n_bound
        risk_cert_01 = inv_kl(emp_01, outer_radius)

        ce_bound = self.compute_bound(
            torch.tensor(emp_nll), torch.tensor(kl), n_bound, delta
        ).item()

        metrics = {
            "empirical_01_risk_mc": emp_01_mc,
            "empirical_01_risk": emp_01,
            "empirical_nll_mc": emp_nll_mc,
            "empirical_nll": emp_nll,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
            "delta_mc": delta_mc,
        }
        return risk_cert_01, metrics
```

This matches the PBB `fclassic` objective while fixing two common implementation slips:
the bounded cross-entropy must be divided by `log(1/pmin)`, and the final 0-1 certificate
must use the Gibbs-risk Monte Carlo average with an inner KL correction, not a majority-vote
error or an uncorrected MC estimate.
