## McAllester / Maurer additive bound (fclassic)

**Problem.** The Catoni rung's free, clamped `λ` co-adapted with the posterior into a small-`λ`,
large-KL corner — KL ballooned to 540–750 on MNIST-FCN and the certificate followed it, with a bimodal
two-of-three-seed blowup. I want to remove the unstable trade-off knob entirely.

**Key idea.** Relax the parent PAC-Bayes-kl through Pinsker (`kl(p‖q) ≥ 2(p−q)²`) instead of Catoni's
tilt, giving the parameter-free additive certificate
```
E_Q[R] <= E_Q[r] + sqrt((KL(Q‖P) + log(2√n/δ)) / (2n)).
```
The `2` from Pinsker lands in the `/(2n)`. No `λ`, nothing for SGD to co-adapt into a bad corner — the
empirical term is weighted exactly 1, so the posterior cannot discount its way into a high-KL state.

**Why it beats Catoni despite a weaker KL penalty.** The additive bound's complexity gradient w.r.t. KL
shrinks as `1/(2√(2n·KL))` — a *sublinear* (weak) penalty, the opposite of Catoni's linear one. But the
Catoni KL runaway was driven by `λ→0` *inflating* the empirical discount, not by a weak penalty.
Removing `λ` removes that mechanism: a weak KL penalty suffices when nothing actively pushes KL up. So
KL stabilizes far smaller and far more uniformly across seeds.

**Why it is not the strongest rung.** The square-root shape caps the certificate at `~√(KL/2n)` even at
zero empirical risk, so once KL is small the remaining looseness *is* the square root — Pinsker is the
loose lower bound on `kl` when the true risk is below 1/4, exactly this regime. That is the next target.

**Why it differs from the generic method.** The textbook fclassic objective divides the bounded NLL by
`1/log(1/pmin)` (loss into `[0,1]`) and reports a nested inner+outer `inv_kl`; this task's edit surface
uses plain *unrescaled* `F.nll_loss` and a single uncorrected `inv_kl` certificate.

**Hyperparameters.** `prior_sigma=0.03`, `pmin=1e-5`, `learning_rate=0.001`, `momentum=0.95`,
`delta=0.025`, `mc_samples=1000`.

```python
class BoundOptimizer:
    """McAllester/Maurer PAC-Bayes bound (fclassic).

    Classic bound: emp_risk + sqrt((KL + log(2*sqrt(n)/delta)) / (2n))
    Training objective: same functional form with NLL surrogate for 0-1 loss.
    Certificate: PAC-Bayes-kl inversion on 0-1 risk.
    """

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin

    def compute_bound(self, empirical_risk, kl, n, delta):
        """McAllester/Maurer bound."""
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        bound = empirical_risk + torch.sqrt(kl_term)
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        """Training objective: McAllester bound with NLL surrogate."""
        output = model(data, sample=True)
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        nll = F.nll_loss(log_probs, target)

        kl = get_total_kl(model)
        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        """Evaluate McAllester risk certificate with PAC-Bayes-kl inversion."""
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

        # 5. CE bound
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
