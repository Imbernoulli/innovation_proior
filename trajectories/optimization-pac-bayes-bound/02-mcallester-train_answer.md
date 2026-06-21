The Catoni run confirmed the failure I worried about, and `kl_divergence` tells the whole story. On MNIST-FCN the KL means 539 across seeds, but the spread is the point: seed 42 sat at 132 while seeds 123 and 456 blew up to 732 and 754 — not noise around a stable operating point but a bimodal outcome where two of three runs let the posterior flee the prior. The certificate tracks it monotonically (0.040 at KL 132, 0.065 at KL 754, same architecture and data, only a different $\lambda$/posterior corner), while the empirical 0-1 risk barely moves across seeds, so the entire gap is divergence. The free, clamped $\lambda$ co-adapted with the posterior into the small-$\lambda$/large-KL corner, and the unrescaled NLL left the empirical term too heavy to act as a counterforce. The lesson is sharp: a free trade-off parameter optimized by SGD with no analytic pinning is a liability on this surface — it hands the optimizer a degree of freedom it uses to discount the empirical term and pay for it in KL. So I want to remove the unstable knob entirely.

I propose the **McAllester / Maurer additive bound (fclassic)**, the parameter-free square-root certificate — which is also the scaffold default, making this rung the precise question of whether removing the trade-off knob and accepting the square-root shape beats the unstable Catoni run. Let me derive it from the same parent so I know exactly what I commit to. Every PAC-Bayes bound starts from a learning algorithm returning a distribution $Q$ over weights, predicting by drawing $h\sim Q$, and wanting a high-probability upper bound on the $Q$-averaged true risk $E_Q[R(h)]$ from the $Q$-averaged empirical risk $E_Q[r(h)]$. The obstruction is that $Q$ is chosen after and because of the data, so a fixed-hypothesis concentration statement does not transfer; the escape is to certify the *distribution*, fixing a data-free reference $P$ and charging complexity as $\mathrm{KL}(Q\|P)$ — zero when $Q=P$, growing as $Q$ flees the prior. The change-of-measure inequality $E_Q[\varphi]\le \mathrm{KL}(Q\|P)+\log E_P[e^{\varphi}]$ transports a per-hypothesis exponential moment under the fixed $P$ onto the data-dependent $Q$ (it is just $\mathrm{KL}\ge 0$ applied to the gap between $Q$ and the Gibbs tilt of $P$). Taking $\varphi(h)=n\cdot\mathrm{kl}(r(h)\|R(h))$, the binary KL between empirical and true Bernoulli risk, with Maurer's sharp moment control $E_S[e^{n\cdot\mathrm{kl}(r\|R)}]\le 2\sqrt{n}$ for $n\ge 8$, plus Markov and Jensen, gives the parent: with probability $1-\delta$, simultaneously for all $Q$,

$$\mathrm{kl}\big(E_Q[r]\,\big\|\,E_Q[R]\big) \;\le\; \frac{\mathrm{KL}(Q\|P)+\log(2\sqrt{n}/\delta)}{n}.$$

The $2\sqrt{n}$ is Maurer's halving of $\log(2n)$ to $\log(2\sqrt{n})$, the same parent the Catoni bound relaxed; I now relax it differently. The relaxation that *removes* the trade-off parameter is Pinsker's inequality $\mathrm{kl}(p\|q)\ge 2(p-q)^2$. Applying it to the left of the parent gives $2(E_Q[R]-E_Q[r])^2 \le (\mathrm{KL}+\log(2\sqrt{n}/\delta))/n$, so $(E_Q[R]-E_Q[r])^2 \le (\mathrm{KL}+\log(2\sqrt{n}/\delta))/(2n)$, and taking the upper root,

$$E_Q[R] \;\le\; E_Q[r] + \sqrt{\frac{\mathrm{KL}(Q\|P)+\log(2\sqrt{n}/\delta)}{2n}}.$$

The factor $2$ from Pinsker's $2(p-q)^2$ lands in the denominator inside the root, turning the parent's $/n$ into $/(2n)$. This certificate has exactly the property I want: it is closed, additive, convex in $Q$ (linear $E_Q[r]$ plus the root of a KL that enters affinely under it), and — crucially — has *no free parameter*. There is nothing for SGD to co-adapt into a bad corner; the trade-off between fit and complexity is fixed by the functional form, not by a knob.

I have to confront the very property that made me leave this bound in the first place. When $E_Q[r]\to 0$ the bound collapses to $\sqrt{\mathrm{KL}/(2n)}$, whose gradient in KL is $1/(2\sqrt{2n\cdot\mathrm{KL}})$ — it *shrinks* as KL grows, the opposite of Catoni's linear penalty, so the additive bound penalizes divergence sublinearly. But the Catoni run taught me the dual lesson: a *strong* (linear) KL penalty with an *unstable* knob is worse than a *weak* (sublinear) penalty with *no* knob, because the instability dominates. Here is why I bet the weak penalty wins. The runaway KL in Catoni was not driven by the penalty being too weak — it was driven by $\lambda$ drifting small, which *simultaneously* discounted the empirical term through $1/(1-\lambda/2)$ and inflated the complexity prefactor, letting the posterior buy a large KL cheaply via the empirical discount. Remove $\lambda$ and that escape route is gone: the empirical term is always weighted exactly $1$, so the posterior cannot discount its way into a high-KL configuration. The additive bound's weak KL gradient is sufficient *as long as nothing is actively pushing KL up* — and removing $\lambda$ removes that mechanism. So I expect McAllester to sit at a far smaller, far more *stable* KL than Catoni, even though its per-nat penalty is weaker.

The implementation is the literal scaffold default and the details are load-bearing. `compute_bound` is the additive formula $\text{empirical\_risk}+\sqrt{(\mathrm{kl}+\log(2\sqrt{n}/\delta))/(2n)}$. `train_step` does a stochastic forward pass, computes the NLL surrogate for the 0-1 loss (`F.nll_loss` on `log_softmax` clamped below at $\log(p_{\min})$), reads the KL from `get_total_kl`, and returns the bound. One deliberate choice matches the edit surface and departs from the textbook recipe: the NLL is *not* rescaled by $1/\log(1/p_{\min})$. The clean derivation requires the loss in $[0,1]$ and the rescaling is what enforces that; feeding the raw clamped NLL means the surrogate can exceed $1$. I keep it unrescaled both to match the default and to make the comparison with Catoni clean — both rungs feed the same unrescaled NLL, so any difference between them is attributable to the bound functional and the $\lambda$, not to the surrogate scaling. Because the additive surrogate *adds* $\sqrt{\text{kl\_term}}$ rather than *dividing* by $1-\lambda/2$, an occasionally-too-large NLL is less damaging here than in Catoni — there is no empirical discount to exploit. (I am holding the rescaling in reserve; it is the calibration fix a later, tighter bound will want.)

The certificate stays separate from training. I train against the additive bound but report the tighter PAC-Bayes-kl inversion: `compute_risk_certificate` MC-samples the empirical 0-1 risk via `compute_01_risk`, reads the KL, forms $c=(\mathrm{KL}+\log(2\sqrt{n}/\delta))/n$ — note the $/n$, not $/(2n)$, because the inversion uses the bare parent budget and the factor $2$ only appeared in the additive *relaxation* — and returns `inv_kl(emp_risk_01, c)`. As in Catoni I keep this single-inversion and uncorrected, no inner Monte-Carlo correction, matching the scaffold style, and report the additive `ce_bound` from empirical NLL and KL. My falsifiable expectation: McAllester collapses the KL by roughly an order of magnitude and stabilizes it across seeds, killing the bimodal corner; the certificate follows down because a far smaller $c$ shrinks the inversion budget. What I do *not* claim is that this is the tightest possible bound — its square-root shape still caps the certificate at $\sim\!\sqrt{\mathrm{KL}/2n}$ even at zero empirical risk, so once KL is stabilized small the remaining looseness is the square root itself, which is the next thing to attack.

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
