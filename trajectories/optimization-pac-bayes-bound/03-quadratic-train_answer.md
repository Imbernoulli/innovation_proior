McAllester did exactly what I bet, and the KL numbers prove it. Removing the $\lambda$ knob collapsed and stabilized the divergence: on MNIST-FCN the mean fell from Catoni's 539 to 81 and — the part I care about most — it is now tight across seeds (71.9 / 80.3 / 91.8), no bimodal blowup; on MNIST-CNN it fell from 192 to a rock-stable 11, on FashionMNIST-CNN from 286 to 32. The certificate followed (MNIST-FCN 0.0558 → 0.0373, MNIST-CNN 0.0250 → 0.0164, FashionMNIST-CNN 0.1215 → 0.1002). But now look at where the additive bound is *stuck*. On MNIST-CNN the empirical 0-1 risk is 0.0119 and the KL is only 11 nats over $n\approx 30000$ — the posterior is essentially perfect and barely moved from the prior — yet the certificate sits at 0.0164. With KL this small, $\text{kl\_term}=(\mathrm{KL}+\log(2\sqrt{n}/\delta))/(2n)\approx 10^{-3}$, and the additive bound adds $\sqrt{10^{-3}}\approx 0.032$ on top of a near-zero empirical risk, so the certificate is essentially $\sqrt{\text{kl\_term}}$ itself. The square root is the floor. McAllester fixed the KL; the residual looseness is now entirely the $\sqrt{\cdot}$, and I need a bound whose complexity contribution is *linear* in $\text{kl\_term}$ at small empirical risk — without reintroducing a free $\lambda$, because Catoni showed me what that costs.

I propose the **quadratic / inverted-kl bound (fquad)**. Going back to the parent PAC-Bayes-kl, $\mathrm{kl}(E_Q[r]\|E_Q[R]) \le (\mathrm{KL}+\log(2\sqrt{n}/\delta))/n$, the reason Pinsker gave a square root is that it lower-bounds the binary KL by $2(p-q)^2$, a *parabola* symmetric in $p$ around $q$, and inverting a parabola gives an additive $\sqrt{\cdot}$. But the true binary KL is *asymmetric*: for $q<p$ — empirical risk below true risk, my case — as $p$ shrinks toward $q$ the true $\mathrm{kl}$ grows faster than the symmetric parabola allows. The sharper, asymmetric lower bound for exactly that side is the *refined Pinsker* inequality $\mathrm{kl}(q\|p)\ge (p-q)^2/(2p)$, valid for $q<p$. Comparing prefactors, refined has $1/(2p)$ and standard has $2$; refined is the tighter lower bound on $\mathrm{kl}$ precisely when $1/(2p)>2$, i.e. $p<1/4$. A true risk below 25% is exactly where a working network lives — McAllester's certificates are all well under 0.13 — so I have been relaxing the parent with the wrong inequality for my regime, and the refined one is sharp here.

Substituting the refined bound into the parent, with $C=(\mathrm{KL}+\log(2\sqrt{n}/\delta))/n$ the bare parent budget, gives $(E_Q[R]-E_Q[r])^2/(2E_Q[R]) \le C$, i.e. $(E_Q[R]-E_Q[r])^2 \le 2E_Q[R]\,C$. Since the certificate case is $E_Q[R]\ge E_Q[r]$, taking the root yields $E_Q[R] \le E_Q[r] + \sqrt{2E_Q[R]\,C}$ — and there is the catch: $E_Q[R]$, the thing I am bounding, sits on *both* sides, under the root on the right. This is not an explicit upper bound. But it is a quadratic — not in $E_Q[R]$, but in $\sqrt{E_Q[R]}$. Set $x=\sqrt{E_Q[R]}$, so $E_Q[R]=x^2$; then $x^2 \le E_Q[r] + \sqrt{2C}\,x$, i.e. $x^2 - \sqrt{2C}\,x - E_Q[r] \le 0$. A quadratic with positive leading coefficient is non-positive between its roots, so $x \le (\sqrt{2C}+\sqrt{2C+4E_Q[r]})/2$. Squaring back and pulling the $1/2$ inside the roots ($\sqrt{2C}/2=\sqrt{C/2}$, $\sqrt{2C+4E_Q[r]}/2=\sqrt{C/2+E_Q[r]}$),

$$E_Q[R] \;\le\; \Big(\sqrt{E_Q[r] + \text{kl\_term}} + \sqrt{\text{kl\_term}}\Big)^2, \qquad \text{kl\_term}=\frac{\mathrm{KL}+\log(2\sqrt{n}/\delta)}{2n},$$

with $\text{kl\_term}=C/2$ — the *same* term as the additive bound, the $2n$ re-explained ($n$ from the parent, $2$ from the refined $(p-q)^2/(2p)$). Crucially $E_Q[R]$ is gone from the right side: the quadratic-in-$\sqrt{E_Q[R]}$ trick simultaneously used the inequality tight in my low-risk regime *and* eliminated the implicit risk, with no free parameter.

What makes this attack the wall is the head-to-head at McAllester's stuck regime, zero empirical risk. At $E_Q[r]=0$ the additive bound gives $\sqrt{\text{kl\_term}}$; fquad gives $(\sqrt{\text{kl\_term}}+\sqrt{\text{kl\_term}})^2 = 4\,\text{kl\_term}$. With $\text{kl\_term}\approx 10^{-3}$, the measured MNIST-CNN operating point, that is $4\times 10^{-3}$ versus $3.2\times 10^{-2}$ — about eight times tighter. The structural reason is the wall diagnosis itself: fquad's complexity contribution is $O(\text{kl\_term})$, *linear* in the complexity, while McAllester's is $O(\sqrt{\text{kl\_term}})$; when $\text{kl\_term}$ is small the linear term wins decisively. This is the fast-rate, realizable-case $1/n$ behavior instead of $1/\sqrt{n}$, the direct payoff of refined Pinsker being sharp below true risk $1/4$. I keep the parameter-free stability I won *and* fix the square-root floor.

There is one implementation detail this rung introduces that the previous two deliberately omitted, and for fquad it is load-bearing. The earlier rungs fed the NLL surrogate *unrescaled*; fquad will not tolerate that. The bound $(\sqrt{E_Q[r]+\text{kl\_term}}+\sqrt{\text{kl\_term}})^2$ assumes $E_Q[r]\in[0,1]$, and if I feed it an unbounded NLL exceeding $1$, the empirical term inside the root is mis-scaled against the $\text{kl\_term}$ complexity term, and the outer $(\cdot)^2$ *amplifies* the miscalibration — the objective and the formula stop upper-bounding anything, and in practice the posterior drifts far from the prior, inflating the very KL I worked to suppress. So for fquad I add the bounded-loss rescaling: clamp `log_softmax` at $\log(p_{\min})$ (capping NLL at $\log(1/p_{\min})$), then multiply by $\text{\_loss\_scale}=1/\log(1/p_{\min})$, landing the surrogate in $[0,1]$. This is the calibration that lets the linear-in-KL form be tighter in practice rather than only on paper. I also clamp both arguments under the roots at zero, defensively, so a floating-point negative does not crash `sqrt`. So `compute_bound` is $(\sqrt{\mathrm{clamp}(\text{emp}+\text{kl\_term})}+\sqrt{\mathrm{clamp}(\text{kl\_term})})^2$, and `train_step` feeds the rescaled bounded NLL through it.

The certificate stays separate. I train against fquad — convex, parameter-free, tight at low risk — but report the tightest valid bound on the learned posterior, the PAC-Bayes-kl inversion: `compute_risk_certificate` MC-samples the empirical 0-1 risk via `compute_01_risk`, reads the KL, forms the *bare* parent budget $c=(\mathrm{KL}+\log(2\sqrt{n}/\delta))/n$ — $/n$, not $/(2n)$, because the $2$ lived only in the fquad relaxation, never in the inversion — and returns `inv_kl(emp_risk_01, c)`. I keep a single, uncorrected inversion with no inner Monte-Carlo correction, matching the scaffold style, and feed the same $\text{\_loss\_scale}$-rescaled empirical NLL into `ce_bound` so it is consistent with training. My falsifiable expectation: the KL stays small and stable (fquad is parameter-free, so the stability win holds) and the bounded-loss rescaling, no longer over-weighting an unrescaled empirical term, should let it drop *further* — toward single digits on FCN, $\sim\!2$ on MNIST-CNN, $\sim\!3$ on FashionMNIST-CNN — while the certificate improves modestly but unmistakably as the shape change from $\sqrt{\text{kl\_term}}$ to $4\,\text{kl\_term}$ lowers the floor, with `ce_bound` falling by a large factor as the most visible signature. This is the strongest rung I can build on this surface: the parent PAC-Bayes-kl is the tightest standard bound, fquad is its sharpest parameter-free convex relaxation at low risk, and the reported certificate is already the kl-inversion of the parent itself.

```python
class BoundOptimizer:
    """Quadratic PAC-Bayes bound (fquad, Rivasplata 2019 / Perez-Ortiz 2021).

    Bound: (sqrt(emp_risk + kl_term) + sqrt(kl_term))^2
    where kl_term = (KL + log(2*sqrt(n)/delta)) / (2n)

    Tighter than McAllester when empirical risk is low.
    """

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        # PBB's loss-bounding constant: maps unbounded NLL into [0,1] via
        # ell_tilde = NLL / log(1/pmin).  See Perez-Ortiz 2021 Sec 5.
        self._loss_scale = 1.0 / math.log(1.0 / self.pmin)

    def compute_bound(self, empirical_risk, kl, n, delta):
        """Quadratic PAC-Bayes bound (fquad)."""
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        # Ensure non-negative under sqrt
        inner = torch.clamp(empirical_risk + kl_term, min=0.0)
        kl_term_clamped = torch.clamp(kl_term, min=0.0)
        bound = (torch.sqrt(inner) + torch.sqrt(kl_term_clamped)) ** 2
        return bound

    def train_step(self, model, data, target, device, n_bound, delta):
        """Training objective: bounded NLL passed through the fquad formula.

        The NLL is rescaled by 1/log(1/pmin) so that the surrogate loss lies in
        [0,1], matching the PBB reference implementation. This is essential
        for fquad to actually be tighter than fclassic in practice.
        """
        output = model(data, sample=True)
        log_probs = F.log_softmax(output, dim=1)
        log_probs = torch.clamp(log_probs, min=math.log(self.pmin))
        # Bounded NLL surrogate, in [0, 1]
        nll = F.nll_loss(log_probs, target) * self._loss_scale

        kl = get_total_kl(model)
        bound = self.compute_bound(nll, kl, n_bound, delta)
        return bound

    def compute_risk_certificate(self, model, bound_loader, device, delta=0.025,
                                 mc_samples=1000):
        """Evaluate quadratic risk certificate with PAC-Bayes-kl inversion."""
        model.eval()
        n_bound = len(bound_loader.dataset)

        # 1. Empirical 0-1 risk via MC sampling
        emp_risk_01 = compute_01_risk(model, bound_loader, device,
                                      mc_samples=mc_samples)

        # 2. Bounded NLL empirical risk (same scaling as training)
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
        emp_nll_bounded = (total_nll / total_samples) * self._loss_scale

        # 3. KL divergence
        with torch.no_grad():
            dummy_data = next(iter(bound_loader))[0][:1].to(device)
            model(dummy_data, sample=True)
            kl = get_total_kl(model).item()

        # 4. PAC-Bayes-kl inversion for 0-1 loss certificate
        c = (kl + math.log(2.0 * math.sqrt(n_bound) / delta)) / n_bound
        risk_cert_01 = inv_kl(emp_risk_01, c)

        # 5. Quadratic bound on bounded CE risk (in [0,1])
        emp_nll_t = torch.tensor(emp_nll_bounded)
        kl_t = torch.tensor(kl)
        ce_bound = self.compute_bound(emp_nll_t, kl_t, n_bound, delta).item()

        metrics = {
            "empirical_01_risk": emp_risk_01,
            "empirical_nll": emp_nll_bounded,
            "kl_divergence": kl,
            "ce_bound": ce_bound,
        }

        return risk_cert_01, metrics
```
