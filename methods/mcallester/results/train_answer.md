A learning algorithm hands me back, after seeing $n$ i.i.d. examples $S = (z_1, \dots, z_n)$, not a single classifier but a distribution $Q$ over hypotheses; to predict I draw $h \sim Q$ and apply it. I can compute the empirical risk of this randomized predictor, $E_{h\sim Q}[r(h)]$ with $r(h) = \frac{1}{n}\sum_i l(h, z_i)$, and what I want is a guarantee on the thing I cannot see, the true risk $E_{h\sim Q}[R(h)]$ with $R(h) = E_{z}[l(h,z)]$ — a printable number such that, with probability at least $1-\delta$ over the draw of $S$, the true risk is at most something computable from the sample. The catch that kills the naive route is adaptivity: $Q$ is chosen after and because of $S$, so any concentration statement proved for a fixed hypothesis does not transfer to the one the algorithm actually returned. Hoeffding gives, for a single fixed $h$, $R(h) \le r(h) + \sqrt{\log(1/\delta)/(2n)}$, but only for an $h$ fixed before the data. The standard patch for a finite class is a union bound — pay $\log|H|$ and the statement holds uniformly over all $h$ — but that charges the complexity of the whole menu, not of the solution; for a continuously parameterized model or a neural network $\log|H|$ is unbounded and the bound degenerates to "true error $\le 1$." Linear change-of-measure certificates already replace $\log|H|$ by a posterior-to-reference divergence but, by collapsing the comparison into the raw gap $R-r$ and a free temperature, stay stuck at a $1/\sqrt{n}$ rate that is blind to the near-zero-risk regime; implicit relative-entropy certificates are sharper there but leave the true risk trapped inside a binary-KL constraint, awkward as a smooth training objective. I want both: the tight implicit form for reporting and a closed differentiable form for training.

I propose the McAllester-style classic PAC-Bayes certificate, built on Maurer's exponential-moment constant. The reframing is to stop certifying a single $h$ and instead certify the distribution $Q$ itself: fix a reference $P$ over $H$ before seeing the bound data (call it a prior, though no Bayesian likelihood is involved — it is merely a yardstick fixed in advance), let the learner return any $S$-dependent $Q$, and measure complexity by how far $Q$ had to move from $P$, namely the relative entropy $KL(Q\|P)$, which is zero when $Q=P$, grows as $Q$ concentrates where $P$ was diffuse, and is $+\infty$ if $Q$ ventures where $P$ has no mass. The device that transports per-hypothesis concentration onto the data-dependent $Q$ is the change-of-measure (Donsker–Varadhan) inequality, $E_{h\sim Q}[\phi(h)] \le KL(Q\|P) + \log E_{h\sim P}[e^{\phi(h)}]$, which is nothing more than $KL(Q\|P_\phi) \ge 0$ for the Gibbs tilt $dP_\phi/dP \propto e^{\phi}$ expanded out: $KL(Q\|P_\phi) = KL(Q\|P) - E_Q[\phi] + \log E_P[e^\phi] \ge 0$, with equality when $Q$ is exactly that tilt. The left side is a $Q$-average of whatever I plug into $\phi$; the right side has the $KL(Q\|P)$ complexity term plus a moment computed under the fixed $P$, the part I can integrate against the data. Choosing $\phi(h)=\lambda(R(h)-r(h))$, running Hoeffding's sub-Gaussian moment $E_S[e^{\lambda(R-r)}]\le e^{\lambda^2/(8n)}$ through Tonelli and Markov, gives the linear bound $E_Q[R] \le E_Q[r] + \lambda/(8n) + (KL(Q\|P)+\log(1/\delta))/\lambda$; optimizing $\lambda \propto \sqrt n$ leaves a penalty of order $\sqrt{(KL+\log(1/\delta))/n}$, always $1/\sqrt n$ regardless of how small the empirical risk is. The structural failure is that collapsing $R$ versus $r$ into a raw symmetric gap throws away the fact that a near-zero-mean Bernoulli has tiny variance.

The fix is to use a discrepancy that is naturally sharp near the boundary: the binary relative entropy $kl(p\|q) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$, which near $q=0$ behaves like $p\log(p/q)$ and explodes, and which is precisely the large-deviation rate for the empirical mean of $n$ Bernoulli$(R)$ variables. To handle this and the linear case with a single pass, take any convex $D:[0,1]^2\to\mathbb{R}$ and set $\phi(h)=n\,D(r(h),R(h))$. The same change-of-measure-and-Markov steps give, with probability $1-\delta$ and simultaneously for all $Q$, $E_{h\sim Q}[D(r(h),R(h))] \le (KL(Q\|P)+\log(M(P)/\delta))/n$, where $M(P)=E_S E_{h\sim P}[e^{nD(r,R)}]$ is the only $D$-specific quantity. Convexity of $D$ then buys the master bound through Jensen, $D(E_Q[r],E_Q[R]) \le E_Q[D(r,R)]$, so
$$ D\big(E_{h\sim Q}[r(h)],\,E_{h\sim Q}[R(h)]\big) \le \frac{KL(Q\|P)+\log(M(P)/\delta)}{n}. $$
Everything now reduces to the constant $M(P)$ for $D=kl$, i.e. the worst-case moment $E[e^{n\,kl(r(h)\|R(h))}]$. This is Maurer's lemma, and the constant is the whole game — carrying $2n$ versus $2\sqrt n$ here directly changes the additive log term. The function $x\mapsto e^{n\,kl(\frac1n\sum x_i\,\|\,\mu)}$ is convex (relative entropy convex in its first argument, precomposed with a linear map, then a convex increasing exponential) and permutation-symmetric, so on $[0,1]^n$ its expectation is maximized at the cube's $\{0,1\}^n$ corners, reducing arbitrary $[0,1]$-valued losses to the Bernoulli case. At a corner with $k$ ones the empirical mean is $k/n$ and $e^{n\,kl(k/n\|\mu)} = ((n-k)/(n(1-\mu)))^{n-k}(k/(n\mu))^k$, so the binomial weights' $\mu$-dependence cancels entirely and leaves $E[e^{n\,kl}] \le \sum_{k=0}^n \binom{n}{k}(k/n)^k((n-k)/n)^{n-k}$. Peeling the $k=0,n$ terms as $+2$, applying Stirling so the $e^k,e^{n-k},e^{-n}$ factors cancel, and recognizing $\sum_{k=1}^{n-1}1/\sqrt{k(n-k)}$ as a Riemann sum for $\int_0^1 dt/\sqrt{t(1-t)} = \pi$ gives $E[e^{n\,kl}] \le e^{1/12n}\sqrt{\pi n/2} + 2 \le 2\sqrt n$ for $n\ge 8$, with a matching lower bound of $\sqrt n$ so the order is genuinely forced. Feeding $M(P)=2\sqrt n$ back yields the sharp PAC-Bayes-kl certificate
$$ kl\big(E_{h\sim Q}[r(h)] \,\|\, E_{h\sim Q}[R(h)]\big) \le \frac{KL(Q\|P)+\log(2\sqrt n/\delta)}{n}, $$
which automatically gets the fast rate when $E_Q[r]$ is near zero: at $E_Q[r]=0$, $kl(0\|q)=-\log(1-q)\ge q$ forces $E_Q[R] \le (KL+\log(2\sqrt n/\delta))/n$, a $1/n$ rate the linear bound could never see.

This kl-form is implicit in the true risk, perfect for reporting but not for backpropagation. Pinsker's inequality $kl(p\|q)\ge 2(p-q)^2$ relaxes the kl-ball to a closed additive form: if $kl(E_Q[r]\|E_Q[R])\le c$ then $(E_Q[R]-E_Q[r])^2 \le c/2$, so with $c=(KL+\log(2\sqrt n/\delta))/n$,
$$ E_{h\sim Q}[R(h)] \le E_{h\sim Q}[r(h)] + \sqrt{\frac{KL(Q\|P)+\log(2\sqrt n/\delta)}{2n}}, $$
the classic McAllester-style square-root certificate — closed, additive, and differentiable in $Q$ through $E_Q[r]$ and $KL(Q\|P)$. The Pinsker $2$ lands in the denominator, turning the kl-form's $/n$ into $/(2n)$; the price is that this surrogate is strictly looser and shows $1/\sqrt n$ rather than $1/n$, so I train against it but report the tighter kl-inversion. Three implementation obstacles remain. The loss must lie in $[0,1]$ for every step from Hoeffding through Maurer's lemma, yet the bounded 0-1 loss is piecewise constant with no gradient and cross-entropy is differentiable but unbounded; I take the cross-entropy surrogate $-\log p_y$ (a valid upper bound on $1-p_y$ since $\log x \le x-1$), clamp $\log p_y$ at $\log(p_{\min})$ so the loss is at most $\log(1/p_{\min})$, then divide by $\log(1/p_{\min})$ to land back in $[0,1]$ — both the clamp and the rescale matter, the clamp for finiteness and the rescale to keep range one so the bound applies and the risk/KL tradeoff stays calibrated. The $KL(Q\|P)$ is the sum of per-coordinate Gaussian relative entropies, kept small by building $P$ data-dependently — train a deterministic ERM model on one held-out split, center $P$ and initialize $Q$ at those weights so $KL$ starts at zero, and evaluate the bound only on the disjoint $n$ examples $P$ never saw. The reported $E_Q[r]$ is itself a posterior average estimable only by Monte Carlo over $m$ Gibbs draws (averaging sampled-classifier 0-1 error, not a majority vote, because the theorem certifies the randomized predictor), so its own sampling error needs a correction: the same kl-concentration one level down gives $kl(\hat{e}_m\|E_Q[r])\le \log(2/\delta')/m$, inverted to bound the true empirical risk before the outer inversion. The fully honest 0-1 certificate is therefore a nested inversion — inner inversion for posterior-sampling error, outer PAC-Bayes-kl inversion for generalization — valid with confidence $1-\delta-\delta'$.

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
