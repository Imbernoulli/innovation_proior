I have a network with far more parameters than training examples, and on MNIST it generalizes beautifully — a few percent test error — yet every guarantee I can write down for it is worthless. VC-dimension, Rademacher complexity, any uniform-convergence argument scales with the capacity of the function class, and with a million parameters against sixty thousand examples the complexity term is enormous: the bound comes out above $1$, and "your error is at most $130\%$" is no guarantee at all. So I have a model that works and no number that says so. What I want is a single number, computed from the training data, that is a genuine high-probability upper bound on the error on unseen data, and small enough — close enough to the test error — to be worth printing. The framework that survives the over-parameterization trap is PAC-Bayes: instead of certifying a fixed weight vector $h_w$, certify a *distribution* $Q$ over weight space, predicting by drawing a fresh $W \sim Q$ each time. The object of interest is then the $Q$-averaged true risk $L(Q) = \mathbb{E}_{W \sim Q}[L(W)]$, with empirical counterpart $\hat L_S(Q) = \mathbb{E}_{W \sim Q}[\hat L_S(W)]$, and the complexity term becomes the Kullback-Leibler divergence $\mathrm{KL}(Q \,\|\, Q_0)$ from a data-free reference $Q_0$ (the "prior") to the learned $Q$ (the "posterior"). That is *not* a parameter count; it measures how far the posterior had to move from the reference, in nats. If the posterior stays near its prior, $\mathrm{KL}(Q \,\|\, Q_0)$ is small no matter how many parameters there are — which is exactly why PAC-Bayes yields non-vacuous numbers where uniform convergence dies. The earlier work that first optimized such a bound by SGD over a Gaussian on the weights produced the first non-vacuous neural-network certificates, but its reported number (on the order of $0.16$–$0.22$ on MNIST) sat far above the few-percent test error: non-vacuous, but loose, and loose precisely in the regime that matters.

The strongest classical statement, the PAC-Bayes-kl bound (Langford-Seeger 2001; Seeger 2002; with the sharp $2\sqrt{n}$ constant of Maurer 2004), controls not the difference but the *binary KL* between the two risks. Writing $\mathrm{kl}(q \,\|\, p) = q \log\frac{q}{p} + (1-q)\log\frac{1-q}{1-p}$ for the relative entropy of $\mathrm{Bernoulli}(q)$ from $\mathrm{Bernoulli}(p)$, with probability at least $1-\delta$ over the size-$n$ sample, simultaneously for all $Q$,
$$\mathrm{kl}\big(\hat L_S(Q) \,\|\, L(Q)\big) \;\le\; \frac{\mathrm{KL}(Q \,\|\, Q_0) + \log\!\big(2\sqrt{n}/\delta\big)}{n} \;=:\; C.$$
This is the inequality everything below relaxes, and I re-derived it to know what I am relaxing: a change of measure (Donsker-Varadhan) moves the data-dependent $Q$ onto the fixed prior $P$, $\mathbb{E}_{h\sim Q}[f] \le \mathrm{KL}(Q\,\|\,P) + \log \mathbb{E}_{h\sim P}[e^f]$ with $f = n\,\mathrm{kl}(\cdot)$; the sharp per-hypothesis moment bound $\mathbb{E}_S[e^{n\,\mathrm{kl}}] \le 2\sqrt{n}$ for $n \ge 8$ (whose $\sqrt{n}$ order is matched by a lower bound, so it cannot be improved) handles the inner expectation; Markov turns the moment bound into a tail statement; and Jensen, using joint convexity of $\mathrm{kl}$, drops $\mathbb{E}_{h\sim Q}[\mathrm{kl}(\cdot)]$ down to $\mathrm{kl}(\hat L_S(Q) \,\|\, L(Q))$. The loosest step is Markov; the rest is essentially tight. The trouble is that pb-kl bounds a binary KL, not $L(Q)$ itself. As a final certificate I can invert it, but as a *training objective* there is no closed form to hand to SGD. So the practical move is to relax pb-kl into an explicit upper bound on $L(Q)$ — and the choice of relaxation decides everything.

The textbook relaxation is standard Pinsker, $\mathrm{kl}(\hat q \,\|\, p) \ge 2(p-\hat q)^2$. Lower-bounding the left side of pb-kl and solving the upper root gives the classic additive bound $L(Q) \le \hat L_S(Q) + \sqrt{(\mathrm{KL} + \log(2\sqrt{n}/\delta))/(2n)}$. It is clean, explicit, and optimizable, and it gave the first non-vacuous bounds — but in the regime a trained network actually reaches, $\hat L_S(Q) \to 0$, it collapses to $\sqrt{\mathrm{kl\_term}}$, with the complexity entering through a *square root* that does not shrink as the empirical loss vanishes. At $\mathrm{kl\_term} = 10^{-3}$ that is a certificate of about $0.032$ while the empirical error is essentially zero — loose exactly where it must be tight. The diagnosis is that standard Pinsker is a *symmetric parabola* lower bound, and the true $\mathrm{kl}$ is asymmetric: for $\hat q < p$ there is a sharper refined inequality $\mathrm{kl}(\hat q \,\|\, p) \ge (p-\hat q)^2/(2p)$, which beats standard Pinsker exactly when $1/(2p) > 2$, i.e. $p < 1/4$ — precisely the below-$25\%$-risk regime a working network lives in. So I propose the **PAC-Bayes-quadratic bound**, $f_{\mathrm{quad}}$: relax pb-kl with the *refined* Pinsker inequality and solve the result exactly.

Substituting $\hat q = \hat L_S(Q)$, $p = L(Q)$ into pb-kl gives, in the upper-tail case $L \ge \hat L_S$ that the certificate needs, $(L - \hat L_S)^2/(2L) \le C$, hence
$$L(Q) \;\le\; \hat L_S(Q) + \sqrt{2\,L(Q)\,C}.$$
Here is the catch everyone hits next: $L(Q)$ appears on both sides, under the square root, so this is not an explicit bound. But it is a *quadratic in $x = \sqrt{L}$*: writing $L = x^2$, it reads $x^2 - \sqrt{2C}\,x - \hat L_S \le 0$, a quadratic with positive leading coefficient that is non-positive between its roots, so $x \le \big(\sqrt{2C} + \sqrt{2C + 4\hat L_S}\big)/2$. Squaring to recover $L = x^2$ and pulling the $\tfrac12$ inside each root ($\sqrt{2C}/2 = \sqrt{C/2}$) eliminates $L$ from the right-hand side and leaves the compact form
$$L(Q) \;\le\; \Big(\sqrt{\hat L_S(Q) + \mathrm{kl\_term}} + \sqrt{\mathrm{kl\_term}}\Big)^2, \qquad \mathrm{kl\_term} = \frac{C}{2} = \frac{\mathrm{KL}(Q \,\|\, Q_0) + \log\!\big(2\sqrt{n}/\delta\big)}{2n}.$$
The factor $2n$ is $n$ from pb-kl times the $2$ that came out of $(L - \hat L_S)^2 \le 2LC$; I verified the algebra by expanding both the compact and larger-root forms to the same $\hat L_S + C + \sqrt{C(2\hat L_S + C)}$. The payoff is the small-loss behavior the wall was about: at $\hat L_S = 0$, $f_{\mathrm{quad}} = (2\sqrt{\mathrm{kl\_term}})^2 = 4\,\mathrm{kl\_term}$, *linear* in the complexity, against the classic $\sqrt{\mathrm{kl\_term}}$ — at $\mathrm{kl\_term} = 10^{-3}$ that is $4\times10^{-3}$ versus $3.2\times10^{-2}$, almost an order of magnitude tighter, the realizable-case fast rate. This advantage holds while the true risk stays below $1/4$ (the refined-Pinsker crossover); above $1/4$ the classic bound is the tighter relaxation, but small loss is my regime. This is what distinguishes $f_{\mathrm{quad}}$ from the related $f_{\lambda}$ (Thiemann et al. 2017), which also starts from refined Pinsker but then applies AM-GM $\sqrt{ab} \le (\lambda a + b/\lambda)/2$ to linearize in $\hat L_S$; that introduces a free $\lambda$ to tune and $\lambda$-dependent slack, whereas solving the quadratic exactly keeps the refined-Pinsker tightness with no extra parameter.

To turn $f_{\mathrm{quad}}$ into a trainable objective I plug in a differentiable surrogate. The $0$-$1$ loss has gradient zero almost everywhere, so I train on cross-entropy $-\log \sigma(u)_y$, justified as a surrogate because (via $\log x \le x-1$) it upper-bounds the mistake probability. But cross-entropy is unbounded above while every PAC-Bayes inequality here assumes a loss in $[0,1]$; feeding an unbounded loss into a $[0,1]$-loss bound makes the bound simply false. So I floor the predicted probability, replacing $\sigma(u)_y$ with $\max(\sigma(u)_y, p_{\min})$, which caps the loss at $\log(1/p_{\min})$, and rescale by $1/\log(1/p_{\min})$ to land in $[0,1]$. That rescaling is load-bearing, not cosmetic: it is exactly what maps the surrogate into the range the $(\sqrt{\cdot}+\sqrt{\cdot})^2$ formula assumes; without it the empirical term and $\mathrm{kl\_term}$ sit on incompatible scales, the formula stops upper-bounding anything, and the posterior drifts from the prior and inflates $\mathrm{KL}$ — loosening the very certificate I am tightening. The posterior $Q = \mathcal{N}(\mu, \mathrm{diag}(\sigma^2))$ is a diagonal Gaussian with $\sigma = \log(1 + e^{\rho})$ (softplus, so $\sigma > 0$ under unconstrained $\rho$); I differentiate through $\mathbb{E}_{W\sim Q}$ by the reparameterization $W = \mu + \sigma \odot V$, $V \sim \mathcal{N}(0,I)$, whose pathwise gradient is unbiased and low-variance and reduces to the ordinary backprop gradients shifted and scaled. The KL between diagonal Gaussians is closed-form and differentiable, $\tfrac12\big(\log\frac{b_0}{b_1} + \frac{(\mu_1-\mu_0)^2}{b_0} + \frac{b_1}{b_0} - 1\big)$ per coordinate with $b = \sigma^2$, summed over coordinates. Because $\mathrm{KL}$ dominates the bound (since $\hat L_S$ is near zero, it is the lever for tightness), I keep it small with a data-dependent prior: split the training data, learn the prior mean by ERM on one part (with dropout only there, to keep the prior from overfitting), and learn the posterior and evaluate the certificate on the disjoint remainder, so the prior is genuinely data-free with respect to the bound; the posterior is initialized at the prior so it starts at the ERM minimizer and need only move a little. Keeping the two subsets disjoint is what avoids the heavier differential-privacy corrections that data reuse would force.

The number I actually report is *not* the quadratic relaxation — for the headline guarantee I invert the sharp pb-kl directly on the $0$-$1$ loss. Define $f^\star(q, c) = \sup\{p \in [q,1] : \mathrm{kl}(q \,\|\, p) \le c\}$; it is well-defined because $\mathrm{kl}(q \,\|\, p)$ is strictly increasing in $p$ on $[q,1]$, so the feasible set is an interval and the supremum is its right endpoint, found by bisection. The certificate is $L(Q) \le f^\star\big(\hat L_S(Q), (\mathrm{KL} + \log(2\sqrt{n}/\delta))/n\big)$, with budget $c$ using $/n$ (bare pb-kl), not the $/2n$ of the relaxation — the $2$ only ever appeared in the quadratic step. Since $\hat L_S(Q)$ is an unobservable expectation over $Q$, I Monte-Carlo it over $m$ weight draws and charge its own sampling error with an inner binary-kl tail $\mathrm{kl}(\hat L_S(\hat Q_m) \,\|\, \hat L_S(Q)) \le \log(2/\delta')/m$, so the reported risk is the nested inversion $f^\star\big(f^\star(\hat L_S(\hat Q_m), \log(2/\delta')/m), c\big)$, valid with confidence $1 - \delta - \delta'$. The quadratic relaxation drives training; the nested sharp inversion delivers the number.

```python
import math
import torch
import torch.nn.functional as F


class BoundObjective:
    """PAC-Bayes-quadratic (fquad) objective and risk certificate."""

    def __init__(self, learning_rate=0.001, momentum=0.95, prior_sigma=0.03,
                 pmin=1e-5, delta=0.025, delta_test=0.01,
                 mc_samples=1000, kl_penalty=1.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prior_sigma = prior_sigma
        self.pmin = pmin
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self._loss_scale = 1.0 / math.log(1.0 / pmin)

    def compute_empirical_risk(self, outputs, targets, bounded=True):
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded:
            empirical_risk = empirical_risk * self._loss_scale
        return empirical_risk

    def compute_losses(self, net, data, target, clamping=True):
        outputs = net(data, sample=True, clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        loss_01 = 1.0 - pred.eq(target.view_as(pred)).sum().item() / target.size(0)
        return loss_ce, loss_01, outputs

    def compute_bound(self, empirical_risk, kl, n, delta):
        kl = kl * self.kl_penalty
        kl_term = (kl + math.log(2.0 * math.sqrt(n) / delta)) / (2.0 * n)
        inner = torch.clamp(empirical_risk + kl_term, min=0.0)
        kl_term_clamped = torch.clamp(kl_term, min=0.0)
        return (torch.sqrt(inner) + torch.sqrt(kl_term_clamped)) ** 2

    def train_step(self, net, data, target, n_posterior, clamping=True):
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net, data, target, clamping)
        train_obj = self.compute_bound(loss_ce, kl, n_posterior, self.delta)
        return train_obj, kl / n_posterior, outputs, loss_ce, loss_01

    def mc_sampling(self, net, input=None, target=None, data_loader=None,
                    device="cuda", clamping=True):
        error, cross_entropy = 0.0, 0.0
        if data_loader is not None:
            batches = 0
            for data_batch, target_batch in data_loader:
                data_batch = data_batch.to(device)
                target_batch = target_batch.to(device)
                ce_mc, err_mc = 0.0, 0.0
                for _ in range(self.mc_samples):
                    loss_ce, loss_01, _ = self.compute_losses(
                        net, data_batch, target_batch, clamping
                    )
                    ce_mc += loss_ce
                    err_mc += loss_01
                cross_entropy += ce_mc / self.mc_samples
                error += err_mc / self.mc_samples
                batches += 1
            return cross_entropy / batches, error / batches

        ce_mc, err_mc = 0.0, 0.0
        for _ in range(self.mc_samples):
            loss_ce, loss_01, _ = self.compute_losses(net, input, target, clamping)
            ce_mc += loss_ce
            err_mc += loss_01
        return ce_mc / self.mc_samples, err_mc / self.mc_samples

    def compute_risk_certificate(self, net, n_posterior, n_bound, input=None,
                                 target=None, data_loader=None, device="cuda",
                                 clamping=True):
        kl = net.compute_kl()
        error_ce, error_01 = self.mc_sampling(
            net, input, target, data_loader, device, clamping
        )

        mc_c = math.log(2.0 / self.delta_test) / self.mc_samples
        empirical_risk_ce = inv_kl(float(error_ce.item()), mc_c)
        empirical_risk_01 = inv_kl(float(error_01), mc_c)

        train_bound = self.compute_bound(
            torch.tensor(empirical_risk_ce, device=kl.device),
            kl,
            n_posterior,
            self.delta,
        )

        # Outer certificate budget is the PAC-Bayes-kl budget: divide by n, not 2n.
        c = (kl.item() + math.log(2.0 * math.sqrt(n_bound) / self.delta)) / n_bound
        risk_ce = inv_kl(empirical_risk_ce, c)
        risk_01 = inv_kl(empirical_risk_01, c)
        return (
            train_bound.item(),
            kl.item() / n_bound,
            empirical_risk_ce,
            empirical_risk_01,
            risk_ce,
            risk_01,
        )


def inv_kl(qs, ks):
    """Inversion of the binary kl: sup{ p in [qs, 1] : kl(qs || p) <= ks }, by bisection."""
    izq, dch = qs, 1 - 1e-10
    qd = 0
    while (dch - izq) / dch >= 1e-5:
        p = (izq + dch) * 0.5
        if qs == 0:
            ikl = ks - ((1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p))
        else:
            ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
```
