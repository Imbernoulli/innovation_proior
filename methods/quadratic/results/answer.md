# PAC-Bayes-quadratic bound (f_quad), distilled

The PAC-Bayes-quadratic bound is an explicit, optimizable upper bound on the true risk of a
randomized (stochastic) predictor `Q`, obtained by relaxing the PAC-Bayes-kl bound with the
*refined* Pinsker inequality and solving the result as a quadratic in `sqrt(L(Q))`. It is
tighter than the classic (additive square-root) PAC-Bayes bound exactly in the small-loss
regime (true risk below `1/4`) that a trained over-parameterized network reaches, where its
complexity term enters linearly rather than as a square root — a fast rate. It is used as a
differentiable training objective; the reported 0-1 risk certificate is then computed by
the sharp PAC-Bayes-kl inversion.

## Problem it solves

Compute a high-probability, *non-vacuous*, *tight* upper bound on the test error of an over-
parameterized network — small enough to be informative, and ideally directly minimizable by
gradient descent during training. Capacity-based certificates (VC, Rademacher) are vacuous
when parameters exceed examples. PAC-Bayes certifies a distribution `Q` over weights and
charges complexity as `KL(Q || Q0)` (distance from a data-free reference), which is small when
the posterior stays near the prior, regardless of parameter count.

## Key idea

The strongest classical inequality, the **PAC-Bayes-kl bound** (Langford-Seeger 2001;
Seeger 2002; sharp `2 sqrt(n)` constant from Maurer 2004): with probability `1 - delta` over
size-`n` samples, simultaneously for all `Q`,

```
kl( Lhat_S(Q) || L(Q) )  <=  ( KL(Q || Q0) + log(2 sqrt(n)/delta) ) / n  =:  C ,
```

bounds a *binary kl*, not `L(Q)`, so it is not directly optimizable. Relaxing it with the
**refined Pinsker inequality** `kl(qh || p) >= (p - qh)^2/(2p)` (valid for `qh < p`, tighter
than standard Pinsker `2(p - qh)^2` when `p < 1/4`) and substituting `qh = Lhat_S(Q)`,
`p = L(Q)` gives, in the upper-tail case `L >= Lhat`,

```
(L - Lhat)^2 / (2L)  <=  C   =>   L  <=  Lhat + sqrt(2 L C) .      (star)
```

`L` appears on both sides of (star). But (star) is a quadratic in `x = sqrt(L)`:
`x^2 - sqrt(2C) x - Lhat <= 0`, so `x <= (sqrt(2C) + sqrt(2C + 4 Lhat))/2`. Squaring and
simplifying (`sqrt(2C)/2 = sqrt(C/2)`) eliminates `L` from the right:

```
L(Q)  <=  ( sqrt( Lhat_S(Q) + kl_term ) + sqrt( kl_term ) )^2 ,
kl_term = C/2 = ( KL(Q || Q0) + log(2 sqrt(n)/delta) ) / (2n) .
```

This is `f_quad`. The factor `2n` is `n` (from pb-kl) times the `2` from `(L - Lhat)^2 <= 2 L C`.

**Why it is tighter at small loss.** Expanding,
`f_quad = Lhat + 2 kl_term + 2 sqrt(kl_term) sqrt(Lhat + kl_term)`. At `Lhat = 0`,
`f_quad = (2 sqrt(kl_term))^2 = 4 kl_term`, while the classic bound is
`f_classic = Lhat + sqrt(kl_term) = sqrt(kl_term)`. For small `kl_term` the complexity enters
`f_quad` *linearly* (`O(kl_term)`) but `f_classic` as a *square root* (`O(sqrt(kl_term))`), so
`f_quad` is much tighter — e.g. at `kl_term = 1e-3`, `4e-3` vs `3.2e-2`. The advantage holds
while the risk is below `1/4` (the refined-Pinsker crossover); above `1/4` the classic bound
is tighter.

## The bound family (all relaxations of pb-kl)

- **f_classic** (McAllester 1999, Maurer constant): standard Pinsker `kl >= 2(p - qh)^2` ->
  `L <= Lhat + sqrt(kl_term)`. Loose at small loss.
- **f_lambda** (Thiemann et al. 2017): refined Pinsker + AM-GM `sqrt(ab) <= (lambda a +
  b/lambda)/2` -> `L <= Lhat/(1 - lambda/2) + (KL + log(2 sqrt(n)/delta))/(n lambda(1 -
  lambda/2))`, uniform over `lambda in (0,2)`. It is optimizable over the extra parameter, but
  the AM-GM step introduces parameter-dependent slack.
- **f_quad** (this method): refined Pinsker solved exactly as a quadratic in `sqrt(L)`.
  It keeps the refined-Pinsker small-loss behavior without introducing `lambda`.

## Training and certificate evaluation

- **Surrogate / loss bounding.** The 0-1 loss has no usable gradient; train on cross-entropy.
  CE is unbounded, but the `[0,1]`-loss bounds require a bounded loss, so floor the predicted
  probability at `pmin` (`-log max(sigma(u)_y, pmin) in [0, log(1/pmin)]`) and rescale by
  `1/log(1/pmin)` to land in `[0,1]`. This rescaling is required for calibration; without it
  the objective and formula are mis-scaled and the posterior drifts (inflating KL).
- **Posterior / gradient.** Diagonal Gaussian `Q = N(mu, diag(sigma^2))`, `sigma =
  log(1 + exp(rho))` (softplus, `sigma > 0`); reparameterization `W = mu + sigma odot V`,
  `V ~ N(0,I)`; pathwise gradient (unbiased, low variance). Closed-form differentiable KL
  per coordinate `(1/2)(log(b0/b1) + (mu1 - mu0)^2/b0 + b1/b0 - 1)`, `b = sigma^2`, summed.
- **Data-dependent prior.** Split the training data: learn the prior mean by ERM on one part
  (dropout only here), learn the posterior + evaluate the certificate on the disjoint
  remainder (so the prior is data-free w.r.t. the bound). Initialize the posterior at the
  prior to keep KL small (KL dominates the bound).
- **Reported certificate (0-1 loss).** Invert the *sharp* pb-kl, not the quadratic relaxation:
  `L(Q) <= f*( Lhat_S(Q), (KL + log(2 sqrt(n)/delta))/n )` where
  `f*(q, c) = sup{ p in [q,1] : kl(q || p) <= c }` (the kl-inversion, computed by bisection).
  Note the certificate's `c` uses `/n` (bare pb-kl), whereas the training objective's `kl_term`
  uses `/2n` (the relaxation). `Lhat_S(Q)` is unobservable, so it is MC-estimated over `m`
  weight draws, then upper-bounded with an inner binary-kl tail
  (`kl(Lhat(Qhat_m) || Lhat(Q)) <= log(2/delta')/m`). The certificate is the nested inversion
  `f*( f*(Lhat(Qhat_m), log(2/delta')/m), c )`, with confidence `1 - delta - delta'`.

## Working code

Fills the bound-machinery slot of the probabilistic-network harness (stochastic forward pass,
`net.compute_kl`, Monte-Carlo risk sampling, and `inv_kl`). Faithful to the PBB implementation.

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
