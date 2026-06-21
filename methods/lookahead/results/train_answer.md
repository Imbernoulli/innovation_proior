The single most fragile knob in training a deep network is the learning rate. With plain SGD, with heavy-ball or Nesterov momentum, or with Adam, the step size is what decides between an iterate that oscillates in the sharp directions and bounces or blows up, and one that crawls so slowly that training never finishes — and the fragility is not a one-time tax, because every new architecture and dataset sends me back to sweeping learning rates and weight decays. The two dominant lines of improvement both attack this by being smarter about the *gradient*: adaptive methods (AdaGrad, Adam) keep a running estimate of the per-coordinate gradient scale and divide by it, and accelerated methods (heavy-ball, Nesterov) keep a velocity that accumulates past gradients to push along consistent directions and damp the rest. Both genuinely help, but both still live inside the same loop — they reshape the step, and they still need that step tuned. I want something orthogonal: a mechanism that sits *outside* whatever optimizer I am already running, does not touch how it computes its step, but makes the whole trajectory lower-variance and robust to a wider band of learning rates, for almost no compute or memory and almost no change to an existing pipeline.

To see where the pain comes from I make it precise on the one model where I can actually compute the steady state, a diagonal noisy quadratic $\hat L(x) = \tfrac{1}{2}(x-c)^\top A (x-c)$ with the optimum itself noisy, $c \sim \mathcal N(0,\Sigma)$, so every coordinate evolves on its own with curvature $a_i$ and noise variance $\sigma_i^2$. The expected loss splits cleanly into a bias, a variance, and an irreducible floor: $\mathbb E[\hat L] = \tfrac{1}{2}\sum_i a_i\big(\mathbb E[\theta_i]^2 + \mathbb V[\theta_i] + \sigma_i^2\big)$. I cannot touch the floor; the fight is bias versus variance. One SGD step is $\theta \leftarrow \theta - \gamma A(\theta-c)$, so the mean contracts as $\mathbb E[x^{(t+1)}] = (I-\gamma A)\mathbb E[x^{(t)}]$ — per coordinate the multiplier $1-\gamma a$ helps the bias only until $\gamma a = 1$, beyond which the mean sign-flips and oscillates while staying stable up to $\gamma a = 2$. The variance follows $\mathbb V[x^{(t+1)}] = (I-\gamma A)^2 \mathbb V[x^{(t)}] + \gamma^2 A^2\Sigma$, which does not vanish but relaxes to a fixed point
$$V^*_{\mathrm{SGD}} = \frac{\gamma^2 A^2\Sigma}{I-(I-\gamma A)^2},$$
and this grows monotonically with $\gamma$ across the whole stable range. So the larger, useful step sizes that drive the bias down fast are exactly the ones that inflate the steady-state jitter and risk overshoot, and short-horizon analyses (Wu et al. 2018) show SGD is pulled toward precisely this large-$\gamma$ regime. If I could knock the variance term down *without* shrinking $\gamma$, I would keep the fast bias contraction and a low floor at once. That is the prize. The classical hammer for variance is averaging the iterates — Ruppert (1988) and Polyak & Juditsky (1992) proved iterate averaging attains the optimal asymptotic variance, and in deep learning this returned as Stochastic Weight Averaging (Izmailov et al. 2018). But the usual averages bother me on three counts: they are flat equal-weight averages that count stale early points the same as recent ones; they demand a *when to start averaging* decision that is itself fragile; and they are a *readout* — a final model produced at the end — that never feeds back into the trajectory the optimizer is actually walking.

I propose Lookahead. Keep two sets of weights: the fast weights $\theta$ that the inner optimizer $A$ grinds on, and slow weights $\phi$. Each outer iteration synchronizes $\theta \leftarrow \phi$, runs $A$ for $k$ steps on fresh minibatches, $\theta \leftarrow \theta + A(L,\theta,d)$, and then takes the slow weights one step toward the final fast weights by linear interpolation, $\phi \leftarrow \phi + \alpha(\theta - \phi)$, before restarting the next window from the new $\phi$. The fast weights go exploring for $k$ steps; the slow weights take one committed step in the direction $\theta_k - \phi$ that the exploration found. This is deliberately the dumbest possible use of the inner trajectory: I keep only the *first and last* iterate, $\theta_0 = \phi$ and $\theta_k$, and combine them with a *fixed* weight $\alpha$, no sub-problem to solve. The minimal alternatives I considered are each worse here — pulling the iterate back every inner step (as accelerated-SVRG / Katyusha does, with an SVRG correction that behaves badly on neural nets) would prevent the fast weights from ever ranging out, which defeats the point of letting them probe and then jumping across the oscillation in one clean interpolation; and keeping all $k$ iterates to solve a least-squares extrapolation (Anderson acceleration, nonlinear extrapolation) costs memory growing with $k$ plus a sub-problem each cycle, for strictly more machinery than I need if the cheap version already cuts variance. Because all I ever use from the inner optimizer is "give me $\theta_k$ after $k$ steps," $A$ can be literally anything — SGD, momentum, Adam — so this wraps existing optimizers rather than competing with them, and that is exactly the orthogonality I wanted, for one extra parameter copy and amortized $O((k+1)/k)$ work.

What makes this the right kind of average falls straight out of unrolling the slow-weight update across windows. Since $\phi_{t+1} = (1-\alpha)\phi_t + \alpha\,\theta_{t,k}$, after $T$ completed windows
$$\phi_T = (1-\alpha)^T\phi_0 + \alpha\sum_{j=0}^{T-1}(1-\alpha)^{T-1-j}\,\theta_{j,k},$$
which is exactly an exponential moving average of the *final* fast weight of each inner loop, decay $(1-\alpha)$, recent-weighted with a fading tail — running from initialization with no start decision, and steering the optimizer because the next window begins from it (the remark from Martens (2014) that an exponentially-decayed average works better than a flat Polyak average is precisely this recency). The limit $\alpha = 1$ collapses it to "just take the fast weights," cleanly recovering the inner optimizer. That it actually reduces variance I verify on the noisy quadratic with SGD inside. Working one coordinate at a time, the mean obeys $\mathbb E[\phi_{t+1}] = [1-\alpha+\alpha(1-\gamma a)^k]\,\mathbb E[\phi_t]$ — a convex combination of $1$ and SGD's $k$-step multiplier. For the variance I need three pieces: unrolling $k$ SGD steps from a fixed start gives $\mathbb V[\theta_{t,k}] = (1-\gamma a)^{2k}\mathbb V[\phi_t] + \sum_{i=0}^{k-1}(1-\gamma a)^{2i}\gamma^2 a^2\sigma^2$; the one-step relation $\mathrm{cov}(\theta_{t,k-1},\theta_{t,k}) = (1-\gamma a)\mathbb V[\theta_{t,k-1}]$ chained back $k$ times gives $\mathrm{cov}(\phi_t,\theta_{t,k}) = (1-\gamma a)^k\mathbb V[\phi_t]$; and substituting these into $\mathbb V[\phi_{t+1}] = (1-\alpha)^2\mathbb V[\phi_t] + \alpha^2\mathbb V[\theta_{t,k}] + 2\alpha(1-\alpha)\mathrm{cov}(\phi_t,\theta_{t,k})$ collapses the $\mathbb V[\phi_t]$ coefficient into a perfect square $[1-\alpha+\alpha(1-\gamma a)^k]^2$, consistent with the mean's contraction factor. Solving the resulting fixed point and dividing by $V^*_{\mathrm{SGD}}$ yields
$$V^*_{\mathrm{LA}} = \frac{\alpha^2\big(I-(I-\gamma A)^{2k}\big)}{\alpha^2\big(I-(I-\gamma A)^{2k}\big) + 2\alpha(1-\alpha)\big(I-(I-\gamma A)^{k}\big)}\cdot V^*_{\mathrm{SGD}}.$$
With $\beta = (1-\gamma a)^k$, stability $|1-\gamma a|<1$ puts $\beta \in (-1,1)$, so $1-\beta^2$ and $1-\beta$ are positive and, for $\alpha\in(0,1)$, the denominator is the numerator plus a strictly positive $2\alpha(1-\alpha)(1-\beta)$ term; the leading factor is therefore strictly between $0$ and $1$ per coordinate, oscillating or not. The slow weights have strictly smaller steady-state variance than the inner SGD at the *same* learning rate — the prize, lowering the floor without touching $\gamma$. The price, visible in the mean, is a slower bias contraction in non-oscillatory coordinates ($1-\alpha+\alpha(1-\gamma a)^k$ versus $(1-\gamma a)^k$ when that factor is positive), but in the large-$\gamma$, variance-limited regime where I actually train, variance is the binding constraint and that slowdown is cheap; in sign-flipping coordinates the interpolation instead damps the oscillation. Deterministically the same interpolation has a geometric reading: with momentum on a quadratic in the under-damped regime the trajectory spirals toward the optimum, so the window's start and end sit on different phases and the straight line between them cuts the corner and lands closer than either endpoint, improving the per-step rate exactly where momentum oscillates (in the over-damped, monotone regime there is nothing to cut across and it can only slow things slightly).

The one knob I introduce is $\alpha$, and I keep it fixed and small-ish rather than optimal. I can in fact solve for the loss-minimizing $\alpha$ on a quadratic, since the interpolation is a one-dimensional line search: $\alpha^* = (\theta_0-x^*)^\top A(\theta_0-\theta_k)/[(\theta_0-\theta_k)^\top A(\theta_0-\theta_k)]$, degenerate only when the endpoints coincide. But this needs the curvature $A$ and the optimum $x^*$; approximating $x^*$ by a Newton step with a diagonal empirical-Fisher curvature $\hat A$ gives a clipped adaptive estimate yet forces me to maintain $\hat A$ even when the inner optimizer is SGD, which breaks the minimal, optimizer-agnostic wrapper. So I keep $\alpha$ fixed — the adaptive derivation only reassures me a small fixed value is a sensible operating point. Likewise the inner optimizer's internal state (momentum buffers, Adam's moments) at interpolation time admits three coherent policies: maintain it (the minimally invasive default), interpolate the momentum buffer the same way as the parameters, or reset it to zero; I default to maintaining and expose the others. End to end, with defaults $k=5$ and $\alpha=0.8$: keep slow weights $\phi$ and fast weights $\theta$; each outer iteration set $\theta\leftarrow\phi$, run the inner optimizer $k$ steps, set $\phi\leftarrow\phi+\alpha(\theta-\phi)$, and repeat.

```python
from collections import defaultdict
import torch
from torch.optim.optimizer import Optimizer


class Lookahead(Optimizer):
    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        # optimizer: any inner optimizer A (SGD, Adam, ...). We only ever call its .step().
        self.optimizer = optimizer
        self._la_step = 0                 # counts inner steps within the current window
        self.la_alpha = la_alpha          # interpolation weight alpha; alpha=1 -> just the inner optimizer
        self._total_la_steps = la_steps   # window length k
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)
        # slow weights phi: a cached copy of the parameters, taken at sync time
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        # swap in the slow weights phi for evaluation
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    def step(self, closure=None):
        # k steps forward: let the inner optimizer take one fast-weight step
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    # 1 step back: phi <- phi + alpha (theta_k - phi), in place on p.data.
                    # p = alpha*theta_k + (1-alpha)*phi  is exactly that interpolation.
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)
                    # commit: the new phi becomes the sync point and the next window's start
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        # interpolate the inner momentum buffer the same way as the params
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)
        return loss
```
