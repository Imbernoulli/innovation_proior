We are training a network of stochastic units — a model that, for each input, emits not a single action but a *distribution* over actions, from which an output is sampled. A Gaussian unit draws $y \sim \mathcal{N}(\mu, \sigma^2)$; a Bernoulli unit flips $\Pr\{y=1\}=p$. We score the whole thing by a scalar performance signal — expected reward $E\{r\mid\theta\}$ in a reinforcement task, or the log-likelihood the model assigns to the right answer in a supervised/imitation setting — and we want to push the parameters up that signal. The tool of choice is the gradient, and for expected reward we have a clean way to estimate it from samples: the score-function identity $\frac{d}{d\theta}E\{r\mid\theta\} = E\!\left[\,r\,\partial_\theta \ln g\,\right]$, where $\partial_\theta \ln g$ is the characteristic eligibility of the parameter. This is REINFORCE: at the end of a trial increment $\Delta\theta = \alpha\,(r-b)\,\partial_\theta \ln g$, a nonnegative rate times the offset reward $(r-b)$ times the eligibility, with a baseline $b$ that cuts variance without biasing the direction. On average it climbs $\nabla_\theta E\{r\mid\theta\}$ exactly, and for a Gaussian unit the eligibilities $\partial_\mu \ln g = (y-\mu)/\sigma^2$ and $\partial_\sigma \ln g = ((y-\mu)^2 - \sigma^2)/\sigma^3$ nudge $\mu$ toward sampled actions that beat the baseline and shrink or grow $\sigma$ depending on whether the realized squared error ran below or above $\sigma^2$. All correct, all principled — and it reliably converges to junk.

The failure is structural, not a bug in the gradient. Because the update is a gradient, it finds local optima, and worse, the search distribution itself is one of the things the gradient is free to move. The same dynamics that pull $\mu$ toward a comfortable region make $(y-\mu)^2$ small there, so the $\sigma$ eligibility turns negative, $\sigma$ falls, samples cluster tighter, and the unit can now only sample near where it already sits. The distribution sharpens toward a near-deterministic output *early*, before it has established that the choice it is collapsing onto is actually best. The instant the spread is gone, exploration is dead — there is no mass left to stumble onto anything better — and the eligibility variance collapses too, so even the learning signal fades. On hierarchically structured tasks, where the agent must keep several options live at once until coordination pays off, this is brutal: the collapse kills the options one by one. The reward-penalty algorithm of Barto's group fights back with an extra $\lambda(1-r)(1-y-p)$ term that shoves a Bernoulli unit toward the opposite of what it just did on failure, but it is a hand-built correction welded to the $0/1$ reward-penalty form; it never says, at the level of the distribution, *how much spread to keep*, and it does not transfer to a Gaussian unit or a likelihood objective. The $\varepsilon$-greedy style of fix bolts exploration on *outside* the objective — a fixed noise knob applied identically whether the agent is wildly over-confident or appropriately uncertain, with the optimized policy still free to collapse underneath it. None of these is a statement about the policy's own shape.

The quantity we actually care about — "spread," "how non-committal the distribution is," "how much it is hedging" — has a name. For a density it is $H(\pi) = -\!\int \pi(a)\log\pi(a)\,da = E_{a\sim\pi}[-\log\pi(a)]$, the Shannon entropy, maximal when the distribution is most spread and falling to its floor as it concentrates. For a Gaussian $H = \tfrac12\log(2\pi e\,\sigma^2)$ rises with $\sigma$, so "shrinking $\sigma$" and "destroying entropy" are literally the same act; for a Bernoulli $H = -p\log p - (1-p)\log(1-p)$ is peaked at $p=\tfrac12$ and zero at $p\in\{0,1\}$. The collapse we keep suffering *is* an entropy collapse, and Jaynes's maximum-entropy principle says preferring high entropy is not a hack but the right default: among all distributions consistent with what you actually know, choose the least-committal one. So I propose entropy regularization — the maximum-entropy / NLL-minus-entropy loss (the policy-gradient form built this way is sometimes called MENT, maximum-entropy exploration). Do not replace the fit objective; bias it toward the maximum-entropy solution among those that fit by adding the entropy back into the objective and maximizing

$$J(\pi) = E_{a\sim\pi}[\,r(a)\,] + \tau\,H(\pi) = E_{a\sim\pi}\!\left[\,r(a) - \tau\log\pi(a)\,\right],\qquad \tau > 0.$$

The first term still climbs reward; the second penalizes collapse; the small temperature $\tau$ trades the two. What makes this principled rather than a vague "encourage randomness" knob is what happens when you solve the idealized problem exactly. Maximize $J$ over a free distribution subject only to $\sum_a \pi(a)=1$ with a Lagrange multiplier: $\partial_{\pi(a)}\big[\sum \pi r - \tau\sum\pi\log\pi + \lambda(\sum\pi-1)\big] = r(a) - \tau(\log\pi(a)+1) + \lambda = 0$, so $\log\pi(a) = (r(a)+\lambda)/\tau - 1$ and

$$\pi^*_\tau(a) = \frac{\exp(r(a)/\tau)}{Z},\qquad Z = \sum_a \exp(r(a)/\tau).$$

The entropy bonus turns the *target* of the optimization into a temperature-$\tau$ Boltzmann softmax over reward. The two limits show $\tau$ is exactly the dial we wanted: as $\tau\to 0$, $\exp(r/\tau)$ is dominated by the largest $r$, so $\pi^*_\tau$ becomes a delta on $\arg\max_a r(a)$ — pure greedy, and $\tau=0$ cleanly recovers plain REINFORCE; as $\tau\to\infty$, $r(a)/\tau\to 0$ and $\pi^*_\tau$ becomes uniform — pure exploration. In between $\tau$ slides continuously from exploit to explore, so we want it small but strictly positive: just enough to defeat premature collapse without dragging the target toward near-uniform and ceasing to fit.

There is a second reading that makes "prevents collapse" sharp rather than hand-wavy. Substituting $r(a) = \tau\log\pi^*_\tau(a) + \tau\log Z$ back into $J$,

$$J(\pi) = \sum_a \pi(a)\big[\tau\log\pi^*_\tau(a) + \tau\log Z - \tau\log\pi(a)\big] = \tau\log Z - \tau\,D_{\mathrm{KL}}\!\big(\pi \,\|\, \pi^*_\tau\big).$$

Since the KL divergence is nonnegative and zero only at $\pi=\pi^*_\tau$, maximizing the regularized objective is *exactly* minimizing the KL divergence to the spread-out, full-support Boltzmann target. Without the $\tau H$ term the implicit target is the delta on $\arg\max$; with it, the destination is a distribution with mass everywhere reward is non-tiny, and pushing $\pi$ toward a full-support target is structurally incapable of driving it to a premature delta. That is the mechanism, derived. To optimize over the parametric $\pi_\theta$ rather than the free $\pi$ we need $\nabla_\theta J$. The reward part is the usual score-function gradient, unchanged. For the entropy part, differentiate $H(\pi_\theta) = -\sum_a \pi_\theta(a)\log\pi_\theta(a)$: using $\nabla_\theta\pi_\theta = \pi_\theta\nabla_\theta\log\pi_\theta$, the term $-\sum_a \pi_\theta\nabla_\theta\log\pi_\theta = -\nabla_\theta\sum_a\pi_\theta = 0$ vanishes because the score has zero mean, leaving

$$\nabla_\theta H(\pi_\theta) = -E_{a\sim\pi_\theta}\!\left[\log\pi_\theta(a)\,\nabla_\theta\log\pi_\theta(a)\right],$$

so that folding everything together gives REINFORCE with an entropy-augmented reward,

$$\nabla_\theta J = E_{a\sim\pi_\theta}\!\left[\big(r(a) - \tau\log\pi_\theta(a)\big)\,\nabla_\theta\log\pi_\theta(a)\right].$$

The undropped form carries an extra $-\tau$ inside the bracket, but that is only a constant baseline — it has zero expectation against the score and changes variance, not direction. The $-\tau\log\pi_\theta$ bonus is large for low-probability actions and disappears only when the distribution has enough spread, so it pushes back precisely against over-confidence. And it does so *adaptively*, which is the real reason it beats a fixed noise floor: for a Gaussian, $H = \tfrac12\log(2\pi e\,\sigma^2)$ gives $\partial H/\partial\sigma = 1/\sigma$, which blows up as $\sigma\to 0$. Near a local optimum the reward gradient is flat by definition while the entropy gradient is enormous, so their sum points firmly back toward spread and the optimizer simply cannot drive $\sigma$ to zero — it is held off at the $\sigma$ where $\tau/\sigma$ balances the small reward gradient. A static external noise floor does not know the policy is collapsing; the entropy bonus steepens exactly as the thing it prevents approaches.

The practical wrinkle is computing $H(\pi_\theta)$ for whatever the model emits. Gaussian and Bernoulli have closed forms, but the distributions we care about are richer — a mixture of Gaussians has $H = -E[\log\sum_k w_k\,\mathcal{N}_k]$, a log-of-a-sum with no closed form. So we fall back on the one estimator that works for anything we can sample from and evaluate the density of: Monte-Carlo, $H(\pi_\theta) \approx \frac{1}{K}\sum_{k=1}^K -\log\pi_\theta(a^{(k)})$ with $a^{(k)}\sim\pi_\theta$, an unbiased estimate of the entropy value. The gradient needs care, because `MixtureSameFamily.sample()` is not reparameterized: backpropagating through $-\log\pi_\theta(a)$ at a fixed sample gives $-E[\nabla\log\pi_\theta]=0$, the entropy value but not its gradient. The score-function gradient $-E[\log\pi_\theta(a)\nabla\log\pi_\theta(a)]$ must instead be carried by a detached $\log\pi_\theta(a)$ multiplying $\nabla\log\pi_\theta(a)$, assembled by the straight-through combination `entropy_value.detach() + entropy_score_grad - entropy_score_grad.detach()`, which has the forward value of the Monte-Carlo estimate and the backward gradient we derived. Finally the sign, which is the entire method. In a likelihood/behavioral-cloning setting the reward is the log-likelihood of the target, so maximizing $J = E[\log\text{-lik}] + \tau H$ becomes, as a loss to *minimize*, the negative log-likelihood **minus** the entropy bonus,

$$\text{loss} = \mathrm{NLL} - \alpha\,H(\pi),\qquad \mathrm{NLL} = -E[\log\pi_\theta(\text{target})],\quad \alpha = \tau.$$

The entropy enters with a minus sign because minimizing $-\alpha H$ maximizes $H$; a plus sign would minimize entropy and *cause* the collapse we are preventing. The weight $\alpha$ is small and positive — on the order of $0.01$ — big enough to keep the distribution honestly spread, small enough that the fit term dominates wherever the data is informative.

```python
import torch
import torch.nn as nn


class EntropyRegularizedLoss(nn.Module):
    """NLL of the target plus a maximum-entropy bonus on the predicted distribution.
    Minimizing (nll - alpha*H) maximizes  log-likelihood + alpha*H(pi): the fit term
    concentrates the distribution while the entropy term keeps it from collapsing to a
    near-deterministic solution. `dist` is any distribution exposing .log_prob and
    .sample (e.g. torch.distributions.MixtureSameFamily, a GMM over actions)."""

    def __init__(self, alpha=0.01, num_entropy_samples=1):
        super().__init__()
        self.alpha = alpha                                # entropy weight (temperature tau)
        self.num_entropy_samples = num_entropy_samples

    def forward(self, dist, target):
        # Fit term: NLL = -(log-likelihood of the target under the predicted distribution).
        nll = -dist.log_prob(target).mean()

        # Entropy bonus H(pi) = E_{a~pi}[ -log pi(a) ], estimated by Monte-Carlo because a
        # mixture has no closed-form entropy. MixtureSameFamily.sample is not reparameterized,
        # so the backward pass uses the score-function identity:
        # grad H = -E[log pi(a) * grad log pi(a)].
        with torch.no_grad():
            samples = dist.sample((self.num_entropy_samples,))
        log_prob = dist.log_prob(samples)
        entropy_value = -log_prob.mean()
        entropy_score_grad = -(log_prob.detach() * log_prob).mean()
        entropy = entropy_value.detach() + entropy_score_grad - entropy_score_grad.detach()

        # Subtract: minimizing nll - alpha*H maximizes H. (A plus sign would *minimize*
        # entropy and drive the collapse this is meant to prevent.)
        return nll - self.alpha * entropy
```
