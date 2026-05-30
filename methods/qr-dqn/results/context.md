# Research question

In reinforcement learning the *return* $Z^\pi=\sum_{t\ge0}\gamma^tR_t$ collected by following a
policy $\pi$ is a random variable; classic value-based methods learn only its mean, the value
$Q^\pi(x,a)=\mathbb{E}[Z^\pi(x,a)]$. A distributional approach instead learns the whole law of
$Z^\pi$. The theory of that approach rests on one fact and one obstruction. The fact: the
distributional Bellman operator
$\mathcal{T}^\pi Z(x,a)\overset{D}{=}R(x,a)+\gamma Z(x',a')$ is a $\gamma$-contraction in the
*maximal Wasserstein* metric $\bar d_p(Z_1,Z_2)=\sup_{x,a}W_p(Z_1(x,a),Z_2(x,a))$ — Wasserstein
being the metric that respects distances between outcomes and is well-behaved under the
disjoint-support situations a Bellman update creates. The obstruction: the Wasserstein distance,
viewed as a loss, *cannot in general be minimized by stochastic gradient descent from sampled
transitions* — the expected gradient of the sample loss does not point at the true minimizer. So
the metric the theory contracts in is precisely the one a sample-based learner cannot descend.

A discrete-distribution agent that achieves strong results sidesteps this by fixing a comb of
support locations $z_1\le\cdots\le z_N$ over a predetermined interval, learning the *probabilities*
on those atoms, *projecting* the Bellman target back onto the comb, and minimizing a KL
divergence after projection. That works in practice but breaks the contract: it minimizes KL,
not Wasserstein, so the contraction theory does not explain why it works, and it needs the support
bounds $[V_{\min},V_{\max}]$ as domain knowledge plus a heuristic projection step. The precise
question: is there a distributional RL algorithm, runnable in the online stochastic-approximation
setting, that genuinely operates *end-to-end* on the Wasserstein metric — keeping the contraction
guarantee while being trainable from single samples — and so removing the projection and the
support hyperparameter?

# Background

**MDPs, returns, value distributions.** For an MDP $(\mathcal{X},\mathcal{A},R,P,\gamma)$,
$\gamma\in[0,1)$, and a policy $\pi$, the return $Z^\pi$ is random because of stochastic rewards,
transitions, and (in control) a shifting policy; the value distribution is its law, capturing the
*intrinsic* randomness of the returns (not parametric/estimation uncertainty). The mean of the
value distribution is the value function. TD methods bootstrap $Q$ through the scalar Bellman
operator $\mathcal{T}^\pi Q=\mathbb{E}[R]+\gamma\mathbb{E}_{P,\pi}[Q(x',a')]$; the distributional
analogue replaces equality of numbers by equality of laws.

**The Wasserstein / Mallows metric.** For distributions $U,Y$ over $\mathbb{R}$ with CDFs
$F_U,F_Y$ and inverse CDFs (quantile functions)
$F^{-1}_Y(\omega)=\inf\{y:\omega\le F_Y(y)\}$, the $p$-Wasserstein metric is the $L^p$ distance
between quantile functions,
$$W_p(U,Y)=\Big(\int_0^1\big|F_Y^{-1}(\omega)-F_U^{-1}(\omega)\big|^p\,d\omega\Big)^{1/p},\qquad
W_\infty(U,Y)=\sup_{\omega\in[0,1]}|F_Y^{-1}(\omega)-F_U^{-1}(\omega)|.$$
For $p=1$ it is the Earth-Mover's Distance and equals the area between the two CDFs. Unlike KL,
it accounts for *both* the probability of and the distance between outcomes, and is finite for
disjoint supports. It was shown (C51 line; Bellemare et al., 2017) that $\bar d_p$ is a metric over
value distributions and that $\mathcal{T}^\pi$ is a $\gamma$-contraction in it; its fixed point is
$Z^\pi$.

**The biased-gradient obstruction.** Bellemare et al. (2017, "Cramér distance") prove that for an
empirical distribution $\hat Y_m=\frac1m\sum\delta_{Y_i}$ of samples from a Bernoulli $B$,
$\arg\min_\mu\mathbb{E}_{Y_{1:m}}[W_p(\hat Y_m,B_\mu)]\ne\arg\min_\mu W_p(B,B_\mu)$: minimizing the
*sample* Wasserstein loss converges to the wrong place. This is why a contraction in Wasserstein
does not hand you a trainable algorithm.

**Quantile regression (Koenker).** A method, standard in econometrics, for estimating a quantile
of a distribution by stochastic approximation. For quantile level $\tau$, the *quantile regression
loss* $\rho_\tau(u)=u(\tau-\mathbb{1}_{u<0})$ is an asymmetric absolute loss — penalty $\tau$ on
underestimation ($u>0$) and $1-\tau$ on overestimation ($u<0$). Its minimizer over $\theta$ of
$\mathbb{E}_{\hat Z\sim Z}[\rho_\tau(\hat Z-\theta)]$ is the quantile $F_Z^{-1}(\tau)$, and its
sample gradient (which depends only on the *sign* of $u$) is unbiased. The Huber loss
(Huber, 1964) $\mathcal{L}_\kappa(u)=\frac12u^2$ for $|u|\le\kappa$ and $\kappa(|u|-\frac12\kappa)$
otherwise is the smooth-at-zero, robust-in-the-tails compromise between squared and absolute loss.

**Function approximation can break contractions.** Tsitsiklis & Van Roy (1997): a Bellman update
*composed with a projection onto an approximation space* may fail to be a contraction. So any
parametric distributional method must show that the *projected* operator still contracts.

# Baselines

**DQN (Mnih et al., 2015).** Convolutional $Q(x,\cdot;\theta)$ trained off-policy from replay by
regressing the scalar Bellman target with a Huber loss (DQN's gradient-clipped squared error is a
Huber loss with $\kappa=1$), RMSProp optimizer, periodically-copied target network,
$\epsilon$-greedy behaviour. **Gap:** learns only the mean return.

**Double DQN (van Hasselt et al., 2016).** Decouples action selection from evaluation in the
target to reduce maximization bias. Still mean-only.

**Dueling architecture (Wang et al., 2016); Prioritized replay (Schaul et al., 2016).**
Orthogonal improvements to credit assignment and sample reuse. Still mean-only.

**C51 (Bellemare, Dabney & Munos, 2017).** The distributional baseline this work most directly
reacts to. Models $Z(x,a)$ as a categorical distribution on a *fixed* comb $z_1\le\cdots\le z_N$
($N=51$) uniformly spaced over a fixed $[V_{\min},V_{\max}]$, with learnable probabilities $q_i$
(softmax logits). Applies a projection $\Phi$ that maps the shifted Bellman target
$\mathcal{T}^\pi Z$ (whose atoms $r+\gamma z_j$ generally miss the comb) back onto the comb by
assigning mass to the nearest support points in proportion to distance, then minimizes the KL
divergence between projected target and prediction. **Gaps:** (i) it minimizes KL, not Wasserstein,
so its strong performance is disconnected from the contraction theory; (ii) it requires the
support bounds $[V_{\min},V_{\max}]$ as a hyperparameter and a uniform resolution; (iii) the
projection step is needed precisely because fixed atoms create disjoint-support problems.

**Parametric (Gaussian/Laplace) return models (Morimura et al., 2010).** Parameterize the value
distribution by the mean and scale of a Gaussian or Laplace and minimize KL between target and
prediction; used for risk-sensitive Q-learning. **Gap:** unimodal family; the Bellman operator is
at best a non-expansion in KL, so guarantees are only asymptotic.

# Evaluation settings

The yardstick is the Arcade Learning Environment (Bellemare et al., 2013): 57 Atari 2600 games,
standard DQN preprocessing (4-frame stacks of $84\times84$ grayscale, frame-skip 4, no-op random
starts up to 30, reward clipping), off-policy replay training with a periodically-updated target
network, 200M frames. Behaviour is $\epsilon$-greedy with $\epsilon$ annealed (to a low value such
as $0.01$). Evaluation freezes learning every 1M frames and averages return over 500K frames
(best-agent protocol uses $\epsilon=0.001$); scores are human-normalized,
$(\text{agent}-\text{random})/(\text{human}-\text{random})$, aggregated as mean/median over the 57
games. A natural diagnostic domain is a small windy gridworld with two rooms and stochastic
transitions producing a genuinely *multimodal* return distribution, where a ground-truth return
distribution can be estimated by many Monte-Carlo rollouts, so a policy-evaluation update can be
checked for actually minimizing $W_1$ to the truth.

# Code framework

The off-policy value-based deep-RL harness already has the Atari wrappers, replay buffer, DQN
convolutional torso, an optimizer, a target network, and the $\epsilon$-greedy loop. The open
slots: what the per-state-action prediction *is* (a scalar? a fixed-atom histogram? something
else), the scalar used for greedy action selection, how the bootstrapped target is formed, and the
loss that pulls prediction toward target.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 200  # resolution of the per-(x,a) prediction -- meaning is the open slot

class QNetwork(nn.Module):
    """DQN conv torso; the head emits N numbers per action, but what those
    numbers parameterize is the open slot."""
    def __init__(self, n_actions, n=N):
        super().__init__()
        self.n_actions, self.n = n_actions, n
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, n_actions * n),
        )

    def prediction(self, x):
        return self.net_out(x).view(-1, self.n_actions, self.n)  # (B, A, N)

    def net_out(self, x):
        return self.torso(x / 255.0)

    def greedy_action(self, x):
        raise NotImplementedError  # TODO: scalar score per action, then argmax

def bootstrap_target(target_net, rewards, next_obs, dones, gamma):
    raise NotImplementedError  # TODO: form the per-(x,a) target object

def loss_fn(online_net, obs, actions, target):
    raise NotImplementedError  # TODO
```
