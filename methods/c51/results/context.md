# Research question

An agent interacts with a time-homogeneous Markov Decision Process
$(\mathcal{X},\mathcal{A},R,P,\gamma)$: from state $x$ it picks action $a$, receives a
random reward $R(x,a)$, and transitions to $X'\sim P(\cdot\,|\,x,a)$. Following a policy
$\pi$, the *return* is the discounted sum of rewards
$Z^\pi(x,a)=\sum_{t\ge0}\gamma^t R(x_t,a_t)$, a *random variable*. The standard goal is to
learn its *expectation*, the value $Q^\pi(x,a)=\mathbb{E}\,Z^\pi(x,a)$, and to act greedily
with respect to the optimal value $Q^*$.

The return is genuinely random — stochastic rewards, stochastic transitions, and (in the
control case) a policy that is itself shifting as it is learned. Its distribution can be
multimodal (a "win" branch and a "lose" branch), skewed, or heavy-tailed; the value
$Q^\pi$ is a single scalar summary of it. The question this raises: what is gained by
predicting more of the *return's distribution* rather than only its mean, within the same
off-policy, bootstrapped, deep-function-approximation setting that mean-based value learning
uses?

# Background

**Bellman's equations and the expected return.** Value-based reinforcement learning rests on
the recursive structure of the expected return (Bellman, 1957; Sutton & Barto, 1998). For a
fixed policy,
$$Q^\pi(x,a)=\mathbb{E}\,R(x,a)+\gamma\,\mathbb{E}_{P,\pi}\,Q^\pi(x',a'),$$
and for control,
$$Q^*(x,a)=\mathbb{E}\,R(x,a)+\gamma\,\mathbb{E}_P\max_{a'}Q^*(x',a').$$
Viewing value functions as vectors in $\mathbb{R}^{\mathcal{X}\times\mathcal{A}}$, the Bellman
operator $\mathcal{T}^\pi$ and optimality operator $\mathcal{T}$ are both $\gamma$-contractions
in the $L_\infty$ norm; by Banach's fixed-point theorem their repeated application converges
geometrically to the unique $Q^\pi$ (resp. $Q^*$) (Bertsekas & Tsitsiklis, 1996). This is the
theoretical backbone behind SARSA and Q-learning: a single contraction guarantees convergence
to a single fixed point.

**Recursions beyond the mean.** Recursions for quantities of the return beyond the mean go back
nearly as far as Bellman's equation. Jaquette (1973) studied a *moment optimality* criterion that
imposes a total ordering on distributions and showed it admits a stationary optimal policy. Sobel
(1982) is usually credited with the first Bellman-style equations for the higher moments (notably
the variance) of the return. Chung & Sobel (1987) analyzed convergence of the distribution of the
return in total-variation distance. White (1988) studied "nonstandard MDP criteria" from the
occupancy perspective. The return distribution has been used for specific downstream purposes: to
represent *parametric uncertainty* about the environment (Dearden et al., 1998, with a Gaussian
value model and a Normal-Gamma prior; Engel et al., 2005, with a Gaussian process; Geist &
Pietquin, 2010, with unscented Kalman filters), and to build *risk-sensitive* algorithms (Morimura
et al., 2010, parametric and nonparametric models of the return's CDF; Tamar et al., 2016,
learning the return variance under linear approximation; Prashanth & Ghavamzadeh, 2013, a
risk-sensitive actor-critic). Mannor & Tsitsiklis (2011) give negative results on computing
variance-constrained optimal solutions.

**Metrics between distributions.** Several established notions of distance between probability
distributions are available off the shelf. Transport-based distances such as the Wasserstein
(Mallows / Kantorovich) metric, for two CDFs $F,G$ over the reals,
$$d_p(F,G)=\inf_{U,V}\|U-V\|_p=\Big(\int_0^1\big|F^{-1}(u)-G^{-1}(u)\big|^p\,du\Big)^{1/p},$$
the infimum over couplings, attained by the inverse-CDF (quantile) coupling. Overlap-based
discrepancies — total variation, Kullback-Leibler divergence, the Kolmogorov sup-CDF distance —
compare probability mass at matched locations.

**Greedy updates under approximation.** Independently of distributions, the greedy
(control) update under approximation has been studied: Tsitsiklis & Van Roy (1996) and
Gordon (1995) document oscillation/"chattering" of greedy policies, and Harutyunyan et al. (2016)
revisit related behavior.

**Empirical motivation.** Veness et al. (2015) obtained very fast learning on Atari by directly
predicting Monte-Carlo returns with a density model (Compress and Control). In supervised
learning, when targets are given, one routinely fits a whole output distribution rather than a
point estimate; in RL the target is itself a guess ("learn a guess from a guess", Sutton &
Barto, 1998).

# Baselines

**DQN (Mnih et al., 2015).** The standard deep value-based agent. A convolutional network
$Q_\theta(x,\cdot)$ maps a stack of 4 grayscale $84\times84$ frames to one scalar action-value
per action (conv 32×8×8 stride 4 → conv 64×4×4 stride 2 → conv 64×3×3 stride 1 → FC 512 → FC
$|\mathcal{A}|$). It is trained off-policy from a replay buffer by regressing the scalar Bellman
target with a squared loss
$\big(r+\gamma\max_{a'}Q_{\tilde\theta}(x',a')-Q_\theta(x,a)\big)^2$, where $\tilde\theta$ is a
periodically-copied *target network*; actions are $\epsilon$-greedy on $Q_\theta$; rewards are
clipped to $[-1,1]$. It learns the *mean* return.

**Double DQN (van Hasselt et al., 2016).** Decouples action selection from evaluation in the
target ($r+\gamma Q_{\tilde\theta}(x',\arg\max_{a'}Q_\theta(x',a'))$) to reduce the maximization
bias of the scalar target.

**Dueling architecture (Wang et al., 2016).** Splits the head into a state-value stream and an
advantage stream, $Q=V+(A-\overline A)$, improving credit assignment across actions.

**Prioritized replay (Schaul et al., 2016).** Samples high-TD-error transitions more often to
speed propagation of the learning signal.

**Gaussian return models (Morimura et al., 2010; Tamar et al., 2016).** Prior attempts to model
the return *distribution* directly. They restrict the distribution to a parametric Gaussian (or
learn only its variance) for policy evaluation / risk sensitivity.

# Evaluation settings

The natural yardstick is the Arcade Learning Environment (Bellemare et al., 2013): the suite of
57 Atari 2600 games, with the standard DQN preprocessing (4-frame stacking of $84\times84$
grayscale frames, frame-skip 4, max-over-2-frames, no-op resets, reward clipping to $[-1,1]$,
episodic-life). The protocol: off-policy training from a replay buffer with an
$\epsilon$-greedy behavior policy, a periodically-updated target network, and per-game scores
reported against random-play and human baselines (e.g. human-normalized mean/median over the
games, and per-game improvement counts). A small subset of training games is used for
hyperparameter selection; the remainder for testing. The ALE is nominally deterministic, but
effective stochasticity arises from state aliasing, from a nonstationary policy during learning,
and from a stochastic action-repeat mechanism that randomly rejects the
agent's chosen action. A small tabular domain such as CliffWalk (Sutton & Barto, 1998), where a
ground-truth return distribution can be estimated by many Monte-Carlo rollouts, is the natural
setting for sanity-checking that a proposed distributional update actually converges to the true
return distribution.

# Code framework

The off-policy value-based deep-RL harness already has an Atari environment with the standard
wrappers, a replay buffer, the DQN convolutional torso, an optimizer, a target network, and an
$\epsilon$-greedy loop. The open slots are the prediction made for each state-action, the scalar
score used for greedy action selection, the bootstrapped target, and the loss.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def make_env(env_id, seed):
    pass

class ReplayBuffer:
    pass

class QNetwork(nn.Module):
    """DQN's convolutional torso; the prediction head is the open slot."""
    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )
        self.head = None  # TODO

    def prediction(self, x):
        h = self.torso(x / 255.0)
        raise NotImplementedError  # TODO

    def get_action(self, x, action=None):
        raise NotImplementedError  # TODO

def bootstrap_target(target_net, rewards, next_obs, dones, gamma):
    raise NotImplementedError  # TODO

def loss_fn(online_net, obs, actions, target):
    raise NotImplementedError  # TODO

def train(env, replay, online_net, target_net, optimizer):
    pass
```
