# Research question

A single value-based deep-RL algorithm (DQN: convolutional Q-network, experience replay, target
network, $\epsilon$-greedy) opened up Atari-from-pixels, and since then a stream of independent
extensions has each fixed a different limitation of it — overestimation bias, sample
inefficiency, poor action-generalization, slow reward propagation, mean-only value estimates,
shallow exploration. Each helps *in isolation*. But they were developed separately and the field
does not know which are *complementary*: whether stacking them yields additive gains or whether
they interfere, and how to integrate them at all when several touch the same machinery (the
bootstrap target, the loss, the network head, the replay distribution, the action-selection
rule). The precise problem: take a set of extensions that each address a *distinct* concern, work
out a single coherent agent that incorporates all of them at once (resolving the conflicts where
two of them want to modify the same object — e.g. a distributional loss vs. a TD-error-based
priority, or a dueling head vs. a categorical output), and determine empirically whether the
combination is largely complementary and where each piece contributes.

# Background

**MDPs and value-based RL.** An MDP $\langle\mathcal{S},\mathcal{A},T,r,\gamma\rangle$; at step
$t$ the agent sees $S_t$, takes $A_t$, receives $R_{t+1}$, discount $\gamma_{t+1}$, next state
$S_{t+1}$. The discounted return is $G_t=\sum_{k=0}^\infty\gamma_t^{(k)}R_{t+k+1}$ with
$\gamma_t^{(k)}=\prod_{i=1}^k\gamma_{t+i}$ (so $\gamma_t^{(0)}=1$ and episode termination is
$\gamma_t=0$). Value-based RL learns $q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]$ and acts
$\epsilon$-greedily. $\epsilon$-greedy's limitation: it is hard to discover courses of action that
extend far into the future, motivating more directed exploration.

**Q-learning and its overestimation bias.** The Bellman optimality target
$R_{t+1}+\gamma_{t+1}\max_{a'}q(S_{t+1},a')$ uses a $\max$, and when $q$ is noisy the $\max$
systematically *overestimates* (Jensen: $\mathbb{E}[\max]\ge\max\mathbb{E}$). The single network
both *selects* and *evaluates* the bootstrap action, coupling the two and amplifying the bias
(van Hasselt, 2010).

**Experience replay.** A buffer of the last $\sim$1M transitions, sampled to decorrelate updates
and reuse data (Lin, 1992). DQN samples it *uniformly*, which spends equal effort on transitions
that are already well-learned and on those with much to learn.

**Multi-step returns and the bias-variance trade-off.** The forward-view truncated $n$-step
return $R_t^{(n)}=\sum_{k=0}^{n-1}\gamma_t^{(k)}R_{t+k+1}$ (Sutton, 1988) bootstraps from
$S_{t+n}$ rather than $S_{t+1}$. Larger $n$ propagates real reward faster to earlier states and
lowers the bias of the bootstrap, at the cost of higher variance; it shifts the bias-variance
trade-off and (with tuned $n$) often speeds learning. (For control it is only strictly on-policy
correct for on-policy returns, but small $n$ is used pragmatically.)

**The distribution of returns.** The return $G_t$ is a random variable; its *distribution*
satisfies a distributional Bellman equation. Modeling the full distribution rather than its mean
$q$ carries more information (multimodality, risk), and was argued to be a useful auxiliary
representation for control even though acting still maximizes the mean.

**Directed exploration via parametric noise.** Injecting *learned* noise into the network weights
makes a single perturbation induce a consistent, state-dependent change in policy across many
steps — coherent exploration that $\epsilon$-greedy's per-step dithering cannot achieve. The noise
scale can be trained, so the network self-anneals exploration, per state and per weight.

**Empirical/diagnostic facts of the time.** DQN reaches human-level on many Atari games but is
slow (its strong final performance takes $\sim$200M frames) and fails outright on
exploration-hard games like Montezuma's Revenge where many actions precede the first reward.
Several extensions had *already* been combined pairwise (Prioritized DDQN, Dueling DDQN, Dueling
DDQN with prioritized replay), establishing that double Q-learning at least co-exists with
prioritization and dueling — but no integration of all the major axes existed.

# Baselines

**DQN (Mnih et al., 2015).** Online net $q_\theta$ (acts $\epsilon$-greedily, receives
gradient) and a periodically-copied target net $q_{\bar\theta}$; minimize
$(R_{t+1}+\gamma_{t+1}\max_{a'}q_{\bar\theta}(S_{t+1},a')-q_\theta(S_t,A_t))^2$ over uniform
minibatches from a 1M replay buffer; RMSProp. **Gap:** overestimation, uniform replay,
single-stream head, 1-step target, mean-only, $\epsilon$-greedy.

**Double DQN (van Hasselt et al., 2016).** Decouple selection from evaluation in the target:
$(R_{t+1}+\gamma_{t+1}q_{\bar\theta}(S_{t+1},\arg\max_{a'}q_\theta(S_{t+1},a'))-q_\theta(S_t,A_t))^2$.
**Gap addressed:** overestimation. Leaves the other five.

**Prioritized experience replay (Schaul et al., 2016).** Sample transition $t$ with probability
$p_t\propto|\delta_t|^\omega$ where $\delta_t$ is the last absolute TD error and $\omega$ shapes
the distribution; new transitions enter at maximum priority. Non-uniform sampling biases the
expected update, corrected by importance-sampling weights with exponent $\beta$ annealed toward 1.
**Gap addressed:** sample efficiency. (Note: a stochastic transition can stay high-priority even
when nothing more is to be learned.)

**Dueling networks (Wang et al., 2016).** Two streams off a shared encoder $f_\xi$ merged by a
mean-subtracting aggregator:
$q_\theta(s,a)=v_\eta(f_\xi(s))+a_\psi(f_\xi(s),a)-\frac{1}{N_{\text{actions}}}\sum_{a'}a_\psi(f_\xi(s),a')$.
The mean-subtraction fixes the identifiability of the value/advantage decomposition. **Gap
addressed:** generalization across actions / valuing states.

**Categorical distributional DQN, C51 (Bellemare et al., 2017).** Fixed support of $N_{\text{atoms}}$
atoms $z^i=v_{\min}+(i-1)\frac{v_{\max}-v_{\min}}{N_{\text{atoms}}-1}$; the network outputs masses
$p^i_\theta(s,a)$ via a per-action softmax. Greedy action on the *mean*
$\bar a^*=\arg\max_a z^\top p_{\bar\theta}(S_{t+1},a)$. The target distribution
$d'_t=(R_{t+1}+\gamma_{t+1}z,\,p_{\bar\theta}(S_{t+1},\bar a^*))$ lands off the support, so it is
projected back by $\Phi_z$ (linear interpolation onto the two nearest atoms), and the loss is
$D_{\text{KL}}(\Phi_z d'_t\,\|\,d_t)$. $N_{\text{atoms}}=51$, $[v_{\min},v_{\max}]=[-10,10]$.
**Gap addressed:** mean-only value estimate.

**Noisy Nets (Fortunato et al., 2017).** Replace a linear layer
$y=b+Wx$ by $y=(b+Wx)+(b_{\text{noisy}}\odot\varepsilon^b+(W_{\text{noisy}}\odot\varepsilon^w)x)$,
with random $\varepsilon^b,\varepsilon^w$ (factorised Gaussian, scale $\sigma_0$) and *learnable*
noise weights. The network can drive the noise toward zero, at different rates in different parts
of the state space, giving state-conditional self-annealing exploration. **Gap addressed:**
exploration.

# Evaluation settings

The 57 Atari 2600 games of the Arcade Learning Environment (Bellemare et al., 2013), training and
evaluation as in Mnih et al. (2015) and van Hasselt et al. (2016): scores evaluated during
training every 1M steps by freezing learning and running 500K frames; episodes truncated at 108K
frames. End-of-training best snapshot re-evaluated under two regimes — *no-op starts* (up to 30
random no-ops) and *human starts* (initial states sampled from human trajectories); the gap
between them measures over-fitting to the agent's own trajectories. Scores are human-normalized
per game ($0\%$ random, $100\%$ human expert), aggregated as the *median* across games (the mean
is dominated by a few games like Atlantis). Tracking the number of games above several human
fractions ($20\%/50\%/100\%/200\%/500\%$) disentangles where median gains come from. The natural
ablation protocol: remove one component at a time from the integrated agent to attribute its
contribution. Hyperparameters held identical across all 57 games.

# Code framework

The off-policy value-based harness has the Atari wrappers, a replay buffer, the conv torso, an
optimizer, a target network, and the loop. Each open slot below is where one or more extensions
will plug in; several extensions touch the *same* slot, which is the integration problem.

```python
import torch
import torch.nn as nn

class ReplayBuffer:
    """Stores transitions. Sampling distribution is an open slot (uniform? weighted?)."""
    def sample(self, batch_size):
        raise NotImplementedError  # TODO
    def update_priorities(self, idxs, priorities):
        pass  # TODO (only if sampling is non-uniform)

class QNetwork(nn.Module):
    """Conv torso fixed; the linear layers, the head structure, and the per-(s,a)
    output object (scalar value? something richer?) are all open slots."""
    def __init__(self, n_actions):
        super().__init__()
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = None  # TODO

    def forward(self, x):
        raise NotImplementedError  # TODO

def act(net, x):
    # action-selection rule (greedy? epsilon-greedy? something else?) -- open slot
    raise NotImplementedError  # TODO

def bootstrap_target(online_net, target_net, batch, gamma):
    # how many steps? selection/evaluation split? target object? -- open slot(s)
    raise NotImplementedError  # TODO

def loss_fn(online_net, batch, target):
    # which loss; whether it also feeds the sampling priority -- open slot
    raise NotImplementedError  # TODO
```
