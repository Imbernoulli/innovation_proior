## Research question

We want an agent that learns to act in high-dimensional, pixel-based environments (the
Atari 2600 suite is the standard testbed) purely from raw frames and a scalar reward,
using a single architecture and a single set of hyper-parameters across dozens of diverse
games. The dominant approach is to learn an action-value function with a deep network and
act greedily with respect to it. The open question we take up is *architectural*: almost
all of the recent progress in this setting has come from better learning algorithms
(better targets, better replay, better exploration) bolted onto a generic feed-forward
network. Could the *network itself* be shaped to fit the structure of value-based control,
so that it learns faster and generalizes better — while keeping exactly the same
input/output interface, so that every existing learning algorithm can drive it unchanged?

Concretely, the value-based network is asked to output one number per action for every
state. In many states the return is overwhelmingly determined by *where you are*, not
*what you do next* — most of the action values in a state are nearly equal, and only the
small differences between them matter for the policy. A generic single-stream network does
not obviously exploit this regularity, and we want any reshaping to leave the learning
algorithm and the input/output interface untouched and to require no extra supervisory
signal.

## Background

**Markov decision processes and value functions.** An agent interacts with an environment
over discrete time steps, choosing actions a from a finite set A and receiving rewards r.
It seeks to maximize the discounted return R_t = Σ_{τ≥t} γ^{τ−t} r_τ, γ ∈ [0,1]. For a
policy π, the action-value and state-value functions are
Q^π(s,a) = E[R_t | s_t=s, a_t=a, π] and V^π(s) = E_{a~π(s)}[Q^π(s,a)]. Q satisfies the
recursive (Bellman) relation Q^π(s,a) = E_{s'}[r + γ E_{a'~π(s')}[Q^π(s',a')]]. The optimal
value is Q*(s,a) = max_π Q^π(s,a); acting greedily, V*(s) = max_a Q*(s,a), and Q* satisfies
the Bellman optimality equation Q*(s,a) = E_{s'}[r + γ max_{a'} Q*(s',a') | s,a].

**The advantage function.** Baird (1993) introduced, in his advantage-updating algorithm,
the quantity A^π(s,a) = Q^π(s,a) − V^π(s), the *advantage* of taking action a in state s
relative to the state's value. By construction E_{a~π(s)}[A^π(s,a)] = 0, and for a
deterministic greedy policy the selected action a* satisfies Q(s,a*) = V(s) and therefore
A(s,a*) = 0. V measures how good a state is; A measures the relative importance of each
action within that state. Baird's advantage updating decomposed the shared
Bellman-residual update into two coupled updates — one for a state value, one for the
advantage — and was shown (Harmon & Baird 1995) to converge faster than Q-learning in
simple continuous-time domains; its successor, advantage learning (Harmon & Baird 1996),
maintained a single advantage function. Advantage functions also have a long history in
policy gradients (Sutton et al. 2000), where they appear as a variance-reduced baseline,
most recently estimated online to reduce policy-gradient variance (Schulman et al. 2015).

**The diagnostic observation that sets up the problem.** In many states the choice of
action has essentially no effect on the immediate dynamics — e.g. in a driving game,
steering only matters when a collision is imminent; the rest of the time every action leads
to nearly the same outcome. In such states the per-action values are nearly equal and only
the state value carries information. Yet a bootstrapping algorithm relies on an accurate
state value at *every* state. A generic network with one output per action has no built-in
way to learn the shared state value once; it must rediscover roughly the same value at each
output, and the shared notion is only nudged through whichever single action was sampled in
a given update.

## Baselines

**Deep Q-Networks (DQN; Mnih et al. 2015).** Approximate Q(s,a;θ) with a convolutional
network — 3 conv layers (32 filters 8×8 stride 4; 64 filters 4×4 stride 2; 64 filters 3×3
stride 1) feeding a single stream of fully-connected layers ending in |A| outputs, one Q
value per action. Train by minimizing the squared temporal-difference error
L(θ) = E[(y^{DQN} − Q(s,a;θ))²] with target y^{DQN} = r + γ max_{a'} Q(s',a';θ⁻), where θ⁻
is a separate *target network* whose parameters are frozen for a fixed number of steps and
periodically copied from the online network (this freezing is what makes the bootstrapped
regression stable). Updates are computed on minibatches drawn uniformly from an
*experience-replay* buffer of past transitions (s,a,r,s'), which re-uses data and
decorrelates consecutive samples. Gap: the network is a generic single stream; it carries
no structure reflecting that the state value is shared across actions, so it relearns each
action's value independently and updates the shared value only through the sampled action.

**Double DQN (DDQN; van Hasselt et al. 2015).** In the DQN target the same network both
*selects* the maximizing action and *evaluates* it, which biases values upward
(over-estimation; van Hasselt 2010). DDQN decouples the two: the online network picks the
action and the target network scores it,
y^{DDQN} = r + γ Q(s', argmax_{a'} Q(s',a';θ); θ⁻). Otherwise identical to DQN. Gap: a
better learning target, but the same generic single-stream architecture.

**Prioritized experience replay (Schaul et al. 2015).** Instead of sampling the replay
buffer uniformly, sample transitions with high expected learning progress more often,
using absolute TD-error as a proxy. Built on top of DDQN, this improved both learning speed
and final quality across the suite. Gap: an algorithmic/sampling improvement, again
orthogonal to the network architecture — and a natural thing a new architecture would need
to *combine* with rather than replace.

## Evaluation settings

The natural yardstick is the Arcade Learning Environment (Bellemare et al. 2013): 57 Atari
2600 games, raw pixel input, one architecture and one hyper-parameter set across all games.
Standard preprocessing: grayscale, resize to 84×84, max-pool over skipped frames, stack 4
frames as a state, clip rewards. A controlled policy-evaluation probe can isolate the
network architecture from exploration and policy-improvement confounds: fix a behavior
policy π, learn Q^π with TD(0) / Expected-SARSA targets
y = r + γ E_{a'~π(s')}[Q(s',a';θ)], and measure squared error against exact values,
Σ_{s,a}(Q(s,a;θ) − Q^π(s,a))². A small corridor MDP is useful for this because its action
set can be enlarged by adding no-op actions while keeping the true Q-values computable.
Standard aggregate metrics are human-normalized scores summarized as mean/median over
games, under both the "30 random no-op" and "human starts" evaluation protocols.

## Code framework

The single-stream value-based harness provides the replay buffer, target network, ε-greedy
behavior, TD loss, and training loop. The open slot is the network module that maps a state
to one value per action; everything around it is fixed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        # Convolutional feature trunk over stacked frames (the established DQN encoder).
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # TODO: the head that turns features into one value per action.
        pass

    def forward(self, x):
        feat = self.feature(x / 255.0)
        # TODO: produce a (batch, |A|) tensor of action values from `feat`.
        pass


def td_target(target_network, q_network, data, gamma):
    # Double-DQN target: online net selects the action, target net evaluates it.
    with torch.no_grad():
        next_actions = torch.argmax(q_network(data.next_observations), dim=1, keepdim=True)
        next_q = target_network(data.next_observations).gather(1, next_actions).squeeze(1)
        return data.rewards.flatten() + gamma * next_q * (1 - data.dones.flatten())


def train_step(q_network, target_network, optimizer, data, gamma):
    y = td_target(target_network, q_network, data, gamma)
    q = q_network(data.observations).gather(1, data.actions).squeeze(1)
    loss = F.mse_loss(y, q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# replay buffer, ε-greedy action selection, periodic target-network sync live outside this step.
```
