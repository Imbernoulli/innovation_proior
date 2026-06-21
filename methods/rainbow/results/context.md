## Research Question

A value-based deep reinforcement learning agent can learn Atari games from pixels with a
convolutional Q-network, replay, and a periodically copied target network. After that base result,
several independent improvements exist, each aimed at a different weakness: overestimated
bootstrap values, uninformative replay samples, poor generalization across actions, slow reward
propagation, mean-only return estimates, and shallow exploration.

The open question is not whether any one of these improvements works alone. The harder question is
whether a single agent can use the useful pieces together without creating contradictory
definitions of the target, the loss, the replay priority, the network head, or the action rule. A
combined agent has to make these choices coherently, and the evidence must then distinguish true
complementarity from redundant or interfering components.

## Base Agent

The starting point is the DQN loop. At time $t$, the agent stores
$(S_t,A_t,R_{t+1},\gamma_{t+1},S_{t+1})$ in a replay buffer, samples minibatches from that buffer,
and optimizes an online network $q_\theta$ against a frozen target network $q_{\bar\theta}$. The
one-step squared loss is

$$
\left(R_{t+1}+\gamma_{t+1}\max_{a'}q_{\bar\theta}(S_{t+1},a')-q_\theta(S_t,A_t)\right)^2.
$$

The target network is copied periodically from the online network and is not directly optimized.
The behavior policy is usually $\epsilon$-greedy with respect to the online Q-values. The replay
buffer stores recent experience and is sampled uniformly, so the update is stable enough for deep
function approximation but does not distinguish high-learning-potential transitions from stale
ones.

## Existing Ingredients

Double Q-learning changes the bootstrap action rule. The online network chooses the maximizing
action and the target network evaluates that chosen action:

$$
R_{t+1}+\gamma_{t+1}q_{\bar\theta}\left(S_{t+1},
\arg\max_{a'}q_\theta(S_{t+1},a')\right).
$$

Prioritized replay changes the sampling distribution. A transition is sampled in proportion to a
power of its last learning error, and importance-sampling weights correct the biased minibatch
gradient as the exponent $\beta$ is annealed toward 1.

Dueling networks change the head. A shared encoder feeds a value stream and an action-advantage
stream, merged as

$$
q_\theta(s,a)=v_\eta(f_\xi(s))+a_\psi(f_\xi(s),a)
-\frac{1}{N_{\text{actions}}}\sum_{a'}a_\psi(f_\xi(s),a').
$$

Multi-step learning changes the scalar target by replacing the single reward with the truncated
return

$$
R_t^{(n)}=\sum_{k=0}^{n-1}\gamma_t^{(k)}R_{t+k+1},
\quad
\gamma_t^{(k)}=\prod_{i=1}^{k}\gamma_{t+i}.
$$

Categorical distributional RL changes the output object. Instead of a scalar action value, the
network predicts probabilities over a fixed support
$z^i=v_{\min}+(i-1)(v_{\max}-v_{\min})/(N_{\text{atoms}}-1)$, acts on the mean
$z^\top p_\theta(s,a)$, projects Bellman-shifted target atoms back to the support, and minimizes a
KL divergence from projected target distribution to prediction.

Noisy linear layers change exploration. A standard linear layer is replaced by a deterministic
stream plus learned parametric noise:

$$
y=(b+Wx)+(b_{\text{noisy}}\odot\epsilon^b+(W_{\text{noisy}}\odot\epsilon^w)x).
$$

With learned noise in the value network, the policy can act greedily while the sampled weights
produce coherent state-dependent exploration.

## Collision Points

Several ingredients touch the same object. Multi-step returns, double Q-learning, and categorical
distributional RL all redefine the bootstrap target. Prioritized replay needs one scalar priority
per sampled transition, but a distributional learner no longer has a squared scalar TD loss as its
native objective. Dueling networks produce scalar Q-values in their original form, so a categorical
agent must decide whether to combine value and advantage before or after atom probabilities are
normalized. Noisy layers replace linear layers and remove the need for random action selection,
which changes both training-time action selection and evaluation behavior.

The combined design therefore has to answer four precise implementation questions: what
distribution is bootstrapped at $S_{t+n}$, which network selects and which network evaluates the
bootstrap action, which scalar is stored as replay priority, and where the dueling aggregation
occurs relative to the categorical softmax.

## Evaluation And Code Frame

The benchmark frame is the 57-game Atari Learning Environment setup inherited from DQN-family
agents: $84\times84$ grayscale frame stacks, action repeat, reward clipping to $[-1,1]$, terminal
life-loss transitions, replay minibatches, a target network, and aggregate human-normalized
metrics. Final evaluation commonly reports both no-op starts and human starts because the gap
between them can reveal overfitting to an agent's own trajectory distribution.

The code frame is an off-policy value-based harness with explicit open slots:

```python
class ReplayBuffer:
    def sample(self, batch_size):
        raise NotImplementedError
    def update_priorities(self, idxs, priorities):
        pass

class QNetwork:
    def forward(self, x):
        raise NotImplementedError

def act(net, x):
    raise NotImplementedError

def bootstrap_target(online_net, target_net, batch, gamma):
    raise NotImplementedError

def loss_fn(online_net, batch, target):
    raise NotImplementedError
```

The method must fill these slots with a single target construction, a single loss, a single replay
priority, a single head definition, and a single exploration rule.
