## Research question

We are given a fixed dataset of trajectories: sequences of states, actions, and rewards collected by some unknown mix of policies, ranging from random flailing to occasional competence. We cannot interact with the environment to gather more data, ask for corrections, or explore. From this static, mixed-quality logbook alone, can we produce a controller that behaves well, ideally better than the typical trajectory in the data?

This is the offline reinforcement learning problem. It matters because in most real settings, including robotics, health, recommendation, and driving, online trial-and-error is expensive, slow, or dangerous, while logged experience is abundant. A solution has to extract useful behavior from data that is often mediocre, avoid the instabilities that plague value learning when fresh corrective interaction is unavailable, and ideally recombine useful fragments of different trajectories into behavior that no single logged trajectory exhibits in full. The central tension is that the obvious supervised approach copies the data and inherits its average quality, while the obvious RL approach bootstraps a value function and becomes fragile exactly when it cannot collect new data to correct its errors.

## Background

**The MDP and the return.** The environment is a Markov decision process $(\mathcal{S}, \mathcal{A}, P, \mathcal{R})$ with transition kernel $P(s'\mid s,a)$ and reward $r=\mathcal{R}(s,a)$. A trajectory is $\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T,a_T,r_T)$. The return from timestep $t$ is the sum of future rewards,

$$R_t=\sum_{t'=t}^{T} r_{t'}.$$

The goal is a policy maximizing expected return $\mathbb{E}[\sum_t r_t]$.

**Temporal-difference learning and the deadly triad.** The dominant control recipe learns a value function by bootstrapping: a Q-learning backup updates

$$Q(s_t,a_t)\leftarrow r_t+\gamma\max_{a'}Q(s_{t+1},a').$$

The target is built from the current estimate at the next state. Reward information therefore propagates backward through Bellman backups one step at a time; in sparse- or delayed-reward problems, useful signal can take many updates to reach early actions and can be distracted by intermediate reward signals (Hung et al., 2019). A discount $\gamma<1$ helps make discounted Bellman operators contractive and keeps infinite-horizon returns finite, but it also biases control toward near-term reward. Most importantly, combining function approximation, bootstrapping, and off-policy data is the classic deadly triad (Sutton & Barto). All three ingredients appear when a neural value function is trained by bootstrapping on a fixed off-policy log.

**Why the offline case is especially fragile.** With a fixed dataset, the value-learning loop has a second failure mode: the max or policy improvement step queries $Q$ at actions that are not well covered by the data. The network extrapolates there, optimistic errors are selected by maximization, and subsequent bootstrapped targets amplify those errors. Because no new rollouts arrive, the policy never receives corrective evidence for the unsupported actions it drifts toward. This extrapolation and overestimation pathology is a central diagnostic in offline RL (Fujimoto et al., 2019; Levine et al., 2020).

**Sequence models that scale.** Separately, autoregressive Transformers (Vaswani et al., 2017; Radford et al., 2018; Brown et al., 2020) show that a single model trained by maximum likelihood can model high-dimensional, multimodal distributions over long sequences. A self-attention layer maps token embeddings $\{x_i\}_{i=1}^n$ to outputs $\{z_i\}_{i=1}^n$ by projecting each token to query, key, and value vectors and forming

$$z_i=\sum_{j=1}^{n}\operatorname{softmax}_j\!\left(\frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}\right)v_j,$$

with causal GPT-style attention restricting the sum to $j\le i$. The relevant property for decision making is that attention can associate positions separated by many timesteps in one layer, rather than relying on step-by-step dynamic-programming propagation. These models also come with a mature training stack: residual blocks, layer normalization, AdamW, warmup schedules, weight decay conventions, and scalable causal masking.

**Conditional generation.** Generative models can be steered by conditioning on a class label, control code, prompt, or other side information. Existing controllable-generation work usually treats the conditioning variable as fixed across a whole sequence. In control, the desired outcome is naturally time-varying: after collecting some reward, the remaining amount to collect should change.

## Baselines

**Q-learning and TD control (Watkins, 1989; DQN, Mnih et al., 2013, 2015).** These methods fit $Q_\theta$ against bootstrapped Bellman targets and act greedily or near-greedily. They are strong online baselines and the backbone of many RL systems, but in an offline dataset they combine off-policy data, function approximation, and bootstrapping, and they are vulnerable to extrapolation error on unsupported actions.

**Offline value methods with conservatism.** BCQ and BEAR (Fujimoto et al., 2019; Kumar et al., 2019) restrict policy improvement toward the data support. CQL (Kumar et al., 2020) lowers values for actions outside the dataset distribution so the learned value is conservative. BRAC (Wu et al., 2019) regularizes toward the behavior policy, and AWR (Peng et al., 2019) performs advantage-weighted regression. Offline Atari comparisons naturally include REM and QR-DQN (Agarwal et al., 2020; Dabney et al., 2018). The shared limitation is that these methods still improve a policy using a learned value estimate, so much of the algorithmic machinery exists to keep that value estimate from being exploited.

**Behavior cloning.** Pure supervised learning regresses actions on states,

$$\max_\theta \sum_t \log \pi_\theta(a_t\mid s_t).$$

It has no Bellman backup, target network, or learned critic, so it is stable. Its limitation is equally direct: it imitates the behavior distribution in the dataset and averages across high- and low-quality trajectories. A percentile variant clones only the top-$X\%$ of trajectories by return, trading data volume for data quality, but choosing $X$ requires evaluation feedback that the offline setting withholds.

**Return-conditioned supervised policies.** Upside-down RL and reward-conditioned policies (Schmidhuber, 2019; Srivastava et al., 2019; Kumar et al., 2019) train a supervised policy conditioned on a desired return or command. They already suggest that outcome information can be used as a conditioning variable rather than as a Bellman target. The common form is close to a single-step supervised policy, leaving open whether a long-context sequence model over trajectory prefixes can use history more effectively.

**Transformers inside RL.** Prior attention-based RL systems (Zambaldi et al., 2018; Parisotto et al., 2020) use Transformer modules as memory or representation components inside actor-critic optimization. They address architecture and training stability, but the optimization loop remains value-based.

## Evaluation settings

- **Offline Atari** (Arcade Learning Environment, Bellemare et al., 2013): the 1% DQN-replay dataset (Agarwal et al., 2020), consisting of 500k of the 50M transitions logged by an online DQN agent. Observations are $84\times84\times4$ stacked frames, actions are discrete, and scores are gamer-normalized following Hafner et al. (2020), where 100 denotes professional gamer score and 0 denotes random policy score. Games: Breakout, Qbert, Pong, Seaquest; 3 seeds.
- **OpenAI Gym continuous control via D4RL** (Fu et al., 2020): HalfCheetah, Hopper, Walker2d, and a sparse goal-reaching Reacher variant. Dataset variants are Medium, Medium-Replay, and Medium-Expert. Scores are normalized so 100 denotes expert performance; 3 seeds.
- **Key-to-Door** (Mesnard et al., 2020): a three-phase grid task where the agent must take a key near the beginning, traverse an empty middle phase, and receive a binary terminal reward at the door only if the key was collected.
- **Delayed-reward continuous control**: locomotion datasets modified so all reward is delivered at the final timestep.
- **Graph shortest path**: a fixed 20-node directed graph posed as an MDP with reward $-1$ per step and $0$ at the goal; training data consists of 1000 random walks of length $T=10$.

## Code framework

Available engineering pieces include PyTorch modules, a causal GPT-style Transformer that accepts input embeddings, an offline trajectory dataset, AdamW, learning-rate warmup, masking for padded timesteps, and a supervised training loop. The open design slot is how to represent a trajectory prefix for a causal sequence model and how to turn the model output into a control action.

```python
import torch
import torch.nn as nn


class CausalTransformer(nn.Module):
    """Stacked causal self-attention blocks over embedded sequences."""
    def __init__(self, hidden_size, n_layer, n_head, max_tokens):
        super().__init__()
        # existing GPT-style implementation

    def forward(self, inputs_embeds, attention_mask=None):
        # returns hidden states shaped (B, L, hidden_size)
        raise NotImplementedError


class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.backbone = CausalTransformer(hidden_size, n_layer=..., n_head=..., max_tokens=...)
        # TODO: choose the trajectory-prefix tokenization and input embeddings
        # TODO: choose the conditioning signal used for controllable generation
        # TODO: choose which hidden positions produce action predictions

    def forward(self, states, actions, rewards, timesteps, conditioning=None, attention_mask=None):
        # TODO: embed a trajectory prefix, run the causal backbone, and predict actions
        raise NotImplementedError

    def get_action(self, states, actions, rewards, timesteps, conditioning=None):
        # TODO: crop/pad history and return the next action for deployment
        raise NotImplementedError


def train_step(model, batch, optimizer, loss_fn):
    states, actions, rewards, dones, timesteps, mask = batch
    # TODO: build any derived conditioning labels from the logged trajectory
    # TODO: compute a masked supervised action loss and update the model
    raise NotImplementedError


def rollout(env, model, target, max_ep_len):
    state = env.reset()
    # TODO: maintain the trajectory prefix and update the conditioning variable online
    raise NotImplementedError
```
