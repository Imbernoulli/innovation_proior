## Research question

We are given a fixed dataset of trajectories: sequences of states, actions, and rewards collected by some unknown mix of policies, ranging from random flailing to occasional competence. We cannot interact with the environment to gather more data, ask for corrections, or explore. From this static, mixed-quality logbook alone, can we produce a controller that behaves well?

This is the offline reinforcement learning problem. It matters because in most real settings, including robotics, health, recommendation, and driving, online trial-and-error is expensive, slow, or dangerous, while logged experience is abundant. The task is to extract useful behavior from data that is often mediocre, using only the fixed log.

## Background

**The MDP and the return.** The environment is a Markov decision process $(\mathcal{S}, \mathcal{A}, P, \mathcal{R})$ with transition kernel $P(s'\mid s,a)$ and reward $r=\mathcal{R}(s,a)$. A trajectory is $\tau=(s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T,a_T,r_T)$. The return from timestep $t$ is the sum of future rewards,

$$R_t=\sum_{t'=t}^{T} r_{t'}.$$

The goal is a policy maximizing expected return $\mathbb{E}[\sum_t r_t]$.

**Temporal-difference learning.** The dominant control recipe learns a value function by bootstrapping: a Q-learning backup updates

$$Q(s_t,a_t)\leftarrow r_t+\gamma\max_{a'}Q(s_{t+1},a').$$

The target is built from the current estimate at the next state, so reward information propagates backward through Bellman backups one step at a time (Hung et al., 2019). A discount $\gamma<1$ makes discounted Bellman operators contractive and keeps infinite-horizon returns finite. Combining function approximation, bootstrapping, and off-policy data is the classic deadly triad (Sutton & Barto), and all three ingredients appear when a neural value function is trained by bootstrapping on a fixed off-policy log.

**The offline setting.** With a fixed dataset, the policy-improvement step queries $Q$ at actions that may not be well covered by the data, and the network extrapolates there. Because no new rollouts arrive, the policy receives no further corrective evidence. Extrapolation and value estimation under a static dataset are central topics in offline RL (Fujimoto et al., 2019; Levine et al., 2020).

**Sequence models that scale.** Separately, autoregressive Transformers (Vaswani et al., 2017; Radford et al., 2018; Brown et al., 2020) show that a single model trained by maximum likelihood can model high-dimensional, multimodal distributions over long sequences. A self-attention layer maps token embeddings $\{x_i\}_{i=1}^n$ to outputs $\{z_i\}_{i=1}^n$ by projecting each token to query, key, and value vectors and forming

$$z_i=\sum_{j=1}^{n}\operatorname{softmax}_j\!\left(\frac{\langle q_i,k_j\rangle}{\sqrt{d_k}}\right)v_j,$$

with causal GPT-style attention restricting the sum to $j\le i$. A single self-attention layer can associate positions separated by many tokens directly. These models come with a mature training stack: residual blocks, layer normalization, AdamW, warmup schedules, weight decay conventions, and scalable causal masking.

**Conditional generation.** Generative models can be steered by conditioning on a class label, control code, prompt, or other side information.

## Baselines

**Q-learning and TD control (Watkins, 1989; DQN, Mnih et al., 2013, 2015).** These methods fit $Q_\theta$ against bootstrapped Bellman targets and act greedily or near-greedily. They are strong online baselines and the backbone of many RL systems.

**Offline value methods with conservatism.** BCQ and BEAR (Fujimoto et al., 2019; Kumar et al., 2019) restrict policy improvement toward the data support. CQL (Kumar et al., 2020) lowers values for actions outside the dataset distribution so the learned value is conservative. BRAC (Wu et al., 2019) regularizes toward the behavior policy, and AWR (Peng et al., 2019) performs advantage-weighted regression. Offline Atari comparisons include REM and QR-DQN (Agarwal et al., 2020; Dabney et al., 2018).

**Behavior cloning.** Pure supervised learning regresses actions on states,

$$\max_\theta \sum_t \log \pi_\theta(a_t\mid s_t).$$

It has no Bellman backup, target network, or learned critic. A percentile variant clones only the top-$X\%$ of trajectories by return, trading data volume for data quality.

**Return-conditioned supervised policies.** Upside-down RL and reward-conditioned policies (Schmidhuber, 2019; Srivastava et al., 2019; Kumar et al., 2019) train a supervised policy conditioned on a desired return or command. The common form is close to a single-step supervised policy that maps the current state and a command to an action.

**Transformers inside RL.** Prior attention-based RL systems (Zambaldi et al., 2018; Parisotto et al., 2020) use Transformer modules as memory or representation components inside actor-critic optimization, addressing architecture and training stability while the optimization loop remains value-based.

## Evaluation settings

- **Offline Atari** (Arcade Learning Environment, Bellemare et al., 2013): the 1% DQN-replay dataset (Agarwal et al., 2020), consisting of 500k of the 50M transitions logged by an online DQN agent. Observations are $84\times84\times4$ stacked frames, actions are discrete, and scores are gamer-normalized following Hafner et al. (2020), where 100 denotes professional gamer score and 0 denotes random policy score. Games: Breakout, Qbert, Pong, Seaquest; 3 seeds.
- **OpenAI Gym continuous control via D4RL** (Fu et al., 2020): HalfCheetah, Hopper, Walker2d, and a sparse goal-reaching Reacher variant. Dataset variants are Medium, Medium-Replay, and Medium-Expert. Scores are normalized so 100 denotes expert performance; 3 seeds.
- **Key-to-Door** (Mesnard et al., 2020): a three-phase grid task where the agent must take a key near the beginning, traverse an empty middle phase, and receive a binary terminal reward at the door only if the key was collected.
- **Delayed-reward continuous control**: locomotion datasets modified so all reward is delivered at the final timestep.
- **Graph shortest path**: a fixed 20-node directed graph posed as an MDP with reward $-1$ per step and $0$ at the goal; training data consists of 1000 random walks of length $T=10$.

## Code framework

Available engineering pieces include PyTorch modules, NumPy preprocessing, a causal GPT-style Transformer that accepts input embeddings, an offline trajectory dataset, AdamW, learning-rate warmup, masking for padded timesteps, and a supervised training loop. The open design slot is how to turn the logged trajectories and the available components into a controller.

```python
import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):
    def __init__(self, state_dim, act_dim, max_length=None):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length

    def forward(self, states, actions, rewards, conditioning, timesteps, attention_mask=None):
        raise NotImplementedError

    def get_action(self, states, actions, rewards, conditioning, timesteps, **kwargs):
        raise NotImplementedError


class CausalTransformer(nn.Module):
    """Stacked causal self-attention blocks over embedded sequences."""
    def __init__(self, hidden_size, n_layer, n_head, max_tokens):
        super().__init__()
        pass

    def forward(self, inputs_embeds, attention_mask=None):
        raise NotImplementedError


class ConditionedTrajectoryModel(TrajectoryModel):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=None, max_ep_len=4096, **kwargs):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.hidden_size = hidden_size
        self.transformer = CausalTransformer(hidden_size, n_layer=..., n_head=..., max_tokens=...)
        # TODO: design the model

    def forward(self, states, actions, rewards, conditioning, timesteps, attention_mask=None):
        raise NotImplementedError

    def get_action(self, states, actions, rewards, conditioning, timesteps, **kwargs):
        raise NotImplementedError


def train_step(model, get_batch, optimizer, batch_size):
    states, actions, rewards, dones, conditioning, timesteps, attention_mask = get_batch(batch_size)
    # TODO: train the model
    raise NotImplementedError


def evaluate_conditioned_episode(
    env, state_dim, act_dim, model, max_ep_len, scale,
    state_mean, state_std, device, target_conditioning, mode="normal"
):
    state = env.reset()
    # TODO: run the controller in the environment
    raise NotImplementedError
```
