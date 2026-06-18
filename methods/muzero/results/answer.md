# MuZero

## Problem it solves

Tree-based planning (MCTS + self-play) reaches superhuman strength in chess, shogi, and Go, but only because it is handed a perfect simulator of the rules — for state transitions, legal actions, and terminal detection. Model-free RL needs no model but cannot do the precise lookahead those domains reward. MuZero keeps the search but learns its model from interaction alone, so the same algorithm plans superhumanly in board games and reaches state-of-the-art on the visually complex Atari suite — with no knowledge of the rules or dynamics.

## Key idea

Don't model the environment's observations; model only the quantities the search reads at a node — reward, value, policy. The search never decodes the internal state back into an observation, so the learned model is trained for *value equivalence*: when unrolled under a hypothetical action sequence, its predicted policy/value/reward must match the targets that real play produces. There is no reconstruction loss and no constraint that the internal state resemble the true environment state; the state is free to encode whatever supports good planning.

Three jointly-trained functions:

- representation `s^0 = h_θ(o_1, …, o_t)`
- dynamics `r^k, s^k = g_θ(s^{k-1}, a^k)` — a deterministic, MDP-shaped recurrence
- prediction `p^k, v^k = f_θ(s^k)`

combined as `μ_θ(o_1,…,o_t, a^1,…,a^k) → (p^k, v^k, r^k)`.

## Planning (latent MCTS)

Run AlphaZero-style MCTS in the latent space. Each edge stores `{N, Q, P, R, S}`. Selection maximizes predictor-UCB:

```
a^k = argmax_a [ Q(s,a) + P(s,a) · (√Σ_b N(s,b) / (1 + N(s,a))) · (c1 + log((Σ_b N(s,b) + c2 + 1)/c2)) ]
```

with `c1 = 1.25`, `c2 = 19652`. A leaf is expanded with one call each to `g` (reward, next state) and `f` (policy, value). Backup uses the `(l−k)`-step discounted return bootstrapped from the leaf value,

```
G^k = Σ_{τ=0}^{l-1-k} γ^τ r_{k+1+τ} + γ^{l-k} v^l,    Q ← (N·Q + G^k)/(N+1),  N ← N+1.
```

Because values are unbounded in general MDPs, `Q` is normalized by the running min/max over the tree before entering the UCB rule. Differences from a rules-based search: learned transitions replace the simulator; legal actions are masked only at the root; terminals get no special treatment (trained as absorbing states). The search returns the visit-count distribution `π_t` and root value `ν_t`; the real action is sampled `a_{t+1} ~ π_t`.

## Targets and loss

- policy target `π_{t+k}` — the search's visit-count distribution (search = policy improvement).
- value target `z_t = u_{t+1} + γu_{t+2} + … + γ^{n-1}u_{t+n} + γ^n ν_{t+n}` — n-step return bootstrapped from the **search** value (board games: `γ=1`, `n` to end ⇒ final outcome; Atari: `n=10`).
- reward target `u_{t+k}` — the observed reward.

Total loss over `K=5` unrolled steps:

```
l_t(θ) = Σ_{k=0}^{K} [ l^r(u_{t+k}, r^k_t) + l^v(z_{t+k}, v^k_t) + l^p(π_{t+k}, p^k_t) ] + c‖θ‖².
```

`l^p(π,p) = -Σ_a π(a) log p(a)`. Board games: `l^v` is squared error and the reward loss is omitted. Atari: `l^v`, `l^r` are cross-entropy losses `-φ(y)^T log q` over a categorical representation `φ(·)` of the scalar — support of 601 integers in `[−300,300]` with an invertible squash `h(x) = sign(x)(√(|x|+1) − 1) + εx`, `ε = 0.001` — which is stable across the wide value/reward scales where MSE is not.

Gradient conditioning for the BPTT unroll: keep the initial observation loss unscaled, scale recurrent-step losses by `1/K`, halve the gradient entering the dynamics function at each recurrent step, and min-max scale the internal state to `[0,1]` after `h` and `g`. The updated pseudocode omits reward loss at `k=0`, where no dynamics transition has been predicted. (Reanalyze variant: re-run MCTS on old trajectories with the latest weights for fresh policy targets and a target-network value, for sample efficiency.)

## Working code

```python
import math
import numpy
import torch
import torch.nn.functional as F


def support_to_scalar(logits, support_size):
    probs = torch.softmax(logits, dim=1)
    support = torch.arange(
        -support_size, support_size + 1, device=logits.device, dtype=probs.dtype
    )
    x = (support * probs).sum(dim=1, keepdim=True)
    eps = 0.001
    x = torch.sign(x) * (((torch.sqrt(1 + 4 * eps * (torch.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1)
    return x


def scalar_to_support(x, support_size):
    eps = 0.001
    original_shape = x.shape
    x = x.reshape(-1, 1)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    lower = (floor + support_size).long()
    upper = lower + 1
    target = torch.zeros(x.shape[0], 2 * support_size + 1, device=x.device, dtype=x.dtype)
    target.scatter_add_(1, lower.clamp(0, 2 * support_size), 1 - prob)
    upper_weight = prob * (upper <= 2 * support_size).to(x.dtype)
    target.scatter_add_(1, upper.clamp(0, 2 * support_size), upper_weight)
    return target.reshape(*original_shape, 2 * support_size + 1)


def scale_gradient(x, scale):
    return x * scale + x.detach() * (1 - scale)


def scale_to_01(s):
    smin = s.min(1, keepdim=True)[0]; smax = s.max(1, keepdim=True)[0]
    return (s - smin) / (smax - smin).clamp_min(1e-5)


class MuZeroNet(torch.nn.Module):
    """h: representation, g: dynamics, f: prediction (fully-connected variant)."""
    def __init__(self, obs_dim, n_actions, enc=64, support=300):
        super().__init__()
        self.n_actions, self.support = n_actions, support
        full = 2 * support + 1
        self.h = torch.nn.Sequential(torch.nn.Linear(obs_dim, enc), torch.nn.ELU(),
                                     torch.nn.Linear(enc, enc))
        self.g_state = torch.nn.Sequential(torch.nn.Linear(enc + n_actions, enc), torch.nn.ELU(),
                                           torch.nn.Linear(enc, enc))
        self.g_reward = torch.nn.Linear(enc, full)
        self.f_policy = torch.nn.Linear(enc, n_actions)
        self.f_value = torch.nn.Linear(enc, full)

    def initial_inference(self, observation):
        s = scale_to_01(self.h(observation))
        reward = scalar_to_support(torch.zeros(len(observation), 1).to(s.device), self.support).squeeze(1)
        return self.f_value(s), reward, self.f_policy(s), s

    def recurrent_inference(self, s, action):
        a = torch.zeros(s.shape[0], self.n_actions, device=s.device)
        a.scatter_(1, action.long(), 1.0)
        s2 = scale_to_01(self.g_state(torch.cat([s, a], dim=1)))
        return self.f_value(s2), self.g_reward(s2), self.f_policy(s2), s2


class MinMaxStats:
    def __init__(self): self.lo, self.hi = float("inf"), -float("inf")
    def update(self, v): self.lo, self.hi = min(self.lo, v), max(self.hi, v)
    def normalize(self, v): return (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else v


class Node:
    def __init__(self, prior, to_play=0):
        self.N, self.value_sum, self.prior = 0, 0.0, prior
        self.to_play = to_play
        self.reward, self.state, self.children = 0.0, None, {}
    def expanded(self): return len(self.children) > 0
    def value(self): return 0 if self.N == 0 else self.value_sum / self.N
    def expand(self, reward, state, policy_logits, actions, to_play=0):
        self.to_play, self.reward, self.state = to_play, reward, state
        ps = torch.softmax(torch.tensor([policy_logits[0][a] for a in actions]), 0).tolist()
        for a, p in zip(actions, ps): self.children[a] = Node(p, to_play)


def ucb(parent, child, mm, c1=1.25, c2=19652, discount=0.997):
    pb_c = (math.log((parent.N + c2 + 1) / c2) + c1) * math.sqrt(parent.N) / (child.N + 1)
    value_score = mm.normalize(child.reward + discount * child.value()) if child.N > 0 else 0
    return pb_c * child.prior + value_score


def backpropagate(path, value, to_play, discount, mm):
    for node in reversed(path):
        node.value_sum += value if node.to_play == to_play else -value
        node.N += 1
        mm.update(node.value())
        value = node.reward + discount * value


def run_mcts(model, observation, legal_actions, num_simulations=50, discount=0.997):
    root = Node(0)
    value, reward, policy_logits, state = model.initial_inference(observation)
    root.expand(support_to_scalar(reward, model.support).item(), state, policy_logits, legal_actions)
    mm = MinMaxStats()
    backpropagate([root], support_to_scalar(value, model.support).item(), 0, discount, mm)
    for _ in range(num_simulations):
        node, path = root, [root]
        while node.expanded():
            action, node = max(node.children.items(), key=lambda kv: ucb(path[-1], kv[1], mm))
            path.append(node)
        parent = path[-2]
        value, reward, policy_logits, state = model.recurrent_inference(parent.state, torch.tensor([[action]]))
        node.expand(support_to_scalar(reward, model.support).item(), state, policy_logits, list(range(model.n_actions)))
        backpropagate(path, support_to_scalar(value, model.support).item(), 0, discount, mm)
    visits = numpy.array([c.N for c in root.children.values()], dtype="float32")
    pi = dict(zip(root.children.keys(), visits / visits.sum()))
    return root, pi, root.value()


def compute_target_value(root_values, rewards, index, td_steps=10, discount=0.997):
    b = index + td_steps
    value = root_values[b] * discount ** td_steps if b < len(root_values) else 0.0
    # Same indexing as the DeepMind pseudocode: rewards[i] is the reward
    # following the action stored at history index i.
    for i, r in enumerate(rewards[index:b]):
        value += r * discount ** i
    return value


def loss_function(value, reward, policy_logits, t_value, t_reward, t_policy):
    lv = -(t_value * F.log_softmax(value, dim=1)).sum(1)
    lr = -(t_reward * F.log_softmax(reward, dim=1)).sum(1)
    lp = -(t_policy * F.log_softmax(policy_logits, dim=1)).sum(1)
    return lv, lr, lp


def update_weights(model, optimizer, batch, K):
    obs, actions, target_v, target_r, target_pi = batch
    target_v = scalar_to_support(target_v, model.support)
    target_r = scalar_to_support(target_r, model.support)
    value, reward, policy, state = model.initial_inference(obs)
    preds = [(1.0, value, reward, policy)]
    for k in range(actions.shape[1]):
        value, reward, policy, state = model.recurrent_inference(state, actions[:, k])
        preds.append((1.0 / K, value, reward, policy))
        state = scale_gradient(state, 0.5)
    loss = 0
    for k, (gradient_scale, value, reward, policy) in enumerate(preds):
        lv, lr, lp = loss_function(value, reward, policy, target_v[:, k], target_r[:, k], target_pi[:, k])
        step_loss = lv + lp + (lr if k > 0 else 0)          # no root reward loss
        loss = loss + scale_gradient(step_loss, gradient_scale)
    loss = loss.mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

(Optimizer: SGD with momentum `0.9`, `weight_decay = 1e-4` realizing the `c‖θ‖²` term. Atari uses a residual-conv body for `h`/`g` with 16 blocks and 256 planes, downsampling 96×96 inputs of 32 stacked frames + 32 actions to a 6×6 latent; prioritized replay with priority `|ν_i − z_i|`.)
