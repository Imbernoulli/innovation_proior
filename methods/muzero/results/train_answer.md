I present MuZero, a reinforcement-learning algorithm that combines the deep lookahead of Monte-Carlo tree search with a learned model of the environment, removing any dependence on a hand-written simulator or the rules of a task. The motivating problem is that classical planning agents such as AlphaZero play chess, shogi, and Go at superhuman level only because they are given a perfect generative model: they need the true state transition, the legal moves at every position, and the exact terminal outcome. Model-free agents such as DQN and its descendants need no such model and excel on visually complex domains, but they cannot perform the precise multi-step planning that wins in board games. MuZero closes this gap by learning, from interaction alone, the latent dynamics that the search needs, and it does so well enough that the same algorithm reaches state-of-the-art on the visually complex Atari suite while also matching AlphaZero's strength in board games.

The central insight is that the search never actually looks at the environment state; it only reads three quantities off each node: an estimate of how good the position is, a policy prior that tells it which actions are promising, and, in general-MDP settings, the reward associated with the transition. Therefore the learned model does not need to reconstruct observations or even to represent the true state. It only needs to be value-equivalent: when it is unrolled under a hypothetical sequence of actions, the policy, value, and reward it predicts must match the targets that real play would produce. Anything that does not serve this planning objective is free to be discarded, which saves capacity compared with pixel-reconstruction approaches and avoids the compounding errors that come from forcing a model to generate observations it will never use.

I implement this as three jointly-trained neural-network functions. The representation function h_θ maps a history of observations into an initial internal state s^0 = h_θ(o_1, …, o_t). The dynamics function g_θ is a deterministic, MDP-shaped recurrence that consumes the current internal state and a hypothetical action and produces the immediate reward and the next internal state: r^k, s^k = g_θ(s^{k−1}, a^k). The prediction function f_θ reads a policy and a value off any internal state: p^k, v^k = f_θ(s^k). Together they define μ_θ(o_1,…,o_t, a^1,…,a^k) → (p^k, v^k, r^k), which is the learned model that the planner consumes.

Planning proceeds as AlphaZero-style MCTS, but entirely inside this latent space. Each edge of the search tree stores visit count N, mean value Q, prior P, reward R, and the cached next state S. Selection follows the predictor-UCB rule a^k = argmax_a [ Q(s,a) + P(s,a) · (√Σ_b N(s,b) / (1 + N(s,a))) · (c1 + log((Σ_b N(s,b) + c2 + 1)/c2)) ], with c1 = 1.25 and c2 = 19652. When a leaf is reached it is expanded with one call to g_θ for the reward and next state and one call to f_θ for the policy and value. During backup I form the (l−k)-step discounted return G^k = Σ_{τ=0}^{l−1−k} γ^τ r_{k+1+τ} + γ^{l−k} v^l for each node along the path and update Q and N accordingly. Because the value scale is unbounded in general MDPs, I normalize Q on the fly using the running minimum and maximum observed anywhere in the tree before feeding it into the UCB rule, so no environment-specific reward scale has to be provided.

Three concessions are forced by the absence of a real simulator. State transitions are learned rather than exact, which is the entire point of the algorithm. Legal-action masking, which AlphaZero applies at every node using the rules, can only be applied at the root in MuZero; deeper in the tree the network's prior is trusted, and illegal actions receive no support because training sees only legal play. Terminal positions receive no special treatment inside the tree, because there is no terminal detector; instead terminal states are trained as absorbing, so passing through them does not corrupt the value estimate.

Training is the part that makes the latent search useful. At every hypothetical unroll step k I compare the model's predictions against three targets. The policy target is the visit-count distribution π_{t+k} produced by running MCTS at the real position at time t+k. Using the search output as a target is crucial because the search is a policy-improvement operator: the visit distribution is better than the raw network prior, so training toward it closes a self-improvement loop. The reward target is simply the observed scalar reward u_{t+k}. The value target is an n-step return bootstrapped off the search value, z_t = u_{t+1} + γu_{t+2} + … + γ^{n−1}u_{t+n} + γ^n ν_{t+n}. Bootstrapping from the search value rather than the raw network value gives a stronger and lower-variance target than model-free alternatives. For board games γ = 1, there are no intermediate rewards, and n is set to reach the end of the game, so z_t collapses to the final outcome z ∈ {−1,0,+1}, exactly as in AlphaZero. For Atari I use n = 10.

The full loss sums these three errors over K = 5 unrolled steps plus L2 regularization: l_t(θ) = Σ_{k=0}^{K} [ l^r(u_{t+k}, r^k_t) + l^v(z_{t+k}, v^k_t) + l^p(π_{t+k}, p^k_t) ] + c||θ||². The policy loss is always cross-entropy. In board games the value loss is mean squared error and the reward loss is omitted because there are no intermediate rewards. In Atari, where rewards and values span many orders of magnitude, both value and reward are represented as categorical distributions over a fixed support of 601 integers from −300 to 300 after applying the invertible squash h(x) = sign(x)(√(|x|+1) − 1) + εx with ε = 0.001. This categorical cross-entropy representation is stable across scales where direct regression would fail.

Because the model is unrolled recurrently and trained by backpropagation through time, I add small gradient-conditioning tricks that do not change the objective but keep optimization stable. Losses on recurrent steps are scaled by 1/K so their total gradient stays roughly constant as K changes. The gradient entering the dynamics function at each recurrent step is halved so the total gradient delivered to g_θ does not compound across the unroll. The internal state is min-max scaled to [0,1] after every application of h and g. These details matter empirically but are not part of the conceptual contribution.

For board games the architecture is similar to AlphaZero: a residual-tower body shared between h and g, with a policy-value head f. For Atari the representation function consumes the last 32 frames and the last 32 actions, because many Atari actions have no visible immediate effect and the action history is genuine information; the input is downsampled through strided convolutions to a compact 6×6 latent so that the latent search remains tractable. The optimizer is SGD with momentum 0.9 and weight decay 1e−4, which realizes the regularization term. Replay is prioritized by |ν_i − z_i|, the absolute gap between the search value and the eventual return, so the learner focuses on positions where the search still disagrees with reality.

In short, MuZero keeps everything that made AlphaZero powerful — MCTS, self-play, a policy-value network, and the idea that the search output should supervise the network — but replaces the hand-written simulator with a learned latent model. The model is trained for value equivalence rather than reconstruction, so it allocates all of its capacity to the quantities the search actually needs. The result is one algorithm that plans at superhuman strength in domains where the rules are known and also learns to plan from pixels in domains where they are not.

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
    smin = s.min(1, keepdim=True)[0]
    smax = s.max(1, keepdim=True)[0]
    return (s - smin) / (smax - smin).clamp_min(1e-5)


class MuZeroNet(torch.nn.Module):
    """h: representation, g: dynamics, f: prediction (fully-connected variant)."""

    def __init__(self, obs_dim, n_actions, enc=64, support=300):
        super().__init__()
        self.n_actions, self.support = n_actions, support
        full = 2 * support + 1
        self.h = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, enc), torch.nn.ELU(), torch.nn.Linear(enc, enc)
        )
        self.g_state = torch.nn.Sequential(
            torch.nn.Linear(enc + n_actions, enc), torch.nn.ELU(), torch.nn.Linear(enc, enc)
        )
        self.g_reward = torch.nn.Linear(enc, full)
        self.f_policy = torch.nn.Linear(enc, n_actions)
        self.f_value = torch.nn.Linear(enc, full)

    def initial_inference(self, observation):
        s = scale_to_01(self.h(observation))
        reward = scalar_to_support(
            torch.zeros(len(observation), 1).to(s.device), self.support
        ).squeeze(1)
        return self.f_value(s), reward, self.f_policy(s), s

    def recurrent_inference(self, s, action):
        a = torch.zeros(s.shape[0], self.n_actions, device=s.device)
        a.scatter_(1, action.long(), 1.0)
        s2 = scale_to_01(self.g_state(torch.cat([s, a], dim=1)))
        return self.f_value(s2), self.g_reward(s2), self.f_policy(s2), s2


class MinMaxStats:
    def __init__(self):
        self.lo, self.hi = float("inf"), -float("inf")

    def update(self, v):
        self.lo, self.hi = min(self.lo, v), max(self.hi, v)

    def normalize(self, v):
        return (v - self.lo) / (self.hi - self.lo) if self.hi > self.lo else v


class Node:
    def __init__(self, prior, to_play=0):
        self.N, self.value_sum, self.prior = 0, 0.0, prior
        self.to_play = to_play
        self.reward, self.state, self.children = 0.0, None, {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return 0 if self.N == 0 else self.value_sum / self.N

    def expand(self, reward, state, policy_logits, actions, to_play=0):
        self.to_play, self.reward, self.state = to_play, reward, state
        ps = torch.softmax(torch.tensor([policy_logits[0][a] for a in actions]), 0).tolist()
        for a, p in zip(actions, ps):
            self.children[a] = Node(p, to_play)


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
        value, reward, policy_logits, state = model.recurrent_inference(
            parent.state, torch.tensor([[action]])
        )
        node.expand(
            support_to_scalar(reward, model.support).item(),
            state,
            policy_logits,
            list(range(model.n_actions)),
        )
        backpropagate(path, support_to_scalar(value, model.support).item(), 0, discount, mm)
    visits = numpy.array([c.N for c in root.children.values()], dtype="float32")
    pi = dict(zip(root.children.keys(), visits / visits.sum()))
    return root, pi, root.value()


def compute_target_value(root_values, rewards, index, td_steps=10, discount=0.997):
    b = index + td_steps
    value = root_values[b] * discount ** td_steps if b < len(root_values) else 0.0
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
        lv, lr, lp = loss_function(
            value, reward, policy, target_v[:, k], target_r[:, k], target_pi[:, k]
        )
        step_loss = lv + lp + (lr if k > 0 else 0)
        loss = loss + scale_gradient(step_loss, gradient_scale)
    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```
