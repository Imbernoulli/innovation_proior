# Ape-X — Distributed Prioritized Experience Replay

## Problem

Scale deep reinforcement learning by using many machines. Standard distributed training parallelizes
gradient computation, but gradients go stale within milliseconds and force tight, low-latency
synchronization (and earlier distributed RL such as Gorila keeps each worker's experience in a
private, uniformly-sampled local replay, so a rare valuable transition found by one worker never
helps the others). The goal: reach state-of-the-art final performance on the Arcade Learning
Environment in a fraction of the wall-clock time, with no per-game tuning, and have it generalize to
continuous control.

## Key idea

Distribute **data generation and selection**, not gradient computation. Decouple **acting** from
**learning**:

- **Many actors** (hundreds, on CPU), each in its own environment copy, run the current policy and
  push experience into **one shared, centralized prioritized replay memory**.
- **One learner** (on GPU) samples prioritized batches from that shared memory, takes gradient steps,
  and periodically publishes updated parameters back to the actors.

Three modifications make this scale:

1. **Shared replay → global priorities.** Because all actors feed one prioritized buffer, a
   high-priority (high-TD-error) transition discovered by *any* actor is immediately preferentially
   sampled by the single learner for the benefit of the whole system.
2. **Actor-computed initial priorities.** New transitions enter with a real priority — the absolute
   n-step TD error computed by the actor from the Q-values it already produced while acting — instead
   of the single-machine "insert at maximum priority, refine on first sample" rule, which at scale
   floods the top of the distribution with fresh data and collapses into training only on the newest
   transitions.
3. **Diverse exploration.** Off-policy learning lets each actor run a different, fixed exploration
   rate `ε_i = ε^{1 + (i/(N−1))·α}` (ε = 0.4, α = 7); the population explores at many scales at once,
   and prioritized shared replay surfaces whichever behaviors paid off.

Because Q-learning is off-policy, experience ages slowly (unlike gradients), so all communication is
batched and latency-tolerant — actors and learner can even run in different datacenters.

## Algorithm

Learner update (Ape-X DQN): multi-step double-Q target with a dueling network `q(·,·,θ)`, target net
`θ⁻`, and importance-sampling-weighted Huber loss.

  Gₜ = R_{t+1} + γ R_{t+2} + … + γⁿ⁻¹ R_{t+n} + γⁿ · q( S_{t+n}, argmaxₐ q(S_{t+n}, a, θ), θ⁻ )
  per-transition TD error  δ = Gₜ − q(Sₜ, Aₜ, θ)
  loss = mean_k [ wₖ · Huber(δₖ) ],  priority pₖ = |δₖ| + ε_p

Sampling: `P(k) = pₖ^α / Σ_j p_j^α` via a sum-tree; IS weight `wₖ = (N·P(k))^{−β}` normalized by the
batch max via a min-tree. Atari settings: n = 3, γ = 0.99, α = 0.6, β = 0.4, replay soft-cap 2M with
periodic FIFO eviction, batch 512, centered RMSProp at lr = 0.00025/4 = 6.25e-5 (decay 0.95,
eps 1.5e-7, no momentum), gradient-norm clip 40, target update every 2500 learner batches, params
refreshed on actors every 400 frames, learning starts after 50k transitions. The learning rate is
cut 4× versus single-machine DQN because prioritization enlarges the typical gradient.

Ape-X DPG (continuous control): replace the DQN head with a DDPG-style critic `q(s,a,ψ)` and policy
`π(s,φ)`; the critic uses the same multi-step TD target with the bootstrap action from the target
policy `π(S_{t+n}, φ⁻)`, and the policy ascends `∇_φ q(Sₜ, π(Sₜ, φ), ψ)`. Exploration is additive
Gaussian noise (σ = 0.3); priority is the critic's absolute TD error.

## Code

```python
import random
from collections import deque
import operator
import numpy as np
import torch
import torch.nn as nn


# ---- function approximator: dueling network, shared by actors and learner ----
class DuelingDQN(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        feat = self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
        self.advantage = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(),
                                       nn.Linear(512, self.num_actions))
        self.value = nn.Sequential(nn.Linear(feat, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        a, v = self.advantage(x), self.value(x)
        return v + a - a.mean(1, keepdim=True)

    def act(self, state, epsilon):
        with torch.no_grad():
            q = self.forward(state.unsqueeze(0))
            action = (q.max(1)[1].item() if random.random() > epsilon
                      else random.randrange(self.num_actions))
        return action, q.numpy()[0]              # Q-values reused for actor-side priorities


# ---- O(log N) segment trees (sum for sampling, min for the max IS weight) ----
class SegmentTree:
    def __init__(self, capacity, operation, neutral):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral] * (2 * capacity)
        self._operation = operation
    def _reduce(self, s, e, node, ns, ne):
        if s == ns and e == ne: return self._value[node]
        mid = (ns + ne) // 2
        if e <= mid: return self._reduce(s, e, 2*node, ns, mid)
        if mid + 1 <= s: return self._reduce(s, e, 2*node+1, mid+1, ne)
        return self._operation(self._reduce(s, mid, 2*node, ns, mid),
                               self._reduce(mid+1, e, 2*node+1, mid+1, ne))
    def reduce(self, s=0, e=None):
        e = self._capacity if e is None else (e + self._capacity if e < 0 else e)
        return self._reduce(s, e - 1, 1, 0, self._capacity - 1)
    def __setitem__(self, idx, val):
        idx += self._capacity; self._value[idx] = val; idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2*idx], self._value[2*idx+1])
            idx //= 2
    def __getitem__(self, idx):
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, c): super().__init__(c, operator.add, 0.0)
    def sum(self, s=0, e=None): return self.reduce(s, e)
    def find_prefixsum_idx(self, prefixsum):
        idx = 1
        while idx < self._capacity:
            if self._value[2*idx] > prefixsum: idx = 2*idx
            else: prefixsum -= self._value[2*idx]; idx = 2*idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, c): super().__init__(c, min, float('inf'))
    def min(self, s=0, e=None): return self.reduce(s, e)


# ---- single shared prioritized replay; add() takes an actor-computed priority ----
class CustomPrioritizedReplayBuffer:
    def __init__(self, size, alpha):
        self._storage = []; self._maxsize = size; self._next_idx = 0; self._alpha = alpha
        cap = 1
        while cap < size: cap *= 2
        self._it_sum = SumSegmentTree(cap); self._it_min = MinSegmentTree(cap)
        self._max_priority = 1.0
    def __len__(self): return len(self._storage)
    def add(self, state, action, reward, next_state, done, priority):
        idx = self._next_idx
        data = (state, action, reward, next_state, done)
        if idx >= len(self._storage): self._storage.append(data)
        else: self._storage[idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._it_sum[idx] = priority ** self._alpha
        self._it_min[idx] = priority ** self._alpha
        self._max_priority = max(self._max_priority, priority)
    def _encode_sample(self, idxes):
        cols = list(zip(*[self._storage[i] for i in idxes]))
        s, a, r, s2, d = cols
        return (list(s), np.array(a), np.array(r), list(s2), np.array(d))
    def _sample_proportional(self, batch_size):
        res = []; p_total = self._it_sum.sum(0, len(self._storage) - 1)
        seg = p_total / batch_size
        for i in range(batch_size):
            res.append(self._it_sum.find_prefixsum_idx(random.random() * seg + i * seg))
        return res
    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_w = (p_min * len(self._storage)) ** (-beta)
        weights = []
        for idx in idxes:
            p = self._it_sum[idx] / self._it_sum.sum()
            weights.append(((p * len(self._storage)) ** (-beta)) / max_w)
        return tuple(list(self._encode_sample(idxes)) + [np.array(weights), idxes])
    def update_priorities(self, idxes, priorities):
        for idx, p in zip(idxes, priorities):
            self._it_sum[idx] = p ** self._alpha
            self._it_min[idx] = p ** self._alpha
            self._max_priority = max(self._max_priority, p)


# ---- per-actor n-step accumulator + initial priority from stored Q-values ----
class BatchStorage:
    def __init__(self, n_steps, gamma=0.99):
        self.n_steps, self.gamma = n_steps, gamma
        self.state_d = deque(maxlen=n_steps); self.action_d = deque(maxlen=n_steps)
        self.reward_d = deque(maxlen=n_steps); self.qval_d = deque(maxlen=n_steps)
        self.reset()
    def reset(self):
        self.states=[]; self.actions=[]; self.rewards=[]; self.next_states=[]
        self.dones=[]; self.q_values=[]; self.next_q_values=[]
    def __len__(self): return len(self.states)
    def multi_step_reward(self, *rewards):
        return sum(r * self.gamma ** i for i, r in enumerate(rewards))
    def add(self, state, reward, action, done, q_values):
        if len(self.state_d) == self.n_steps or done:
            self.states.append(self.state_d[0]); self.actions.append(self.action_d[0])
            self.rewards.append(self.multi_step_reward(*self.reward_d, reward))
            self.next_states.append(state); self.dones.append(np.float32(done))
            self.q_values.append(self.qval_d[0]); self.next_q_values.append(q_values)
        if done:
            self.state_d.clear(); self.reward_d.clear()
            self.action_d.clear(); self.qval_d.clear()
        else:
            self.state_d.append(state); self.reward_d.append(reward)
            self.action_d.append(action); self.qval_d.append(q_values)
    def compute_priorities(self):
        actions = np.array(self.actions); rewards = np.array(self.rewards); dones = np.array(self.dones)
        q = np.stack(self.q_values); nq = np.stack(self.next_q_values)
        q_a = q[range(len(q)), actions]; next_q_a = nq.max(1)
        target = rewards + (self.gamma ** self.n_steps) * next_q_a * (1 - dones)
        return np.abs(target - q_a) + 1e-6
    def make_batch(self):
        prios = self.compute_priorities()
        return [self.states, self.actions, self.rewards, self.next_states, self.dones], prios


# ---- learner loss: multi-step double-Q, IS-weighted Huber, refreshed priorities ----
def compute_loss(model, tgt_model, batch, n_steps, gamma=0.99):
    states, actions, rewards, next_states, dones, weights = batch
    q_a = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_actions = model(next_states).max(1)[1].unsqueeze(1)             # select: online net
    next_q_a = tgt_model(next_states).gather(1, next_actions).squeeze(1) # evaluate: target net
    target = rewards + (gamma ** n_steps) * next_q_a * (1 - dones)
    td = torch.abs(target.detach() - q_a)
    prios = (td + 1e-6).data.cpu().numpy()
    loss = torch.where(td < 1, 0.5 * td ** 2, td - 0.5)
    return (loss * weights).mean(), prios


# ---- the two decoupled loops ----
def actor_loop(env, model, shared_buffer, param_source, actor_id, n_actors, args):
    epsilon = args.eps_base ** (1 + actor_id / (n_actors - 1) * args.eps_alpha)
    storage = BatchStorage(args.n_steps, args.gamma)
    model.load_state_dict(param_source.latest())
    state = env.reset(); step = 0
    while True:
        action, q_values = model.act(torch.FloatTensor(np.array(state)), epsilon)
        next_state, reward, done, _ = env.step(action)
        storage.add(state, reward, action, done, q_values)
        state = env.reset() if done else next_state
        step += 1
        if step % args.update_interval == 0:
            model.load_state_dict(param_source.latest())
        if len(storage) >= args.send_interval:
            batch, prios = storage.make_batch()
            for sample, p in zip(zip(*batch), prios):
                shared_buffer.add(*sample, p)
            storage.reset()


def learner_loop(model, tgt_model, shared_buffer, optimizer, param_sink, args):
    learn_idx = 0
    while True:
        *batch, idxes = shared_buffer.sample(args.batch_size, args.beta)
        loss, prios = compute_loss(model, tgt_model, batch, args.n_steps, args.gamma)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        shared_buffer.update_priorities(idxes, prios)
        learn_idx += 1
        if learn_idx % args.target_update_interval == 0:
            tgt_model.load_state_dict(model.state_dict())
        if learn_idx % args.publish_param_interval == 0:
            param_sink.publish(model.state_dict())


# learner optimizer: centered RMSProp at lr cut 4x because prioritization enlarges gradients
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00025/4,
#                                 alpha=0.95, eps=1.5e-7, centered=True)
# defaults: n_steps=3, gamma=0.99, alpha=0.6, beta=0.4, batch_size=512,
#           eps_base=0.4, eps_alpha=7, update_interval=400, send_interval=50,
#           target_update_interval=2500, max_norm=40, replay size 2_000_000
```
