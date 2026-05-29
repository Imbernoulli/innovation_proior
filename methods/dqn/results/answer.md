# Deep Q-Network (DQN)

## Problem

Learn good control policies for many different tasks directly from raw high-dimensional sensory input (Atari 2600 pixels), using only a scalar reward, with a single architecture and shared hyperparameters across games apart from the emulator frame-skip visibility exception — no hand-engineered features.

## Key idea

Use a deep convolutional network as the action-value function Q(s,a;θ) and train it with Q-learning, but break the two things that make deep-net value learning unstable — correlated samples and a self-shifting training distribution — with **experience replay**: store transitions in a large memory and take gradient steps on uniformly sampled minibatches instead of on the live stream. Replay decorrelates updates, averages the training distribution over many past policies, and reuses each transition many times. Because replayed samples come from older policies, learning must be off-policy, which is exactly what Q-learning's max-target provides.

Two design choices make it efficient: the network takes only the state as input and emits **one Q-value per action** (all actions scored in a single forward pass), and partial observability is handled by stacking the **last 4 preprocessed frames** as a fixed-length input rather than using a recurrent net over the full history.

## Algorithm

Initialize replay memory D (capacity N) and Q-network weights θ. For each step:

1. With probability ε pick a random action, else a = argmax_a Q(φ(s),a;θ) (ε annealed 1 → 0.1 over the first 1M frames, then fixed at 0.1).
2. Execute a (repeated over k skipped frames; k=4, k=3 for Space Invaders), observe r (clipped to {−1,0,+1} in training) and the next frame; form the next stacked state.
3. Store (φ, a, r, φ', done) in D.
4. Sample a random minibatch from D. Set the target from the same online network

   y = r                          if φ' is terminal
   y = r + γ · max_{a'} Q(φ',a';θ)   otherwise

   and take a gradient-descent (RMSProp) step on the half-squared TD error 1/2(y − Q(φ,a;θ))², with y treated as a stop-gradient. The semi-gradient is (Q(φ,a;θ) − y)∇_θ Q(φ,a;θ), so the descent update is proportional to +(y − Q(φ,a;θ))∇_θ Q(φ,a;θ).

Preprocessing φ: RGB → grayscale, downsample to 110×84, crop 84×84, stack last 4 frames → 84×84×4. Architecture: Conv 16×(8×8, stride 4)–ReLU maps 84→20, Conv 32×(4×4, stride 2)–ReLU maps 20→9, then flatten 32×9×9=2592 → FC 256–ReLU → linear output (one unit per action). Replay memory 1M recent experiences, minibatch 32, γ=0.99, 10M training frames. This variant uses the online weights θ directly in the target (no separate target network).

## Code

```python
import random
import numpy as np
import torch
import torch.nn as nn

class ActionValueModel(nn.Module):
    """State -> one Q-value per action (all actions scored in a single forward pass)."""
    def __init__(self, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4), nn.ReLU(),   # 84x84x4 -> 20x20x16
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),  # -> 9x9x32
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)

class ReplayMemory:
    """Store recent transitions; sample uniform minibatches to decorrelate updates."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def add(self, s, a, r, s_next, done):
        item = (
            np.asarray(s, dtype=np.uint8).copy(),
            int(a),
            float(r),
            np.asarray(s_next, dtype=np.uint8).copy(),
            float(done),
        )
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item          # overwrite oldest
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)

def exploration_rate(step, start=1.0, end=0.1, duration=1_000_000):
    return start + min(step / duration, 1.0) * (end - start)

def phi(frames):
    arr = np.asarray(frames)
    if arr.shape == (84, 84, 4):
        arr = arr.transpose(2, 0, 1)
    elif arr.shape != (4, 84, 84):
        arr = np.stack(frames, axis=0)
    if arr.shape != (4, 84, 84):
        raise ValueError(f"expected 4 stacked 84x84 frames, got {arr.shape}")
    return arr.astype(np.uint8, copy=False)

def reset_frames(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out

def step_env(env, action):
    out = env.step(action)
    if len(out) == 3:
        return out
    if len(out) == 4:
        frames, reward, done, _ = out
        return frames, reward, done
    frames, reward, terminated, truncated, _ = out
    return frames, reward, bool(terminated or truncated)

def select_action(net, state, epsilon, num_actions, device):
    if random.random() < epsilon:
        return random.randrange(num_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(net(state_t).argmax(dim=1).item())

def unpack_batch(batch, device):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = torch.as_tensor(np.stack(states), dtype=torch.float32, device=device)
    next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
    return states, actions, rewards, next_states, dones

def compute_targets(net, rewards, next_states, dones, gamma):
    with torch.no_grad():
        max_next = net(next_states).max(dim=1).values
        return rewards + gamma * max_next * (1.0 - dones)   # y=r at terminal

def values_for_taken_actions(net, states, actions):
    return net(states).gather(1, actions[:, None]).squeeze(1)

def optimize_step(net, optimizer, batch, gamma, device):
    states, actions, rewards, next_states, dones = unpack_batch(batch, device)
    target = compute_targets(net, rewards, next_states, dones, gamma)
    pred = values_for_taken_actions(net, states, actions)
    loss = 0.5 * ((target - pred) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(env, num_actions=None, total_frames=10_000_000, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if num_actions is None:
        num_actions = env.num_actions if hasattr(env, "num_actions") else env.action_space.n
    q = ActionValueModel(num_actions).to(device)
    optimizer = torch.optim.RMSprop(q.parameters(), lr=2.5e-4, alpha=0.95, eps=0.01)
    memory = ReplayMemory(capacity=1_000_000)
    gamma, batch_size = 0.99, 32

    state = phi(reset_frames(env))
    for step in range(total_frames):
        eps = exploration_rate(step)
        a = select_action(q, state, eps, num_actions, device)

        # env.step applies frame-skip (repeat action on skipped frames) and reward clipping
        next_frames, r, done = step_env(env, a)
        next_state = phi(next_frames)
        memory.add(state, a, r, next_state, float(done))
        state = phi(reset_frames(env)) if done else next_state

        if len(memory) < batch_size:
            continue

        optimize_step(q, optimizer, memory.sample(batch_size), gamma, device)
    return q
```

The code keeps one online action-value model only: the same module selects actions, builds the max-next bootstrap under `no_grad`, and receives the TD update. There is no separate target network.
