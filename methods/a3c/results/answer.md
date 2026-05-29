# Asynchronous Advantage Actor-Critic (A3C)

## Problem it solves

Online deep RL is unstable because a single agent's consecutive updates are strongly correlated and bootstrapped targets chase a moving network. The standard fix, a large experience replay buffer, forces the algorithm to be off-policy, costs memory and per-step compute, and pushes toward GPUs or large clusters. A3C removes replay and instead decorrelates learning by running many actor-learners in parallel on a single multi-core CPU.

## Key idea

Replay's real job is to make the gradient stream look decorrelated and stationary. The same effect can be obtained by running many workers in parallel, each in its own environment instance with its own exploration trajectory, all updating one shared parameter vector asynchronously. At any instant the workers occupy different states, so the aggregate update is an average over a diverse mixture of experience, but the data is still fresh enough for on-policy updates. That makes Sarsa, n-step methods, and actor-critic usable with deep networks without a replay buffer.

The shared skeleton supports async one-step Q-learning, async one-step Sarsa, async n-step Q-learning, and the advantage actor-critic instance called A3C.

## The algorithm

Maintain a shared policy π(a|s;θ) and value function V(s;θ_v), with most parameters shared: one visual/recurrent body, a softmax policy head, and a linear value head. Each worker, in its own environment:

1. Sync thread-local θ′, θ′_v from the shared θ, θ_v.
2. Roll out up to t_max steps, sampling a_t ∼ π(·|s_t;θ′).
3. Bootstrap R = V(s_t;θ′_v) for the last non-terminal state, else R = 0.
4. Walk backward over the rollout, R ← r_i + γR. Then
   Â_i = R − V(s_i;θ′_v)
   = Σ_{j=0}^{k-1} γ^j r_{i+j} + γ^k V(s_{i+k};θ′_v) − V(s_i;θ′_v), k ≤ t_max.
5. Minimize the per-step losses
   L_π = −log π(a_i|s_i;θ′) stopgrad(Â_i) − βH(π(·|s_i;θ′)),
   L_v = 0.5 [stopgrad(R_i) − V(s_i;θ′_v)]²,
   usually as L_π + c_v L_v.
6. Copy the local gradients onto the shared parameters and apply an asynchronous lock-free optimizer step.

The baseline is mathematically safe: for a fixed state, E_a[∇_θ log π(a|s)b(s)] = b(s)Σ_a∇_θπ(a|s) = b(s)∇_θ1 = 0. Choosing b(s) ≈ V(s) cuts variance and makes R − V(s) an advantage estimate. In the actor loss this advantage is stopped; in the value loss only the target return R_i is stopped, so the value prediction still learns. The entropy term H(π) = −Σ_aπ(a|s)logπ(a|s) discourages premature collapse to a deterministic policy.

The value-based variants reuse the same rollout skeleton with a target network θ⁻: one-step Q uses target r + γ max_{a'} Q(s',a';θ⁻); one-step Sarsa uses r + γ Q(s',a';θ⁻) with the actually-taken a′; n-step Q bootstraps R = max_a Q(s_t,a;θ⁻) and minimizes (R − Q(s_i,a_i;θ′))² while walking backward through the rollout. Each value-based worker uses ε-greedy exploration, with ε varied across workers to increase diversity.

The algorithmic optimizer is shared non-centered RMSProp. For an accumulated loss gradient Δθ, g ← αg + (1−α)Δθ² and θ ← θ − ηΔθ/√(g+ε), elementwise, with g shared across workers. The PyTorch code below uses the same shared-adaptive-state idea with Adam: shared first moments, shared second moments, and lock-free steps on shared parameters.

## Working code

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, action_space):
        super().__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, action_space.n)

        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 3 * 3)
        hx, cx = self.lstm(x, (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0):
        super().__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()
    state = torch.from_numpy(env.reset())
    done = True
    episode_length = 0

    while True:
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values, log_probs, rewards, entropies = [], [], [], []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.item())
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()
        values.append(R)

        policy_loss, value_loss = 0, 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            # With lambda = 1, this is the same finite n-step advantage.
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss \
                - log_probs[i] * gae.detach() \
                - args.entropy_coef * entropies[i]

        optimizer.zero_grad(set_to_none=True)
        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        ensure_shared_grads(model, shared_model)
        optimizer.step()
```

The harness shares the model and optimizer state, then launches one evaluator and several worker processes:

```python
env = create_atari_env(args.env_name)
shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
shared_model.share_memory()

if args.no_shared:
    optimizer = None
else:
    optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

counter = mp.Value('i', 0)
lock = mp.Lock()
processes = []

p = mp.Process(target=test, args=(args.num_processes, args,
                                  shared_model, counter))
p.start()
processes.append(p)

for rank in range(args.num_processes):
    p = mp.Process(target=train,
                   args=(rank, args, shared_model, counter, lock, optimizer))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
```

The PyTorch defaults here are 4 worker processes, 20 rollout steps, γ = 0.99, GAE λ = 1.0, entropy weight 0.01, value-loss weight 0.5, max gradient norm 50, 42×42 normalized Atari observations, and SharedAdam unless the non-shared optimizer path is selected. With λ = 1, the GAE recursion telescopes to the same finite n-step advantage R − V(s_i).
