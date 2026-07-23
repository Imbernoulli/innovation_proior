# SimBa: Simplicity Bias For Scalable Deep RL Networks

SimBa is a network replacement for deep RL actor-critics. It keeps the RL algorithm fixed and
changes only the function class:

1. Normalize observations with running per-coordinate mean and variance.
2. Embed the normalized input with an orthogonal linear layer.
3. Run pre-LayerNorm residual feedforward blocks:
   `x <- x + Linear_down(ReLU(Linear_up(LayerNorm(x))))`, with a `4x` expansion.
4. Apply one final LayerNorm before the actor or critic head.

The critic receives `[normalized_obs, action]`; the actor receives `normalized_obs`. For SAC, the
actor head is a squashed Gaussian policy. For DDPG, the same encoder can feed a deterministic tanh
head. The algorithmic losses, replay buffer, targets, and target-network update are not part of the
method change.

## Equations

Running-stat observation normalization, for a single new observation coordinate:

```text
delta_t = o_t - mu_{t-1}
mu_t = mu_{t-1} + delta_t / t
sigma_t^2 = ((t - 1) / t) * (sigma_{t-1}^2 + delta_t^2 / t)
RSNorm(o_t) = (o_t - mu_t) / sqrt(sigma_t^2 + epsilon)
```

In implementation this is usually the equivalent batch/parallel running-moments update.

Encoder:

```text
x_0 = Linear(RSNorm(o))
x_{l+1} = x_l + W_2 ReLU(W_1 LayerNorm(x_l)),  l = 0,...,L-1
z = LayerNorm(x_L)
```

Simplicity-bias diagnostic:

```text
c(f) = sum_k f_tilde(k) * k / sum_k f_tilde(k)
s(f) ~= E_{theta ~ Theta_0}[1 / c(f_theta)]
```

This score is a diagnostic of the random initialized function class, not an additional RL loss.

## Canonical SAC Settings

```text
normalize_observation = true
actor:  residual blocks = 1, hidden dim = 128
critic: residual blocks = 2, hidden dim = 512
actor/critic optimizer = AdamW(lr=1e-4, weight_decay=1e-2)
target critic tau = 5e-3
SAC temperature init = 1e-2
SAC temperature lr = 1e-4
SAC target entropy coefficient in code = -0.5 * action_dim
replay ratio = 2
clipped double Q = true on HumanoidBench/episodic settings, false otherwise
```

## Faithful PyTorch Equivalent

My reference version uses JAX/Flax. This is the same architecture boundary in PyTorch: running
statistics are an observation wrapper, while the actor and critic modules contain the residual
encoder and heads.

```python
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class RunningMeanStd(nn.Module):
    def __init__(self, shape, eps_count=1e-4, dtype=torch.float32):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=dtype))
        self.register_buffer("var", torch.ones(shape, dtype=dtype))
        self.register_buffer("count", torch.tensor(float(eps_count), dtype=dtype))

    @torch.no_grad()
    def update(self, x):
        x = x.detach()
        if x.ndim == self.mean.ndim:
            x = x.unsqueeze(0)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device, dtype=x.dtype)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)


class ObservationNormalizer(nn.Module):
    def __init__(self, obs_shape, eps=1e-8):
        super().__init__()
        self.rms = RunningMeanStd(obs_shape)
        self.eps = eps

    def forward(self, obs, update=False):
        if update:
            self.rms.update(obs)
        return (obs - self.rms.mean) / torch.sqrt(self.rms.var + self.eps)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.up = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.down = nn.Linear(4 * hidden_dim, hidden_dim)
        nn.init.kaiming_normal_(self.up.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="relu")
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.bias)

    def forward(self, x):
        y = self.norm(x)
        y = torch.relu(self.up(y))
        y = self.down(y)
        return x + y


class SimbaEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        nn.init.orthogonal_(self.embed.weight, gain=1.0)
        nn.init.zeros_(self.embed.bias)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class SimbaSACActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128, num_blocks=1):
        super().__init__()
        self.encoder = SimbaEncoder(obs_dim, hidden_dim, num_blocks)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        nn.init.orthogonal_(self.mean.weight, gain=1.0)
        nn.init.orthogonal_(self.log_std.weight, gain=1.0)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.bias)
        self.log_std_min = -10.0
        self.log_std_max = 2.0

    def forward(self, obs, temperature=1.0, deterministic=False):
        z = self.encoder(obs)
        mean = self.mean(z)
        raw_log_std = self.log_std(z)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (1.0 + torch.tanh(raw_log_std))
        if deterministic or temperature == 0.0:
            return torch.tanh(mean)
        scale = torch.exp(log_std) * temperature
        base = Independent(Normal(mean, scale), 1)
        return TransformedDistribution(base, [TanhTransform(cache_size=1)])


class SimbaCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512, num_blocks=2):
        super().__init__()
        self.encoder = SimbaEncoder(obs_dim + action_dim, hidden_dim, num_blocks)
        self.q = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.q.weight, gain=1.0)
        nn.init.zeros_(self.q.bias)

    def forward(self, normalized_obs, action):
        z = self.encoder(torch.cat([normalized_obs, action], dim=-1))
        return self.q(z)


def build_simba_sac(obs_dim, action_dim):
    normalizer = ObservationNormalizer((obs_dim,))
    actor = SimbaSACActor(obs_dim, action_dim)
    critic = SimbaCritic(obs_dim, action_dim)
    target_critic = SimbaCritic(obs_dim, action_dim)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-4, weight_decay=1e-2)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=1e-2)
    log_temperature = torch.log(torch.tensor(1e-2)).requires_grad_()
    temperature_opt = torch.optim.AdamW([log_temperature], lr=1e-4, weight_decay=0.0)
    target_entropy = -0.5 * action_dim
    return {
        "normalizer": normalizer,
        "actor": actor,
        "critic": critic,
        "target_critic": target_critic,
        "actor_opt": actor_opt,
        "critic_opt": critic_opt,
        "log_temperature": log_temperature,
        "temperature_opt": temperature_opt,
        "target_entropy": target_entropy,
        "target_tau": 5e-3,
    }
```
