The question I set out to answer is narrow and concrete: can a continuous-control actor-critic use a much larger policy and value network without touching the reinforcement-learning algorithm itself, and have those extra parameters actually pay off in sample efficiency and final return rather than merely train without diverging? In practice the standard actor-critic is a shallow MLP, and that is not an accident. Value learning is bootstrapped, the data comes off a replay buffer, and the state distribution drifts as the policy changes, so the regression target is far less stable than a supervised label. When the plain MLP is widened or deepened, the critic gains the capacity to fit transient structure in the buffer and then feeds its own error back through the Bellman target; the policy in turn optimizes against that distorted value surface. Raw capacity in this regime amplifies the deadly triad instead of producing the clean scaling we see in vision and language. The existing algorithmic tools each address a piece of the difficulty without isolating the network change I want: SAC supplies a stable stochastic off-policy backbone with entropy regularization, TD3-style clipped double Q curbs overestimation, and high update-to-data ratios, ensembles, distributional heads, resets, and optimistic exploration can improve efficiency — but every one of them alters the training protocol. Normalization and spectral-constraint work shows that architecture matters, yet none of it isolates a minimal, plug-in function-class change that makes parameter scaling itself useful while leaving the algorithm recognizably the same.

The lesson I take from supervised learning is that overparameterized models generalize not because they are big but because their architecture and optimizer bias training toward simple functions — residual paths make identity-like behavior cheap, normalization keeps activations on a controlled scale, and ReLU feedforward blocks with careful initialization shape the function before any gradient step. So I propose SimBa, a network replacement built entirely around engineering a simplicity bias into the actor and critic, while keeping the RL objective, targets, replay buffer, exploration, and target-network update untouched. SimBa has three load-bearing ingredients, and the method is their combination, not any one in isolation. The first is input-side control. A state vector is heterogeneous — a joint angle here, a velocity there, a contact signal elsewhere — and a large network left to see raw values will latch onto scale rather than meaning, with the problem compounded by online distribution drift. I cannot use fixed dataset statistics, so I normalize per coordinate with running mean and variance maintained over the stream. For a single new observation with $\delta_t = o_t - \mu_{t-1}$,
$$\mu_t = \mu_{t-1} + \frac{\delta_t}{t}, \qquad \sigma_t^2 = \frac{t-1}{t}\left(\sigma_{t-1}^2 + \frac{\delta_t^2}{t}\right),$$
and the network is fed $\mathrm{RSNorm}(o_t) = (o_t - \mu_t)/\sqrt{\sigma_t^2 + \epsilon}$. The axis is what makes this distinct: statistics are taken per observation coordinate across samples, not per sample across features, which is precisely why this is not RMSNorm. In code the same effect is obtained with the standard parallel running-moments update over batches.

The second ingredient makes an identity-like map the default for the body of the network. A deep plain MLP forces information through every nonlinearity, so even a nearly linear value function must be reconstructed by a chain of layers that all have to cooperate. A residual block $x + F(x)$ flips that default: $F$ can be small and the block behaves like the identity, so the nonlinear part is an additive correction rather than the only path through the layer. The large network can express a complicated correction but is never forced to use one. Where the normalization sits inside the block is a deliberate choice. Normalizing after the residual addition would interfere with the identity path; normalizing the input to the nonlinear branch instead lets the branch see controlled features while the skip carries the residual stream directly. Each block is therefore pre-LayerNorm,
$$x_{l+1} = x_l + W_2\,\mathrm{ReLU}\!\left(W_1\,\mathrm{LayerNorm}(x_l)\right), \qquad l = 0,\dots,L-1,$$
where the branch is the standard inverted bottleneck Linear$(d_h, 4d_h)$, ReLU, Linear$(4d_h, d_h)$. The factor of four is a real constant, not decoration: it is where most of the scalable parameters live. These branch linears use He-normal (ReLU-compatible fan-in) initialization with zero biases, and the input embedding is an orthogonal linear map with gain $1$. The third ingredient addresses what remains after $L$ residual additions: the stream is a sum of the embedded input plus every learned correction, and I do not want the prediction head to chase the growing scale of that sum as depth or width increases. So after the last block I apply one final LayerNorm before the head,
$$x_0 = \mathrm{Linear}(\mathrm{RSNorm}(o)), \qquad z = \mathrm{LayerNorm}(x_L).$$
This post-LN is not a replacement for the pre-LN inside the branch; the two have different jobs. Input statistics handle drifting coordinates, the pre-LN residual blocks make nonlinear corrections optional, and the post-LN standardizes the head input. Collapse any of these roles together and the design is lost — and the ablations confirm that removing any one of the three hurts.

The same encoder serves both networks. The actor receives normalized observations; the critic receives the normalized observation concatenated with the raw action, encodes, and reads off a scalar $Q$ from a linear head. A subtle but important software boundary follows from the online normalizer: the running statistics live in a wrapper that updates when actions are sampled during training and normalizes observations for both acting and replay batches, so the actor and critic modules must not contain a second RSNorm or they would double-normalize. For the heads I follow the off-policy reference, SAC, where the actor is a squashed Gaussian — one linear head for the mean, one for the log standard deviation, the raw log-std squashed by tanh into $[-10, 2]$, and the sampled normal action passed through tanh — though the very same encoder also feeds a deterministic tanh head when paired with DDPG. When clipped double Q is enabled there are two independently parameterized critics and the SAC target and actor loss use their minimum; in the canonical configuration this is on for episodic HumanoidBench settings and off otherwise. The constants reflect that the critic carries the harder bootstrapped regression: the actor uses one residual block at width $128$, the critic two blocks at width $512$, both trained with AdamW at learning rate $10^{-4}$ and weight decay $10^{-2}$, target critic momentum $\tau = 5\times 10^{-3}$, SAC temperature initialized at $10^{-2}$ with its own $10^{-4}$ learning rate, a target-entropy coefficient set in code to $-0.5\,\dim(a)$ (the sign kept explicit), and replay ratio $2$; DDPG reuses the architecture and optimizer constants with Gaussian exploration noise of standard deviation $0.1$. To argue the design without leaning on the performance curves, I use a diagnostic of functional simplicity at initialization: sample random parameters, evaluate the random function on a grid, decompose it in frequency space, and form a frequency-weighted complexity $c(f) = \sum_k \tilde f(k)\,k \,/\, \sum_k \tilde f(k)$, with a simplicity score $s(f) \approx \mathbb{E}_{\theta \sim \Theta_0}[1/c(f_\theta)]$. This is not a training objective and not a proof that every larger model improves; it measures the architectural bias. The stronger, more modest claim is that running the same RL algorithm with this normalized residual encoder makes scaling the critic beneficial precisely over the range where a plain MLP degrades.

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
