I propose the canonical name Rainbow for this method. Rainbow is a single deep reinforcement learning agent that unifies six previously separate improvements to the DQN family into one coherent value-based architecture. The central insight is that the categorical return distribution should act as the spine of the design, with every other ingredient attached to that distribution rather than to an independent scalar Q-value. By making this choice explicit, the method avoids the contradictions that arise when different components try to redefine the target, the loss, the replay priority, the network head, or the exploration rule in incompatible ways.

The starting point is the ordinary DQN loop. An online network produces action values, a target network provides a stable bootstrap, and transitions are stored in a replay buffer that is sampled uniformly. The scalar target is the one-step Bellman update with a max over next-action values. This setup is stable enough for deep function approximation, but it leaves several weaknesses unaddressed: overestimated bootstrap values, uniform sampling that ignores learning potential, poor generalization across actions, slow credit assignment, mean-only return estimates, and shallow exploration driven by a fixed epsilon schedule.

To fix these weaknesses together, Rainbow folds in Double Q-learning, prioritized replay, dueling networks, multi-step learning, categorical distributional reinforcement learning, and noisy linear layers. The order in which these pieces are combined matters because several of them touch the same object. Multi-step returns, Double Q-learning, and categorical learning all change how the bootstrap target is built. Prioritized replay needs a scalar priority, but the categorical learner no longer optimizes a squared TD error. Dueling networks alter the head, and noisy layers replace standard linear layers while also changing how exploration is generated.

The construction begins with the categorical return distribution. Instead of predicting one scalar value per action, the network predicts a probability mass function over a fixed grid of atoms. The support is fixed, typically fifty-one atoms evenly spaced between negative ten and positive ten in the canonical Atari setting. Because reward shifting and discounting can move target atoms off this support, the Bellman target must be projected back onto the support with the categorical projection before it is compared to the prediction. Once this distributional object is in place, the other ingredients attach naturally.

Multi-step learning extends the target from a single reward to the discounted truncated return collected over the next n steps. In distribution space, the same operation shifts the target atoms by the n-step return and contracts them by the cumulative discount. If the sampled trajectory terminates before n steps, the bootstrap term is masked out and the projected target collapses to the observed partial return. This keeps the multi-step target consistent with the distributional object rather than treating it as an afterthought applied to scalar means.

Double Q-learning then determines how the bootstrap action is selected. The online network chooses the action that maximizes the mean of the predicted return distribution at the next state, and the target network supplies the full probability distribution for that chosen action. This decouples selection from evaluation even though both networks now predict entire distributions. The online network is used only to pick the action; its probabilities are not mixed into the target distribution.

Prioritized replay must derive its scalar priority from the native objective. Using the absolute TD error computed from means would be convenient, but it would not reflect what the agent is actually optimizing. Instead, the priority is the per-sample categorical loss, which is the cross-entropy form of the KL divergence between the projected target distribution and the predicted distribution. The replay buffer stores this loss, raises it to a power to control how aggressively transitions are prioritized, and applies importance-sampling weights to correct the biased sampling when computing gradients. The importance-sampling exponent is annealed toward one during training, while the stored priority itself remains based on the raw categorical loss.

The dueling head is reinterpreted to respect the categorical object. In the scalar version, a shared torso feeds a value stream and an advantage stream, and the final Q-value is formed by subtracting the mean advantage. With categorical outputs, the value stream produces one logit per atom, the advantage stream produces one logit per action per atom, and the aggregation happens per atom before applying a softmax over atoms for each action. This means the mean subtraction is performed on logits for each atom independently, not on already-normalized probabilities and not on scalar action means.

Noisy linear layers replace the fully connected layers in the value and advantage streams. A standard linear transformation is augmented with learned parametric noise drawn from a factorized distribution. During training, the sampled noise provides state-dependent exploration, so the behavior policy can be greedy with epsilon set to zero. The convolutional torso remains deterministic, and the noise is resampled in the training loop before acting and before constructing the target distribution. The noise scale is initialized so that the network begins with a modest amount of exploratory variance that can be learned down or up as training progresses.

The integrated target distribution takes the following form. The n-step return shifts the atom support, the cumulative discount contracts it, and the target network evaluates the distribution for the action selected by the online network mean. The projected target is compared to the online prediction with a cross-entropy loss, which is also used to set replay priorities. This single update subsumes all six improvements in a way that keeps each component faithful to the central distributional representation.

The canonical hyperparameters inherited from the original Rainbow paper are as follows. Learning starts after eighty thousand frames, which is earlier than in vanilla DQN because prioritized replay makes early samples more useful. The optimizer is Adam with learning rate 6.25e-5 and epsilon 1.5e-4. The target network is copied every thirty-two thousand frames. The priority exponent is 0.5, and the importance-sampling exponent anneals from 0.4 to 1.0. The multi-step horizon is three, the categorical support has fifty-one atoms spanning [-10, 10], the noisy-layer initial scale is 0.5, and epsilon-greedy exploration is disabled because the noisy weights provide exploration.

The code below gives a compact, runnable illustration of the core pieces: a noisy linear layer, a dueling categorical network head, greedy action selection by expected return, and the multi-step categorical projection loss that produces both gradients and replay priorities. The snippet is not a full Atari training loop, but it captures the exact target construction and update that distinguish Rainbow from its ingredients.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0
N_STEP = 3


class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma0=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer("weight_epsilon", torch.empty(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer("bias_epsilon", torch.empty(out_f))
        self.sigma0 = sigma0
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 / math.sqrt(self.in_f))
        self.bias_sigma.data.fill_(self.sigma0 / math.sqrt(self.out_f))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_f)
        eps_out = self._scale_noise(self.out_f)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        return F.linear(x, self.weight_mu, self.bias_mu)


class RainbowNet(nn.Module):
    def __init__(self, n_actions, n_atoms=N_ATOMS):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, n_atoms
        self.register_buffer("z", torch.linspace(V_MIN, V_MAX, n_atoms))
        self.torso = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        self.fc_h_v = NoisyLinear(3136, 512)
        self.fc_h_a = NoisyLinear(3136, 512)
        self.fc_z_v = NoisyLinear(512, n_atoms)
        self.fc_z_a = NoisyLinear(512, n_actions * n_atoms)

    def forward(self, x, log=False):
        phi = self.torso(x).view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(phi))).view(-1, 1, self.n_atoms)
        a = self.fc_z_a(F.relu(self.fc_h_a(phi))).view(-1, self.n_actions, self.n_atoms)
        logits = v + a - a.mean(dim=1, keepdim=True)
        return F.log_softmax(logits, dim=2) if log else F.softmax(logits, dim=2)

    def reset_noise(self):
        for module in (self.fc_h_v, self.fc_h_a, self.fc_z_v, self.fc_z_a):
            module.reset_noise()


def act(net, x):
    with torch.no_grad():
        probs = net(x)
        return (probs * net.z).sum(dim=2).argmax(dim=1)


def learn(online, target, obs, actions, n_returns, next_obs, nonterminal, gamma, weights):
    batch = actions.numel()
    arange = torch.arange(batch, device=actions.device)
    z = online.z
    delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)

    log_ps_a = online(obs, log=True)[arange, actions]

    with torch.no_grad():
        next_online = online(next_obs)
        a_star = (next_online * z).sum(dim=2).argmax(dim=1)

        target.reset_noise()
        next_target = target(next_obs)[arange, a_star]

        Tz = n_returns[:, None] + nonterminal.view(-1, 1) * (gamma ** N_STEP) * z[None, :]
        Tz = Tz.clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / delta_z
        lower = b.floor().long()
        upper = b.ceil().long()

        lower[(upper > 0) & (lower == upper)] -= 1
        upper[(lower < N_ATOMS - 1) & (lower == upper)] += 1

        m = torch.zeros(batch, N_ATOMS, device=obs.device)
        offset = (torch.arange(batch, device=obs.device) * N_ATOMS).unsqueeze(1)
        m.view(-1).index_add_(
            0, (lower + offset).reshape(-1),
            (next_target * (upper.float() - b)).reshape(-1))
        m.view(-1).index_add_(
            0, (upper + offset).reshape(-1),
            (next_target * (b - lower.float())).reshape(-1))

    per_sample_loss = -(m * log_ps_a).sum(dim=1)
    loss = (weights * per_sample_loss).mean()
    priorities = per_sample_loss.detach().cpu().numpy()
    return loss, priorities
```
