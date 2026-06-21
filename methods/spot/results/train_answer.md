We are handed a fixed batch $D = \{(s,a,r,s')\}$ collected by some unknown behavior policy $\pi_\beta$, no environment to interact with, and asked to extract the best continuous-control policy we can. The reflex is to run an off-the-shelf off-policy actor-critic on the batch — fit twin critics $Q_\theta$ to a bootstrapped Bellman target and train a deterministic actor to maximize $Q_\theta(s, \pi_\phi(s))$ — and the reflex fails in a precise, diagnosable way. The Bellman target $r + \gamma\, Q_{\text{target}}(s', \pi_\phi(s'))$ evaluates the critic at $\pi_\phi(s')$, an action the actor chose rather than one the data contains. Online this self-corrects: a silly proposed action gets punished by the environment and the critic is fixed. Offline there is no correction loop, so when $\pi_\phi(s')$ lands on an action the batch never saw, $Q_\theta$ is extrapolating off the data manifold, and a neural net's extrapolation there is anchored to nothing — it can be arbitrarily large. The actor is a maximizer, so it is *attracted* to exactly those out-of-distribution actions with spuriously inflated value; those values bootstrap back through the backup into neighboring states, the next actor update chases them harder, and the whole thing diverges. The disease, then, is not "small dataset" or "sparse reward" in the first instance — it is that the $\max$ over actions reaches outside the data where the critic lies. The cure must forbid the actor from leaving the region where $\pi_\beta$ actually put mass.

That region is the *support* of $\pi_\beta$: in state $s$, only allow actions with $\pi_\beta(a|s) > \epsilon$. This restriction is provably safe. With the support-restricted backup $T_\epsilon Q(s,a) = \mathbb{E}_{s'}[r + \gamma \max_{a':\,\pi_\beta(a'|s')>\epsilon} Q(s',a')]$ and $\alpha(\epsilon) = \|T Q^* - T_\epsilon Q^*\|_\infty$, the supported optimal value function $Q^*_\epsilon$ obeys $\|Q^* - Q^*_\epsilon\|_\infty = \|T Q^* - T_\epsilon Q^*_\epsilon\|_\infty \le \|T_\epsilon Q^* - T_\epsilon Q^*_\epsilon\|_\infty + \|T Q^* - T_\epsilon Q^*\|_\infty \le \gamma\|Q^* - Q^*_\epsilon\|_\infty + \alpha(\epsilon)$, using that $T_\epsilon$ is a $\gamma$-contraction in sup-norm; solving gives $$\|Q^* - Q^*_\epsilon\|_\infty \le \frac{\alpha(\epsilon)}{1-\gamma}.$$ Confinement to the support costs at most $\alpha(\epsilon)/(1-\gamma)$ in value, with $\epsilon$ the single lever trading "stay where the critic is trustworthy" (a tight support shrinks extrapolation risk but raises the backup-perturbation $\alpha(\epsilon)$) against "keep enough room to be optimal" (a loose support does the reverse). The existing ways to *enforce* this split cleanly, and each falls short. The parameterization camp — BCQ, PLAS, EMaQ — bakes support into the policy's architecture: BCQ fits a conditional VAE of $\pi_\beta$, samples candidate actions, perturbs them, and defines the policy as the argmax of $Q$ over the perturbed candidates. These respect a genuine density/support notion, but every action selection at training and deployment must sample the generative model, perturb, and score with the critic — an intrusive, slow inference path that welds the algorithm to its generative component and blocks dropping in a better online RL algorithm later. The regularization camp — BEAR, TD3+BC — keeps a plain policy network and adds a penalty pulling it toward $\pi_\beta$: BEAR penalizes the sampled MMD to dataset actions, TD3+BC adds the behavior-cloning term $-(\pi(s)-a)^2$. This is pluggable — one forward pass at inference — but it enforces the *wrong notion*. A divergence or a BC squared-error measures how close the learned *distribution* is to the behavior *distribution*, which is not the *support* condition the theory asked for. Matching the distribution is strictly more restrictive: if $\pi_\beta$ is broad or near-uniform, forcing $\pi_\phi$ close to it in MMD or L2 drags the policy toward random behavior even when the data clearly supports a sharp, strong policy. BEAR's claim that its MMD really matches *support* leans on a fragile, sample-count-dependent curiosity and in practice leaks OOD actions; TD3+BC's $(\pi(s)-a)^2$ is even more directly an imitation of the single logged action, over-constraining on suboptimal or multimodal data. So parameterization gets the right notion but is intrusive, and regularization is pluggable but enforces a divergence proxy; neither is both at once.

I propose SPOT — Supported Policy OpTimization — which is simultaneously pluggable and a direct match to the density-based support condition. The mismatch in the prior art is the clue: the reason every regularizer is "indirect" is that it measures a distance between distributions and hopes that controls the support, but the constraint $\pi_\beta(a|s) > \epsilon$ is not a distance at all — it is a statement about the *value of the behavior density at the specific action the policy takes*. So instead of measuring a divergence, evaluate the behavior density at $\pi_\phi(s)$ directly and require it be large. Written over the policy parameters, with $\log$ so the density enters additively and $\hat\epsilon := \log\epsilon$, the support constraint is exactly $$\max_\phi\ \mathbb{E}_{s\sim D}[Q_\theta(s, \pi_\phi(s))]\quad\text{s.t.}\quad \min_s \log\pi_\beta(\pi_\phi(s)|s) > \hat\epsilon.$$ The $\min$ over states is a separate constraint at every state of a continuous, effectively infinite state space, which is impractical, so I relax the per-state hard constraint to one on the *average* over the state distribution — the standard move used in TRPO, advantage-weighted regression and BEAR — trading "every state in-support" for "in-support on average" while keeping the density itself as the constrained object. Forming the Lagrangian, treating the constraint as a penalty with multiplier $\lambda$, and flipping signs to a loss gives $$J_\pi(\phi) = \mathbb{E}_{s\sim D}\big[-Q_\theta(s, \pi_\phi(s)) - \lambda\,\log\pi_\beta(\pi_\phi(s)|s)\big].$$ This is the whole regularizer, and it is pluggable by construction: an extra term on the standard actor loss, the policy still a plain network, inference still a single forward pass. The penalty *is* the behavior log-density at the action taken — a direct density/support constraint, not a divergence. Here $\lambda$ is not literally the threshold $\epsilon$; it is the soft actor-loss coefficient governing the same conservatism-versus-optimality tradeoff: crank it up and the policy is pushed onto high-density actions (tight support), turn it down and the value term dominates (loose support).

The catch is that I do not have $\pi_\beta$, only samples from it, so I must *estimate* the behavior density at arbitrary $(s,a)$ points, including the off-data actions the policy probes. Offline behavior policies are routinely multimodal — mixtures of experts, replays of many policies — and a single Gaussian would smear a bimodal behavior into one blob and call the valley between modes "in support," so I model $\pi_\beta(a|s) \approx p_\psi(a|s) = \int p_\psi(a|z,s)\,p(z|s)\,dz$ with a conditional VAE and fixed prior $p(z|s)=\mathcal{N}(0,I)$. The marginal is intractable, so the VAE lower-bounds it: introducing an approximate posterior $q_\varphi(z|a,s)$ and applying Jensen to the concave $\log$, $$\log p_\psi(a|s) \ge \mathbb{E}_{q_\varphi(z|a,s)}[\log p_\psi(a|z,s)] - \mathrm{KL}(q_\varphi(z|a,s)\,\|\,p(z|s)) =: -L_{\text{ELBO}}(s,a).$$ The gap of this bound is exactly $\mathrm{KL}(q_\varphi(z|a,s)\,\|\,p_\psi(z|a,s)) \ge 0$, so $-L_{\text{ELBO}}$ is a genuine *lower* bound on the log-density — which is conservative in the right direction, since I constrain $\log\pi_\beta$ to be *above* a threshold: if the lower bound is high, the true density is at least that high. So I substitute $-L_{\text{ELBO}}$ for $\log\pi_\beta$ in the penalty. Burda's $L$-sample importance-weighted estimator $\log\frac{1}{L}\sum_l p_\psi(a,z_l|s)/q_\varphi(z_l|a,s)$ tightens the bound monotonically toward $\log p_\psi$ as $L\to\infty$, giving a knob on tightness — but the density term is a soft *constraint signal*, not a calibrated reporting metric; it only needs to rise on in-support actions and fall off-support with a stable gradient. Setting $L=1$ collapses the estimator back to the plain ELBO with a real bonus: the KL term stays *analytic* (closed form for Gaussian $q$ and Gaussian prior) rather than sampled, so the gradient has lower variance. I therefore use the $L=1$ ELBO as the penalty, $\texttt{neg\_log\_beta} = L_{\text{ELBO}}(s,\pi_\phi(s))$. Concretely the encoder maps $(s,a)$ to $q_\varphi(z|a,s) = \mathcal{N}(\text{mean}, \text{std}^2)$ via a shared trunk and mean/log-std heads (log-std clamped for stability), reparameterizes $z = \text{mean} + \text{std}\cdot\text{noise}$, and the decoder maps $(s,z)$ to $u = \texttt{max\_action}\cdot\tanh(\text{MLP}(s,z))$. With a Gaussian decoder of fixed variance the reconstruction term is, up to a positive scale and constants, a negative squared error, so the reconstruction *loss* is $\text{recon} = \text{mean}((u-a)^2)$; the two-Gaussian KL has the closed form $\text{KL} = -0.5\,\text{mean}(1 + \log\text{std}^2 - \text{mean}^2 - \text{std}^2)$. The practical loss is $\text{recon} + \beta\cdot\text{KL}$ with $\beta = 0.5$, a beta-VAE down-weighting of the KL so the model spends enough capacity on faithful reconstruction; this same quantity is both what trains the VAE and what is evaluated as the density penalty.

The base algorithm is TD3, not SAC, and deliberately so. My enemy is overestimation on OOD actions, and TD3 was built to suppress exactly that: the $\min$ of clipped twin critics in the target caps the bootstrap, target policy smoothing keeps the critic from latching onto a sharp spurious peak, and delayed actor updates let the value settle before the actor chases it. SAC is the opposite on two counts — its stochastic actor *samples* actions whose tails reach into OOD regions and pull in erroneous values, and its entropy bonus actively *rewards spreading out*, pushing the policy toward and past the support edge, precisely the behavior I am forbidding. So: TD3 base, deterministic actor, twin critics. One more piece comes from TD3+BC. My actor loss $-Q + \lambda\cdot\texttt{neg\_log\_beta}$ mixes two terms on incompatible scales — $Q$ has the magnitude of returns (reward-dependent), $\texttt{neg\_log\_beta}$ the magnitude of a log-density — so a fixed $\lambda$ would have to be re-tuned per task. I normalize the value term by the mean absolute $Q$ over the minibatch, $\texttt{norm\_q} = 1/\overline{|Q|}$ detached (it is a scaling, not part of the objective), giving $$J_\pi = -\texttt{norm\_q}\cdot\overline{Q} + \lambda\cdot\overline{\texttt{neg\_log\_beta}},$$ which keeps the value gradient order-1 regardless of reward scale so a single $\lambda$ schedule transfers across tasks. This yields a two-phase algorithm: phase one pretrains the VAE on $(s,a)$ pairs by minimizing $\text{recon}+\beta\,\text{KL}$ for order $10^5$ iterations — before policy training, because the constraint term is meaningless until the density model is good, and frozen afterward to keep the constraint stationary — and phase two runs ordinary TD3 with the augmented actor loss every $\texttt{policy\_freq}=2$ steps. The pay-off for the offline-to-online setting falls out for free: at $\lambda = 0$ the penalty vanishes and the loss is just $-\texttt{norm\_q}\cdot\overline{Q}$, which is exactly the standard online TD3 actor objective. SPOT is therefore plain TD3 with a knob that, at zero, returns it to online TD3 — a smoothness a parameterized policy cannot offer. So fine-tuning is just cooling $\lambda$: as fresh online interactions shift the data toward the policy's own actions (the very actions the critic can now learn about from real feedback), the reason for the constraint erodes, and holding $\lambda$ fixed would pin the policy to the *offline* support and cap improvement. I decay it linearly, $\lambda_t = \lambda\cdot\max(\lambda_{\text{end}},\,1 - t_{\text{online}}/10^6)$, holding a floor $\lambda_{\text{end}} > 0$ rather than going to zero because on the hardest sparse-reward, high-dimensional tasks bootstrapping error stays dangerous and a residual constraint keeps the critic stable. And I *freeze the VAE* online: the online data is a moving, policy-dependent distribution, not the fixed $\pi_\beta$ the density model captured, so re-fitting it would chase a target that no longer means "behavior support" — the frozen offline VAE keeps a stable notion of where the offline data was, and the decaying $\lambda$ controls how much it still binds.

```python
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    """Conditional VAE for the behavior-density penalty."""

    def __init__(self, state_dim, action_dim, latent_dim, max_action, hidden_dim=750):
        super().__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device

    def encode(self, state, action):
        h = F.relu(self.e1(torch.cat([state, action], -1)))
        h = F.relu(self.e2(h))
        log_std = self.log_std(h).clamp(-4, 15)
        return self.mean(h), torch.exp(log_std)

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5, 0.5)
        a = F.relu(self.d1(torch.cat([state, z], -1)))
        a = F.relu(self.d2(a))
        if self.max_action is not None:
            return self.max_action * torch.tanh(self.d3(a))
        return self.d3(a)

    def forward(self, state, action):
        mean, std = self.encode(state, action)
        z = mean + std * torch.randn_like(std)
        return self.decode(state, z), mean, std

    def elbo_loss(self, state, action, beta, num_samples=1):
        mean, std = self.encode(state, action)
        mean_s = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std_s = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean_s + std_s * torch.randn_like(std_s)
        state_r = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        u = self.decode(state_r, z)
        recon = ((u - action_r) ** 2).mean(dim=(1, 2))
        kl = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(-1)
        return recon + beta * kl

    def iwae_loss(self, state, action, beta, num_samples=10):
        return -self.importance_sampling_estimator(state, action, beta, num_samples)

    def importance_sampling_estimator(self, state, action, beta, num_samples=500):
        mean, std = self.encode(state, action)
        mean_enc = mean.repeat(num_samples, 1, 1).permute(1, 0, 2)
        std_enc = std.repeat(num_samples, 1, 1).permute(1, 0, 2)
        z = mean_enc + std_enc * torch.randn_like(std_enc)
        state_r = state.repeat(num_samples, 1, 1).permute(1, 0, 2)
        action_r = action.repeat(num_samples, 1, 1).permute(1, 0, 2)
        mean_dec = self.decode(state_r, z)
        std_dec = torch.ones_like(mean_dec) * math.sqrt(beta / 4)
        log_qzx = td.Normal(mean_enc, std_enc).log_prob(z)
        log_pz = td.Normal(torch.zeros_like(z), torch.ones_like(z)).log_prob(z)
        log_pxz = td.Normal(mean_dec, std_dec).log_prob(action_r)
        w = log_pxz.sum(-1) + log_pz.sum(-1) - log_qzx.sum(-1)
        return w.logsumexp(dim=-1) - math.log(num_samples)
```

```python
def weights_init_(m, init_w=3e-3):
    if isinstance(m, nn.Linear):
        m.weight.data.uniform_(-init_w, init_w)
        m.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, dropout=None,
                 hidden_dim=256, init_w=None):
        super().__init__()
        if dropout:
            self.l1 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.Dropout(dropout))
            self.l2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout))
        else:
            self.l1 = nn.Linear(state_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action
        if init_w:
            weights_init_(self.l3, init_w)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return self.max_action * torch.tanh(a) if self.max_action is not None else a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_w=None):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)
        if init_w:
            weights_init_(self.l3, init_w)
            weights_init_(self.l6, init_w)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        return self.l3(q1)


class SPOT_TD3:
    def __init__(self, vae, state_dim, action_dim, max_action, device="cuda",
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 policy_freq=2, beta=0.5, lambd=1.0, lr=3e-4, actor_lr=None,
                 without_Q_norm=False, num_samples=1, iwae=False,
                 actor_hidden_dim=256, critic_hidden_dim=256, actor_dropout=0.1,
                 actor_init_w=None, critic_init_w=None,
                 lambd_cool=False, lambd_end=0.2, max_online_steps=1_000_000):
        self.device = device
        self.total_it = 0
        self.vae = vae.eval()
        self.actor = Actor(state_dim, action_dim, max_action, dropout=actor_dropout,
                           hidden_dim=actor_hidden_dim, init_w=actor_init_w).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr or lr)
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden_dim,
                             init_w=critic_init_w).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.max_action = max_action
        self.discount, self.tau = discount, tau
        self.policy_noise, self.noise_clip = policy_noise, noise_clip
        self.policy_freq = policy_freq
        self.beta, self.lambd, self.num_samples = beta, lambd, num_samples
        self.iwae = iwae
        self.without_Q_norm = without_Q_norm
        self.lambd_cool, self.lambd_end = lambd_cool, lambd_end
        self.max_online_steps = max_online_steps
        self.online_it = 0

    def select_action(self, state):
        with torch.no_grad():
            self.actor.eval()
            s = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            action = self.actor(s).cpu().data.numpy().flatten()
            self.actor.train()
            return action

    def _train_step(self, replay_buffer, batch_size, lambd):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + not_done * self.discount * torch.min(target_q1, target_q2)
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state)
            q = self.critic.Q1(state, pi)
            neg_log_beta = (self.vae.iwae_loss(state, pi, self.beta, self.num_samples)
                            if self.iwae else
                            self.vae.elbo_loss(state, pi, self.beta, self.num_samples))
            if self.without_Q_norm:
                actor_loss = -q.mean() + lambd * neg_log_beta.mean()
            else:
                norm_q = 1.0 / q.abs().mean().detach()
                actor_loss = -norm_q * q.mean() + lambd * neg_log_beta.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def train(self, replay_buffer, batch_size=256):
        self._train_step(replay_buffer, batch_size, self.lambd)

    def train_online(self, replay_buffer, batch_size=256, max_online_steps=None):
        self.online_it += 1
        max_online_steps = max_online_steps or self.max_online_steps
        # THUML computes the same decay with total_it/1000000 inside train_online;
        # this standalone form names the online-step counter explicitly.
        lambd = (self.lambd * max(self.lambd_end, 1.0 - self.online_it / max_online_steps)
                 if self.lambd_cool else self.lambd)
        self._train_step(replay_buffer, batch_size, lambd)
```

```python
def pretrain_vae(vae, replay_buffer, iterations=100_000, batch_size=256, beta=0.5, lr=1e-3):
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.train()
    for _ in range(iterations):
        state, action, *_ = replay_buffer.sample(batch_size)
        loss = vae.elbo_loss(state, action, beta).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    vae.eval()
    return vae
```
