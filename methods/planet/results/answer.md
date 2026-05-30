# PlaNet (Deep Planning Network)

## Problem

Control an agent from raw $64\times64\times3$ images while spending as few real
environment episodes as possible. The setting is a POMDP — a single frame does not
reveal the full state — and the goal is to maximize $\mathbb E[\sum_{t=1}^T r_t]$.
Planning is data-efficient and powerful when dynamics are known; PlaNet learns the
dynamics from experience and plans inside the learned model, using **no policy or
value network** — the policy is online planning (MPC).

## Key idea

Learn a latent dynamics model whose reward is a function of the latent state, so
thousands of candidate action sequences can be scored entirely in latent space
without rendering images; plan with the cross-entropy method and re-plan every
step.

**1. Recurrent state-space model (RSSM).** Split the latent state into a
deterministic recurrent part $h_t$ (a memory highway) and a stochastic part $s_t$
(uncertainty/multimodality):
$$h_t=f(h_{t-1},s_{t-1},a_{t-1})\ \text{(GRU)},\quad s_t\sim p(s_t\mid h_t),\quad o_t\sim p(o_t\mid h_t,s_t),\quad r_t\sim p(r_t\mid h_t,s_t),$$
with a filtering posterior (encoder) $q(s_t\mid h_t,o_t)$. A purely stochastic
transition cannot reliably remember; a purely deterministic recurrence cannot
represent multiple futures — the split fixes both. All observation information must
pass through the sampled $s_t$ (no deterministic shortcut to reconstruction). The
transition is Gaussian (feedforward net), the observation model Gaussian with a
deconvolutional mean and identity covariance, the reward a unit-variance scalar
Gaussian; under unit variance, the reconstruction log-likelihoods are mean squared
errors. Features fed downstream are $[\,s_t;\,h_t\,]$.

**2. Variational training objective.** Maximize the bound
$$\ln p(o_{1:T}\mid a_{1:T})\ \ge\ \sum_{t=1}^T\Big(\underbrace{\mathbb E_{q(s_t\mid o_{\le t},a_{<t})}[\ln p(o_t\mid s_t)]}_{\text{reconstruction}}-\underbrace{\mathbb E_{q(s_{t-1}\mid\cdot)}\big[\mathrm{KL}\big(q(s_t\mid o_{\le t},a_{<t})\,\|\,p(s_t\mid s_{t-1},a_{t-1})\big)\big]}_{\text{complexity}}\Big),$$
(derived by importance-weighting under $q$ and Jensen's inequality), with the same
term for reward by analogy. Outer expectations are estimated with a single
reparameterized sample. The KL is granted **3 free nats** (clipped below that
value) and is not otherwise scaled against reconstruction.

**3. Latent overshooting (a general fix for latent sequence models).** The one-step
bound only trains one-step transitions. Define the $d$-step prior
$p(s_t\mid s_{t-d})=\mathbb E_{p(s_{t-1}\mid s_{t-d})}[p(s_t\mid s_{t-1})]$ and the
$d$-step bound
$$\ln p_d(o_{1:T})\ \ge\ \sum_t\Big(\mathbb E_{q(s_t\mid o_{\le t})}[\ln p(o_t\mid s_t)]-\mathbb E_{p(s_{t-1}\mid s_{t-d})\,q(s_{t-d}\mid o_{\le t-d})}\big[\mathrm{KL}\big(q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})\big)\big]\Big),$$
then average over $d=1\ldots D$ with $\beta_d$ weights ($\beta$-VAE style):
$$\tfrac1D\textstyle\sum_{d=1}^D\ln p_d(o_{1:T})\ \ge\ \sum_t\Big(\mathbb E_q[\ln p(o_t\mid s_t)]-\tfrac1D\sum_{d=1}^D\beta_d\,\mathbb E\big[\mathrm{KL}\big(q(s_t\mid o_{\le t})\,\|\,p(s_t\mid s_{t-1})\big)\big]\Big).$$
This is a pure latent-space regularizer (no extra images); gradients on posteriors
for $d>1$ are stopped so multi-step predictions are trained toward the informed
posteriors. The data-processing inequality (latent chain is Markov,
$\mathbb I(s_t;s_{t-d})\le\mathbb I(s_t;s_{t-1})$) shows a multi-step bound is also a
bound on the true one-step likelihood in expectation over data. With the RSSM the
deterministic path already carries memory, so the final agent does not require
overshooting.

**4. Planning (CEM + MPC).** From the current state belief, run the cross-entropy
method over the next $H$ actions, $a_{t:t+H}\sim\mathcal N(\mu,\sigma^2\mathbb I)$,
starting from $\mathcal N(0,I)$: sample $J$ sequences, roll the **prior** forward in
latent space and sum mean predicted rewards (one trajectory per sequence, no image
generation), keep the top $K$, refit $\mu,\sigma$; after $I$ iterations execute
$\mu_t$ and re-plan next step from $\mathcal N(0,I)$ (resetting avoids local optima).

**5. Experience loop.** Start with $S=5$ random seed episodes; alternate $C=100$
gradient updates with collecting one new planned episode; add small Gaussian action
noise $\epsilon\sim\mathcal N(0,0.3)$; repeat each action $R$ times.

## Hyperparameters

GRU with 200 units; all other functions are two fully-connected layers of 200 with
ReLU; latent is a 30-dimensional diagonal Gaussian (softplus stddev + min 0.1);
images reduced to 5-bit depth; Adam with learning rate $10^{-3}$, $\epsilon=10^{-4}$,
gradient-clip norm 1000; batches of $B=50$ chunks of length $L=50$; 3 free nats. CEM:
$H=12$, $I=10$, $J=1000$, $K=100$. Action repeat per domain: cart-pole 8, reacher 4,
cheetah 4, finger 2, cup 4, walker 2.

## Code

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, kl_divergence

class Encoder(nn.Module):                       # 64x64x3 -> 1024
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 4, 2); self.c2 = nn.Conv2d(32, 64, 4, 2)
        self.c3 = nn.Conv2d(64, 128, 4, 2); self.c4 = nn.Conv2d(128, 256, 4, 2)
    def forward(self, o):
        h = F.relu(self.c1(o)); h = F.relu(self.c2(h))
        h = F.relu(self.c3(h)); h = F.relu(self.c4(h))
        return h.reshape(h.size(0), -1)

class RSSM(nn.Module):
    def __init__(self, state_dim=30, action_dim=1, rnn_dim=200, hidden=200, min_std=0.1):
        super().__init__()
        self.rnn_dim, self.min_std = rnn_dim, min_std
        self.fc_input  = nn.Linear(state_dim + action_dim, hidden)
        self.gru       = nn.GRUCell(hidden, rnn_dim)            # h_t = f(h, s, a)
        self.fc_prior  = nn.Linear(rnn_dim, hidden)
        self.prior_mu  = nn.Linear(hidden, state_dim)
        self.prior_std = nn.Linear(hidden, state_dim)
        self.fc_post   = nn.Linear(rnn_dim + 1024, hidden)
        self.post_mu   = nn.Linear(hidden, state_dim)
        self.post_std  = nn.Linear(hidden, state_dim)
    def prior(self, state, action, h):                          # p(s_{t+1} | h_{t+1})
        x = F.relu(self.fc_input(torch.cat([state, action], -1)))
        h = self.gru(x, h)
        f = F.relu(self.fc_prior(h))
        return Normal(self.prior_mu(f), F.softplus(self.prior_std(f)) + self.min_std), h
    def posterior(self, h, embed):                              # q(s_{t+1} | h_{t+1}, o_{t+1})
        f = F.relu(self.fc_post(torch.cat([h, embed], -1)))
        return Normal(self.post_mu(f), F.softplus(self.post_std(f)) + self.min_std)

class ObsModel(nn.Module):                                      # p(o | s, h), deconv mean
    def __init__(self, state_dim=30, rnn_dim=200):
        super().__init__()
        self.fc = nn.Linear(state_dim + rnn_dim, 1024)
        self.d1 = nn.ConvTranspose2d(1024, 128, 5, 2); self.d2 = nn.ConvTranspose2d(128, 64, 5, 2)
        self.d3 = nn.ConvTranspose2d(64, 32, 6, 2);    self.d4 = nn.ConvTranspose2d(32, 3, 6, 2)
    def forward(self, s, h):
        x = self.fc(torch.cat([s, h], -1)).view(-1, 1024, 1, 1)
        x = F.relu(self.d1(x)); x = F.relu(self.d2(x)); x = F.relu(self.d3(x))
        return self.d4(x)

class RewardModel(nn.Module):                                   # p(r | s, h), unit-var scalar
    def __init__(self, state_dim=30, rnn_dim=200, hidden=200):
        super().__init__()
        self.f1 = nn.Linear(state_dim + rnn_dim, hidden); self.f2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, s, h):
        x = F.relu(self.f1(torch.cat([s, h], -1))); x = F.relu(self.f2(x))
        return self.out(x).squeeze(-1)

def model_loss(obs, act, rew, enc, rssm, obs_m, rew_m, free_nats=3.0):
    T, B = obs.shape[0], obs.shape[1]
    embed = enc(obs.reshape(-1, 3, 64, 64)).view(T, B, -1)
    s = obs.new_zeros(B, 30); h = obs.new_zeros(B, rssm.rnn_dim)
    states, hiddens, kl = [], [], 0.0
    for t in range(T - 1):
        prior, h = rssm.prior(s, act[t], h)
        post     = rssm.posterior(h, embed[t + 1])
        s = post.rsample(); states.append(s); hiddens.append(h)
        kl = kl + kl_divergence(post, prior).sum(-1).clamp(min=free_nats).mean()
    kl = kl / (T - 1)
    s_seq = torch.stack(states); h_seq = torch.stack(hiddens)
    recon = obs_m(s_seq.reshape(-1, 30), h_seq.reshape(-1, rssm.rnn_dim)).view(T - 1, B, 3, 64, 64)
    obs_loss = 0.5 * F.mse_loss(recon, obs[1:], reduction='none').mean([0, 1]).sum()
    rew_pred = rew_m(s_seq.reshape(-1, 30), h_seq.reshape(-1, rssm.rnn_dim)).view(T - 1, B)
    rew_loss = 0.5 * F.mse_loss(rew_pred, rew[:-1])
    return obs_loss + rew_loss + kl

def plan(belief_h, enc, rssm, rew_m, obs, action_dim, H=12, I=10, J=1000, K=100):
    embed = enc(obs); post = rssm.posterior(belief_h, embed)
    mu = obs.new_zeros(H, action_dim); std = obs.new_ones(H, action_dim)
    for _ in range(I):
        cand = Normal(mu, std).sample([J]).transpose(0, 1)      # (H, J, action_dim)
        s = post.sample([J]).squeeze(1); h = belief_h.repeat(J, 1)
        ret = obs.new_zeros(J)
        for t in range(H):
            prior, h = rssm.prior(s, cand[t], h)
            s = prior.sample(); ret = ret + rew_m(s, h)         # latent-only rollout
        elite = ret.argsort(descending=True)[:K]; top = cand[:, elite, :]
        mu = top.mean(1); std = (top - mu.unsqueeze(1)).abs().sum(1) / (K - 1)
    return mu[0]
```
