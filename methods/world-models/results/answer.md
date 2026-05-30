# World Models

## Problem

Let an RL agent benefit from a large, expressive predictive model of its
environment without paying the credit-assignment cost of training that model
against reward. Solution: split the agent into a **large world model** trained
unsupervised (dense, gradient-friendly next-frame prediction) and a **tiny
controller** that is the only part optimized against reward — small enough to be
optimized by a derivative-free evolution strategy. Demonstrated on pixel-based car
racing and a first-person dodging task.

## Key idea: V, M, C

**V — Vision (convolutional VAE).** Compresses each $64\times64\times3$ frame into
a latent $z$: a conv encoder outputs the mean/std of a diagonal Gaussian,
$z=\mu+\sigma\odot\epsilon$, and a deconv decoder reconstructs the frame. Trained on
reconstruction + KL toward $\mathcal N(0,I)$. The Gaussian prior limits per-frame
information capacity and keeps the latent space smooth/robust to the (slightly off)
$z$ vectors the predictive model will generate. Latent size 32 (racing), 64 (doom).

**M — Memory (MDN-RNN).** An LSTM whose mixture-density output layer models the
*distribution* of the next latent given the current latent, action, and hidden
state:
$$P(z_{t+1}\mid a_t,z_t,h_t)=\sum_{k=1}^K \pi_k\,\mathcal N\big(z_{t+1}\mid \mu_k,\sigma_k\big),$$
trained by the negative log-likelihood of $z_{t+1}$ under the mixture (factored
Gaussians, no cross-correlation). A *mixture* (not a single Gaussian) because the
dynamics are multimodal — a fireball forms or it doesn't — and a point prediction
would blur distinct futures. A temperature $\tau$ scales the sampling distribution.
For the dodging task M also predicts a done flag. Hidden size 256 (racing) / 512
(doom); $K=5$ mixtures.

**C — Controller (single linear layer).** $a_t=W_c\,[\,z_t\,;\,h_t\,]+b_c$, with
$\tanh$ bounding the action. It sees the present ($z_t$) *and* M's hidden state
$h_t$, which summarizes the predicted future — so it can act reflexively without
explicit planning rollouts. Deliberately tiny ($\sim$867 params for racing, 1088 for
doom). Trained separately from V and M.

**Optimization of C.** CMA-ES (works well up to a few thousand parameters), using
only the scalar cumulative return of rollouts — no gradients through the
environment, trivially parallel. Population 64; each candidate run for 16 rollouts
with different seeds; fitness = average cumulative reward.

**Rollout.**
```
z = V.encode(obs); a = C.action([z, h]); obs, r, done = env.step(a); h = M.forward([a, z, h])
```

## Learning inside the dream

Because M predicts $P(z_{t+1}, d_{t+1}\mid a_t, z_t, h_t)$, it *is* a latent-space
environment: wrap M as the environment and train C **entirely inside its
hallucination** (sample $z_{t+1}$ from the mixture, feed it back; never render a
frame, never run V). Transfer the learned controller back to the real environment.

**Cheating the world model and the temperature fix.** A controller optimized against
an *approximate* M will find adversarial policies that exploit M's errors (e.g. move
so monsters never fire) and that fail in reality — and training *inside* M even
exposes M's hidden states to the controller. M's stochasticity prevents this: since
M emits a *distribution* over futures, raising the temperature $\tau$ makes the dream
noisier and un-exploitable, forcing a robust policy. $\tau$ too low ($\approx0.1$)
causes mode collapse (no fireballs ever — trivially winnable dream, fails in
reality); $\tau$ a bit above 1 transfers best. Agents that succeed at high $\tau$
transfer well to the cleaner real environment.

## Iterative training (harder tasks)

1. Initialize M, C randomly. 2. Roll out the current C in the real environment $N$
times, logging actions and observations. 3. Train M to model
$P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1}\mid x_t, a_t, h_t)$ and train C inside M.
4. Repeat. M's own prediction error can be turned into a curiosity reward (flip the
sign of M's loss) to drive exploration toward states that would most improve the
model. One pass suffices for the simple tasks here.

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class VAE(nn.Module):                                   # V
    def __init__(self, img_channels=3, latent_size=32):
        super().__init__()
        self.c1 = nn.Conv2d(img_channels, 32, 4, 2); self.c2 = nn.Conv2d(32, 64, 4, 2)
        self.c3 = nn.Conv2d(64, 128, 4, 2);          self.c4 = nn.Conv2d(128, 256, 4, 2)
        self.fc_mu = nn.Linear(2*2*256, latent_size); self.fc_logsig = nn.Linear(2*2*256, latent_size)
        self.fc_dec = nn.Linear(latent_size, 1024)
        self.d1 = nn.ConvTranspose2d(1024, 128, 5, 2); self.d2 = nn.ConvTranspose2d(128, 64, 5, 2)
        self.d3 = nn.ConvTranspose2d(64, 32, 6, 2);    self.d4 = nn.ConvTranspose2d(32, img_channels, 6, 2)
    def encode(self, x):
        h = F.relu(self.c1(x)); h = F.relu(self.c2(h)); h = F.relu(self.c3(h)); h = F.relu(self.c4(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logsig(h)
    def decode(self, z):
        h = F.relu(self.fc_dec(z)).unsqueeze(-1).unsqueeze(-1)
        h = F.relu(self.d1(h)); h = F.relu(self.d2(h)); h = F.relu(self.d3(h))
        return torch.sigmoid(self.d4(h))
    def forward(self, x):
        mu, logsig = self.encode(x); sigma = logsig.exp()
        z = mu + sigma * torch.randn_like(sigma)
        return self.decode(z), mu, logsig

def vae_loss(recon, x, mu, logsig):
    rec = F.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + 2*logsig - mu.pow(2) - (2*logsig).exp())
    return rec + kld

class MDNRNN(nn.Module):                                # M
    def __init__(self, latents=32, actions=3, hiddens=256, gaussians=5):
        super().__init__()
        self.latents, self.gaussians = latents, gaussians
        self.rnn = nn.LSTM(latents + actions, hiddens)
        self.head = nn.Linear(hiddens, (2*latents + 1)*gaussians + 2)
    def forward(self, actions, latents):
        T, B = actions.size(0), actions.size(1)
        out, _ = self.rnn(torch.cat([actions, latents], dim=-1))
        o = self.head(out); s = self.gaussians * self.latents
        mu    = o[:, :, :s].view(T, B, self.gaussians, self.latents)
        sigma = o[:, :, s:2*s].view(T, B, self.gaussians, self.latents).exp()
        logpi = F.log_softmax(o[:, :, 2*s:2*s + self.gaussians], dim=-1)
        return mu, sigma, logpi, o[:, :, -2], o[:, :, -1]    # +reward, done logit

def gmm_loss(target, mu, sigma, logpi):
    target = target.unsqueeze(-2)
    log_g = logpi + Normal(mu, sigma).log_prob(target).sum(-1)
    m = log_g.max(dim=-1, keepdim=True)[0]
    log_prob = m.squeeze(-1) + torch.log(torch.exp(log_g - m).sum(-1))
    return -log_prob.mean()

class Controller(nn.Module):                            # C
    def __init__(self, latents, hiddens, actions):
        super().__init__()
        self.fc = nn.Linear(latents + hiddens, actions)
    def action(self, z, h):
        return torch.tanh(self.fc(torch.cat([z, h], dim=-1)))

def rollout(controller, env, vae, rnn_cell):
    obs = env.reset(); h, c = rnn_cell.initial_state(); total = 0.0; done = False
    while not done:
        z, _ = vae.encode(preprocess(obs))
        a = controller.action(z, h)
        obs, reward, done, _ = env.step(a); total += reward
        _, _, _, _, _, (h, c) = rnn_cell(a, z, (h, c))
    return total

def train_controller(env_factory, vae, rnn, popsize=64, rollouts_per=16):
    es = CMAES(num_params=controller_param_count(), popsize=popsize)
    while not solved:
        candidates = es.ask(); fitness = []
        for params in candidates:                       # independent -> parallel
            c = controller_from(params)
            returns = [rollout(c, env_factory(), vae, rnn) for _ in range(rollouts_per)]
            fitness.append(sum(returns) / len(returns))  # average return
        es.tell(candidates, fitness)
    return es.best()
# Dream training: replace env_factory with a wrapper whose step() samples
# z_{t+1} ~ mixture(M) at temperature tau, returns predicted reward, ends when
# the done logit > 0.5 -- pure latent rollouts, no frames, no V.
```
