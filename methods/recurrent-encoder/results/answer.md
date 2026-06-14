# The recurrent belief encoder (VariBAD), distilled

VariBAD is a meta-RL method that learns approximately Bayes-optimal online behaviour by
meta-learning amortized variational inference over a latent task variable. Its defining
component — the one the meta-RL benchmark's `recurrent_encoder` baseline instantiates — is a
**recurrent context encoder** `q_φ(m|τ_{:t})`: an RNN that consumes the agent's ordered
experience `(s, a, r)` one transition at a time, carries a hidden state as the online sufficient
statistic of the past, and reads off a posterior (mean and variance) over a low-dimensional
stochastic latent task embedding `m`. The policy conditions on this posterior (state + belief),
so it can trade off exploration and exploitation online; the encoder is trained by a variational
ELBO whose decoder reconstructs past *and future* states/rewards.

## Problem it solves

Maximize expected return *during learning* on a task drawn from a known family `p(M)` with
unknown reward/transition functions — i.e. trade off exploration and exploitation online from the
first steps. The principled optimum is the Bayes-optimal policy of the Bayes-adaptive MDP
(BAMDP), which conditions on a belief over the task, but the BAMDP is intractable (unknown
parameterization of `R`,`T`; intractable belief update; intractable belief-space planning), and
the tractable shortcut, posterior sampling, explores inefficiently. Given a meta-training
distribution, the goal is to learn an encoder that turns experience into a calibrated, online
belief and a policy that acts approximately Bayes-optimally on it, with no test-time planning.

## Key idea

- Represent the task by a low-dimensional **stochastic latent** `m` (Gaussian), with the reward
  and transition functions `R(·;m)`, `T(·;m)` **shared across tasks**; only `m` varies. This
  shrinks the belief from a posterior over functions to a posterior over a small vector and lets
  the shared model pool data over all tasks.
- **Amortize** the intractable posterior with an inference network `q_φ(m|τ_{:t})` (the VAE move):
  one forward pass gives the belief at runtime.
- Make the encoder **recurrent** over the ordered trajectory: the belief `p(m|τ_{:t})` depends on
  the whole ordered history and must be updated online, which is exactly what a GRU hidden state
  does — `O(1)` update per transition, hidden state = running belief. (Contrast: a
  permutation-invariant product-of-Gaussians encoder is order-blind and treats transitions as
  independent given `m`; convenient for off-policy replay but it discards the sequential structure
  the belief needs.)
- Train with a **variational ELBO at every context length `t`**, with the decoder reconstructing
  the **full trajectory including the future**, and the prior for step `t` set to the previous
  posterior (online Bayesian updating).
- The probabilistic latent + KL is the inductive bias that turns recurrent meta-RL (RL²) into a
  calibrated, uncertainty-aware belief; removing the decoder + KL recovers RL².

## Objective

Per-context-length ELBO (encode the past `τ_{:t}`, decode the whole trajectory `τ_{:H⁺}`):

```
ELBO_t = E_{q_φ(m|τ_{:t})}[ log p_θ(τ_{:H⁺} | m) ] − KL( q_φ(m|τ_{:t}) || p_θ(m) ),

log p_θ(τ_{:H⁺}|m) = log p(s_0|m) + Σ_i [ log p(s_{i+1}|s_i,a_i,m) + log p(r_{i+1}|s_i,a_i,s_{i+1},m) ].
```

Prior `p_θ(m)` for step `t` is the previous posterior `q_φ(m|τ_{:t-1})`; initial prior `N(0,I)`.
Overall objective combines the RL return and the summed ELBO:

```
L(φ, θ, ψ) = E_{p(M)}[ J(ψ, φ) + λ Σ_{t=0}^{H⁺} ELBO_t(φ, θ) ].
```

Latent sampled by reparameterization `m = μ + exp(½ logvar) ⊙ ε`, `ε ~ N(0,I)`. In practice
subsample the `t`'s if `H⁺` is large. The RL loss is **not** backpropagated through the encoder
(avoids gradient interference and the cost of re-encoding for every on-policy minibatch); VAE and
policy use separate optimizers, learning rates, and data buffers. At meta-test time the decoder is
discarded; the policy acts on the streaming belief with no gradient adaptation and no planning.

## Derivation of the bound

```
E_ρ[ log p_θ(τ_{:H⁺}) ] = E_ρ[ log ∫ p_θ(τ_{:H⁺}, m) (q_φ(m|τ_{:t})/q_φ(m|τ_{:t})) dm ]
                        = E_ρ[ log E_{q_φ(m|τ_{:t})}[ p_θ(τ_{:H⁺}, m)/q_φ(m|τ_{:t}) ] ]
        (Jensen)        ≥ E_{ρ, q_φ}[ log p_θ(τ_{:H⁺}|m) + log p_θ(m) − log q_φ(m|τ_{:t}) ]
                        = E_ρ[ E_{q_φ}[ log p_θ(τ_{:H⁺}|m) ] − KL( q_φ(m|τ_{:t}) || p_θ(m) ) ] = ELBO_t.
```

## Architecture (encoder)

Per-modality feature extractors (state / action / reward each embedded separately, since they have
different scales/dims) → optional FC → **single-layer GRU** (gated, handles long horizons,
lighter than LSTM; orthogonal recurrent-weight init, zero bias init) → optional FC → **two
separate heads** for `μ` and `logvar`. Latent dim is small (≈5; task variation is
low-dimensional, so a tiny latent gives a sharp, stable posterior, unlike a large deterministic
recurrent hidden state that drifts across resets). The decoder is a learned reward function
(and optionally transition function) conditioned on `m`.

## Working code

The recurrent belief encoder, grounded in the canonical implementation: per-modality embeddings,
a GRU core whose hidden state is the online belief, two Gaussian heads, reparameterized sampling,
an online forward (one transition + previous hidden state), and a `t=0` prior path.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class FeatureExtractor(nn.Module):
    """Embed one modality (state / action / reward) before fusing."""
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        self.activation = activation
        self.fc = nn.Linear(input_size, output_size) if output_size != 0 else None

    def forward(self, x):
        return self.activation(self.fc(x)) if self.fc is not None else torch.zeros(0)


class RNNEncoder(nn.Module):
    """Recurrent belief encoder q_phi(m | tau_{:t})."""
    def __init__(self,
                 layers_before_gru=(), hidden_size=64, layers_after_gru=(),
                 latent_dim=5,
                 action_dim=2, action_embed_dim=16,
                 state_dim=2, state_embed_dim=32,
                 reward_size=1, reward_embed_size=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size

        self.state_encoder = FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = FeatureExtractor(reward_size, reward_embed_size, F.relu)

        curr = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList(
            [nn.Linear(curr if i == 0 else layers_before_gru[i - 1], layers_before_gru[i])
             for i in range(len(layers_before_gru))])
        if layers_before_gru:
            curr = layers_before_gru[-1]

        self.gru = nn.GRU(input_size=curr, hidden_size=hidden_size, num_layers=1)
        for name, p in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(p, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(p)

        curr = hidden_size
        self.fc_after_gru = nn.ModuleList(
            [nn.Linear(curr if i == 0 else layers_after_gru[i - 1], layers_after_gru[i])
             for i in range(len(layers_after_gru))])
        if layers_after_gru:
            curr = layers_after_gru[-1]

        self.fc_mu = nn.Linear(curr, latent_dim)
        self.fc_logvar = nn.Linear(curr, latent_dim)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def prior(self, batch_size, device, sample=True):
        h = torch.zeros((1, batch_size, self.hidden_size), device=device)
        for fc in self.fc_after_gru:
            h = F.relu(fc(h))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        m = self.reparameterise(mu, logvar) if sample else mu
        return m, mu, logvar, torch.zeros((1, batch_size, self.hidden_size), device=device)

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True):
        # inputs (seq_len, batch, dim); one online step has seq_len == 1
        if return_prior:
            m0, mu0, lv0, hidden_state = self.prior(actions.shape[1], actions.device)

        h = torch.cat((self.action_encoder(actions),
                       self.state_encoder(states),
                       self.reward_encoder(rewards)), dim=2)
        for fc in self.fc_before_gru:
            h = F.relu(fc(h))

        output, _ = self.gru(h, hidden_state)        # carry hidden state forward
        gru_h = output
        for fc in self.fc_after_gru:
            gru_h = F.relu(fc(gru_h))

        mu, logvar = self.fc_mu(gru_h), self.fc_logvar(gru_h)
        m = self.reparameterise(mu, logvar) if sample else mu

        if return_prior:
            m = torch.cat((m0, m)); mu = torch.cat((mu0, mu)); logvar = torch.cat((lv0, logvar))
            output = torch.cat((hidden_state, output))
        return m, mu, logvar, output
```

Filling the per-transition encoder slot of a PEARL-style harness (ordered context `(task, seq,
feat)`; the agent applies `softplus` to the variance and a product of Gaussians): collapse the
per-modality embedders into one per-transition MLP, take the last recurrent state, emit
`2·latent_dim` numbers (`μ`, pre-softplus `σ²`) in one head — the recurrence is the only
structural change from the per-transition baseline.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import PyTorchModule


def _identity(x):
    return x


class CustomContextEncoder(PyTorchModule):
    """Recurrent belief encoder in the per-transition encoder slot."""
    IS_RECURRENT = True

    def __init__(self, hidden_sizes, input_size, output_size,
                 init_w=3e-3, hidden_activation=F.relu,
                 output_activation=_identity, hidden_init=ptu.fanin_init,
                 b_init_value=0.1, **kwargs):
        self.save_init_params(locals())
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size               # 2 * latent_dim
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        in_size = input_size
        self.fcs = []
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        self.hidden_dim = self.hidden_sizes[-1]
        self.register_buffer('hidden', torch.zeros(1, 1, self.hidden_dim))
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim,
                            num_layers=1, batch_first=True)

    def forward(self, input, return_preactivations=False):
        task, seq, feat = input.size()
        out = input.view(task * seq, feat)
        for fc in self.fcs:
            out = self.hidden_activation(fc(out))
        out = out.view(task, seq, -1)

        if self.hidden.size(1) != task:
            self.reset(task)
        zeros = torch.zeros(self.hidden.size()).to(ptu.device)
        out, (hn, cn) = self.lstm(out, (self.hidden, zeros))
        self.hidden = hn
        out = out[:, -1, :]

        preactivation = self.last_fc(out)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        return output

    def reset(self, num_tasks=1):
        self.hidden = self.hidden.new_full((1, num_tasks, self.hidden_dim), 0)
```

## Relation to prior methods

- **RL² (Duan et al. 2016; Wang et al. 2016)** — recurrent black-box meta-RL: a GRU/LSTM consumes
  `(s, a_{t-1}, r_{t-1}, d)` and adaptation lives in the recurrent dynamics. Removing the decoder
  and the KL term recovers RL²; the recurrent belief encoder adds a stochastic latent (uncertainty
  inductive bias) and a decoder reconstructing past *and future* (auxiliary task-grounding loss).
- **PEARL (Rakelly et al. 2019)** — permutation-invariant per-transition encoder, product of
  Gaussians `q_φ(z|c_{1:N}) ∝ Π_n Ψ_φ(z|c_n)`, posterior-sampling exploration, off-policy (SAC).
  Order-invariant (good for replay) but blind to trajectory order; gives posterior-sampling-style
  exploration rather than online-belief-conditioned Bayes-optimal exploration.
- **Posterior sampling (Strens 2000; Osband et al. 2013)** — sample one MDP from the posterior,
  act optimally for it per episode; tractable but explores less efficiently than acting on the
  running belief.
- **VAE (Kingma & Welling 2013)** — supplies the amortized-inference ELBO and reparameterization
  the encoder is trained with; the difference here is encode-the-past / decode-past-and-future and
  the per-context-length online objective with the prior set to the previous posterior.
