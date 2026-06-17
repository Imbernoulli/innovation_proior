# Layer Normalization

## Problem it solves

Deep-net training is slowed by the fact that each layer's summed-input distribution drifts as the layers
below it change, forcing small learning rates. Batch normalization stabilizes this by standardizing each
unit's summed input across the minibatch, but its statistic lives on the batch axis: it degrades at small
batch size, is undefined at batch size one / online, differs between train and test (running averages), and
needs separate statistics per time step in an RNN — breaking on sequences longer than any seen in training.
Layer normalization keeps batch normalization's activation-statistic stabilization while removing the
dependence on the batch.

## Key idea

Transpose the normalization axis. Instead of computing the mean and variance of each feature *across the
training cases*, compute a single mean and variance *across all the units in a layer* for a **single**
training case. All units in the layer share that mean and variance; different cases get different ones. The
statistic then depends only on the current example's own summed inputs, so it is independent of batch size,
identical at train and test, and trivially applicable to RNNs at any sequence length. A per-neuron learned
gain and bias, applied after normalization and before the nonlinearity, restore representational power.

## Final form

For the summed inputs `a = (a_1, ..., a_H)` to a layer of `H` units (one training case):

```
mu    = (1/H) * sum_{i=1}^H a_i
sigma = sqrt( (1/H) * sum_{i=1}^H (a_i - mu)^2 )          # biased variance (divide by H)
h_i   = f( (g_i / sigma) * (a_i - mu) + b_i )             # gain g, bias b per neuron, before the nonlinearity f
```

As a reusable function on a vector `z`, with an epsilon floor for numerical stability:

```
LN(z; g, b) = ( (z - mu) / sqrt(sigma^2 + eps) ) ⊙ g + b ,
   mu = (1/D) sum_i z_i ,   sigma^2 = (1/D) sum_i (z_i - mu)^2 .
```

Defaults: `g` initialized to 1, `b` initialized to 0 (so the transform starts as pure normalization), `eps`
≈ 1e-5 inside the square root. Apply to the summed inputs `Wh` (after the linear map, before the
nonlinearity); applying it to the input before the weights would forfeit invariance to weight
re-scaling/re-centering.

**Recurrent use.** With `a^t = W_hh h^{t-1} + W_xh x^t`, normalize per time step with one shared gain/bias:
`h^t = f[ (g/sigma^t) ⊙ (a^t - mu^t) + b ]`. Because the transform is invariant to positive rescaling of all
of `a^t`, it pins the hidden-to-hidden signal scale and tames exploding/vanishing dynamics over long
sequences. For an LSTM, normalize the two projections separately and the cell on the way out:
`(f,i,o,g)^T = LN(W_h h^{t-1}) + LN(W_x x^t) + b`, `h^t = sigmoid(o) ⊙ tanh(LN(c^t))`.

## Why it stabilizes (invariance + geometry)

- **Invariances.** Layer norm is invariant to re-scaling the whole weight matrix, to **re-centering** the
  whole weight matrix (`W -> δW + 1γᵀ`; the shared shift `γᵀx` cancels in `a - mu`), and to **re-scaling a
  single training case** (`x -> δx`; `mu, sigma` scale with it and cancel). It is *not* invariant to scaling
  a single weight vector (batch/weight norm are) — a genuine trade. Batch norm lacks per-case-rescale and
  weight-recentering invariance; weight norm re-centers nothing.
- **Implicit learning-rate control.** On a GLM (lifting to deep nets via a block-diagonal Fisher), the
  normalized Fisher weight-weight block carries `g_i g_j/(sigma_i sigma_j)` along with the derivative of the
  normalized input. If `||w_i||` grows along a scale-invariant direction, the model output is unchanged but
  the relevant `sigma_i` grows with the summed-input scale, so the directional curvature shrinks; in the
  doubling case, the directional curvature changes by `1/2`. The weight norm acts as a per-weight effective
  learning rate, an "early-stopping" effect that stabilizes learning as it proceeds. Learning the gain `g` has a
  metric depending on the prediction-error covariance and the *normalized* activities, not on the raw
  input/weight scale, so it is robust to input and parameter rescaling.

## Where to apply it

Strong on fully-connected and recurrent hidden layers, where all units in a layer contribute similarly and
the layer-wide nuisance is shared. Weaker on convolutional layers, where boundary-receptive-field units fire
rarely and have different statistics, breaking the share-a-statistic premise. Keep it off the output head:
the scale of the logits / value carries the prediction confidence and should not be normalized away.

## Code (drop-in for a standard actor-critic MLP)

```python
import torch
import torch.nn as nn
from torch.distributions import Normal


class LayerNorm(nn.Module):
    """Standardize the summed inputs over the H features of one example; learned gain & bias."""

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(num_features))    # g, init 1
        self.bias = nn.Parameter(torch.zeros(num_features))   # b, init 0

    def forward(self, x):                                      # x: (..., num_features)
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)     # biased: divide by H
        return self.gain * (x - mu) / torch.sqrt(var + self.eps) + self.bias


def build_mlp(input_dim, hidden_dims, output_dim, activation=nn.ELU()):
    layers, dims = [], [input_dim] + list(hidden_dims)
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(d_in, d_out))
        layers.append(nn.LayerNorm(d_out))                    # normalize summed inputs, before activation
        layers.append(activation)
    layers.append(nn.Linear(dims[-1], output_dim))            # output head: un-normalized
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=(512, 256, 128), critic_hidden_dims=(512, 256, 128),
                 init_noise_std=1.0, activation=nn.ELU()):
        super().__init__()
        self.actor = build_mlp(num_obs, actor_hidden_dims, num_actions, activation)
        self.critic = build_mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None

    def update_distribution(self, obs):
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        return self.actor(obs)

    def evaluate(self, critic_obs):
        return self.critic(critic_obs)
```

`nn.LayerNorm(d_out)` computes exactly the feature-wise transform above (mean/variance over the last
dimension, biased variance, `eps=1e-5` inside the square root, gain init 1, bias init 0). The only change
from a standard actor-critic is the inserted normalization after each hidden linear, before the activation,
with the output head left un-normalized. Existing `nn.Linear` bias settings are otherwise left as the base
implementation defines them; the method-specific affine parameters are the LayerNorm weight and bias.
