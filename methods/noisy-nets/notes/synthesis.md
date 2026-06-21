# Noisy Nets Synthesis

This file is the short synthesis; the auditable reconstruction is in
`notes/discovery_synthesis.md` and the source-by-source evidence table is in
`notes/source_matrix.md`.

Core method:

- Replace action-space dithering with learnable parameter noise in network heads.
- Use `theta = mu + Sigma * epsilon`, with fixed-statistics zero-mean noise and learnable
  `(mu, Sigma)`.
- Optimize `Lbar(zeta) = E_epsilon[L(mu + Sigma * epsilon)]` with a single-sample
  reparameterized gradient.
- This is not a Bayesian posterior approximation; the paper explicitly distinguishes it from
  variational weight-uncertainty methods.

Noisy linear layer:

```text
y = (mu_w + sigma_w * epsilon_w) x + mu_b + sigma_b * epsilon_b
```

Noise cases:

- Independent Gaussian: `pq + q` iid standard-normal draws; main paper choice for A3C.
- Factorised Gaussian: draw `p` input noises and `q` output noises, transform with
  `f(x)=sign(x)sqrt(abs(x))`, and set
  `epsilon_w[j,i]=f(epsilon_out[j])f(epsilon_in[i])`, `epsilon_b[j]=f(epsilon_out[j])`.
  For `Z ~ N(0,1)`, `E[f(Z)^2]=sqrt(2/pi)`, so a factorised weight entry has variance
  `2/pi`, not unit variance.

Initialization:

- Independent: `mu ~ U[-sqrt(3/p), sqrt(3/p)]`, `sigma = 0.017`.
- Factorised paper statement: `mu ~ U[-1/sqrt(p), 1/sqrt(p)]`,
  `sigma = sigma_0/sqrt(p)`, `sigma_0=0.5`.
- Saved Rainbow reference implementation:
  `weight_sigma = std_init/sqrt(in_features)`,
  `bias_sigma = std_init/sqrt(out_features)`.

Integration:

- DQN/Dueling: remove `epsilon`-greedy, use factorised noisy fully connected layers, hold one
  sample fixed across a replay batch, and resample after each replay/optimization step. In the
  paper's DQN/Dueling loop this implies resampling before each action.
- Noisy DQN uses independent online and target noise samples.
- Noisy Dueling uses online sampled action selection and target sampled action evaluation.
- A3C removes the entropy bonus, uses noisy policy/value heads, and fixes one noise sample for
  the whole rollout to preserve on-policy consistency.

Reference-code checks:

- `code/kaixhin_rainbow_model.py` is a community Rainbow implementation, not an official
  DeepMind source release.
- The repaired answer code follows its `NoisyLinear`: epsilon buffers, device-aware
  `_scale_noise`, `epsilon_out.ger(epsilon_in)`, deterministic eval path, and recursive
  `reset_noise` over noisy fully connected modules.
