# Context: private gradient mechanisms for DP-SGD under a fixed budget

## Research question

We want to train a neural network with a hard, externally checked
`(epsilon, delta)`-differential-privacy guarantee while giving up as little test accuracy as
possible. The privacy is enforced inside the gradient step: each per-sample gradient is
clipped to a bounded `L2` norm so that one example can move the aggregate only so far, the
clipped gradients are averaged, and zero-mean Gaussian noise calibrated to that bound is added
before the optimizer sees the gradient. A privacy accountant calibrates the noise level to the
target budget and tracks how much budget each step consumes.

The design question is what clipping-and-noise policy to run over the course of training — one
that respects a fixed total `(epsilon, delta)`, stays compatible with the accountant the harness
already uses, and returns gradients to the optimizer in the ordinary DP-SGD shape.

## Background

**Differential privacy and the Gaussian mechanism.** A randomized function `F` is
`(epsilon, delta)`-DP if for any two datasets differing in one example and any output set `O`,
`Pr[F(d) in O] <= e^epsilon Pr[F(d') in O] + delta`. The standard way to privatize a
vector-valued query `g` of bounded `L2` sensitivity `C = max_{d,d'} ||g(d) - g(d')||_2` is the
Gaussian mechanism: return `g(d) + N(0, sigma^2 C^2 I)`. The multiplier `sigma` controls how
much the output distribution can shift when one example changes; a single Gaussian release with
multiplier `sigma` satisfies `(epsilon, delta)`-DP once `sigma >= sqrt(2 ln(1.25/delta)) / epsilon`.
In DP-SGD the sensitivity is enforced by clipping each per-sample gradient to `L2` norm `C`, so
replacing one example changes the clipped sum by at most `C`.

**Accountants compose per-step costs additively, and the per-step cost scales like `1/sigma^2`.**
A training run is a long sequence of subsampled Gaussian mechanisms, one per step, and the job
of an accountant is to turn that sequence into one `(epsilon, delta)` number. The moments
accountant (Abadi et al. 2016) and its Rényi-DP formulation (Mironov 2017) both track a
per-step privacy cost for the subsampled Gaussian mechanism and *add* those costs across steps;
for the Gaussian mechanism the per-step cost is proportional to `1/sigma^2`, and Poisson
subsampling at rate `q = b/n` keeps that inverse-square dependence with a roughly `q^2`
amplification. Truncated concentrated DP (tCDP; Bun, Dwork, Rothblum & Steinke 2018) gives a
closed-form version of the same shape through four properties:
- *(Gaussian mechanism)* a Gaussian step of sensitivity `C` and multiplier `sigma` is
  `(C^2 / (2 sigma^2), infinity)`-tCDP;
- *(composition)* `(rho_1, omega_1)`-tCDP composed with `(rho_2, omega_2)`-tCDP is
  `(rho_1 + rho_2, min(omega_1, omega_2))`-tCDP — the `rho`'s add;
- *(conversion)* `(rho, omega)`-tCDP implies `(rho + 2 sqrt(rho ln(1/delta)), delta)`-DP for
  `delta >= 1 / exp((omega - 1)^2 rho)`;
- *(subsampling amplification)* running a `(rho, omega)`-tCDP mechanism on uniformly random
  `hn` of `n` entries gives `(13 h^2 rho, log(1/h)/(4 rho))`-tCDP, under the side conditions
  `rho, h in (0, 0.1]`, `log(1/h) >= 3 rho (2 + log(1/rho))`, `omega >= log(1/h)/(2 rho)`.
The recurring shape across all of these is that each step contributes a privacy cost that grows
like `1/sigma^2` and that the costs add up across steps.

**A predefined noise schedule is free.** A schedule that sets the noise level as a fixed
function of the epoch index — known before training, never looking at the data or the model —
spends no privacy beyond the Gaussian noise it adds at each step. Only the per-step noise costs
budget; choosing *how* the per-step noise level varies over time is a public decision and is
not itself a query on the data.

**Gradient signal during training.** As a model trains, the average per-sample gradient norm
tends to change over epochs. The noise added to the averaged gradient has standard deviation
`sigma C / b`, an absolute quantity independent of the true gradient magnitude. It is observed
empirically that early and late iterations can have different sensitivities to the noise level,
with validation accuracy behaving differently across training phases.

## Baselines

**Standard DP-SGD (Abadi et al. 2016).** Clip each per-sample gradient,
`g_i <- g_i * min(1, C / ||g_i||_2)`, average the clipped gradients, add Gaussian noise scaled
by a constant multiplier `sigma`, and step the optimizer on the noised average. One `sigma` and
one `C` are calibrated to the budget before training via the moments accountant and reported to
it identically for every step. Abadi et al. noted that clipping to roughly the median gradient
norm tends to work well, hinting that the clip scale is a training-time quantity.

**Linearly decaying noise for private medical training (Zhang et al. 2021).** This approach
lowers the noise *variance* geometrically by epoch, `sigma_{e}^2 = R sigma_{e-1}^2`
with `R in (0, 1)` — i.e. `sigma_e^2 = sigma_0^2 R^e` — so that less noise is added as training
proceeds, and accounts for the now-varying Gaussian steps with tCDP rather than the moments
accountant, because tCDP's additive composition handles a changing noise level cleanly. Using
the four tCDP properties above (Gaussian step, subsampling, additive composition, conversion),
it obtains a closed-form total privacy expression for the geometric schedule and calibrates
`sigma_0` to a fixed budget.

**Adaptive clipping and global-scaling methods (Andrew et al. 2021; Bu et al. 2021; Esipova
et al. 2022).** Rather than a single hand-tuned `C`, these track or reshape the clipping rule.
Adaptive clipping privately estimates a quantile of the per-sample norm so the threshold is
less brittle. Global-scaling methods (DP-Global, DP-Global-Adapt) use a *lower* clipping bound
`c_0` as the sensitivity and an *upper* threshold `z_e`: gradients below `z_e` are scaled by
`c_0/z_e`, and DP-Global-Adapt additionally raises `z_e` with a geometric update rule so that
all per-sample norms stay below it, keeping the global sensitivity at `c_0` while moving the
upper threshold.

## Evaluation settings

The yardstick is image classification trained under a fixed target privacy budget. The
MLS-Bench task uses MNIST, Fashion-MNIST, and CIFAR-10 with one shared model/data/optimizer
harness for every candidate mechanism, scoring test accuracy and the privacy spent by the
accountant at the requested `epsilon = 3.0`, `delta = 1e-5`. Broader DP-SGD studies of the same
family also report MNIST (a two-conv 20/50-channel net with a 500-hidden classifier), CIFAR-10
and CIFAR-100 (fine-tuning a pretrained backbone with only the final layer reinitialized), and
class-imbalanced settings (an artificially unbalanced MNIST and a real imbalanced
additive-manufacturing image dataset) for fairness, typically run for 100 epochs at batch size
64 with a one-cycle learning-rate schedule and an AdamW optimizer, sweeping `epsilon in
{1, 3, 5, 8, 10}` at `delta = 1e-5`. Metrics are test accuracy for utility and, for fairness,
ROC-AUC, accuracy parity across groups, and the privacy-cost gap. Implementations use PyTorch
with Opacus.

The harness exposes per-sample gradients as a list of tensors shaped `[B, *param_shape]`. The
fixed accountant accepts a step count, a single reported noise multiplier, the sampling rate
`q = batch_size / dataset_size`, and `delta`. A mechanism therefore must return noised
batch-average gradients to the optimizer and a single scalar accounting value to the existing
`compute_epsilon` call.

## Code framework

The mechanism plugs into an otherwise fixed private-training loop. The data pipeline, model,
optimizer, loss, and the accountant `compute_epsilon` already exist; what is open is the policy
inside the gradient mechanism — how to clip, how much noise to add, and how that varies across
training — together with the scalar it must hand the fixed accountant.

```python
import torch


def compute_epsilon(steps, sigma, q, delta):
    """Existing accountant for a subsampled Gaussian mechanism: per-step cost ~ 1/sigma^2,
    summed over `steps` and converted to (epsilon, delta). Assumes a single uniform sigma."""
    pass


class DPMechanism:
    """Editable gradient mechanism inside a fixed private-training loop.
    Given per-sample gradients it must return the noised batch-average gradients, and it must
    report to the fixed accountant a scalar noise multiplier for the steps it has run."""

    def __init__(self, max_grad_norm, noise_multiplier, n_params,
                 dataset_size, batch_size, epochs, target_epsilon, target_delta):
        self.max_grad_norm = max_grad_norm          # initial per-sample clipping knob
        self.noise_multiplier = noise_multiplier    # calibrated uniform multiplier sigma
        self.n_params = n_params
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        # TODO: any state needed by the policy.

    def clip_and_noise(self, per_sample_grads, step, epoch):
        # `epoch` is the zero-based public epoch index used to choose the current public policy.
        batch_size = per_sample_grads[0].shape[0]
        flat = torch.cat([g.reshape(batch_size, -1) for g in per_sample_grads], dim=1)
        norms = flat.norm(2, dim=1)                 # per-sample L2 norms, [B]
        # TODO: clip the per-sample gradients, aggregate to a batch average, and add Gaussian
        #       noise. Return one noised average per parameter, shape [*param_shape].
        pass

    def get_effective_sigma(self, step, epoch):
        # TODO: return the scalar multiplier the fixed accountant should use for the completed
        #       steps so the reported (epsilon, delta) matches the privacy actually spent.
        pass
```

The fixed accountant only ever sees one scalar `sigma`, charges a cost that grows like
`steps / sigma^2`, and converts that to `(epsilon, delta)`. The slot to fill is the policy in
`clip_and_noise` and the scalar in `get_effective_sigma`.
