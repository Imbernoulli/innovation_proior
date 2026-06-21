# Context: machine-learned convective parameterization for climate models (circa 2018–2021)

## Research question

A global climate model (a GCM, here the Community Atmosphere Model, CAM) divides the atmosphere
into columns ~100–300 km wide and steps them forward every 20–30 minutes. The processes that
actually make weather and rain — deep convective storms, shallow clouds, turbulence, radiative
heating — happen on scales of a few kilometres, far below a grid cell, and are not resolved
directly. Their net effect on the resolved column is supplied by a hand-built
"parameterization", a physically-motivated approximation.

A superparameterized model (SPCAM) instead embeds a small cloud-resolving model (CRM) inside
every column, so convection is resolved explicitly. Machine learning exploits this: run SPCAM
once, record for every column and timestep the large-scale state going in and the sub-grid
tendencies coming out, and learn the map. Concretely the learning problem is a nonlinear
regression `Y = f(X)` from a large-scale column state `X` (vertical profiles of temperature and
humidity plus a few surface/forcing scalars) to the sub-grid tendencies `Y` (vertical profiles
of diabatic heating `dT/dt` and moistening `dq/dt`, radiative fluxes at the model top and
surface, and precipitation rate). The dimensions are modest — a few dozen inputs, a few dozen
outputs.

The question here is how to learn this state-to-tendency map `f` in a form that, alongside
reproduction skill, offers a low-dimensional summary one can inspect to relate large-scale
conditions to kinds of convection, and that represents the natural variability of convection.

## Background

By this time it is established that deep feed-forward neural networks can learn the SPCAM
convective map. Gentine, Pritchard, Rasp, Reinaudi & Yacalis (2018) first showed, on an
aquaplanet, that a deep fully-connected net reproduces SPCAM's convective heating and
moistening with good offline skill — strongest in the tropics and along the mid-latitude storm
tracks, weakest in the planetary boundary layer. Rasp, Pritchard & Gentine (2018) took the next
step: they trained such a net and then *coupled it back into the GCM*, replacing the physics
package, and obtained stable multi-year prognostic runs that reproduced the mean climate and
much of its variability. These two results frame everything that follows — the input/output
variable choice, the normalization recipe, the depth, and the optimizer all become the
inherited precedent.

Two diagnostic facts about these deterministic emulators are on record. First, the networks
that work are large and flat: nine fully-connected layers of 256 units, on the order of half a
million weights. The representation the net learns is distributed across hundreds of thousands
of weights; reading off how a large-scale variable influences a predicted tendency is done with
a saliency / attribution method applied to the trained net. Second, a deterministic net trained
to minimize mean squared error against targets that are partly chaotic predicts, sample by
sample, the conditional mean: when the boundary-layer tendency at a given large-scale state is a
spread of outcomes (the sub-grid CRM is itself turbulent and only weakly determined by the
coarse state), the MSE-optimal point prediction is the average of that spread. The reported
behaviour matches this — the deterministic emulator's heating and moistening fields are
*smoother* than SPCAM's, with reduced variance at the shallow-cloud level (~900 hPa) and in the
boundary layer. This is read in the literature as an under-representation of the *stochastic*
variability of shallow and deep convection.

The conceptual machinery available:

- **Dimensionality reduction.** The classical tool is principal component analysis: project the
  data onto the leading directions of variance. PCA is linear, orthogonal, and interpretable.
  There is also a long tradition in atmospheric science of *reduced-complexity* models —
  multi-cloud models, the quasi-equilibrium tropical circulation model — built on the premise
  that the effective number of degrees of freedom governing convection is small, far smaller
  than the raw state vector.

- **Latent-variable generative models.** A variational autoencoder (Kingma & Welling, 2014)
  pairs a recognition network (encoder) `q_φ(z|x)`, which maps a datapoint to a distribution
  over a low-dimensional latent `z`, with a generative network (decoder) `p_θ(x|z)`. It is
  trained by maximizing the evidence lower bound
  `L = E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))`, the first term a reconstruction
  term and the second a regularizer pulling the latent posterior toward a fixed prior `p(z) =
  N(0,I)`. Two pieces of its machinery are standard knowledge: the **reparameterization trick**
  — write a Gaussian sample as `z = μ + σ·ε` with `ε ~ N(0,1)`, so the randomness is pushed
  into an input-independent noise variable and gradients flow through `μ` and `σ` — and the
  **closed-form Gaussian KL**: for a diagonal-Gaussian posterior `N(μ, σ²)` against an `N(0,I)`
  prior, `−D_KL = ½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`. A known failure mode of training such a
  model is *posterior collapse*: if the KL term is given full weight from the start, the encoder
  is driven to ignore the data and emit the prior, and the latent carries no information; a
  common remedy is to anneal the KL weight up from zero over the first few epochs (Alemi et
  al., 2018).

- **Activations.** ReLU, `max(0,x)`, is the deep-net default; LeakyReLU, `max(0.3x, x)`, was the
  activation that gave the lowest training loss for the flat emulator. The exponential linear
  unit (Clevert, Unterthiner & Hochreiter, 2016), `ELU(x) = x` for `x>0` and `α(e^x − 1)` for
  `x≤0` (with `α=1`), is smooth at the origin, admits negative outputs that push mean
  activations toward zero, saturates gently to `−α` for very negative inputs, and avoids the
  dying-ReLU problem.

The simulation that supplies the data is a two-year aquaplanet run of SPCAM v3.0 in the
configuration of Pritchard et al. (2014): a coarse GCM (~300 km, 30-minute step, 30 vertical
levels) with, inside each column, eight nested 2-D CRM columns of 4 km grid spacing in which
deep convection is resolved every 20 s with a 1.5-order turbulence closure and one-moment
microphysics. Sea-surface temperatures are fixed and zonally symmetric, the solar forcing is
perpetual austral summer with a diurnal cycle. This setup reproduces a realistic ITCZ, tropical
wave spectra, and an MJO-like signal.

## Baselines

These are the prior methods a new emulator would be measured against.

**Deep fully-connected emulator (Gentine et al. 2018; Rasp, Pritchard & Gentine 2018).** The
established approach. Stack the large-scale state into an input vector `x = [T(z), Q(z), V(z),
P_s, S_in, H, E]` of length 94 and the sub-grid tendencies into `y = [ΔT_phy, ΔQ_phy, F_rad, P]`
of length 65; learn `ŷ = N(x)` with a network of nine fully-connected layers, 256 units each
(~567k parameters), LeakyReLU activations, trained with Adam under a mean-squared-error loss
(18 epochs, batch 1024, learning rate `1e-3` divided by five every three epochs). Inputs are
normalized by subtracting each element's mean and dividing by the larger of its range and its
across-levels standard deviation — chosen so that upper-level humidity, which is tiny, is not
divided by a near-zero number. Outputs are rescaled so heating, moistening, fluxes and
precipitation sit at a common order of magnitude, because the *magnitude* of each output sets
its weight in the quadratic loss. Depth matters for more than fit: shallow nets (one or two
hidden layers) produced unstable modes and unrealistic artifacts when coupled back into the
GCM, and four layers was the minimum for a good prognostic run.

**Principal component analysis / linear dimensionality reduction.** Project `X` onto its leading
variance directions and regress `Y` on the scores (principal-component regression). This yields
an orthogonal, interpretable low-dimensional set of components and is the textbook way to
summarize a high-dimensional state, giving the dominant *linear* modes of the state.

**Variational autoencoder (Kingma & Welling 2014).** A latent-variable generative model that
learns a smooth, regularized low-dimensional code: encoder `q_φ(z|x)`, decoder `p_θ(x|z)`,
trained on the evidence lower bound `L = E_{q_φ}[log p_θ(x|z)] − D_KL(q_φ(z|x)‖p(z))` with the
reparameterization trick and the closed-form Gaussian KL above. In the atmospheric sciences it
has so far been used to *compress and classify* — to find phases of the polar vortex, to detect
convective regimes from resolved updraft profiles — i.e. as an unsupervised analysis of one
field that reconstructs its own input.

## Evaluation settings

The natural yardsticks already in use for SPCAM emulators:

- **Offline reproduction skill.** Train on a shuffled multi-month block of the SPCAM run,
  validate and test on disjoint later months (here ~3 months each, space-time shuffled for
  training, unshuffled for validation/test, with the test year having no overlap with training).
  The primary metric is mean squared error of the predicted sub-grid fields under a fixed output
  normalization, reported on train/validation/test; the complementary metric is the coefficient
  of determination `R² = 1 − MSE/Var`, computed per field per grid cell and mapped or
  averaged. Heating and moistening tendencies are examined level-by-level, with particular
  attention to the lower-troposphere (~700 hPa) fields where neural emulators are known to be
  weakest.
- **Variability / spectra.** Whether the emulator preserves the *variance* of the tendencies
  (not just their mean), and whether it preserves the spatio-temporal variability of tropical
  convection — diagnosed with a Wheeler-Kiladis wavenumber-frequency spectrum of outgoing
  longwave radiation in the tropics (15° N–15° S), checking that Kelvin waves and the MJO
  signal are not damped or distorted.
- **Latent-space interpretability.** For a model with a low-dimensional code: whether the latent
  separates physically meaningful convective regimes (deep / shallow / suppressed convection),
  diurnal day/night structure, and geographic origin (tropics vs poles), assessed by projecting
  the latent (e.g. via a 2-D PCA of it for visualization) and computing conditional averages of
  physical variables — precipitation, outgoing longwave radiation, solar insolation, surface
  temperature — across the latent manifold, and by traversing a single latent coordinate and
  reading the generated profiles.

## Code framework

The new emulator plugs into the SPCAM data pipeline and training loop that already exist for the
flat deterministic baseline. The data generator yields normalized batches of the large-scale
state `x` and the sub-grid target; the input normalization (per-variable, per-level mean
subtraction and division by range) and the output normalization (rescaling each field to a
common order of magnitude) are fixed by precedent; the optimizer (Adam), the squared-error
reconstruction objective, the learning-rate schedule and the batching are inherited. The
architecture of the regressor itself — the network that consumes the normalized state and
produces the predicted target — is the thing to be designed; the substrate is the generic
cross-domain-regression machinery, with one empty slot where the model goes.

```python
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf

from cbrain.data_generator import DataGenerator


INPUT_DIM = 64       # q(p), T(p), PS, SOLIN, SHFLX, LHFLX after normalization
SUBGRID_DIM = 65     # PHQ, TPHYSTND, FSNT, FSNS, FLNT, FLNS, PRECT
BATCH_SIZE = 714
EPOCHS = 40

in_vars = ["QBP", "TBP", "PS", "SOLIN", "SHFLX", "LHFLX"]
subgrid_vars = ["PHQ", "TPHYSTND", "FSNT", "FSNS", "FLNT", "FLNS", "PRECT"]


def make_generator(data_fn, output_vars, output_transform, shuffle):
    return DataGenerator(
        data_fn=data_fn,
        input_vars=in_vars,
        output_vars=output_vars,
        norm_fn="../preprocessed_data/000_norm_1_month.nc",
        input_transform=("mean", "maxrs"),
        output_transform=output_transform,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def build_emulator(input_dim=INPUT_DIM, output_dim=SUBGRID_DIM):
    inputs = Input(shape=(input_dim,), name="emulator_input")
    # TODO: the architecture we will design goes here.
    raise NotImplementedError


def schedule(epoch):
    if epoch < 7:
        return 0.00074594
    if epoch < 14:
        return 0.00074594 / 5
    if epoch < 21:
        return 0.00074594 / 25
    if epoch < 28:
        return 0.00074594 / 125
    if epoch < 35:
        return 0.00074594 / 625
    return 0.00074594 / 3125


def train(model, train_gen, val_gen):
    model.compile(tf.keras.optimizers.Adam(lr=1e-4), loss=mse, metrics=["mse"])
    model.fit(
        train_gen,
        validation_data=(val_gen, None),
        epochs=EPOCHS,
        shuffle=False,
        callbacks=[LearningRateScheduler(schedule, verbose=1)],
    )
```

The data pipeline supplies normalized `(x, target)` pairs and the loop optimizes a squared-error
objective; the single empty slot is the model architecture itself, and the loss is left open in
case the design needs to add a term to it.
