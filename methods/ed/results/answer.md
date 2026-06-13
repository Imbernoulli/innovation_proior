# Encoder-Decoder (ED) and Variational Encoder-Decoder (VED), distilled

ED is a deep fully-connected encoder-decoder that learns a sub-grid convective parameterization
`Y = f(X)` for a climate model by funneling the large-scale column state `X` through a tiny
latent bottleneck and expanding it back out to the sub-grid tendencies. The encoder narrows the
input to a 5-node code; the decoder reconstructs both the sub-grid target `Y` *and* the
large-scale input `X`, which forces the code to encode the climate state and makes it
interpretable as a nonlinear-PCA summary of the convective problem. VED is the same backbone
made probabilistic: the latent is a Gaussian `q(z|x) = N(μ, σ²)`, sampled via the
reparameterization trick, regularized toward an `N(0,I)` prior by a KL term, and trained as a
variational autoencoder adapted to map *across* domains (state → tendencies) rather than to
reconstruct its input. ED is exactly VED with the KL and the probabilistic latent removed.

## Problem it solves

Sub-grid convection (storms, shallow clouds, turbulence, radiation) is too fine to resolve in a
climate model and must be predicted from the resolved large-scale state. Deep flat emulators do
this with skill but are opaque (≈0.5M unstructured weights, no low-dimensional handle on the
drivers of convection) and, being deterministic under MSE, regress to the conditional mean and
under-represent the variability of chaotic boundary-layer convection. ED/VED keep the skill
while adding a small, inspectable latent (ED) and a distribution-aware generative latent (VED).

## Key idea

Put a small bottleneck in the middle of a cross-domain regressor. Encode `X` down through a
geometric-halving funnel to a 5-node code; decode the code back up to `O = [Y ; X-recon]`.

- The bottleneck is a **nonlinear PCA**: a few nonlinear layers compress the state to a code that
  retains what is needed to reproduce the output. A linear-activation version degenerates to
  principal-component regression; a wide-enough bottleneck approaches the flat emulator.
- The **X-reconstruction term** in the output is what makes the code interpretable: predicting
  only `Y` yields a code that tracks convective magnitude alone, whereas also reconstructing `X`
  forces the code to encode the large-scale climate state, so its coordinates separate convective
  regimes (deep/shallow/suppressed), diurnal structure, and geography.
- **ELU on the output** (not ReLU) because the targets are signed real-valued tendencies and
  fluxes; ReLU would clip all negatives. **ReLU** on the hidden layers.
- The 5-node bottleneck is the selected compression point: small enough for coordinate-level
  inspection, but not so small that every physical driver has to collapse onto one scalar.

The variational extension (VED) on top of ED:

- **Probabilistic latent.** Encoder emits a mean `μ` and a log-variance `log σ²` (log so the
  unconstrained linear head maps to a strictly positive variance). Sample with the
  **reparameterization trick** `z = μ + exp(½·log σ²)·ε`, `ε ~ N(0,1)`, so gradients flow through
  `μ` and `σ` despite the sampling.
- **Objective** `= reconstruction MSE + λ·KL`, where the KL of the diagonal-Gaussian posterior
  from the `N(0,I)` prior has the closed form `KL = −½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`
  (the `2π` constants cancel; `KL = 0` exactly when `μ=0, σ²=1`). The KL regularizes the latent
  toward a common, smooth prior geometry, which makes sampling and traversal well behaved.
- **KL annealing.** `λ` starts at 0 and ramps to 1 (here: 0 until epoch 2, then +1/5 per epoch to
  full strength by epoch 7) to avoid **posterior collapse** — if the KL is at full strength from
  the start, the encoder just emits the prior and the latent carries no information.

## Architecture (canonical hyperparameters)

- Input `X` dim 64: temperature and specific-humidity profiles (30 levels each) + solar
  insolation, surface latent and sensible heat fluxes, surface pressure.
- Output `O` dim 129 = `Y` (65: `dT/dt`, `dq/dt` on 30 levels; shortwave/longwave fluxes at top
  and surface; precipitation) concatenated with the 64-dim reconstruction of `X`. Latent 5.
- Encoder node sizes `[64, 463, 463, 232, 116, 58, 29, 5]`; decoder `[5, 29, 58, 116, 232, 463,
  463, 129]`. Hidden activations ReLU; decoder output ELU; (VED) `μ`, `log σ²` linear.
- Optimizer Adam, reconstruction loss MSE, scheduler value `0.00074594` divided by 5 every 7
  epochs, 40 epochs, batch size 714. The scripts compile Adam with `lr=1e-4` and then let the
  scheduler set the effective learning rate. Input normalization: per-variable per-level mean
  subtracted, divided by range.
  Output normalization: each field rescaled to a common order of magnitude (profiles by a
  near-surface long-term std; `dT/dt` by the std near 845 hPa, the top of the boundary layer).
- ED ≈ 832k parameters; VED encoder/decoder ≈ 388k / 418k.

## Relation to prior methods

- **Flat deep emulator** = the wide-bottleneck limit (latent width → input dim; no compression).
- **PCA / linear reduced model (PCR)** = ED with all activations linear (latent not orthogonal).
- **ED** = VED with the probabilistic latent and the KL term removed; loss = reconstruction only.

## Working code — canonical (Keras functional)

```python
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from tensorflow.keras import backend as K
import tensorflow as tf

original_dim_input  = 64           # large-scale state X
original_dim_output = 65 + 64      # O = [Y ; X-reconstruction] = 129
intermediate_dim = 463            # first/last hidden width; the funnel halves from here
latent_dim = 5                    # bottleneck
epochs = 40


def encoder_backbone(inputs):
    h = Dense(intermediate_dim, activation='relu')(inputs)
    h = Dense(intermediate_dim, activation='relu')(h)
    h = Dense(int(np.round(intermediate_dim / 2)),  activation='relu')(h)  # 232
    h = Dense(int(np.round(intermediate_dim / 4)),  activation='relu')(h)  # 116
    h = Dense(int(np.round(intermediate_dim / 8)),  activation='relu')(h)  # 58
    h = Dense(int(np.round(intermediate_dim / 16)), activation='relu')(h)  # 29
    return h


def build_decoder():
    dz = Input(shape=(latent_dim,), name='decoder_input')
    y = Dense(int(np.round(intermediate_dim / 16)), activation='relu')(dz)  # 29
    y = Dense(int(np.round(intermediate_dim / 8)),  activation='relu')(y)   # 58
    y = Dense(int(np.round(intermediate_dim / 4)),  activation='relu')(y)   # 116
    y = Dense(int(np.round(intermediate_dim / 2)),  activation='relu')(y)   # 232
    y = Dense(intermediate_dim, activation='relu')(y)
    y = Dense(intermediate_dim, activation='relu')(y)
    out = Dense(original_dim_output, activation='elu')(y)   # signed real outputs -> ELU
    return Model(dz, out, name='decoder')


def schedule(epoch):                                       # lr /5 every 7 epochs
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


# ============================ ED (deterministic) ============================
inputs = Input(shape=(original_dim_input,), name='encoder_input')
h = encoder_backbone(inputs)
encoder_out = Dense(latent_dim, name='encoder_output')(h)  # linear projection to 5
encoder = Model(inputs, encoder_out, name='encoder')
decoder = build_decoder()
ED = Model(inputs, decoder(encoder(inputs)))
ED.compile(tf.keras.optimizers.Adam(lr=1e-4), loss=mse, metrics=['mse'])   # reconstruction only
ED.fit(train_gen, validation_data=(val_gen, None), epochs=epochs, shuffle=False,
       callbacks=[LearningRateScheduler(schedule)])

# ============================ VED (variational) ============================
def sampling(args):                                        # z = mu + exp(0.5*log_var) * eps
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))  # eps ~ N(0, I)
    return z_mean + K.exp(0.5 * z_log_var) * eps

inputs = Input(shape=(original_dim_input,), name='encoder_input')
h = encoder_backbone(inputs)
z_mean    = Dense(latent_dim, name='z_mean')(h)            # linear head: posterior mean
z_log_var = Dense(latent_dim, name='z_log_var')(h)         # linear head: posterior log-variance
z = Lambda(sampling, name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = build_decoder()
emul_outputs = decoder(encoder(inputs)[2])

# closed-form Gaussian KL:  KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
weight = K.variable(0.0)                                   # KL annealing coefficient lambda

VED = Model(inputs, emul_outputs)
VED.add_loss(K.mean(kl_loss * weight))                     # total = mse reconstruction + lambda*KL
VED.add_metric(kl_loss, name='kl_loss', aggregation='mean')

klstart, kl_annealtime = 2, 5                              # lambda: 0 until epoch 2, then +1/5 to 1
class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight

    def on_epoch_end(self, epoch, logs=None):
        if epoch > klstart:
            new_weight = min(K.get_value(self.weight) + 1.0 / kl_annealtime, 1.0)
            K.set_value(self.weight, new_weight)

encoder.load_weights('./saved_models/VAE_climate_encoding/encoder_JAS.h5')
VED.compile(tf.keras.optimizers.Adam(lr=1e-4), loss=mse, metrics=['mse'])  # mse = reconstruction
VED.fit(train_gen, validation_data=(val_gen, None), epochs=epochs, shuffle=False,
        callbacks=[LearningRateScheduler(schedule), AnnealingCallback(weight)])
```
