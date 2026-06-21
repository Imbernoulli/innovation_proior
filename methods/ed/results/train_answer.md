The task is to replace the hand-built convective parameterization in a global climate model with a learned map from the resolved large-scale column state to the unresolved sub-grid tendencies. Existing deep feed-forward emulators already fit this map well, but they leave two persistent problems. First, they are opaque: a nine-layer, 256-unit network stores its knowledge in roughly half a million unstructured weights, so asking which large-scale conditions drive shallow-cloud heating requires an expensive, fragile post-hoc attribution method. Second, because they are trained deterministically with mean-squared error, they regress every prediction toward the conditional mean. The boundary-layer and shallow-cloud tendencies are genuinely stochastic for a given coarse state, so the emulator smooths away real variability exactly where the target is most uncertain. Principal-component regression would give an interpretable low-dimensional summary, but it is linear and the convection map is strongly nonlinear, so it underfits badly. A plain variational autoencoder learns a low-dimensional code, yet it is trained to reconstruct its own input, whereas the parameterization problem is a cross-domain map from large-scale state to sub-grid tendencies. The right model should combine a small, inspectable latent with the nonlinear predictive power of a deep regressor.

I propose an Encoder-Decoder, ED, with a Variational Encoder-Decoder, VED, as its probabilistic extension. ED funnels the large-scale state through a geometric-halving fully-connected bottleneck to a five-node latent code and then expands the code back out to the target. The output is not just the sub-grid tendencies but the tendencies concatenated with a reconstruction of the large-scale input itself. Reconstructing both forces the latent to encode the climate state alongside the convective response, which makes the coordinates interpretable: they separate deep, shallow, and suppressed convection, diurnal structure, and geographic origin. A model trained to predict only the tendencies would learn a code that tracks convective magnitude alone and would discard the large-scale drivers. The architecture is symmetric: the encoder descends 64 -> 463 -> 463 -> 232 -> 116 -> 58 -> 29 -> 5, and the decoder mirrors those widths back up to 129 outputs. Hidden layers use ReLU; the final output uses ELU so that signed tendencies, such as cooling or drying, are not clipped to zero. When the bottleneck is made wide and the activation linear, the construction degenerates to the flat deep emulator or to principal-component regression, so ED sits naturally between those extremes.

VED adds a variational latent on top of the same backbone. Instead of a single deterministic vector, the encoder emits a mean and a log-variance for each of the five latent coordinates, and the code is sampled from that diagonal Gaussian via the reparameterization trick. The objective is reconstruction mean-squared error plus the KL divergence between the posterior and a standard normal prior, with the KL weight annealed from zero so the encoder learns an informative code before the prior regularizer is turned on. This avoids posterior collapse, where the encoder would simply emit the prior and discard the input. VED therefore represents a distribution over outcomes for a given large-scale state rather than a single conditional-mean point, addressing the lost-variability problem, while ED is simply VED with the probabilistic latent and KL term removed. Both models inherit the same input and output normalization, the same Adam optimizer, and the same stepped learning-rate schedule used by the flat emulator, so any difference in behavior can be attributed to the architecture itself.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.callbacks import LearningRateScheduler, Callback

original_dim_input = 64        # X: q(p), T(p) profiles + PS, SOLIN, SHFLX, LHFLX
original_dim_output = 65 + 64  # O = [Y tendencies ; X reconstruction]
intermediate_dim = 463
latent_dim = 5
epochs = 40


def encoder_backbone(inputs):
    h = Dense(intermediate_dim, activation='relu')(inputs)
    h = Dense(intermediate_dim, activation='relu')(h)
    h = Dense(int(np.round(intermediate_dim / 2)), activation='relu')(h)   # 232
    h = Dense(int(np.round(intermediate_dim / 4)), activation='relu')(h)   # 116
    h = Dense(int(np.round(intermediate_dim / 8)), activation='relu')(h)   # 58
    h = Dense(int(np.round(intermediate_dim / 16)), activation='relu')(h)  # 29
    return h


def build_decoder():
    dz = Input(shape=(latent_dim,), name='decoder_input')
    y = Dense(int(np.round(intermediate_dim / 16)), activation='relu')(dz)  # 29
    y = Dense(int(np.round(intermediate_dim / 8)), activation='relu')(y)    # 58
    y = Dense(int(np.round(intermediate_dim / 4)), activation='relu')(y)    # 116
    y = Dense(int(np.round(intermediate_dim / 2)), activation='relu')(y)    # 232
    y = Dense(intermediate_dim, activation='relu')(y)
    y = Dense(intermediate_dim, activation='relu')(y)
    out = Dense(original_dim_output, activation='elu')(y)
    return Model(dz, out, name='decoder')


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


# Deterministic Encoder-Decoder (ED)
inputs = Input(shape=(original_dim_input,), name='encoder_input')
h = encoder_backbone(inputs)
encoder_out = Dense(latent_dim, name='encoder_output')(h)
encoder = Model(inputs, encoder_out, name='encoder')
decoder = build_decoder()
ED = Model(inputs, decoder(encoder(inputs)), name='ED')
ED.compile(tf.keras.optimizers.Adam(lr=1e-4), loss=mse, metrics=['mse'])

# Variational Encoder-Decoder (VED)
def sampling(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], K.int_shape(z_mean)[1]))
    return z_mean + K.exp(0.5 * z_log_var) * eps

inputs = Input(shape=(original_dim_input,), name='encoder_input')
h = encoder_backbone(inputs)
z_mean = Dense(latent_dim, name='z_mean')(h)
z_log_var = Dense(latent_dim, name='z_log_var')(h)
z = Lambda(sampling, name='z')([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = build_decoder()
emul_outputs = decoder(encoder(inputs)[2])

kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
weight = K.variable(0.0)

VED = Model(inputs, emul_outputs, name='VED')
VED.add_loss(K.mean(kl_loss * weight))
VED.add_metric(kl_loss, name='kl_loss', aggregation='mean')

klstart, kl_annealtime = 2, 5

class AnnealingCallback(Callback):
    def __init__(self, weight):
        self.weight = weight

    def on_epoch_end(self, epoch, logs=None):
        if epoch > klstart:
            new_weight = min(K.get_value(self.weight) + 1.0 / kl_annealtime, 1.0)
            K.set_value(self.weight, new_weight)

VED.compile(tf.keras.optimizers.Adam(lr=1e-4), loss=mse, metrics=['mse'])

# Example fit calls (train_gen and val_gen must be provided by the data pipeline):
# ED.fit(train_gen, validation_data=(val_gen, None), epochs=epochs, shuffle=False,
#        callbacks=[LearningRateScheduler(schedule)])
# VED.fit(train_gen, validation_data=(val_gen, None), epochs=epochs, shuffle=False,
#         callbacks=[LearningRateScheduler(schedule), AnnealingCallback(weight)])
```
