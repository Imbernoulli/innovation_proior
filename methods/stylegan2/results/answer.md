# StyleGAN2

## Problem

A style-based GAN generator produces state-of-the-art high-resolution images but carries two
families of characteristic artifacts: stereotyped "water-droplet" blobs present in essentially
all intermediate feature maps, and location-locked / "phase" details that stick to fixed pixel
positions. StyleGAN2 diagnoses both, fixes them without losing style-based controllability, and
adds a regularizer that makes the latent→image mapping smooth (improving perceived quality and
making the generator invertible).

## Key ideas

1. **Weight demodulation** replaces AdaIN. The blob is the generator exploiting AdaIN's
   per-feature-map instance normalization — it plants a dominant spike so that dividing by the
   map's measured standard deviation rescales everything else as it likes, smuggling relative
   feature magnitudes past the normalization. The fix removes the *data-dependence*: normalize by
   the *expected* output standard deviation computed analytically from the weights, so there is no
   measured statistic to game, while the style scale is preserved (style mixing still works).

2. **Path-length regularization** encourages a fixed-size step in W to produce a fixed-magnitude
   image change in every direction, i.e. a well-conditioned (locally isometric) mapping. This
   correlates with perceived image quality (which FID/P&R miss) and makes inversion reliable.

3. **Lazy regularization** evaluates the (slowly-changing) regularizers once every k steps in a
   separate pass, cutting compute and memory.

4. **No progressive growing.** A skip-connection generator and residual discriminator preserve the
   coarse-to-fine training behavior without ever changing topology, removing the phase artifacts;
   doubling high-resolution feature maps fixes an observed capacity shortfall.

## Final method

**Weight demodulation.** Fold the per-input-map style scale into the conv weights and divide each
output map's weights by their L2 norm:

- Modulate:  `w'_{ijk} = s_i · w_{ijk}`
- Output std under i.i.d. unit-variance inputs:  `σ_j = sqrt( Σ_{i,k} (w'_{ijk})² )`
- Demodulate:  `w''_{ijk} = w'_{ijk} / sqrt( Σ_{i,k} (w'_{ijk})² + ε )`

(i = input map, j = output map, k = kernel footprint.) Applied to all convolutions except the
to-RGB outputs, which are modulated but not demodulated. Activations are scaled to preserve unit
variance; per-sample weights are run via grouped convolution. Bias and noise are moved outside the
style block.

**Path-length regularizer.**

```
L_pl = E_{w, y~N(0,I)} ( ‖J_wᵀ y‖₂ − a )²,     J_wᵀ y = ∇_w ( g(w) · y )
```

with y a random image (∝ N(0,I)), `a` the exponential moving average of the observed lengths
(β=0.99, init 0), and weight `γ_pl = ln 2 / ( r² (ln r − ln 2) )` for output resolution r. In high
dimension this prior is minimized when J_w is orthogonal up to a global scale (equal singular
values everywhere), so the mapping locally preserves lengths.

**Lazy regularization.** Step each regularizer every k iterations (k=16 for D's R1, k=8 for G's
path length), sharing Adam state; correct hyperparameters with c = k/(k+1): `λ′ = cλ`,
`β₁′ = β₁^c`, `β₂′ = β₂^c`, and multiply the regularizer by k.

**Architecture.** Skip-connection generator (per-resolution to-RGB outputs upsampled and summed),
residual discriminator, no progressive growing; residual merges scaled by 1/√2 to cancel variance
doubling; bilinear up/downsampling; doubled feature maps at resolutions 64²–1024².

## Code

```python
import numpy as np
import tensorflow as tf
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act

def get_weight(shape, gain=1, use_wscale=True, lrmul=1, weight_var='weight'):
    fan_in = np.prod(shape[:-1]); he_std = gain / np.sqrt(fan_in)
    init_std, runtime_coef = (1.0/lrmul, he_std*lrmul) if use_wscale else (he_std/lrmul, lrmul)
    w = tf.get_variable(weight_var, shape=shape,
                        initializer=tf.initializers.random_normal(0, init_std))
    return w * runtime_coef

def dense_layer(x, fmaps, weight_var='weight', **kw):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    return tf.matmul(x, tf.cast(get_weight([x.shape[1].value, fmaps], weight_var=weight_var, **kw), x.dtype))

def apply_bias_act(x, act='linear', gain=None, lrmul=1, bias_var='bias'):
    b = tf.get_variable(bias_var, shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, gain=gain)

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, **kw):
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kw)
    if up:   return upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    if down: return conv_downsample_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    return tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')

# ---- Weight demodulation: the entire style block as one convolution -------------------
def modulated_conv2d_layer(x, w_latent, fmaps, kernel, up=False, demodulate=True,
                           resample_kernel=None, fused_modconv=True):
    w  = get_weight([kernel, kernel, x.shape[1].value, fmaps])
    ww = w[np.newaxis]                                                  # [B,k,k,I,O]
    s  = dense_layer(w_latent, fmaps=x.shape[1].value, weight_var='mod_weight')
    s  = apply_bias_act(s, bias_var='mod_bias') + 1                     # [B,I], affine bias init 1
    ww *= tf.cast(s[:, np.newaxis, np.newaxis, :, np.newaxis], w.dtype) # w'_ijk = s_i w_ijk
    if demodulate:
        d = tf.rsqrt(tf.reduce_sum(tf.square(ww), axis=[1,2,3]) + 1e-8) # [B,O] = 1/sigma_j
        ww *= d[:, np.newaxis, np.newaxis, np.newaxis, :]              # w''_ijk = w'_ijk / sigma_j
    if fused_modconv:                                                  # per-sample weights via groups
        x = tf.reshape(x, [1, -1, x.shape[2], x.shape[3]])
        w = tf.reshape(tf.transpose(ww, [1,2,3,0,4]),
                       [ww.shape[1], ww.shape[2], ww.shape[3], -1])
    else:
        x *= tf.cast(s[:, :, np.newaxis, np.newaxis], x.dtype)
    if up: x = upsample_conv_2d(x, tf.cast(w, x.dtype), data_format='NCHW', k=resample_kernel)
    else:  x = tf.nn.conv2d(x, tf.cast(w, x.dtype), data_format='NCHW', strides=[1,1,1,1], padding='SAME')
    if fused_modconv:   x = tf.reshape(x, [-1, fmaps, x.shape[2], x.shape[3]])
    elif demodulate:    x *= tf.cast(d[:, :, np.newaxis, np.newaxis], x.dtype)
    return x

# ---- Skip generator (no progressive growing) ------------------------------------------
def G_synthesis(dlatents_in, resolution=1024, num_channels=3, resample_kernel=[1,3,3,1]):
    res_log2 = int(np.log2(resolution))
    nf = lambda stage: int(np.clip(16<<10 >> stage, 1, 512))
    def layer(x, idx, fmaps, kernel, up=False):
        x = modulated_conv2d_layer(x, dlatents_in[:, idx], fmaps, kernel, up=up,
                                   resample_kernel=resample_kernel)
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]], dtype=x.dtype)
        x += noise * tf.get_variable('noise_strength', shape=[], initializer=tf.initializers.zeros())
        return apply_bias_act(x, act='lrelu')
    def torgb(x, y, res):
        t = apply_bias_act(modulated_conv2d_layer(x, dlatents_in[:, res*2-3],
                           fmaps=num_channels, kernel=1, demodulate=False))
        return t if y is None else y + t
    def block(x, res):
        x = layer(x, res*2-5, nf(res-1), 3, up=True)
        return layer(x, res*2-4, nf(res-1), 3)
    x = tf.tile(tf.cast(tf.get_variable('const', shape=[1, nf(1), 4, 4],
                initializer=tf.initializers.random_normal()), dlatents_in.dtype),
                [tf.shape(dlatents_in)[0], 1, 1, 1])
    x = layer(x, 0, nf(1), 3)
    y = torgb(x, None, 2)
    for res in range(3, res_log2 + 1):
        x = block(x, res)
        y = upsample_2d(y, k=resample_kernel)
        y = torgb(x, y, res)
    return tf.identity(y, name='images_out')

# ---- Residual discriminator -----------------------------------------------------------
def D_net(images_in, resolution=1024, num_channels=3, resample_kernel=[1,3,3,1],
          mbstd_group_size=4):
    res_log2 = int(np.log2(resolution))
    nf = lambda stage: int(np.clip(16<<10 >> stage, 1, 512))
    def block(x, res):
        t = x
        x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-1), kernel=3), act='lrelu')
        x = apply_bias_act(conv2d_layer(x, fmaps=nf(res-2), kernel=3, down=True,
                           resample_kernel=resample_kernel), act='lrelu')
        t = conv2d_layer(t, fmaps=nf(res-2), kernel=1, down=True, resample_kernel=resample_kernel)
        return (x + t) * (1 / np.sqrt(2))                      # cancel variance doubling
    x = apply_bias_act(conv2d_layer(images_in, fmaps=nf(res_log2-1), kernel=1), act='lrelu')
    for res in range(res_log2, 2, -1):
        x = block(x, res)
    x = minibatch_stddev_layer(x, mbstd_group_size)
    x = apply_bias_act(conv2d_layer(x, fmaps=nf(1), kernel=3), act='lrelu')
    x = apply_bias_act(dense_layer(x, fmaps=nf(0)), act='lrelu')
    return tf.identity(apply_bias_act(dense_layer(x, fmaps=1)), name='scores_out')

# ---- G loss + path-length regularizer -------------------------------------------------
def G_logistic_ns_pathreg(G, D, training_set, minibatch_size,
                          pl_minibatch_shrink=2, pl_decay=0.01, pl_weight=2.0):
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels  = training_set.get_random_labels_tf(minibatch_size)
    fake, dlat = G.get_output_for(latents, labels, is_training=True, return_dlatents=True)
    loss = tf.nn.softplus(-D.get_output_for(fake, labels, is_training=True))

    pl_n = minibatch_size // pl_minibatch_shrink
    fake, dlat = G.get_output_for(tf.random_normal([pl_n] + G.input_shapes[0][1:]),
                                  training_set.get_random_labels_tf(pl_n),
                                  is_training=True, return_dlatents=True)
    y = tf.random_normal(tf.shape(fake)) / np.sqrt(np.prod(G.output_shape[2:]))     # /sqrt(#pixels)
    pl_grads = tf.gradients(tf.reduce_sum(fake * y), [dlat])[0]                     # J^T y
    pl_lengths = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(pl_grads), axis=2), axis=1))
    pl_mean_var = tf.Variable(0.0, trainable=False, name='pl_mean')                # target a (EMA)
    pl_mean = pl_mean_var + pl_decay * (tf.reduce_mean(pl_lengths) - pl_mean_var)
    with tf.control_dependencies([tf.assign(pl_mean_var, pl_mean)]):
        reg = tf.square(pl_lengths - pl_mean) * pl_weight                          # (||J^T y|| - a)^2
    return loss, reg

# ---- R1 for the discriminator ---------------------------------------------------------
def D_logistic_r1(G, D, training_set, minibatch_size, reals, labels, gamma=10.0):
    fake = G.get_output_for(tf.random_normal([minibatch_size] + G.input_shapes[0][1:]),
                            labels, is_training=True)
    rs = D.get_output_for(reals, labels, is_training=True)
    fs = D.get_output_for(fake,  labels, is_training=True)
    loss = tf.nn.softplus(fs) + tf.nn.softplus(-rs)
    real_grads = tf.gradients(tf.reduce_sum(rs), [reals])[0]
    reg = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3]) * (gamma * 0.5)
    return loss, reg

# Lazy regularization (training loop): step `reg` every k iters in a separate pass
# sharing Adam state; with c = k/(k+1): lr *= c, beta1 **= c, beta2 **= c, reg *= k.
# k = 8 (G, path length), k = 16 (D, R1).
```
