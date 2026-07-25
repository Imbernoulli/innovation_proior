I propose the canonical method name **StyleGAN2**. It denotes a redesigned style-based generator that removes the characteristic artifacts of its predecessor while preserving, and even improving, the controllability that made style-based generation attractive.

The starting point is StyleGAN, which already produced compelling high-resolution images, but two families of artifacts kept betraying the outputs. The first is a stereotyped water-droplet blob that appears in essentially every intermediate feature map from 64×64 upward; the second is a location-locked or phase artifact, where details such as teeth or eyes stick to preferred pixel positions and jump discretely rather than gliding smoothly. A successful redesign has to eliminate both without sacrificing style mixing, and it should do so without making training appreciably slower.

The blob is the key diagnostic. It is too consistent to be noise, and the discriminator ought to be able to penalize it if it were merely a defect. That the blob persists, and that removing it sometimes severely corrupts the image, suggests the generator relies on it. The mechanism that creates it is adaptive instance normalization, AdaIN. AdaIN normalizes each feature map independently by its own mean and standard deviation, computed from the actual contents of that map, and then re-applies a style-driven scale and bias. The normalization is what makes the style scale dominate each map and thereby enables style mixing. But because the divisor is data-dependent, the generator can game it: by planting a single dominant spike in a feature map, it makes the measured standard deviation essentially the size of that spike, so dividing by it rescales the rest of the map by a chosen factor. The spike becomes a private channel for relative feature magnitudes, and that spike is the blob.

Removing normalization entirely does make the blob vanish, but it also removes the per-sample scale control that style mixing requires; later layers receive activations at wildly inconsistent magnitudes and cannot be controlled by feeding different styles to different layers. The right move is therefore to keep the effect of normalization while removing its data-dependence. Instead of measuring the standard deviation of each actual output map, I predict the standard deviation that the output map would have under a unit-variance-input assumption, and I do so analytically from the weights and the style.

The derivation is straightforward. Modulation multiplies input feature map i by the style scalar s_i. I can fold that scaling into the convolution weights: w'_{ijk} = s_i · w_{ijk}, where i is an input map, j is an output map, and k indexes the kernel footprint. If the inputs are independent with unit variance, the variance of output map j is the sum of squared weights feeding it, so the predicted standard deviation is σ_j = sqrt( Σ_{i,k} (w'_{ijk})² ). Demodulation simply divides by that predicted value, w''_{ijk} = w'_{ijk} / sqrt( Σ_{i,k} (w'_{ijk})² + ε ). The whole style block, modulation, convolution, and normalization, collapses into a single convolution with per-sample adjusted weights. Because the divisor is a deterministic function of the style and the weights, there is no measured content statistic for the generator to game, so the blob loses its purpose and disappears. At the same time, the style scale s is still fully present in w', so scale-specific control and style mixing remain intact.

A few details keep the system calibrated. The to-RGB output layers should be modulated by the style but not demodulated, because they produce an image rather than features feeding another layer. Bias and noise are moved outside the style block so their effect does not depend on the current style magnitude. The activation function is scaled so that leaky ReLU preserves unit variance, which keeps the demodulation assumptions valid as the signal flows deeper. Because the effective weights are different for every sample, I run them through grouped convolution, reshaping the batch dimension into groups so that the convolution sees one sample per group; the reshape is a view and adds no copies.

The second major addition is path-length regularization. Standard metrics such as FID and precision/recall have a known blind spot: two generators can score identically while one looks clearly better to humans. The difference tends to track perceptual path length, which measures how much the generated image changes under a small latent step. Generators with lower PPL generally look better. The intuition is that without a smoothness pressure, the adversarial objective can improve average quality by squeezing bad images into tiny regions of latent space and stretching good regions; those squeezed regions create violent changes in the latent-to-image map, which degrades overall quality and makes inversion unreliable.

I therefore regularize the mapping so that a fixed-size step in latent space produces a fixed-magnitude image change in every direction. Let g be the generator and J_w its Jacobian at w. For a random image-direction y drawn from a standard normal, the vector-Jacobian product J_w^T y is obtained by back-propagating through the scalar g(w)·y. The regularizer is the squared deviation of the length of J_w^T y from a target a, averaged over w and y. The target a is not fixed by hand; it is an exponential moving average of the observed lengths, so the optimizer equalizes the spectrum around whatever scale the network already has rather than fighting an arbitrary scale from initialization. In high dimensions this prior is minimized when the Jacobian is orthogonal up to a global scale, meaning all singular values are equal and the map is a local isometry. Straight latent interpolations then follow geodesics on the image manifold, and inversion by optimizing a single latent code becomes far more reliable.

Both the discriminator's R1 gradient penalty and the new path-length term change slowly, so I evaluate them lazily rather than every iteration. The path-length term runs once every eight generator steps, and R1 once every sixteen discriminator steps. To keep the optimizer state consistent, I adjust the hyperparameters with c = k/(k+1): the learning rate becomes c·λ, the Adam momenta become β_1^c and β_2^c, and the regularizer is multiplied by k so its accumulated gradient magnitude matches what it would have been if applied every step. The path-length term can also be computed on a fraction of the minibatch to save memory, since it is only a regularizer.

The remaining architectural change is to abandon progressive growing, which is the source of the phase artifacts. Progressive growing is good at enforcing a coarse-to-fine schedule, but because each resolution momentarily serves as the output resolution during fade-in, it is pushed to emit maximal-frequency detail at that stage. That leaves the intermediate layers with excessive high frequencies and breaks shift invariance. I keep the coarse-to-fine behavior by using a skip generator: every resolution block has its own to-RGB layer, and the final image is the sum of upsampled per-resolution RGB contributions. The discriminator is made residual, with residual merges scaled by 1/√2 to cancel the variance doubling of adding two paths. A sweep confirms that skip connections in the generator improve perceptual path length, while a residual discriminator improves FID, so the final design is skip generator plus residual discriminator with no progressive growing. Because the skip generator exposes how much each resolution actually contributes, I noticed that the highest resolutions were under-used; doubling the number of feature maps from 64×64 up to 1024×4 fixes that capacity shortfall and improves both FID and recall.

Putting these pieces together gives the StyleGAN2 generator: weight demodulation removes the blob while keeping style control; path-length regularization smooths the latent-to-image map and improves perceived quality and invertibility; lazy regularization keeps the computation affordable; and the skip generator with residual discriminator removes phase artifacts without progressive growing. The result is a method that fixes both artifact families and retains the signature capability of controlling images by feeding different styles to different layers.

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
