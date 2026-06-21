# Context

## Research question

Unconditional generation of high-resolution photographic images with a generative
adversarial network has reached the point where the best generators produce convincing
faces and objects at 1024×1024. The leading style-based generator, while achieving
state-of-the-art image quality, exhibits two families of **characteristic, recognizable
artifacts** visible in its outputs:

- **Blob / "water-droplet" artifacts.** A consistent blob-shaped anomaly appears in the
  generated images and in essentially *all* of the generator's intermediate feature maps.
  It starts to appear around 64×64 resolution and grows stronger at higher resolutions.
  In a small fraction of images (~0.1%) the blob is absent and the image is then severely
  corrupted — suggesting the network depends on the blob.
- **Location-locked / "phase" artifacts.** Details such as teeth or eyes that should glide
  smoothly across the frame as the latent moves instead tend to stay pinned to preferred
  pixel positions and then jump to the next preferred position.

Two further questions arise from this:

1. The standard distribution-quality metrics may not fully capture perceived image quality.
   Generators with *identical* density/coverage scores can differ markedly in perceived
   quality.
2. Inverting the generator (finding the latent code that reproduces a given image) is
   valuable for editing and for attribution.

How can the style-based generator's image quality and controllability be improved?

## Background

**Generative adversarial networks.** A generator G maps a latent code to an image and a
discriminator D scores images as real or fake; the two are trained against each other
(Goodfellow et al. 2014). The non-saturating logistic loss is standard. Training is
notoriously unstable, and a body of regularizers exists to tame it.

**Style-based generation.** The current best high-resolution generator is unconventional:
rather than feeding the latent z directly into the convolutional stack, a *mapping
network* f (eight fully-connected layers, trained with a ~100× lower learning rate) first
transforms z∈Z into an intermediate code w∈W. Per-layer *affine transforms* turn w into
*styles*, and the styles control each layer of a *synthesis network* g. Stochastic detail
is injected via per-pixel Gaussian noise maps added inside each layer; the synthesis
network starts from a learned constant 4×4 tensor rather than from z. The intermediate
space W is empirically far less entangled than Z, which is why all analysis here is framed
in terms of W.

**Adaptive instance normalization (AdaIN).** The mechanism that injects a style is AdaIN
(Huang & Belongie 2017; Dumoulin et al. 2016; Ghiasi et al. 2017). For each feature map it
removes the per-map mean and standard deviation and then re-applies a learned, style-driven
scale and bias:
`AdaIN(x_i, s) = s_{scale,i} · (x_i − μ(x_i)) / σ(x_i) + s_{bias,i}`,
with μ, σ computed *per feature map, per sample*. The normalization makes the style scale
the dominant control over each map's contribution and is what enables style mixing — it
normalizes each map *separately*, using data-dependent statistics.

**Initializer-style variance analysis.** A long line of work analyzes signal variance
through a network analytically rather than empirically. Glorot & Bengio (2010) and He et al.
(2015) set initial weight scales so that activation variance is preserved layer to layer,
under the assumption of independent, unit-variance inputs.

**Weight normalization** (Salimans & Kingma 2016) reparameterizes a weight vector as a
direction times a scalar magnitude, which incidentally divides by the L2 norm of the
weights. It has been reported beneficial in GAN training (Xiang & Li 2017).

**Progressive growing.** To stabilize high-resolution training, the field grows both
networks resolution by resolution: train at 4×4, then fade in 8×8, then 16×16, and so on
(Karras et al. 2017). This gives a coarse-to-fine training schedule and is highly effective
for stability.

**Architectures for multi-scale generation.** Skip connections (Ronneberger et al. 2015;
Karnewar & Wang 2019), residual networks (He et al. 2015b; Gulrajani et al. 2017; Miyato et
al. 2018), and Laplacian-pyramid / hierarchical generators (Denton et al. 2015; Zhang et al.
2016, 2017) all provide ways to combine information across resolutions without growing the
network during training. MSG-GAN (Karnewar & Wang 2019) connects matching resolutions of G
and D with multiple skip connections, the generator emitting a multi-resolution image stack.

**Conditioning of the generator mapping.** Odena et al. (2018) ask whether the conditioning
of the generator's input→output Jacobian is causally related to GAN performance, and propose
a Jacobian clamping regularizer that pushes Jacobian-vector-product magnitudes toward a
target using finite differences. Spectral normalization (Miyato et al. 2018; applied to
generators by Zhang et al. 2018) constrains the *largest* singular value of a layer's
weights.

**Perceptual path length (PPL).** Introduced to quantify the smoothness of the latent→image
map, PPL measures the average LPIPS perceptual distance (Zhang et al. 2018) between images
produced under small steps in latent space. Generators with the same FID and precision/recall
but lower PPL tend to look better to humans, so a smoother mapping (lower PPL) appears to
correlate with perceived quality.

**FID and precision/recall.** FID (Heusel et al. 2017) and precision/recall
(Sajjadi et al. 2018; Kynkäänniemi et al. 2019) are computed in the feature space of
ImageNet-trained classifiers (InceptionV3, VGG-16; Simonyan & Zisserman 2014). Such
classifiers have been shown to base decisions on texture more than shape (Geirhos et al.
2018), whereas humans weight shape heavily (Landau et al. 1988).

## Baselines

**Style-based generator with AdaIN (the immediate predecessor).** Architecture as above:
mapping net f, per-layer affine styles, synthesis net g with AdaIN modulation, per-pixel
noise, learned constant input, trained with progressive growing. Core algorithm of a style
block: normalize each feature map (subtract mean, divide by std), then modulate by the
style's scale and bias.

**Progressive growing of GANs** (Karras et al. 2017). Trains at increasing resolutions,
fading each new resolution in.

**Removing normalization entirely.** Simply dropping instance normalization removes the
blob and even improves FID slightly (Kynkäänniemi et al. 2019).

**R1 gradient penalty** (Mescheder et al. 2018). Penalizes the squared gradient norm of D on
real data, `(γ/2)·E‖∇_x D(x)‖²`, improving convergence. It is written together with the
main loss and thus computed every step.

**Jacobian clamping** (Odena et al. 2018). Regularizes the generator's input→output Jacobian
toward a target conditioning using finite-difference estimates of Jacobian-vector products.

**Spectral normalization** (Miyato et al. 2018; Zhang et al. 2018 for generators). Divides
each weight tensor by its largest singular value.

**MSG-GAN** (Karnewar & Wang 2019). Multiple skip connections between matching resolutions
of G and D, generator emitting a resolution stack.

## Evaluation settings

- **Datasets.** FFHQ faces at 1024×1024 (augmented with horizontal flips, ~70k→140k); LSUN
  categories (Car, Cat, Church, Horse) at their native resolutions (no augmentation).
- **Metrics.** Fréchet inception distance (FID) over 50k samples; precision and recall
  (P&R) in the improved formulation; perceptual path length (PPL), computed as average LPIPS
  distance under small latent perturbations (here over the full image, sampling around
  w∼f(z)). For inversion quality, LPIPS distance between an image and its re-synthesis from
  the recovered latent.
- **Protocol.** Non-saturating logistic loss with R1 regularization; Adam (β1=0, β2=0.99,
  ε=1e-8, minibatch 32); equalized learning rate on all trainable parameters; exponential
  moving average of generator weights; leaky ReLU (α=0.2); bilinear filtering in all
  up/downsampling; style-mixing regularization; minibatch-standard-deviation layer at the end
  of D. Training on 8×V100.

## Code framework

The pre-existing scaffold below holds everything that is fixed before the redesign: the
mapping network, the affine style transforms, the loss, the optimizer, and the training
loop. The slots that the redesign will fill — how a style actually conditions a convolution,
how the synthesis and discriminator stacks pass information across resolutions, and any
generator-side regularizer — are left empty.

```python
import numpy as np
import tensorflow as tf

# --- Primitives -------------------------------------------------------------

def get_weight(shape, gain=1, use_wscale=True, lrmul=1):
    fan_in = np.prod(shape[:-1])
    he_std = gain / np.sqrt(fan_in)              # He init
    if use_wscale:
        init_std, runtime_coef = 1.0 / lrmul, he_std * lrmul   # equalized LR
    else:
        init_std, runtime_coef = he_std / lrmul, lrmul
    w = tf.get_variable('weight', shape=shape,
                        initializer=tf.initializers.random_normal(0, init_std))
    return w * runtime_coef

def dense_layer(x, fmaps, **kw):
    w = get_weight([x.shape[1].value, fmaps], **kw)
    return tf.matmul(x, tf.cast(w, x.dtype))

def conv2d_layer(x, fmaps, kernel, up=False, down=False, resample_kernel=None, **kw):
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kw)
    # plain / bilinear-up / bilinear-down conv (NCHW)
    ...

def apply_bias_act(x, act='linear', alpha=None, gain=None, lrmul=1):
    b = tf.get_variable('bias', shape=[x.shape[1]],
                        initializer=tf.initializers.zeros()) * lrmul
    return fused_bias_act(x, b=tf.cast(b, x.dtype), act=act, alpha=alpha, gain=gain)

def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    ...   # appends a feature map of group std; sits at the end of D

# --- Mapping network: z -> w ------------------------------------------------

def G_mapping(latents_in, labels_in, latent_size=512, dlatent_size=512,
              mapping_layers=8, mapping_lrmul=0.01, dlatent_broadcast=None, **kw):
    x = latents_in
    x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + 1e-8)  # normalize z
    for i in range(mapping_layers):
        fmaps = dlatent_size if i == mapping_layers - 1 else latent_size
        x = apply_bias_act(dense_layer(x, fmaps, lrmul=mapping_lrmul),
                           act='lrelu', lrmul=mapping_lrmul)
    if dlatent_broadcast is not None:
        x = tf.tile(x[:, np.newaxis], [1, dlatent_broadcast, 1])
    return x

# --- SLOT 1: how a style conditions one convolution -------------------------

def style_conv_layer(x, style, fmaps, kernel, up=False, **kw):
    # TODO: produce one conv layer whose behavior is set by `style`,
    #       such that style mixing keeps working. (To be designed.)
    pass

# --- SLOT 2: synthesis network ----------------------------------------------

def G_synthesis(dlatents_in, resolution=1024, num_channels=3, **kw):
    # constant 4x4 input, then per-resolution blocks of style_conv_layer + noise,
    # producing an RGB image.
    # TODO: how the per-resolution blocks are organized into the final image.
    #       (To be designed.)
    pass

# --- SLOT 3: discriminator --------------------------------------------------

def D_net(images_in, labels_in, resolution=1024, num_channels=3, **kw):
    # per-resolution downsampling blocks, minibatch-stddev, dense head.
    # TODO: how do blocks across resolutions connect? (To be designed.)
    pass

# --- Loss + regularizers ----------------------------------------------------

def G_logistic_ns(G, D, training_set, minibatch_size):
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels  = training_set.get_random_labels_tf(minibatch_size)
    fake    = G.get_output_for(latents, labels, is_training=True)
    scores  = D.get_output_for(fake, labels, is_training=True)
    loss = tf.nn.softplus(-scores)               # -log sigmoid(D(fake))
    reg  = None
    # TODO: an optional generator-side regularizer to be designed.
    return loss, reg

def D_logistic_r1(G, D, training_set, minibatch_size, reals, labels, gamma=10.0):
    fake  = G.get_output_for(tf.random_normal([minibatch_size] + G.input_shapes[0][1:]),
                             labels, is_training=True)
    rs = D.get_output_for(reals, labels, is_training=True)
    fs = D.get_output_for(fake,  labels, is_training=True)
    loss = tf.nn.softplus(fs) + tf.nn.softplus(-rs)
    real_grads = tf.gradients(tf.reduce_sum(rs), [reals])[0]
    reg = tf.reduce_sum(tf.square(real_grads), axis=[1,2,3]) * (gamma * 0.5)
    return loss, reg

# --- Optimizer / training loop ----------------------------------------------
# Adam(beta1=0, beta2=0.99, eps=1e-8); equalized LR; EMA of G weights;
# style-mixing regularization; main loss + reg are summed and stepped together.
```
