# SimCLR

## Problem

Learn a visual representation from unlabeled images that is good enough that a linear classifier trained on the frozen features rivals a supervised network of the same architecture — using a simple, general recipe with **no specialized architecture, no memory bank, and no hand-crafted pretext heuristic**. Quality is measured by the linear-evaluation protocol: freeze the encoder, train one linear classifier on top, read off accuracy.

## Key idea

Learn by **maximizing agreement between two differently augmented views of the same image** via a contrastive loss in a latent space. Two questions are answered cleanly:

- **The predictive task comes from data augmentation, not architecture.** Random cropping (with resize) alone subsumes the global-to-local and adjacent-view prediction tasks that prior methods baked into the receptive field, so a standard, powerful ResNet is used unchanged. A *composition* of augmentations defines the task; crop + color distortion is the essential pair (crop-only is solvable by matching color histograms, a shortcut that color distortion destroys).
- **The negatives come from a large batch, not a memory bank or queue.** Treat the other 2(N−1) augmented examples in the minibatch as negatives. With a batch of thousands, every negative is fresh (produced by the *current* encoder) and fully back-proppable — no staleness, no momentum encoder.

Four components: (1) a stochastic augmentation module producing two views; (2) a base encoder f (ResNet) giving h; (3) a small **nonlinear projection head** g giving z = g(h); (4) the **NT-Xent** contrastive loss on the z's. After training, **g is discarded** and the representation **h (before the head)** is used downstream, because the loss trains z to be invariant to augmentation — discarding augmentation-relevant information (color, orientation, position) that h, shielded by g, retains.

## The objective: NT-Xent

For a positive pair (i, j) among the 2N views in a batch, with cosine similarity sim(u,v)=uᵀv/(‖u‖‖v‖) and temperature τ:

  ℓ_{i,j} = −log [ exp(sim(z_i, z_j)/τ) / Σ_{k=1}^{2N} 1[k≠i] exp(sim(z_i, z_k)/τ) ]

The total loss averages ℓ over both directions of every positive pair. This is InfoNCE specialized to: positive = the other augmentation of the same image, similarity = temperature-scaled cosine, negatives = the rest of the batch, applied symmetrically.

**Why this loss.** Posing "which candidate is the positive" as Bayes classification, the posterior is P(pos=i|X,c) = [p(x_i|c)/p(x_i)] / Σ_k [p(x_k|c)/p(x_k)], so the optimal score is proportional to the density ratio p(x|c)/p(x) — the correct loss shape is the log-sum-exp softmax, not a margin. The gradient w.r.t. a normalized anchor u is

  ∂ℓ/∂u = −(1/τ)(1 − p⁺) v⁺ + (1/τ) Σ_{v⁻} p⁻ v⁻,   p = softmax probability,

so the descent step pushes each negative away with weight equal to its own softmax probability p⁻ ∝ exp(uᵀv⁻/τ): **hard negatives are up-weighted automatically, relative to the other candidates in the same denominator**, with no semi-hard mining (triplet and pairwise logistic losses do not supply this candidate-set softmax weighting). τ sets the sharpness of this weighting; ℓ2-normalization bounds similarity to [−1,1] so τ is a clean knob and the model can't game the loss via vector magnitude. More candidates raise the log N ceiling of the standard bound I ≥ log N − L_N, motivating large batches.

## Default configuration

- **Augmentation:** Inception random resized crop to 224×224 (+50% horizontal flip), color distortion (brightness/contrast/saturation/hue + probabilistic grayscale, strength s), Gaussian blur (σ∈[0.1,2.0], kernel ≈10% of image side).
- **Encoder:** ResNet-50 (h = post-average-pool output); unconstrained.
- **Projection head:** paper default is a 2-layer MLP, `Linear→BN→ReLU` then final `Linear→BN` with no bias/center and no ReLU, producing a 128-d z. The released code generalizes this with `FLAGS.num_proj_layers`.
- **Loss:** NT-Xent, τ≈0.1 (ImageNet) / ≈0.5 (CIFAR), ℓ2-normalized embeddings.
- **Optimization:** paper default is LARS, LR = 0.3·BatchSize/256, weight decay 1e-6, 10-epoch linear warmup + cosine decay, batch size 4096 for the main ablations and up to 8192 in the batch-size study. BN and bias are excluded from LARS weight decay in the released code. **Global BN** aggregates BN statistics across all devices to remove the local-batch-statistics shortcut.

## Algorithm

```
input: batch size N, temperature τ, encoder f, head g, augmentation family 𝒯
for a sampled minibatch {x_k}_{k=1..N}:
    for k = 1..N:
        draw t, t' ~ 𝒯
        x̃_{2k-1} = t(x_k);  h_{2k-1} = f(x̃_{2k-1});  z_{2k-1} = g(h_{2k-1})
        x̃_{2k}   = t'(x_k); h_{2k}   = f(x̃_{2k});    z_{2k}   = g(h_{2k})
    for i,j in 1..2N:  s_{i,j} = z_iᵀz_j / (‖z_i‖‖z_j‖)
    ℓ(i,j) = -log( exp(s_{i,j}/τ) / Σ_{k≠i} exp(s_{i,k}/τ) )
    L = (1/2N) Σ_{k=1..N} [ ℓ(2k-1, 2k) + ℓ(2k, 2k-1) ]
    update f and g to minimize L
return encoder f(·); discard g(·)
```

## Code (canonical TensorFlow v1 fragments)

These are extracted google-research/simclr-style snippets, not standalone notebook cells. They run in the canonical module layout with `tensorflow.compat.v1 as tf`, `xla.replica_id`, `absl.flags.FLAGS`, `tensorflow.python.tpu.tpu_function`, `lars_optimizer.LARSOptimizer`, and local helpers such as `linear_layer` / `resnet.batch_norm_relu`.

NT-Xent as a single cross-entropy: each anchor's "correct class" is its counterpart view; self-similarity is masked with a large negative constant; across devices the other view's features are gathered so every sample sees all in-batch negatives.

```python
from absl import flags
import tensorflow.compat.v1 as tf
from tensorflow.compiler.tf2xla.python import xla

FLAGS = flags.FLAGS
LARGE_NUM = 1e9

def tpu_cross_replica_concat(tensor, tpu_context=None):
    if tpu_context is None or tpu_context.num_replicas <= 1:
        return tensor
    num_replicas = tpu_context.num_replicas
    with tf.name_scope('tpu_cross_replica_concat'):
        ext_tensor = tf.scatter_nd(
            indices=[[xla.replica_id()]],
            updates=[tensor],
            shape=[num_replicas] + tensor.shape.as_list())
        ext_tensor = tf.tpu.cross_replica_sum(ext_tensor)
        return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])

def add_contrastive_loss(hidden, hidden_norm=True, temperature=1.0,
                         tpu_context=None, weights=1.0):
    # hidden: (2N, dim) — first N are view a, last N are view b
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)        # cosine similarity; clean temperature
    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]

    if tpu_context is not None:                            # gather negatives across replicas
        hidden1_large = tpu_cross_replica_concat(hidden1, tpu_context)
        hidden2_large = tpu_cross_replica_concat(hidden2, tpu_context)
        enlarged = tf.shape(hidden1_large)[0]
        replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, enlarged * 2)
        masks  = tf.one_hot(labels_idx, enlarged)
    else:
        hidden1_large, hidden2_large = hidden1, hidden2
        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks  = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM             # mask self-similarity (k=i)
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature  # positives here
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

    loss_a = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ab, logits_aa], 1), weights=weights)
    loss_b = tf.losses.softmax_cross_entropy(labels, tf.concat([logits_ba, logits_bb], 1), weights=weights)
    return loss_a + loss_b, logits_ab, labels
```

Augmentation is implemented as crop/resize, optional flip, color jitter with probabilistic grayscale, and batched blur. In the TF1 code, `FLAGS.color_jitter_strength=1.0`, `use_blur=True`, jitter is applied with probability 0.8, grayscale with probability 0.2, and blur with probability 0.5.

```python
def color_jitter(image, strength, random_order=True, impl='simclrv2'):
    brightness = 0.8 * strength
    contrast = 0.8 * strength
    saturation = 0.8 * strength
    hue = 0.2 * strength
    if random_order:
        return color_jitter_rand(image, brightness, contrast, saturation, hue, impl=impl)
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue, impl=impl)

def random_color_jitter(image, p=1.0, impl='simclrv2'):
    def _transform(image):
        jitter = functools.partial(
            color_jitter, strength=FLAGS.color_jitter_strength, impl=impl)
        image = random_apply(jitter, p=0.8, x=image)
        return random_apply(to_grayscale, p=0.2, x=image)
    return random_apply(_transform, p=p, x=image)

def random_blur(image, height, width, p=1.0):
    del width
    def _transform(image):
        sigma = tf.random.uniform([], 0.1, 2.0, dtype=tf.float32)
        return gaussian_blur(image, kernel_size=height // 10, sigma=sigma, padding='SAME')
    return random_apply(_transform, p=p, x=image)

def preprocess_for_train(image, height, width, color_distort=True, crop=True, flip=True, impl='simclrv2'):
    if crop:
        image = random_crop_with_resize(image, height, width)
    if flip:
        image = tf.image.random_flip_left_right(image)
    if color_distort:
        image = random_color_jitter(image, impl=impl)
    return tf.clip_by_value(tf.reshape(image, [height, width, 3]), 0., 1.)
```

Projection head (keep h = the layer before the head for downstream); global BN; LARS. `linear_layer` is the local helper that applies `tf.layers.dense` followed by optional batch norm.

```python
from tensorflow.python.tpu import tpu_function
from lars_optimizer import LARSOptimizer

def projection_head(hiddens, is_training, name='head_contrastive'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        mid_dim = hiddens.shape[-1]
        out_dim = FLAGS.proj_out_dim
        hiddens_list = [hiddens]                          # out[0] == h
        if FLAGS.proj_head_mode == 'none':
            pass
        elif FLAGS.proj_head_mode == 'linear':
            hiddens = linear_layer(
                hiddens, is_training, out_dim,
                use_bias=False, use_bn=True, name='l_0')
            hiddens_list.append(hiddens)
        elif FLAGS.proj_head_mode == 'nonlinear':
            for j in range(FLAGS.num_proj_layers):
                if j != FLAGS.num_proj_layers - 1:
                    dim, bias_relu = mid_dim, True
                else:
                    dim, bias_relu = FLAGS.proj_out_dim, False
                hiddens = linear_layer(
                    hiddens, is_training, dim,
                    use_bias=bias_relu, use_bn=True, name='nl_%d' % j)
                hiddens = tf.nn.relu(hiddens) if bias_relu else hiddens
                hiddens_list.append(hiddens)
        else:
            raise ValueError('Unknown head projection mode {}'.format(
                FLAGS.proj_head_mode))
        if FLAGS.train_mode == 'pretrain':
            return hiddens_list[-1]                       # z = g(h) for the loss
        return hiddens_list[FLAGS.ft_proj_selector]       # h (out[0]) for downstream

class BatchNormalization(tf.layers.BatchNormalization):  # global BN
    def _cross_replica_average(self, t):
        n = tpu_function.get_tpu_context().number_of_shards
        return tf.tpu.cross_replica_sum(t) / tf.cast(n, t.dtype)

    def _moments(self, inputs, axes, keep_dims, mask=None):
        shard_mean, shard_var = super(BatchNormalization, self)._moments(
            inputs, axes, keep_dims=keep_dims, mask=mask)
        n = tpu_function.get_tpu_context().number_of_shards
        if n and n > 1:
            group_mean = self._cross_replica_average(shard_mean)
            group_var = self._cross_replica_average(shard_var)
            group_var += self._cross_replica_average(tf.square(group_mean - shard_mean))
            return group_mean, group_var
        return shard_mean, shard_var

optimizer = LARSOptimizer(learning_rate, momentum=FLAGS.momentum,
                          weight_decay=FLAGS.weight_decay,
                          exclude_from_weight_decay=['batch_normalization', 'bias', 'head_supervised'])
```

The training loop is an ordinary supervised loop with the classification head and cross-entropy swapped for the projection head and the contrastive loss; after pretraining, g is discarded and only the encoder is kept for evaluation.

## Why it works (design rationale, condensed)

| Choice | Why this, not the alternative |
|---|---|
| Agreement-under-augmentation pretext | Doesn't bake in a fixed invariance the way rotation/jigsaw do; the invariance set is chosen by the augmentation family. |
| Task from augmentation, not architecture | Random crop subsumes global→local and adjacent-view, so a standard ResNet is usable unchanged — unlike CPC's patch+PixelCNN or DIM/AMDIM's constrained receptive field. |
| Composition of augmentations (crop + color) | Crop-only lets the net match views by shared **color histogram** — a shortcut that solves the task without semantics. Color distortion destroys it; composition makes the task hard. Implies contrastive learning needs *stronger* augmentation than supervised. |
| Nonlinear projection head g; keep h | The loss makes z invariant to augmentation, discarding color/orientation/position; g absorbs the invariance so h retains downstream-useful info. Nonlinear > linear > none. |
| NT-Xent (softmax) loss | Optimal critic ∝ density ratio ⇒ softmax shape; its descent direction auto-weights hard negatives by softmax probability, so no semi-hard mining (unlike triplet/logistic). |
| Temperature τ + ℓ2-normalization | τ sets hardness/concentration; ℓ2-norm bounds similarity to [−1,1] so τ is meaningful and magnitude can't be gamed. |
| Large batch / many in-batch negatives | Raises the log N ceiling in I ≥ log N − L_N; negatives are fresh and back-proppable. Avoids stale bank (InstDisc) / queue+momentum (MoCo); cost moves to engineering. |
| Global BN | Local per-device BN leaks which samples are co-located (an info-leak shortcut); global aggregation removes it. |
| LARS + warmup + cosine; ResNet | LARS stabilizes the large-batch regime where linearly-scaled SGD is unstable; ResNet is a strong, unconstrained standard encoder. |
