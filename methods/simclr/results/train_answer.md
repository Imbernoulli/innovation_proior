The canonical method I am describing is SimCLR, short for A Simple Framework for Contrastive Learning of Visual Representations. Its purpose is to learn a visual representation from unlabeled images that is good enough to rival a supervised network of the same architecture, measured by the standard linear-evaluation protocol: freeze the encoder, train a single linear classifier on top, and read off accuracy. The design is intentionally minimal. It uses no specialized encoder, no memory bank, no queue, and no hand-crafted pretext heuristic. Instead, every design choice is forced by a small number of linked requirements: the task must come from somewhere other than labels, the negatives must come from somewhere other than an external storage structure, and the architecture must remain a plain, powerful ResNet.

The starting point is the observation that labels are expensive while unlabeled images are abundant. Supervised training produces representations that transfer well because the classification objective pulls semantically similar images together and pushes dissimilar ones apart. SimCLR replicates this organization without labels by asking a different classification question: given one augmented view of an image, which among a set of candidate views came from the same original image? The positive pair is simply two different augmentations of the same source image, and every other view in the batch serves as a negative. This is instance discrimination, but implemented in the simplest possible way.

The predictive task is manufactured entirely by data augmentation. Two independent random crops of the same image, each resized to the target resolution, already create a rich family of related views. A large crop and a small crop of the same object play the role of global-to-local prediction; two adjacent, barely overlapping crops play the role of adjacent-view prediction. Prior methods baked these tasks into the architecture, either by chopping the image into a fixed grid and running a PixelCNN context network or by strangling the receptive field with 1x1 convolutions. SimCLR observes that a single augmentation operation, random cropping, subsumes both of those architectural contrivances, so a standard ResNet can be dropped in unchanged. The task therefore lives in the data pipeline, not in the network design.

Cropping alone, however, leaves open a shortcut. Two crops of the same image share almost the same color distribution, so the network can solve the contrastive task by matching color histograms rather than learning semantics. To close this loophole, color distortion is applied after cropping: random brightness, contrast, saturation, hue, and a small probability of dropping the image to grayscale. The two views of one source now have independently scrambled colors, which forces the network to rely on shape, texture, and other semantic cues. A small amount of Gaussian blur is added as a further high-frequency shortcut blocker. The composition of augmentations is load-bearing; crop plus color distortion is the essential pair, and neither alone is sufficient. This also implies that contrastive learning benefits from stronger augmentation than supervised learning, because augmentation is not a safety net but the only thing defining the task.

The loss is NT-Xent, the normalized temperature-scaled cross-entropy. Given L2-normalized embeddings z_i and z_j for a positive pair and a batch of 2N views, the loss for anchor i is negative log of the softmax probability assigned to its positive counterpart j, with every other view in the batch treated as a negative. The similarity is cosine similarity divided by a temperature tau. The loss is computed symmetrically, treating both (i,j) and (j,i) as positive pairs, so gradient flows from both directions. This softmax form is not arbitrary. If one derives the optimal score for identifying the positive among a set containing one positive and many negatives, the optimal critic is proportional to the density ratio p(positive|context)/p(positive), and the correct loss shape is a log-sum-exp softmax, not a margin or a pairwise logistic loss. The gradient of NT-Xent with respect to an anchor shows that each negative is pushed away with weight equal to its own softmax probability, which means hard negatives are up-weighted automatically relative to the other candidates. This internal hard-negative weighting is the reason SimCLR needs no external semi-hard mining, unlike the triplet loss. The temperature tau controls the sharpness of this weighting: smaller tau concentrates the gradient on the hardest negatives, while larger tau flattens it. L2 normalization bounds similarity to the interval [-1, 1], so tau is a clean sharpness knob rather than an arbitrary scale absorber. Without normalization the network could game the loss by growing vector magnitudes instead of improving directions.

A nonlinear projection head sits between the encoder and the loss. The base encoder f produces a representation h, and a small MLP g maps h to z, on which the loss is computed. After pretraining, g is discarded and the downstream classifier is built on h, not z. This is counter-intuitive but crucial. The contrastive loss trains z to be invariant to augmentation, which means z learns to discard color, orientation, position, and high-frequency detail. If the loss were applied directly to h, h would be forced to discard the same information, hurting downstream tasks that need it. By inserting g, the invariance burden is absorbed by the projection head, leaving h rich in the very cues that augmentation varies. Empirically, a nonlinear head performs better than a linear head, which in turn performs better than no head at all.

The negatives come from the minibatch itself. At a batch size of several thousand, each positive pair sees tens of thousands of fresh negatives, all produced by the current encoder and fully back-propagable. This avoids the staleness of a memory bank or the approximate consistency of a momentum queue. The price is engineering: very large batches require global batch normalization and a large-batch optimizer. Local per-device batch normalization leaks information about which samples are co-located on the same device, giving the network a shortcut to identify the positive. Global batch normalization aggregates statistics across all devices, removing that leak. LARS, with layer-wise adaptive learning rates, stabilizes training at these batch sizes, while linear warmup and cosine decay keep the optimization from detonating. Batch-normalization and bias parameters are excluded from weight decay.

The full recipe is therefore: draw two augmented views of every image in a batch, encode both through the same ResNet, map the representations through a nonlinear projection head, compute NT-Xent on the projected features, and update the encoder and projection head with LARS. After training, discard the projection head and use the pre-head representation for downstream linear evaluation, semi-supervised fine-tuning, or transfer learning. The method is SimCLR.

The loss is implemented as a single softmax cross-entropy rather than an explicit sum of log-ratios. `hidden` holds the 2N projected features for one training step, the first N rows being view a and the last N being view b of the same N source images; after the optional L2-normalization I split it into `hidden1` and `hidden2`. Four blocks of pairwise cosine similarities at temperature-scaled magnitude are then formed: `logits_aa` and `logits_bb` compare each half against itself, and `logits_ab` / `logits_ba` compare the two halves against each other, so the positive for anchor k in `logits_ab` sits at column k. Self-similarity inside the aa/bb blocks is masked out with the constant `LARGE_NUM` so an anchor can never match itself, and `tpu_cross_replica_concat` gathers every replica's embeddings across devices before the similarities are formed, so a given anchor's negative set is the whole distributed batch rather than just its local shard — the labels and masks are widened accordingly when a `tpu_context` is present. The two symmetric cross-entropy terms, `loss_a` against `[logits_ab, logits_aa]` and `loss_b` against `[logits_ba, logits_bb]`, are added to give the direction-symmetric NT-Xent loss:

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
