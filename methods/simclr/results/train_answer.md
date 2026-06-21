The canonical method I am describing is SimCLR, short for A Simple Framework for Contrastive Learning of Visual Representations. Its purpose is to learn a visual representation from unlabeled images that is good enough to rival a supervised network of the same architecture, measured by the standard linear-evaluation protocol: freeze the encoder, train a single linear classifier on top, and read off accuracy. The design is intentionally minimal. It uses no specialized encoder, no memory bank, no queue, and no hand-crafted pretext heuristic. Instead, every design choice is forced by a small number of linked requirements: the task must come from somewhere other than labels, the negatives must come from somewhere other than an external storage structure, and the architecture must remain a plain, powerful ResNet.

The starting point is the observation that labels are expensive while unlabeled images are abundant. Supervised training produces representations that transfer well because the classification objective pulls semantically similar images together and pushes dissimilar ones apart. SimCLR replicates this organization without labels by asking a different classification question: given one augmented view of an image, which among a set of candidate views came from the same original image? The positive pair is simply two different augmentations of the same source image, and every other view in the batch serves as a negative. This is instance discrimination, but implemented in the simplest possible way.

The predictive task is manufactured entirely by data augmentation. Two independent random crops of the same image, each resized to the target resolution, already create a rich family of related views. A large crop and a small crop of the same object play the role of global-to-local prediction; two adjacent, barely overlapping crops play the role of adjacent-view prediction. Prior methods baked these tasks into the architecture, either by chopping the image into a fixed grid and running a PixelCNN context network or by strangling the receptive field with 1x1 convolutions. SimCLR observes that a single augmentation operation, random cropping, subsumes both of those architectural contrivances, so a standard ResNet can be dropped in unchanged. The task therefore lives in the data pipeline, not in the network design.

Cropping alone, however, leaves open a shortcut. Two crops of the same image share almost the same color distribution, so the network can solve the contrastive task by matching color histograms rather than learning semantics. To close this loophole, color distortion is applied after cropping: random brightness, contrast, saturation, hue, and a small probability of dropping the image to grayscale. The two views of one source now have independently scrambled colors, which forces the network to rely on shape, texture, and other semantic cues. A small amount of Gaussian blur is added as a further high-frequency shortcut blocker. The composition of augmentations is load-bearing; crop plus color distortion is the essential pair, and neither alone is sufficient. This also implies that contrastive learning benefits from stronger augmentation than supervised learning, because augmentation is not a safety net but the only thing defining the task.

The loss is NT-Xent, the normalized temperature-scaled cross-entropy. Given L2-normalized embeddings z_i and z_j for a positive pair and a batch of 2N views, the loss for anchor i is negative log of the softmax probability assigned to its positive counterpart j, with every other view in the batch treated as a negative. The similarity is cosine similarity divided by a temperature tau. The loss is computed symmetrically, treating both (i,j) and (j,i) as positive pairs, so gradient flows from both directions. This softmax form is not arbitrary. If one derives the optimal score for identifying the positive among a set containing one positive and many negatives, the optimal critic is proportional to the density ratio p(positive|context)/p(positive), and the correct loss shape is a log-sum-exp softmax, not a margin or a pairwise logistic loss. The gradient of NT-Xent with respect to an anchor shows that each negative is pushed away with weight equal to its own softmax probability, which means hard negatives are up-weighted automatically relative to the other candidates. This internal hard-negative weighting is the reason SimCLR needs no external semi-hard mining, unlike the triplet loss. The temperature tau controls the sharpness of this weighting: smaller tau concentrates the gradient on the hardest negatives, while larger tau flattens it. L2 normalization bounds similarity to the interval [-1, 1], so tau is a clean sharpness knob rather than an arbitrary scale absorber. Without normalization the network could game the loss by growing vector magnitudes instead of improving directions.

A nonlinear projection head sits between the encoder and the loss. The base encoder f produces a representation h, and a small MLP g maps h to z, on which the loss is computed. After pretraining, g is discarded and the downstream classifier is built on h, not z. This is counter-intuitive but crucial. The contrastive loss trains z to be invariant to augmentation, which means z learns to discard color, orientation, position, and high-frequency detail. If the loss were applied directly to h, h would be forced to discard the same information, hurting downstream tasks that need it. By inserting g, the invariance burden is absorbed by the projection head, leaving h rich in the very cues that augmentation varies. Empirically, a nonlinear head performs better than a linear head, which in turn performs better than no head at all.

The negatives come from the minibatch itself. At a batch size of several thousand, each positive pair sees tens of thousands of fresh negatives, all produced by the current encoder and fully back-propagable. This avoids the staleness of a memory bank or the approximate consistency of a momentum queue. The price is engineering: very large batches require global batch normalization and a large-batch optimizer. Local per-device batch normalization leaks information about which samples are co-located on the same device, giving the network a shortcut to identify the positive. Global batch normalization aggregates statistics across all devices, removing that leak. LARS, with layer-wise adaptive learning rates, stabilizes training at these batch sizes, while linear warmup and cosine decay keep the optimization from detonating. Batch-normalization and bias parameters are excluded from weight decay.

The full recipe is therefore: draw two augmented views of every image in a batch, encode both through the same ResNet, map the representations through a nonlinear projection head, compute NT-Xent on the projected features, and update the encoder and projection head with LARS. After training, discard the projection head and use the pre-head representation for downstream linear evaluation, semi-supervised fine-tuning, or transfer learning. The method is SimCLR.

```python
import numpy as np

# A small, self-contained illustration of SimCLR's NT-Xent loss.
# It uses random unit-normalized vectors to stand in for the projected
# features z = g(h) that SimCLR computes on two augmented views.

def nt_xent_loss(z, temperature=0.5):
    """
    z: array of shape (2*N, d). Rows 0..N-1 are view a,
       rows N..2N-1 are view b for the same N source samples.
    Returns the scalar NT-Xent loss and the anchor-positive logits.
    """
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)  # L2-normalize
    n = z.shape[0] // 2
    sim = z @ z.T / temperature                                 # cosine / tau
    # Mask self-similarities by subtracting a large value on the diagonal.
    np.fill_diagonal(sim, -1e9)

    # Positive indices: view a's positive is the matching view b and vice versa.
    pos_a = np.arange(n, 2 * n)
    pos_b = np.arange(0, n)

    # Softmax over all 2N-1 candidates for each anchor.
    exp_sim = np.exp(sim - np.max(sim, axis=1, keepdims=True))
    exp_sum = exp_sim.sum(axis=1)
    loss = 0.0
    for anchor, pos in enumerate(pos_a):
        loss += -np.log(exp_sim[anchor, pos] / exp_sum[anchor])
    for anchor, pos in enumerate(pos_b):
        loss += -np.log(exp_sim[n + anchor, pos] / exp_sum[n + anchor])
    return loss / (2 * n)

# Synthetic demo: 8 source samples, 16 views total, 128-dimensional features.
np.random.seed(0)
N, d = 8, 128
h = np.random.randn(2 * N, d)
loss = nt_xent_loss(h, temperature=0.5)
print(f"NT-Xent loss on random features: {loss:.4f}")

# Sanity check: if all positives were perfectly aligned and negatives were
# orthogonal, the loss for each anchor would approach -log(1/(2*N-1)).
print(f"Ideal lower bound for N={N}: {-np.log(1.0 / (2 * N - 1)):.4f}")

# Contrast with perfectly aligned positives (same vector for each pair)
# and random negatives. The loss should drop toward the lower bound.
perfect = np.zeros((2 * N, d))
for i in range(N):
    v = np.random.randn(d)
    v = v / np.linalg.norm(v)
    perfect[i] = v
    perfect[N + i] = v
print(f"NT-Xent loss with aligned positives: {nt_xent_loss(perfect):.4f}")
```
