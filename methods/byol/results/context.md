## Research question

We want to learn good image representations without labels. A network should be pre-trained on a large pool of unlabeled images so that the features it produces — typically the output of the final pooling layer of a convolutional backbone — are immediately useful for downstream tasks: a linear classifier trained on top of the frozen features should be accurate, the features should fine-tune well from few labels, and they should transfer to detection, segmentation, and depth.

By 2020 the strongest approach to this is contrastive learning. It does not just *predict* one view of an image from another, it *discriminates* the right view from a crowd of wrong ones. That discrimination uses negative examples — representations of other images that the model pushes away — which are supplied by very large batches, a memory bank / queue of stored features, or hard-negative mining.

So the precise question is: **how to learn high-quality image representations from unlabeled images by relating two augmented views of the same image**, evaluated under the standard linear-evaluation protocol on ImageNet.

## Background

**The cross-view prediction principle.** A long line of self-supervised work (going back to Becker & Hinton, 1992) learns representations by predicting one view of an input from another view of the same input. For images, the "views" are random augmentations (crops, color changes) of one image. To predict view A's content from view B, the network must encode the stable, semantic content shared across views.

**Prediction in representation space.** Prediction can be cast directly in representation space: the representation of one augmented view is made predictive of the representation of another view of the same image. This objective has a constant representation across all inputs as a global optimum, which is trivially predictive of itself.

**How contrastive methods handle collapse.** Contrastive methods reformulate prediction as *discrimination*. From the representation of one view, the model picks out the representation of the matching view (the positive) against the representations of many views of *other* images (the negatives). A constant representation cannot discriminate the positive from the negatives, so the negatives act as a repulsive force that holds the representation apart. Concretely, these methods minimize a temperature-scaled softmax (InfoNCE / NT-Xent) over cosine similarities of l2-normalized embeddings: pull the positive pair together, push every negative away.

**The role of negatives.** For the discrimination task to be hard enough to learn good features, the method uses many negatives, ideally negatives close to the positive. The engineering of the field at the time reflects this:
- *Large batches.* If negatives are drawn from the current minibatch, the batch is very large — sizes of several thousand, giving tens of thousands of negatives per positive.
- *Memory banks / queues.* A running queue of feature vectors from recent batches serves as negatives, decoupling the negative count from the batch size.
- *Hard-negative mining.* Customized strategies to retrieve informative negatives.

**Augmentations.** Contrastive performance is sensitive to the augmentation set. With only random cropping, the contrastive task can be largely solved by matching color histograms, because two crops of the same image tend to share a color histogram while different images differ in color. The standard practice adds strong color distortion to the augmentations.

**Prediction toward a fixed target.** One can regress toward the representation produced by a *fixed* network, which is never trained. An observation: if the fixed target is a randomly initialized network, training a second network to predict that random target's features yields a representation with substantially higher linear-evaluation accuracy than the frozen random network it was trained against.

**Slow-moving target networks in reinforcement learning.** In deep RL, bootstrapped targets (the Bellman target depends on the network's own current estimate) are computed with a slow copy of the network — either a periodically frozen copy or a soft exponential moving average (EMA) of the weights, θ_target ← τ θ_target + (1−τ) θ.

**EMA targets in self-supervised and semi-supervised learning.**
- A momentum-encoder approach in contrastive learning keeps the "key" encoder as an EMA of the "query" encoder so that the queued negative features stay consistent as the encoder drifts.
- In semi-supervised learning, the *mean teacher* keeps a "teacher" network as an EMA of a "student" network and adds a consistency loss (an l2 distance between teacher and student predictions) on top of a supervised classification loss on a handful of labels.

**Bootstrapping representations.** Some non-contrastive methods bootstrap the representation itself: cluster the data using the current representation and use cluster indices as classification targets for the next representation.

## Baselines

**SimCLR (Chen et al., 2020).** A contrastive framework. For each image, sample two augmentations to form two views; pass each through an encoder f (a ResNet) to get a representation y, then through a small MLP projection head g to get a projection z. l2-normalize the projections and compute the NT-Xent loss: for a positive pair (z_i, z'_i), the loss is −log of a softmax over cosine similarities, with the matching view in the numerator and all other 2(N−1) augmented examples in the minibatch as negatives in the denominator, scaled by a temperature τ. The representation kept for downstream use is y (before the projection head), because applying the loss on the projection but evaluating on the pre-projection features works better.

**MoCo (He et al., 2019).** Contrastive learning with a queue of negatives and a momentum encoder. A query encoder is trained; a key encoder is maintained as an EMA of the query encoder, θ_k ← m θ_k + (1−m) θ_q. Keys (including a large queue of past keys) are encoded by the slow key encoder so they stay consistent across steps; the InfoNCE loss treats the matching key as positive and the queue as negatives.

**Mean Teacher (Tarvainen & Valpola, 2017).** Semi-supervised. A student network is trained; a teacher is the EMA of the student's weights. The total loss is a supervised classification loss on labeled examples plus an l2 consistency loss between teacher and student outputs on all examples.

**DeepCluster (Caron et al., 2018).** Non-contrastive, bootstrap-based: alternately cluster the current features and train to predict cluster assignments, with precautions such as reassigning empty clusters.

**Handcrafted-pretext-task methods.** Relative patch location, colorization, inpainting, jigsaw, rotation prediction.

## Evaluation settings

- **Pretraining data:** the ImageNet ILSVRC-2012 training set (~1.28M images), used without labels for self-supervised pretraining.
- **Linear evaluation on ImageNet:** freeze the pretrained encoder (no weight or batch-statistic updates); train a linear classifier on top of the frozen features. Train-time augmentation is random crop + resize to 224×224 and random flip; test-time is a 224×224 center crop of an image resized to 256 on the short side; color channels normalized by ImageNet statistics. Optimize cross-entropy with SGD + Nesterov momentum over 80 epochs, batch 1024, sweeping the learning rate. Report top-1 / top-5 accuracy.
- **Semi-supervised on ImageNet:** initialize from the pretrained encoder and fine-tune on fixed 1% and 10% labeled subsets; report top-1 / top-5.
- **Transfer (classification):** linear evaluation and fine-tuning on Food-101, CIFAR-10, CIFAR-100, Birdsnap, SUN397, Stanford Cars, FGVC Aircraft, VOC2007, DTD, Oxford-IIIT Pets, Caltech-101, Oxford 102 Flowers; standard per-dataset metrics (top-1, mean-per-class, 11-point mAP).
- **Transfer (other vision tasks):** VOC2012 semantic segmentation (FCN, mIoU), VOC2007 object detection (Faster R-CNN, AP50), NYU v2 depth estimation (relative error, rms error, and percent of pixels within 1.25^n).
- **Backbones:** ResNet-50 (1×) as the standard encoder, plus wider (up to 4×) and deeper (101/152/200) ResNets.

## Code framework

A standard self-supervised training harness supplies paired augmented views, a Haiku model wrapper, a large-batch optimizer, a cosine schedule, and a training loop.

```python
import jax
import jax.numpy as jnp
import haiku as hk
import optax

def augment(images, rng, view_id):
    pass

def make_optimizer(learning_rate):
    pass

def cosine_lr(step, base_lr, warmup_steps, total_steps):
    pass

def network(inputs, is_training):
    # TODO: define the representation model and any training-only outputs.
    pass

def loss_fn(params, state, view_1, view_2):
    # TODO: map two augmented views to a scalar self-supervised loss.
    pass

def update_fn(state, step, view_1, view_2):
    # TODO: apply the parameter update for one training step.
    pass

def main(dataset):
    rng = jax.random.PRNGKey(0)
    state = init(rng)
    for step in range(total_steps):
        images = next(dataset)
        rng, r1, r2 = jax.random.split(rng, 3)
        view_1 = augment(images, r1, view_id=1)
        view_2 = augment(images, r2, view_id=2)
        state = update_fn(state, step, view_1, view_2)
    return state
```
