## Research question

We want to learn good image representations without labels. A network should be pre-trained on a large pool of unlabeled images so that the features it produces — typically the output of the final pooling layer of a convolutional backbone — are immediately useful for downstream tasks: a linear classifier trained on top of the frozen features should be accurate, the features should fine-tune well from few labels, and they should transfer to detection, segmentation, and depth.

By 2020 the strongest answer to this is contrastive learning, and it works by a specific trick: it does not just *predict* one view of an image from another, it *discriminates* the right view from a crowd of wrong ones. That discrimination requires negative examples — representations of other images that the model must push away. And that requirement is expensive and brittle. It forces very large batches (to have enough negatives in view), or a memory bank / queue of stored features, or hand-tuned hard-negative mining. It also makes the method unusually sensitive to the exact set of data augmentations used.

So the precise question is: **are negative pairs indispensable for preventing representation collapse, or can a method learn high-quality representations purely by predicting one view's representation from another's — with no negatives, no large-batch requirement, and less dependence on the augmentation recipe?**

A solution would have to (i) avoid the trivial collapsed solution (a constant representation that predicts everything perfectly) without using negatives as the anti-collapse force, and (ii) be competitive with the contrastive state of the art under the standard linear-evaluation protocol on ImageNet.

## Background

**The cross-view prediction principle.** A long line of self-supervised work (going back to Becker & Hinton, 1992) learns representations by predicting one view of an input from another view of the same input. For images, the "views" are random augmentations (crops, color changes) of one image. The hope is that to predict view A's content from view B, the network must encode the stable, semantic content shared across views.

**Why naive prediction in representation space collapses.** Suppose we cast prediction directly in representation space: the representation of one augmented view should be predictive of the representation of another view of the same image. This objective has a degenerate global optimum — a representation that is constant across all inputs. A constant is trivially, perfectly predictive of itself, and it carries no information. So any method that *only* minimizes prediction error in representation space, with both sides free to move, can slide into this collapsed solution. This is the central obstacle.

**How contrastive methods dodge collapse.** Contrastive methods reformulate prediction as *discrimination*. From the representation of one view, the model must pick out the representation of the matching view (the positive) against the representations of many views of *other* images (the negatives). A constant representation can no longer be optimal, because constant features cannot discriminate the positive from the negatives. The negatives are, in effect, a repulsive force that holds the representation apart and rules out collapse. Concretely, these methods minimize a temperature-scaled softmax (InfoNCE / NT-Xent) over cosine similarities of l2-normalized embeddings: pull the positive pair together, push every negative away.

**The cost of negatives.** For the discrimination task to be hard enough to learn good features, you need many negatives, and ideally negatives close to the positive. This drives the engineering of the field at the time:
- *Large batches.* If negatives are drawn from the current minibatch, you need a very large batch — batch sizes of several thousand, giving tens of thousands of negatives per positive. Performance degrades as the batch (and so the negative count) shrinks.
- *Memory banks / queues.* An alternative keeps a running queue of feature vectors from recent batches to serve as negatives, decoupling the negative count from the batch size — but then those stored features go stale as the encoder changes.
- *Hard-negative mining.* Customized strategies to retrieve informative negatives.

**A diagnostic about augmentations.** Contrastive performance is unusually sensitive to the augmentation set. A telling observation: if the only augmentation is random cropping, the contrastive task can be largely solved by matching color histograms, because two crops of the same image tend to share a color histogram while different images differ in color. The representation is then not pushed to encode anything beyond color statistics. The standard fix is to add strong color distortion to the augmentations so this shortcut is broken — but this means the method leans heavily on a carefully chosen augmentation pipeline.

**A diagnostic about a fixed target.** There is a simple way to avoid collapse without any negatives: regress toward the representation produced by a *fixed* network. The target cannot collapse because it is never trained. A striking observation makes this more than a curiosity: if the fixed target is a randomly initialized network, training a second network to predict that random target's features yields a representation that is *substantially better than the random target itself* (a large gap in linear-evaluation accuracy between the trained network and the frozen random network it was trained against). So prediction toward a fixed target, with no negatives, both avoids collapse and *improves* on the target.

**Slow-moving target networks as a stabilizer.** In deep reinforcement learning, bootstrapped targets (the Bellman target depends on the network's own current estimate) are stabilized by computing the target with a slow copy of the network — either a periodically frozen copy or a soft exponential moving average (EMA) of the weights, θ_target ← τ θ_target + (1−τ) θ. This decouples the moving target from the rapidly changing online weights and makes bootstrapping stable.

**EMA targets in self-supervised and semi-supervised learning.**
- A momentum-encoder approach in contrastive learning keeps the "key" encoder as an EMA of the "query" encoder so that the queued negative features stay consistent as the encoder drifts. Here the EMA serves the *contrastive* objective — it keeps the dictionary of negatives coherent.
- In semi-supervised learning, the *mean teacher* keeps a "teacher" network as an EMA of a "student" network and adds a consistency loss (an l2 distance between teacher and student predictions) *on top of* a supervised classification loss on a handful of labels. The classification loss grounds the training; the consistency-to-an-EMA-teacher term is a regularizer.

**Bootstrapping representations.** Some non-contrastive methods avoid negatives by bootstrapping the representation itself: cluster the data using the current representation and use cluster indices as classification targets for the next representation. This avoids negatives but requires an expensive clustering step and special precautions against collapse.

## Baselines

**SimCLR (Chen et al., 2020).** A clean contrastive framework. For each image, sample two augmentations to form two views; pass each through an encoder f (a ResNet) to get a representation y, then through a small MLP projection head g to get a projection z. l2-normalize the projections and compute the NT-Xent loss: for a positive pair (z_i, z'_i), the loss is −log of a softmax over cosine similarities, with the matching view in the numerator and all other 2(N−1) augmented examples in the minibatch as negatives in the denominator, scaled by a temperature τ. The representation kept for downstream use is y (before the projection head), because applying the loss on the projection but evaluating on the pre-projection features works better. *Gaps it leaves:* needs a very large batch for enough negatives (accuracy drops sharply as batch size falls); relies on strong color augmentation to avoid the color-histogram shortcut.

**MoCo (He et al., 2019).** Contrastive learning with a queue of negatives and a momentum encoder. A query encoder is trained; a key encoder is maintained as an EMA of the query encoder, θ_k ← m θ_k + (1−m) θ_q. Keys (including a large queue of past keys) are encoded by the slow key encoder so they stay consistent across steps; the InfoNCE loss treats the matching key as positive and the queue as negatives. *Gap it leaves:* still fundamentally relies on negatives; the EMA exists to keep the *negative* dictionary consistent, not to remove negatives. It does, however, demonstrate that a slow EMA copy of the network is a useful, stable source of targets.

**Mean Teacher (Tarvainen & Valpola, 2017).** Semi-supervised. A student network is trained; a teacher is the EMA of the student's weights. The total loss is a supervised classification loss on labeled examples plus an l2 consistency loss between teacher and student outputs on all examples. *Gap it leaves:* it is not a self-supervised method — the classification loss on labels is what keeps the representation from collapsing. Without that supervised term, an EMA-teacher consistency objective on its own has nothing stopping the student and teacher from agreeing on a constant.

**DeepCluster (Caron et al., 2018).** Non-contrastive, bootstrap-based: alternately cluster the current features and train to predict cluster assignments. *Gap it leaves:* needs a costly clustering phase and explicit tricks (e.g., reassigning empty clusters) to avoid collapse.

**Handcrafted-pretext-task methods.** Relative patch location, colorization, inpainting, jigsaw, rotation prediction. *Gap they leave:* even with good architectures, they are outperformed by contrastive methods.

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
