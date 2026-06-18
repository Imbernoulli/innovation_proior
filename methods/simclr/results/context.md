# Context: learning visual representations from unlabeled images

## Research question

Can we learn a visual representation from images **without any human labels** that is good enough to rival a supervised network of the same architecture — and do it with a *simple, general* recipe that needs no specialized architecture, no external memory structure, and no hand-engineered pretext heuristic?

The pain point is concrete. Labeling data at ImageNet scale is expensive, and the overwhelming majority of the world's images carry no labels at all. The dominant paradigm trains a network with cross-entropy against human annotations, producing features that transfer well to downstream tasks; we would like features of comparable quality "for free" from the pixels alone. The community has converged on a single yardstick for "quality": the **linear-evaluation protocol** — freeze the learned encoder, train one linear classifier on top of its frozen features, and read off the test accuracy. The accuracy is a proxy for how *linearly separable*, hence how semantically organized, the representation is: if a single hyperplane per class can carve the feature space, the network has done the hard work of disentangling the semantics.

A solution must therefore (a) push that linear-probe number close to a supervised network of the same backbone, (b) keep improving as the model and data scale up rather than saturating, and (c) avoid depending on a bespoke architecture or a fragile auxiliary mechanism — every such crutch couples the learning signal to engineering and limits generality. Concretely, any candidate method has to answer two clean questions on its own terms: **where does the predictive task come from** (what is the network asked to predict, constructed without labels?) and **where do the contrastive negatives come from** (how do we supply enough informative "this is not the same thing" signal to shape the feature geometry?).

## Background

Unsupervised visual representation learning splits into two families. **Generative** approaches — restricted Boltzmann machines / deep belief nets (Hinton et al. 2006), variational autoencoders (Kingma & Welling 2013), generative adversarial nets (Goodfellow et al. 2014) — model or reconstruct the pixel distribution. Modeling pixels faithfully is computationally heavy and arguably solves a harder problem than we need: we want a representation, not a renderer, and pixel-level fidelity spends capacity on texture and noise that linear separability does not care about. **Discriminative** approaches instead reuse the supervised machinery (a network minimizing a classification-style loss) but *manufacture* both the inputs and the targets from unlabeled data through a **pretext task**.

The recent renaissance of self-supervision opened with **hand-crafted pretext tasks**: predict the relative position of two image patches (Doersch et al. 2015), reassemble a shuffled jigsaw (Noroozi & Favaro 2016), colorize a grayscale image (Zhang et al. 2016), or predict which of four rotations was applied (Gidaris et al. 2018). These work — especially with bigger backbones and longer training (Kolesnikov et al. 2019) — but each bakes a *specific* cue into the task by fiat. Rotation prediction, for example, forces the network to become rotation-*sensitive*, which is the opposite of what a general-purpose representation usually wants. The heuristic nature of these tasks caps the generality of the features they produce.

The line that pulled ahead is **contrastive learning in a latent space**. The seed is old: Becker & Hinton (1992) proposed a network whose two views of the same input should *agree*, maximizing the mutual information between the outputs of spatially adjacent patches. Hadsell, Chopra & LeCun (2006) turned "agree" into a trainable objective (DrLIM): pull the representations of a positive pair together, push negatives apart past a margin, like a system of springs. The modern instantiations frame the task as **instance discrimination** and lean on the **InfoNCE** objective and on **mutual-information** intuition. By late 2019 the prevailing wisdom held that good contrastive learning needs *many* negatives and is (loosely) maximizing a lower bound on the mutual information between views — though it was already contested whether the gains come from the mutual information per se or from the specific loss form (Tschannen et al. 2019).

The load-bearing technical concepts the field rests on:

- **Noise-contrastive estimation / InfoNCE.** Given a query and a candidate set containing one positive (drawn from the conditional) and many negatives (drawn from the marginal), classify which candidate is the positive via a softmax over similarity scores. Minimizing this categorical cross-entropy drives the score function toward the density ratio p(x|c)/p(x) and gives a lower bound of the form I ≥ log N − L_N, so more candidates raise the maximum mutual information the bound can certify.
- **Non-parametric softmax with a temperature and ℓ2-normalization.** Rather than a learned class weight per instance, the candidates are scored by the (normalized) features themselves, with a temperature scaling the logits.
- **Batch normalization** (Ioffe & Szegedy 2015) as a standard ResNet component, with the subtlety that under data-parallel training its mean/variance statistics are computed per-device.
- **Large-batch optimization machinery:** linear learning-rate scaling with warmup (Goyal et al. 2017), the LARS layer-wise adaptive optimizer (You et al. 2017), and cosine learning-rate decay (Loshchilov & Hutter 2016).
- A rich **data-augmentation toolbox:** Inception-style random resized crop (Szegedy et al. 2015), color jitter/dropping (Howard 2013; Szegedy et al. 2015), cutout (DeVries & Taylor 2017), Gaussian blur, Sobel filtering, and learned policies such as AutoAugment (Cubuk et al. 2019).

## Baselines

**Exemplar CNN (Dosovitskiy et al. 2014).** Treat each image (and its augmentations) as its own class and train a classifier to recognize which exemplar an augmented patch came from. This is *parametric* instance discrimination: one weight vector per instance in the softmax. The classifier head therefore grows linearly with the dataset, becomes intractable at millions of images, and the per-instance weights do not generalize to unseen instances.

**InstDisc — non-parametric instance discrimination (Wu et al. 2018).** Removes the scaling wall by replacing the per-class weight w_j with the ℓ2-normalized *feature itself* v_j, giving a non-parametric softmax P(i|v) = exp(vᵢᵀv/τ) / Σ_j exp(vⱼᵀv/τ) with temperature τ and ‖v‖=1. Computing the denominator over all instances is prohibitive, so it approximates the objective with noise-contrastive estimation and keeps a **memory bank** storing every instance's most recent feature, refreshed as the encoder trains, with the partition function estimated by Monte Carlo. Gap: bank features are *stale* — written by an earlier version of the encoder than the one currently producing the query — and the bank is extra machinery with its own update rule and approximation knobs.

**CPC / InfoNCE (Oord et al. 2018).** The pivotal loss ancestor. An autoregressive context c_t predicts a future latent; among a set X containing the true future (from p(x_{t+k}|c_t)) and N−1 negatives (from p(x_{t+k})), the model classifies the positive via L_N = −E[ log f_k(x_{t+k}, c_t) / Σ_{x_j∈X} f_k(x_j, c_t) ]. This is exactly the categorical cross-entropy of a positive-identification task; its optimum makes f_k proportional to the density ratio, and minimizing it bounds the mutual information between context and future. Gap: CPC ties the task to a specific pipeline — deterministically split the image into a grid of patches, run a PixelCNN context-aggregation network, and let the encoder see only small patches. Architecture-heavy.

**DIM / AMDIM (Hjelm et al. 2018; Bachman et al. 2019).** Maximize mutual information between global and local features by *constraining the receptive field in the architecture* (e.g. replacing many spatial convolutions with 1×1 convolutions), use a tanh-clipped, regularized critic, and rely on a learned augmentation policy. Because the global-to-local prediction task lives in the network design, a standard powerful backbone cannot be used.

**MoCo — momentum contrast (He et al. 2019).** Decouples the number of negatives from the batch size by maintaining a **queue** of keys from previous minibatches (enqueue current, dequeue oldest). Because one cannot backprop through the whole queue and naively reusing an evolving encoder makes queued keys inconsistent, MoCo updates a separate key encoder by momentum, θ_k ← m·θ_k + (1−m)·θ_q with m≈0.999, so the slowly-moving encoder keeps the queued keys mutually consistent. Gap: still a separate mechanism (queue + momentum encoder), the keys are not produced by the *current* query encoder, and the consistency the momentum buys is only approximate.

**N-pair loss and in-batch negatives (Sohn 2016).** Generalizes the triplet loss from one negative to N−1 negatives via a softmax / log-sum-exp form, and points directly at reusing the *other in-batch examples* as negatives (also Doersch & Zisserman 2017; Ye et al. 2019; Ji et al. 2019). Framed for deep metric learning / retrieval rather than as a full self-supervised framework, and still often paired with negative mining.

**FaceNet / triplet loss with semi-hard mining (Schroff et al. 2015).** The margin triplet loss max(0, d(a,p) − d(a,n) + m). It treats all margin-violating negatives alike — it does *not* weight a negative by how hard it is — so in practice it requires explicit **semi-hard negative mining** (selecting negatives inside the margin but farther than the positive) to train at all. The natural baseline against which a softmax-style loss can be contrasted.

## Evaluation settings

Pretraining (learning the encoder without labels) is done primarily on **ImageNet ILSVRC-2012** (Russakovsky et al. 2015), with additional small-scale confirmation on **CIFAR-10** (Krizhevsky 2009). Representation quality is read out under three protocols, all of which predate any new method and form the natural yardstick:

- **Linear evaluation:** freeze the pretrained encoder, train a linear classifier on top of the frozen features, report top-1 / top-5 accuracy (Zhang et al. 2016; Oord et al. 2018; Bachman et al. 2019; Kolesnikov et al. 2019).
- **Semi-supervised fine-tuning:** fine-tune the whole network on a class-balanced 1% or 10% subset of ImageNet labels.
- **Transfer learning:** linear-probe and fine-tune on a suite of natural-image datasets (Food-101, CIFAR-10/100, Birdsnap, SUN397, Cars, Aircraft, VOC2007, DTD, Pets, Caltech-101, Flowers), following the protocol of Kornblith et al. (2019).

## Code framework

The substrate is TensorFlow targeting data-parallel accelerators. A bare self-supervised harness contains a standard encoder, an input-transformation slot, a training-signal slot, an optimizer with its schedule, a training loop, and the linear-probe yardstick.

```python
import tensorflow.compat.v1 as tf

# --- Encoder: standard ResNet backbone. ---
# Returns h, the post-average-pool feature vector we will probe downstream.
encoder = resnet_v1(resnet_depth=50, width_multiplier=1)   # He et al. 2016


# --- Input transformation: the unlabeled training inputs we will construct. ---
def augment(image, height, width):
    # TODO: fill in the unlabeled input generator.
    pass


# --- Training signal: the self-supervised objective. ---
class RepresentationObjective(object):
    def __call__(self, encoder, batch, is_training):
        # TODO: consume a minibatch and produce a scalar loss.
        pass


# --- Optimizer + schedule: standard training machinery. ---
def learning_rate_schedule(base_lr, num_examples):
    # warmup/decay machinery already used by large-scale supervised training
    ...

def make_optimizer(learning_rate):
    # an existing SGD-style optimizer slot
    ...


# --- Linear-probe evaluation: the yardstick, which predates any method. ---
def linear_probe_eval(encoder, labeled_data):
    # freeze the encoder; train a single linear classifier on its frozen
    # features; report test accuracy as the representation-quality proxy.
    ...


# --- Training loop: an ordinary minibatch loop. ---
def train(dataset, num_examples):
    objective = RepresentationObjective()
    lr = learning_rate_schedule(base_lr=base_learning_rate, num_examples=num_examples)
    optimizer = make_optimizer(lr)
    for batch in dataset:                                  # a raw minibatch of images
        training_inputs = augment(batch, height, width)
        loss = objective(encoder, training_inputs, is_training=True)
        optimizer.minimize(loss)                           # update trainable parameters
    return encoder                                         # keep for linear_probe_eval
```
