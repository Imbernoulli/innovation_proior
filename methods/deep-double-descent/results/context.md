# Context: reconciling model size, training time, and test error

## Research question

There is a contradiction at the heart of how the field thinks about model complexity and
generalization. Classical statistical learning theory and the lived experience of deep learning
practitioners give opposite advice about whether bigger models, longer training, and more data help
or hurt — and both have strong empirical support in their own regimes. The question is whether these
competing intuitions can be unified into a single coherent picture of *test error as a function of
complexity*, and if so, what the right notion of "complexity" is.

Concretely: is there one axis along which test error behaves predictably across model size, training
time, regularization, and dataset size simultaneously — one that explains both the classical
"larger is worse past a threshold" U-curve and the modern "larger is better" experience, and that
locates *where* increasing complexity flips from helpful to harmful? A satisfying answer would also
have to explain anomalies that neither classical theory nor modern folklore predicts: settings where
training *longer* first hurts then helps, and settings where adding *more data* hurts.

## Background

**The classical bias–variance tradeoff.** Classical statistical learning theory (Hastie et al.) holds
that as model complexity rises, bias falls but variance grows; past a threshold the variance term
dominates and test error increases. So test error vs. complexity is U-shaped, and the prescription is
"don't make the model too big" and "stop training before you overfit." Conventional wisdom: *larger
models are worse* once you pass the sweet spot, and early stopping helps.

**The modern over-parameterized regime.** Deep networks routinely have far more parameters than
training samples — enough to fit even randomly-labeled data (Zhang et al. 2016) — yet they generalize
well, and across vision and language, *larger models are better* (the consistent experience behind
AlexNet, Inception, GPipe, GPT). Training all the way to zero training error often improves test
performance rather than harming it. This directly contradicts the classical U-curve.

**The point both camps share.** Classical statisticians and deep learning practitioners agree on one
thing: *more data is always better*. Any complete account of the complexity–error relationship will
have to say whether even this holds universally.

**The interpolation threshold and "double descent."** The reconciling observation (Belkin et al.
2019, building on earlier high-dimensional analyses — Opper; Advani & Saxe; the "jamming" transition
of Spigler et al. and Geiger et al.) is that the classical U-curve is only the *first half* of the
story. As model complexity increases, test error follows the classical U up to the point where the
model is just barely able to *interpolate* the training data (reach ≈0 training error) — the
*interpolation threshold* — and then, as complexity increases *further* into the over-parameterized
regime, test error *descends again*. The full curve is therefore "double descent": down, up to a peak
at the interpolation threshold, then down again. Belkin et al. demonstrated this for decision trees,
random features, and small two-layer networks with ℓ2 loss on MNIST/CIFAR. The diagnostic limitation
of this prior framing is that it is organized entirely around the *number of parameters* and shown
mostly on simple models — it does not address modern deep networks trained with SGD, nor the roles of
training time, data augmentation, or regularization, which are not "parameters" but plainly affect
how a model fits.

**Mechanism from linear models.** In the tractable case of linear least-squares / random-feature
regression (Belkin et al.; Hastie et al.; Bartlett et al.; Mei & Montanari; Muthukumar et al.), the
peak at the interpolation threshold is understood: when the number of samples equals the model
dimension there is essentially a *unique* interpolating solution, which is highly sensitive to noise
and mis-specification; in the over-parameterized regime there are many interpolating solutions, and
the minimum-norm one (which gradient descent from zero finds) generalizes well. The peak appears even
without label noise whenever the model family mis-specifies the true distribution.

**Random Fourier Features (Rahimi & Recht 2008).** A random-feature model — a two-layer network whose
first layer is a fixed random map of width d and whose second layer is trained with MSE — is the
clean analytic testbed: its effective complexity is exactly its width d.

## Baselines

The prior accounts of the complexity–error relationship that a unified view must subsume or correct:

- **Bias–variance U-curve (classical).** Test error U-shaped in the number of parameters; complexity
  helps then hurts. Gap: only describes the *under*-parameterized side; it predicts monotone harm
  past the threshold, contradicting the over-parameterized regime where error descends again.

- **"Bigger is better" (modern deep learning practice).** Monotone benefit from scaling parameters
  and training to zero training error. Gap: ignores the peak — there *are* realistic regimes where a
  bigger model or more training is worse — and it has no account of *where* the benefit kicks in.

- **Parameter-count double descent (Belkin et al. 2019).** Test error double-descends as a function of
  the number of parameters, peaking at the interpolation threshold. Gap: tied to parameter count
  alone; says nothing about training time, augmentation, regularization, or sample count as
  complexity-altering knobs, and is demonstrated mostly on simple (non-deep, non-SGD) models.

- **Classical complexity measures (Rademacher complexity, VC dimension).** Quantify a model family's
  capacity to fit (e.g. randomly-labeled) data. Gap as a locator of the peak: they depend only on the
  architecture and data *inputs*, not on the *true labels* (so they cannot explain why adding label
  noise *moves* the peak) and not on the *training procedure* (so they cannot capture epoch-wise or
  data-augmentation effects).

## Evaluation settings

The phenomena are studied by sweeping a complexity knob and recording train and test error to
completion:

- **Architectures, each scaled by a width parameter k:** ResNet18 with convolutional widths
  [k, 2k, 4k, 8k] (standard ResNet18 is k = 64); a 5-layer CNN with four conv layers [k, 2k, 4k, 8k]
  plus a fully-connected layer; a 6-layer encoder–decoder Transformer scaled by its embedding
  dimension d_model with feed-forward width d_ff = 4·d_model.
- **Datasets:** CIFAR-10, CIFAR-100 (image classification); IWSLT'14 German–English (≈160K sentences)
  and WMT'14 English–French (subsampled to 200K) for translation.
- **Optimization:** cross-entropy for the vision nets, trained either with Adam (learning rate 1e-4,
  4K epochs) or SGD (learning rate ∝ 1/√T, 500K steps); Transformers trained 80K steps with 10% label
  smoothing and no dropout.
- **Label noise:** with probability p a training example keeps its correct label, otherwise gets a
  uniformly-random wrong one; the noise is drawn once (not re-sampled per epoch). Test error is
  reported on the clean (or noisy) distribution.
- **Knobs swept:** model width, number of training epochs, dataset size, label-noise level, and the
  presence of data augmentation / regularization.

Metrics: test error / per-token perplexity, alongside *training* error (needed to locate the
interpolation threshold), and early-stopping behavior.

## Code framework

The primitives that exist: trainable architectures whose size is set by a width hyperparameter, the
optimizers, a loss, a label-noising step, and a training loop that records train and test error over
epochs. What is missing is the *organizing quantity* — the single complexity measure along which test
error is to be understood — and the harness that sweeps a knob and measures both that quantity and
the error. The scaffold leaves the complexity measure as an empty slot.

```python
import numpy as np

def make_model(width):
    """Architecture scaled by a width parameter (e.g. ResNet18 widths [k,2k,4k,8k])."""
    # TODO
    pass

def add_label_noise(labels, p, num_classes, rng):
    """Each label kept w.p. (1-p), else replaced by a uniform random label. Drawn ONCE."""
    flip = rng.random(len(labels)) < p
    noisy = labels.copy()
    noisy[flip] = rng.integers(0, num_classes, size=flip.sum())
    return noisy

def train(model, data, optimizer, num_steps):
    """Train; return train_error and test_error (record per-epoch for epoch-wise studies)."""
    # TODO
    pass

def effective_complexity(training_procedure, distribution, epsilon=0.1):
    """The organizing quantity: the largest sample count on which the procedure reaches
    (approximately) zero training error. THE missing definition. TODO."""
    pass

def sweep(knob_values, fixed):
    """Vary one knob (model width, #epochs, or #samples), training to completion,
    and record train/test error to expose how error behaves along the complexity axis. TODO."""
    results = []
    for v in knob_values:
        # build procedure with this knob, train, record (train_err, test_err)
        pass
    return results
```
