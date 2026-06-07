# Deep Double Descent synthesis

arXiv 1912.02292 (verified). Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever. ICLR 2020 (also JSTAT 2021).
This is an EMPIRICAL-FIRST / framework paper: the "method" is a conceptual definition (EMC) + a
hypothesis, validated by a large experimental sweep. No new algorithm/architecture.

## The two conventional wisdoms in tension (motivation)
1. Classical statistics (bias-variance tradeoff, Hastie et al.): higher complexity → lower bias but
   higher variance; past a threshold variance dominates → test error U-shaped → "larger models are
   worse." Conventional wisdom: stop before overfitting; early stopping helps.
2. Modern deep learning: nets with millions of params (enough to fit random labels, Zhang et al.
   2016) generalize well; "larger models are better" (AlexNet, GPipe, Inception, GPT). And training
   to zero train error often helps.
3. Both camps agree: "more data is always better."
These contradict: classical U-curve says bigger is worse; practice says bigger is better.

## Prior observation: double descent (Belkin et al. 2019, "Reconciling modern ML with bias-variance")
Belkin et al. postulated in generality the "double descent" curve: test error vs model complexity is
U-shaped UNTIL the interpolation threshold (where train error hits ~0), then DESCENDS AGAIN as
complexity increases past it. Demonstrated for decision trees, random features, 2-layer nets with
ℓ2 loss on MNIST/CIFAR. Earlier hints: Opper, Advani & Saxe (high-dim dynamics), Spigler/Geiger
(jamming transition). Limitation: framed around NUMBER OF PARAMETERS only; demonstrated mostly on
simple models, not modern deep nets / SGD / standard practice.

## The conceptual move: Effective Model Complexity (EMC)
The number of parameters is too narrow a complexity axis — training time, data augmentation,
regularization all change behavior but aren't "parameters." Need ONE axis that subsumes them.
Definition (training procedure T = anything mapping a labeled train set S to a classifier T(S)):
  EMC_{D,ε}(T) := max { n | E_{S~D^n}[ Error_S(T(S)) ] ≤ ε },
the MAXIMUM number of training samples on which T achieves ≈0 (≤ε) TRAIN error on average.
ε used heuristically = 0.1.
Crucially EMC depends on (a) data distribution, (b) architecture, AND (c) the training procedure —
so increasing TRAINING TIME increases EMC (train longer → fit more samples), and so does width,
data-aug off, less regularization, etc. This is the unifying axis.

Contrast with classical complexity (Rademacher, VC): EMC (1) depends on the TRUE LABELS of the
distribution (Rademacher uses random labels), and (2) depends on the TRAINING PROCEDURE not just
architecture. These two features are why EMC can locate the peak where Rademacher/VC cannot — adding
label noise shifts the peak (so label-dependence needed), and epoch-wise/data-aug effects need
training-procedure dependence.

## Generalized Double Descent Hypothesis (informal)
For natural distribution D, NN training procedure T, small ε, predicting n samples:
- Under-parameterized (EMC ≪ n): any perturbation of T that INCREASES effective complexity DECREASES
  test error. (classical regime, complexity helps)
- Over-parameterized (EMC ≫ n): any perturbation that increases effective complexity DECREASES test
  error. (modern regime, bigger/longer helps)
- Critically parameterized (EMC ≈ n): increasing effective complexity MIGHT increase OR decrease
  test error. (the danger zone)
The test error PEAKS around EMC ≈ n — the "interpolation threshold." There's a "critical interval"
around it whose width depends on D and T (not formally characterized). "Sufficiently smaller/larger"
left informal.

## Three corollaries (each = one way to vary EMC across n)
1. MODEL-WISE double descent: fix large #steps, vary model SIZE (width). Test error U then peaks at
   interpolation threshold then descends again. Peak strongest with label noise; sometimes a plateau
   (no noise) that becomes a peak with noise. Realistic settings where BIGGER MODELS ARE WORSE.
   All modifications that raise the interpolation threshold (label noise, data-aug, more samples)
   shift the peak toward LARGER models.
2. EPOCH-WISE double descent: fix a large model, vary TRAINING TIME. Since training longer raises
   EMC, a big enough model goes under→over parameterized over training. Test error: decreases, then
   INCREASES near interpolation, then decreases AGAIN — "training longer can correct overfitting."
   Medium models: classical U (early stop best). Small models: monotone decrease.
3. SAMPLE-WISE non-monotonicity: fix model+procedure, vary n. More samples (a) shrinks overall error
   but (b) SHIFTS the peak right (fitting more samples needs bigger model). Near critical regime
   these cancel → MORE DATA DOESN'T HELP, and in some settings (Transformers on IWSLT) MORE DATA
   HURTS. By increasing n the SAME T goes from over- to under-parameterized.

## Corollary: early stopping
Optimal early stopping often REMOVES the double-descent phenomena — consistent with the hypothesis,
because if early stopping prevents reaching ≈0 train error, EMC never reaches n. (One setting shows
model-wise DD even with optimal early stopping: ResNet CIFAR-100 no noise.) Early stopping only helps
in the narrow critical regime.

## Mechanism intuition (in-frame, from linear-model theory)
At interpolation threshold, there's effectively ONLY ONE model that fits the train data → extremely
sensitive to label noise / model mis-specification: barely able to fit, so forcing it to fit slightly
noisy labels destroys global structure → high test error. Over-parameterized: MANY interpolating
models fit the train set, and SGD finds one that "absorbs/memorizes" the noise while keeping good
distributional performance (for linear: minimum-norm solution, = GD from 0 init). Theoretically
justified for linear least squares / random features. Manifests even WITHOUT label noise under model
mis-specification.
Label noise isn't fundamental — it's a proxy for "harder distribution" / more model mis-specification
(even pseudorandom noise invertible to Bayes-optimal would give same DD).

## Random Fourier Features case study (clean analytic anchor)
RFF (Rahimi & Recht 2008) = 2-layer net, e^{-ix} activation, first layer ~N(0,1/d) FIXED, width d =
model size, second layer init 0, trained MSE (gradient flow → min-norm). Here EMC = d exactly. Test
error grid over (n, d): peak follows n = d. Model-wise (vary d) and sample-wise (vary n) DD both
appear. Proof that DD is not deep-net-specific; even linear/kernel methods show it. (More data can
hurt even for linear models.)

## Experimental setup (the harness = the "code")
Architectures, each scaled by a width parameter k:
- ResNet18: conv layer widths [k,2k,4k,8k], standard = k=64.
- 5-layer CNN: 4 conv layers [k,2k,4k,8k] + 1 FC; k=64 → >90% CIFAR-10 test acc w/ aug.
- Transformer: 6-layer encoder-decoder (Vaswani), scale embedding d_model, d_ff = 4·d_model.
Datasets: CIFAR-10, CIFAR-100, IWSLT'14 de-en (160K), WMT'14 en-fr (subsampled 200K).
Training: ResNets/CNNs cross-entropy; (1) Adam lr=1e-4 for 4K epochs; (2) SGD lr ∝ 1/√T for 500K
steps. Transformers: 80K steps, 10% label smoothing, no dropout.
Label noise prob p: each sample correct w.p. (1-p), else uniformly-random WRONG label; sampled ONCE
(not per epoch). Plot test error either on noisy or clean distribution (linear rescaling).
To MEASURE EMC: it's the max n on which the procedure reaches ≤ε train error — practically read off
as the interpolation threshold (the model size / epoch / sample count where train error hits ~0).

## Design-decision → why
- Define EMC via max-n-reaching-zero-train-error (not param count): one scalar that unifies width +
  training time + data-aug + regularization + samples; everything that affects "how much it can fit."
- EMC measured at TRAIN error ≤ ε: interpolation (zero train error) is the empirical transition point.
- EMC depends on true labels + training procedure: needed to locate the peak (label noise shifts it;
  epoch/aug effects exist) — Rademacher/VC lack both and can't.
- ε = 0.1: heuristic threshold for "≈ zero train error."
- Add label noise to make DD prominent: it's a controllable proxy for model mis-specification /
  distribution hardness, which is the real driver.
- Vary EMC three ways (size, epochs, samples): each yields a corollary phenomenon, all from one
  hypothesis — strong evidence the unifying axis is right.
- Train to completion (fixed large #steps) for model-wise: ensures EMC reflects the procedure's full
  fitting capacity, not premature stopping.
