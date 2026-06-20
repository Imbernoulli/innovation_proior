## Research question

CIFAR-10 is the most-trained dataset in machine learning, and the cost of every experiment that
touches it is dominated by how long one training run takes. The task here is a *speedrun*: train a
neural net from scratch to a fixed test accuracy on CIFAR-10 as fast as possible on a single GPU.
The accuracy bar is **94% mean test accuracy** (the threshold used by the 2017–2020 Stanford
DAWNBench track, and the level of human accuracy reported for CIFAR-10), with documented **95%** and
**96%** variants as harder targets. The hardware is held fixed: one **NVIDIA A100**. Timing begins
the moment the method is first given the training data and ends when it emits test-set predictions;
a method is valid only if its *mean* accuracy over repeated runs clears the bar. Warmup runs on dummy
data (to prime the GPU) are permitted and not counted, and arbitrary test-time augmentation is allowed.

Everything that defines the *problem* is frozen: the 50,000-image train set, the 10,000-image test
set, the 94% bar, and the single-A100 budget. The single free variable is the **training method** —
the network architecture, the initialization, the optimizer, the data augmentation, and the inference
procedure. Because compute is capped at one GPU and the data is fixed, you cannot buy accuracy with
scale; the only lever is algorithmic. The metric the ladder is ranked on is **A100-seconds to reach
the accuracy bar** (lower is better); each rung is a real record that drove that number down.

## Prior art before the first rung

The lineage the ladder climbs out of is a chain of public speedrun records, each one a faithful
single-GPU reproduction that then got faster:

- **David Page / Myrtle "How to train your ResNet" (2019).** A carefully engineered ResNet-style
  training that reached 94% in **26 V100-seconds** (≈ 10.4 A100-seconds), winning the DAWNBench track.
  It established the playbook: a small VGG-flavoured residual net, Nesterov SGD with a triangular
  learning-rate schedule, label smoothing, horizontal-flip + random-translation augmentation, biases
  disabled on convolutions, and the decoupled batch-size/lr/momentum/wd parametrization that lets each
  hyperparameter be tuned independently. It also introduced two of the tricks the ladder inherits:
  *patch-whitening* of the first layer and a *64×* learning-rate boost on the BatchNorm biases.
- **tysam-code/hlb-CIFAR10 (2023).** A from-scratch rewrite that pushed the record to 94% in
  **6.3 A100-seconds** — the prior state of the art. It supplies the concrete network this ladder
  uses (a 2×2-stride first conv, three conv blocks, GELU, BatchNorm), Dirac/identity-style conv
  initialization, the frozen patch-whitening layer, the final-layer scaling, the Lookahead optimizer,
  and horizontal-flip TTA.

The baseline the ladder actually starts from is the airbench *unwhitened* baseline: this exact
network, trained with Nesterov SGD, a triangular LR schedule, label smoothing 0.2, and flip+translate
augmentation, with optimized lr/momentum/wd. With **no whitening and none of the speedup features
added**, it reaches 94% mean accuracy in **45 epochs taking 18.3 A100-seconds**. That is the number
every rung below has to beat.

## The fixed substrate

The network is a VGG-like convolutional net of ≈1.97M parameters: a 2×2-stride first convolution
with no padding (shrinking the spatial path 31→15→7→3 rather than the usual 32→16→8→4), then three
blocks of 3×3 convolutions with `MaxPool2d(2)`, BatchNorm, and GELU activations, a final max-pool,
and a bias-free linear head whose output is scaled by 1/9. Convolutional and linear biases are
disabled; BatchNorm weights are frozen at 1 and only its biases train. Training is Nesterov SGD at
batch size 1024 with label smoothing 0.2 and a triangular LR schedule (start at 0.2× the max, peak
20% of the way through, decay toward zero); augmentation is horizontal flip plus 2-pixel reflection-
padded random translation; evaluation uses horizontal-flip TTA. This loop, its hyperparameters
expressed in decoupled form, is the frozen scaffold. Each rung below is a single named change to it
that moves the seconds-to-94% number down — except the last two rungs, which trade time to clear the
harder 96% bar and then claw that time back.

## Evaluation settings

The ranking metric is **A100-seconds to clear the accuracy bar** (lower is better), reported as the
mean over many runs (n in the hundreds to thousands, per record). Each rung's feedback is the real
published number from the airbench repo or its accompanying paper (arXiv:2404.00498): the epochs-to-bar
and the wall-clock seconds it took, plus the mean accuracy attained. The 94%-target rungs (1–6, and
the Muon finale) are all measured on the same 94% bar so their seconds are directly comparable; the
two higher-accuracy rungs (residual-scaling and data-filtering) are measured on the 96% bar and are
labeled as such. No number here is re-run by us — every figure is the repo's own published record.
