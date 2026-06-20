## Research question

Train **ResNet-50 on ImageNet to a fixed target top-1 accuracy at the lowest wall-clock time and dollar
cost.** Everything about the learning problem that *defines* the task is frozen: the model family is
ResNet-50, the dataset is ImageNet-1k, and the bar to clear is the standard validation accuracy that the
library's reference schedule reaches — **76.6% top-1**. The single free variable is *which
training-efficiency methods are stacked on top of the one fixed training job* — the algorithmic speedups
(model-surgery transforms, data augmentations, optimizers, schedule and data-pipeline changes) layered onto
an otherwise ordinary ResNet-50 / ImageNet run. A "better" rung is one that pushes the
time-accuracy-cost frontier: it either reaches the same accuracy in less wall-clock time, or buys back
accuracy headroom that lets the schedule be shortened, or both. The yardstick is the same throughout —
top-1 accuracy on the held-out ImageNet validation split, against wall-clock training time on a fixed 8×A100
machine.

The reference point that anchors the whole exercise: a plain ResNet-50 trained for the standard schedule in
this library reaches 76.6% validation accuracy, and on an 8×A100 box a careful baseline takes on the order
of 3.5 hours. The recipe that this ladder builds toward reaches that **same 76.6%** in a small fraction of
that time. Each rung is one named method added to the stack; the question at every step is "what is the next
method that moves the time-accuracy frontier, and why."

## Prior art before the first rung

The starting point is a vanilla ResNet-50 / ImageNet recipe — the well-worn 2015-era convolutional
classifier trained the standard way, with nothing layered on. This is the floor the ladder climbs out of,
and the precedents it reacts to are the components of that floor.

- **ResNet-50 trained with SGD + momentum, cosine or step LR decay, standard crops/flips.** The default
  recipe: 50-layer residual network, stochastic gradient descent with momentum 0.9, a learning-rate
  schedule decayed over ~90 epochs, weight decay as L2 regularization, and the usual random-resized-crop +
  horizontal-flip augmentation. It reaches ~76% top-1 and is the universal reference. Gap: it is *slow* and
  *expensive* — every training step runs the full-resolution image through the full-capacity network, the
  GPU is not used in its most efficient memory layout, and the only knob for trading time against accuracy
  is the number of epochs. Nothing in the recipe exploits the structure of the problem to get the same
  accuracy for less compute.

- **L2 regularization folded into SGD as "weight decay."** In the textbook SGD update, weight decay and L2
  regularization are treated as the same thing, and the decay term is scaled by the learning rate. Gap:
  this *couples* the learning rate and the weight decay — change one in a sweep and the effective value of
  the other moves with it — which makes the regularization strength hard to tune independently and is a
  known footgun when stacking other schedule-altering methods on top.

- **The single lever everyone reaches for first: train longer or train shorter (schedule scaling).** The
  one universally available time-accuracy knob is the length of training. Scaling the schedule down by a
  ratio shrinks cost roughly proportionally but costs accuracy: training ResNet-50 on ImageNet for the
  standard schedule reaches 76.6%, while halving the schedule (scale-schedule ratio 0.5) drops it to 75.6%.
  Gap: this is a *pure* tradeoff along a single line — it buys speed only by giving up accuracy point-for-
  point, with no structural insight. To move the *frontier* rather than slide along it, you need methods
  that change *how* each step spends its compute, not just *how many* steps there are.

The ladder here is exactly the set of training-efficiency methods that get stacked onto this fixed
ResNet-50 / ImageNet job, ordered so that each rung adds one distinct method and the stack as a whole walks
the time-accuracy-cost frontier toward "76.6% in a fraction of the baseline time."

## The fixed substrate

A standard ResNet-50 / ImageNet supervised-classification loop is frozen: a 50-layer residual network,
cross-entropy loss, SGD-with-momentum optimization, a learning-rate schedule with warmup, batched training
over the ImageNet train split, top-1 accuracy measured on the held-out validation split. The methods on the
ladder are inserted into this loop in three places — by **model surgery** (swapping modules in the network,
e.g. replacing strided convolutions or pooling layers, or inserting attention blocks), by **batch-level
transforms** (operating on the input/target tensors each step, e.g. resizing or augmenting), and by
**optimizer / schedule / data-pipeline** changes (e.g. a decoupled-weight-decay optimizer, an averaged set
of weights for evaluation, or a faster data loader). The trainer exposes each method as an `algorithm`
object dropped into a list; the trainer runs it at the appropriate point in the loop. No rung changes the
model family, the dataset, or the accuracy bar.

## Evaluation settings

The metric is **top-1 validation accuracy on ImageNet-1k** (higher is better), reported against
**wall-clock training time** on a fixed 8×A100 machine (lower is better) and the corresponding **dollar
cost** of that machine-time. The reference schedule reaches 76.6% top-1; the recipe being assembled targets
that same accuracy at a fraction of the time and cost. Per-method effects are reported the way the library
characterizes them: an **accuracy delta** on ResNet-50 / ImageNet (in percentage points of top-1), and/or a
**throughput / wall-clock delta** (samples-per-second or percent change in training time). Some methods are
*pure quality* levers (accuracy up, throughput roughly unchanged or slightly down), some are *pure speed*
levers (throughput up, accuracy roughly flat or slightly down), and the point of the ladder is to compose
them so the *frontier* moves. Each rung is judged on whether it improves the attainable
time-vs-accuracy tradeoff for ResNet-50 on ImageNet.
