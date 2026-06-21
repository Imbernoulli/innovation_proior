## Research question

Train **ResNet-50 on ImageNet-1k to 76.6% top-1 validation accuracy at the lowest wall-clock time and dollar cost.** The problem definition is frozen: the model family is ResNet-50, the dataset is ImageNet-1k, and the accuracy bar is the 76.6% reached by the reference schedule. The only free variable is which training-efficiency methods are layered onto that fixed job.

A better method moves the time-accuracy-cost frontier: it reaches the same accuracy in less wall-clock time, or it recovers enough accuracy headroom that the schedule can be shortened without dropping below the bar. The yardstick is top-1 accuracy on the held-out ImageNet validation split against wall-clock training time on a fixed 8×A100 machine. The reference baseline — a plain ResNet-50 trained with the standard recipe — takes on the order of 3.5 hours to reach 76.6%. The question at each step is: what is the next method that improves the frontier, and why?

## Prior art / Background / Baselines

The starting point is the vanilla ResNet-50 / ImageNet supervised recipe with no efficiency methods stacked on top.

- **Vanilla ResNet-50 with SGD + momentum, cosine/step LR decay, and standard crops/flips.** Core idea: a 50-layer residual network is trained with SGD with momentum 0.9, a learning rate decayed over roughly 90 epochs, L2 weight decay, and random-resized-crop plus horizontal-flip augmentation, reaching about 76% top-1. Gap: it is slow and expensive — every step passes a full-resolution image through the full-capacity network, the only time-accuracy knob is the number of epochs, and nothing in the recipe exploits problem structure to get the same accuracy for less compute.

- **Weight decay folded into SGD as L2 regularization.** Core idea: the optimizer penalizes weight magnitude through a loss term whose update scaling is tied to the learning rate. Gap: this couples the learning rate and the effective regularization strength, so changing one during a sweep changes the other and makes independent tuning difficult when other schedule changes are stacked on top.

- **Schedule scaling: training longer or shorter.** Core idea: vary the number of epochs to trade accuracy for wall-clock time. Gap: halving the schedule roughly halves training time but drops accuracy (for example, from 76.6% to about 75.6%), so it slides along the existing frontier rather than pushing it outward.

## Fixed substrate / Code framework

The scaffold is a standard ResNet-50 / ImageNet supervised-classification loop that must stay fixed: a 50-layer residual network, cross-entropy loss, SGD with momentum, a learning-rate schedule with warmup, batched training over the ImageNet train split, and top-1 accuracy on the held-out validation split. Efficiency methods are inserted through three hooks:

- **Model surgery** — swapping modules inside the network (e.g., replacing strided convolutions or pooling layers, or inserting attention blocks).
- **Batch-level transforms** — operating on input/target tensors each step (e.g., resizing or augmenting).
- **Optimizer / schedule / data-pipeline changes** — e.g., a different weight-decay treatment, an averaged weight copy for evaluation, or a faster data loader.

No method may change the model family, the dataset, or the 76.6% accuracy target.

## Editable interface

The editable surface is the ordered list of `Algorithm` objects passed to the trainer and their hyperparameters. The trainer loop, model architecture, dataset, loss, evaluation metric, and hardware setting are read-only. To add a method, instantiate or implement an `Algorithm` and append it to the list; the trainer invokes it at events such as `INIT`, `AFTER_DATALOADER`, `BEFORE_LOSS`, `AFTER_BACKWARD`, `BATCH_END`, `EPOCH_END`, and `EVAL_START`. A method may modify the model state, the current batch, the loss, the optimizer, or maintain state across steps. The baseline list is empty.

## Evaluation settings

The metric is **top-1 validation accuracy on ImageNet-1k** (higher is better), reported against **wall-clock training time** on a fixed 8×A100 machine (lower is better) and the corresponding **dollar cost** of that machine-time. The reference schedule reaches 76.6% top-1; every method is judged by whether it improves the attainable time-vs-accuracy tradeoff for ResNet-50 on ImageNet at that target. Per-method effects are reported as an **accuracy delta** (percentage points of top-1) and/or a **throughput / wall-clock delta** (samples per second or percent change in training time). Some methods are pure quality levers, some are pure speed levers, and the goal is to compose them so the frontier moves.
