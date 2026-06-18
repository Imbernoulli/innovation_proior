## Research question

A deep classifier `f(·; w)` was trained on the full dataset `D`. Someone invokes the right to be
forgotten: remove the influence of a designated forget set `D_f` — here a whole class — while keeping
everything the model knows about the rest, `D_r = D \ D_f`. Retraining from scratch on `D_r` is the
gold standard, but it costs a full training run per deletion request, so it is exactly the thing to
avoid. The single thing being designed is the **unlearning update rule**: a per-step parameter update,
applied to the already-trained weights, that drives forget-set memorization down while holding
retained-task accuracy up. Everything else about the harness is fixed.

## Prior art before the first rung (approximate-unlearning lineage)

The first rung reacts to a line of approximate-unlearning ideas — each removes `D_f`'s influence
without retraining, and each leaves a gap the ladder will work through.

- **Retraining from scratch on `D_r` (the reference, not a method here).** Train a fresh model on the
  retained data only. This is the behavior every approximate method tries to imitate — a model that
  *never studied* `D_f` and so sits at generalization-level uncertainty there. Gap: a full training
  run per request, infeasible at deletion scale.
- **SISA (Bourtoule et al. 2021).** Shard the training set, train one model per shard, ensemble; on a
  deletion only the shard that held the point is retrained. Gap: a *training-time* decision — it
  dictates how the model must have been trained and pays an accuracy cost from ensembling weak shards.
  Useless for a model already trained as one monolith.
- **Fisher / NTK scrubbing (Golatkar, Achille, Soatto 2020).** Write a closed-form weight update — a
  noise injection sized by the Fisher information, or an NTK linearization of the SGD dynamics — that
  pushes the parameters toward the retrain-from-scratch distribution. Gap: it forms/inverts a
  Hessian-or-Fisher (quadratic in samples) and rests on the edit being *small*; forgetting a whole
  class (≈10% of the data) is not a small edit, and the linearization cracks exactly there.
- **Amnesiac unlearning (Graves et al. 2021).** Log every batch's parameter update during training and
  subtract the ones that touched `D_f`. Gap: store the entire update history and stay tied to
  training-time bookkeeping.

What they share as constraints: a demand on *how the model was trained*, or heavy second-order
objects, or only one forgetting mode. The ladder starts from a method that takes the trained model as
a black box of weights and edits it with the minibatches the harness hands it, then works through the
forgetting *mechanism* — passive erosion, active ascent, distillation — toward something that forgets
sharply without wrecking utility.

## The fixed substrate

The harness is frozen and must not be touched. It (1) builds a standard vision model — ResNet-20,
VGG-16-BN, or MobileNetV2 (CIFAR-adapted, BatchNorm, ReLU); (2) pretrains it on the **full** training
set for 80 epochs with SGD + cosine annealing (`lr=0.1`, momentum `0.9`, weight decay `5e-4`); (3)
designates one class as `D_f` and splits train/test into retain and forget; (4) runs the unlearning
rule for 20 epochs, feeding it one retain minibatch and one forget minibatch per step (batch size 128,
both already on device); (5) evaluates. The unlearning optimizer is a fixed `Adam(model.parameters(),
lr=0.001)` instance — the rule does not get to choose the optimizer or its learning rate. The harness
also provides `import torch` and `import torch.nn.functional as F` at module top, so the editable
region may use `torch` and `F` without importing them.

## The editable interface

Exactly one region is editable — the `UnlearningMethod` class in
`bench/unlearning/custom_unlearning.py`. The contract:

```python
class UnlearningMethod:
    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        # retain_batch: (images, labels) from D_r, on device
        # forget_batch: (images, labels) from D_f, on device
        # optimizer:    the fixed Adam(lr=0.001) instance
        # return: dict with at least "loss"
        ...
```

Every method on the ladder is a fill of this same contract — one `unlearn_step` that, given the two
minibatches and the fixed optimizer, performs one parameter update. The architecture, pretraining,
forget split, and evaluation probes are fixed; only the update rule changes.

The starting point is the scaffold default: **retain-only finetuning** — train on the retained
minibatch and ignore the forget minibatch entirely.

```python
# EDITABLE region of custom_unlearning.py -- default fill (retain-only finetuning)
class UnlearningMethod:
    """Default retain-only finetuning update."""

    def __init__(self):
        self.forget_weight = 0.0

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        retain_x, retain_y = retain_batch
        logits = model(retain_x)
        loss = F.cross_entropy(logits, retain_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
```

## Evaluation settings

Three benchmarks, each forgetting class 0, single seed {42}:

- `resnet20-cifar10-class0`: ResNet-20 on CIFAR-10.
- `vgg16bn-cifar100-class0`: VGG-16-BN on CIFAR-100.
- `mobilenetv2-fmnist-class0`: MobileNetV2 on FashionMNIST (held out / hidden).

Reported per benchmark:
- `retain_acc`: accuracy on non-forget test data (higher is better).
- `forget_acc`: accuracy on forget-class test data (lower is better).
- `forget_mia_auc`: membership-inference-attack AUC on the forget set (lower is better) — members are
  forget-train confidences, non-members are forget-test confidences, scored by the Mann-Whitney AUC of
  max-softmax confidence.
- `unlearn_score = (retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3` (primary, higher is
  better).
