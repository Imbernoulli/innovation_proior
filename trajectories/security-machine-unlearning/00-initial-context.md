## Research question

A deep classifier `f(·; w)` is given, already trained on the full dataset `D`. A deletion request removes the influence of a designated forget set `D_f` — here a whole class — while preserving everything the model knows about the rest, `D_r = D \ D_f`. Retraining from scratch on `D_r` gives the desired behavior but costs a full training run per request. The object of design is the **unlearning update rule**: a per-step parameter update applied to the already-trained weights that drives down forget-set memorization while holding retained-task accuracy. Everything else in the harness is fixed.

## Prior art / Background / Baselines

Existing approximate-unlearning methods remove `D_f` without retraining, but each leaves a concrete gap:

- **Retraining from scratch on `D_r` (reference, not a method here).** Train a fresh model on the retained data only. Gap: a full training run per request, infeasible at deletion scale.
- **SISA.** Shard the training set, train one model per shard, and ensemble; only the shard containing the deleted point is retrained. Gap: it requires a training-time sharding decision and sacrifices accuracy by ensembling weak shards; it cannot be applied to an already-trained monolithic model.
- **Fisher / NTK scrubbing.** Inject noise or apply a closed-form weight update sized by the Fisher information or an NTK linearization. Gap: it forms or inverts a large second-order matrix and assumes the edit is small; forgetting a whole class is a large edit, and the linearization fails.
- **Amnesiac unlearning.** Record every parameter update during training and subtract those that touched `D_f`. Gap: must store the entire update history and depends on training-time bookkeeping.

The open question is how to edit an already-trained monolithic model given only the deletion-time minibatches.

## Fixed substrate / Code framework

The harness is frozen. It (1) builds a standard vision model — ResNet-20, VGG-16-BN, or MobileNetV2 (CIFAR-adapted, BatchNorm, ReLU); (2) pretrains it on the full training set for 80 epochs with SGD + cosine annealing (`lr=0.1`, momentum `0.9`, weight decay `5e-4`); (3) designates one class as `D_f` and splits train/test into retain and forget; (4) runs the unlearning rule for 20 epochs, feeding it one retain minibatch and one forget minibatch per step (batch size 128, both already on device); (5) evaluates. The unlearning optimizer is a fixed `Adam(model.parameters(), lr=0.001)` instance — the rule does not choose the optimizer or its learning rate. The harness also provides `import torch` and `import torch.nn.functional as F` at module top, so the editable region may use `torch` and `F` without importing them.

## Editable interface

Exactly one region is editable — the `UnlearningMethod` class in `bench/unlearning/custom_unlearning.py`. The contract:

```python
class UnlearningMethod:
    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        # retain_batch: (images, labels) from D_r, on device
        # forget_batch: (images, labels) from D_f, on device
        # optimizer:    the fixed Adam(lr=0.001) instance
        # return: dict with at least "loss"
        ...
```

Every method fills this same contract — one `unlearn_step` that, given the two minibatches and the fixed optimizer, performs one parameter update. The architecture, pretraining, forget split, and evaluation probes are fixed; only the update rule changes.

The starting point is the scaffold default: **retain-only finetuning** — train on the retained minibatch and ignore the forget minibatch entirely.

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
- `forget_mia_auc`: membership-inference-attack AUC on the forget set (lower is better) — members are forget-train confidences, non-members are forget-test confidences, scored by the Mann-Whitney AUC of max-softmax confidence.
- `unlearn_score = (retain_acc + (1 - forget_acc) + (1 - forget_mia_auc)) / 3` (primary, higher is better).
