# Context: removing data influence from a trained deep network (circa 2021-2022)

## Research question

A deep classifier `M` has been trained on a dataset `D_c = {(x_i, y_i)}`. A user — or a regulator acting
for one — now invokes the *right to be forgotten* (GDPR, CCPA): some subset of the training data, the
**forget set** `D_f`, must have its influence removed from the deployed model, while the influence of
the **retain set** `D_r = D_c \ D_f` is left intact. The forget set can be an entire class, several
classes, a cohort inside one class, or a random subset spread across classes.

One reference answer is to retrain from scratch on `D_r` alone — the *retrained* (or *gold*) model.
Training a modern deep net is expensive, and deletion requests arrive repeatedly. The problem studied here
is **approximate unlearning**: starting from the already-trained weights, produce a model that *behaves
like* the retrained model — having lost its specific knowledge of `D_f` while keeping its accuracy on
`D_r` — at a small fraction of the cost of retraining.

The setting is an *already-trained* monolithic deep net, with no prior reorganization of training and no
recorded information about how the model was optimized. The intended behavior on `D_f` is the
*generalization-level* behavior of a model that never saw it — uncertain rather than adversarially wrong,
since a model that is *confidently wrong* on the forget data advertises that those samples were specially
treated, which a membership-inference attacker can exploit. We also want a way to *measure* whether
forgetting happened.

## Background

**Memorization and the deletion obligation.** Deep nets memorize individual training samples
(Feldman 2020; Carlini et al. 2019), so a sample's influence persists in the weights and is removed by an
active procedure rather than by withholding the data. Machine unlearning as a named problem was introduced
by Cao & Yang (2015), who recast learning algorithms into a summation form so a deleted point's
contribution could be subtracted; this applies to statistical-query-style learners.

**The retrained model as ground truth.** The probabilistic notion of unlearning (Ginart et al. 2019),
borrowing the differential-privacy template, asks that the output distribution of the unlearned model be
close to that of the model retrained without `D_f`. Subsequent theory (Guo et al. 2020 certified removal;
Neel et al. 2021 descent-to-delete; Sekhari et al. 2021) works in the convex / linear regime and gives
guarantees there. The retrained model is the thing every method imitates.

**The tradeoff and its empirical signature.** There is a tension between forgetting and retain accuracy:
pushing hard on forgetting moves retain accuracy, while staying conservative leaves measurable traces of
`D_f`. A diagnostic that recurs in this literature: an over-aggressive method drives forget-set accuracy to
*exactly zero*, whereas the retrained model typically retains some nonzero, near-chance accuracy on the
forget class — the class is not pathologically *avoided*, merely not specifically learned. Confidently-wrong
predictions on `D_f` are themselves information.

**Knowledge distillation.** A standard technique is distillation (Hinton, Vinyals & Dean 2015): a *student*
network is trained to match a *teacher's* output distribution rather than the hard labels. With
temperature-softened softmax `p_i(z; T) = softmax(z_i / T)`, the teacher's soft probabilities carry "dark
knowledge" — the relative mass it places on the wrong classes — and the student minimizes the divergence to
those soft targets,

```
L_KD = KL( p^teacher(T) || p^student(T) ) ,   p(T) = softmax(logits / T),
```

(commonly weighted by `T^2` when mixed with a hard-label term, to keep the softened-loss gradient on the
same scale). Distillation makes the student *copy whatever distribution the teacher emits on a given
input*. Jensen–Shannon divergence, `JS(p, q) = ½ KL(p‖m) + ½ KL(q‖m)` with `m = (p+q)/2`, is the
symmetric, bounded variant used to *compare* two output distributions.

## Baselines

The prior deep-network unlearning methods a new method would be measured against.

**SISA — sharded training (Bourtoule et al. 2021).** Partition `D_c` into disjoint shards, train one
sub-model per shard, and aggregate (e.g. by voting); cache intermediate checkpoints. A deletion request
forces retraining only of the single shard (and slice) that held the point, so cost drops by the shard
count. It is a *training-time* architecture: the sharding and checkpointing are set up from the beginning,
and the sub-models and checkpoints are stored.

**Fisher scrubbing and its linearized variants (Golatkar, Achille & Soatto 2020 "eternal sunshine";
Golatkar et al. 2020 NTK; Golatkar et al. 2021 mixed-linear).** Derive a closed-form weight update that
moves the parameters toward the distribution a retrained model would have, using the Fisher information
matrix (a noise injection that scrubs the readable information) or a neural-tangent-kernel linearization
of training dynamics; the mixed-linear version trains an auxiliary linearized model alongside the original.
The scrubbing derivation uses the SGD optimization trajectory; the Fisher / Hessian approximations are
second-order; and the NTK / mixed-linear variants train and store an extra approximating model.

**UNSIR — impair-then-repair (Tarun et al. 2021).** For class unlearning, learn an
error-maximizing *noise* tensor for the forget class, fine-tune the model on that noise to impair its
forget-class behavior, then briefly fine-tune on retain data to repair collateral damage. It operates at
the class level via the noise construction, and drives forget-class accuracy to 0.

**Amnesiac unlearning (Graves et al. 2021).** Log, during training, the parameter update contributed by
each batch; to unlearn, subtract back the logged updates that touched the forget data. It keeps the
per-batch update history throughout training and subtracts the logged contributions.

## Evaluation settings

The yardsticks in use for deep unlearning:

- **Datasets / architectures.** Image classification on CIFAR-10, CIFAR-100 (and its 20 super-classes),
  and FashionMNIST, with standard backbones — ResNet-(18/20/34), VGG-style and AllCNN convolutional
  nets, MobileNetV2, and Vision Transformers — plus non-vision settings (LSTM for human-activity
  recognition, small DNN for epileptic-seizure detection) to test modality breadth.
- **Forgetting protocols.** Single-class forgetting (designate one class as `D_f`), multi-class,
  sub-class-from-superclass, and random-subset forgetting across classes; also *sequential* requests
  (forget several cohorts one after another).
- **Metrics.** Accuracy on the retain test set (higher is better) and on the forget test set (closer to
  the retrained model — typically low but not adversarially zero — is better); membership-inference-attack
  success/AUC on `D_f` (lower is better); and, where a retrained model is available, activation distance
  to it and relearn time. A combined unlearning score balancing retain accuracy against forget accuracy
  and forget-MIA is the natural single summary.
- **Protocol.** Pretrain the backbone to convergence on `D_c`; freeze that as the original model; then
  run the candidate unlearning procedure for a small budget over all of `D_f` plus a chosen subset of
  `D_r`. Record the retained-data fraction, batch size, optimizer, learning rate, epoch count, and any
  distillation temperature, because these knobs control how much the update moves the already-trained
  weights. Evaluate the resulting model on the metrics above.

## Code framework

The unlearning procedure plugs into a fixed harness that has already pretrained the backbone and split the
training data into a forget set and a retained pool. The harness can mark each example with an unlearning
label, combine the requested forget data with a retained subset, build a shuffled loader, and optimize only
the live model. The open slot is the objective that turns a batch, its unlearning labels, and any auxiliary
model outputs the update rule elects to use into a scalar loss.

```python
import torch
from torch.utils.data import Dataset, DataLoader


class UnLearningData(Dataset):
    """Combine the requested forget data with a retained subset and expose only
    an input plus an unlearning label: 1 means forget, 0 means retain."""

    def __init__(self, forget_data, retain_data):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.forget_len + self.retain_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            return x, 1
        x = self.retain_data[index - self.forget_len][0]
        return x, 0


def candidate_unlearning_loss(output, labels, support_logits, temperature):
    # TODO: the distribution-matching objective we will design.
    pass


def unlearning_step(model, support_models, unlearn_data_loader, optimizer, device, temperature):
    losses = []
    for x, labels in unlearn_data_loader:
        x, labels = x.to(device), labels.to(device)
        with torch.no_grad():
            support_logits = [support_model(x) for support_model in support_models]
        output = model(x)

        optimizer.zero_grad()
        loss = candidate_unlearning_loss(output, labels, support_logits, temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
    return torch.stack(losses).mean()


def run_unlearning(model, support_models, retain_data, forget_data, optimizer, device, epochs,
                   temperature, batch_size):
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearn_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        unlearning_step(model, support_models, unlearn_loader, optimizer, device, temperature)
```

The harness supplies a mixed unlearning loader, optional support-model outputs, and an optimizer over the
live model; `candidate_unlearning_loss` is where the update rule will live.
