# Context: removing data influence from a trained deep network (circa 2021-2022)

## Research question

A deep classifier `M` has been trained on a dataset `D_c = {(x_i, y_i)}`. A user — or a regulator acting
for one — now invokes the *right to be forgotten* (GDPR, CCPA): some subset of the training data, the
**forget set** `D_f`, must have its influence removed from the deployed model, while the influence of
the **retain set** `D_r = D_c \ D_f` is left intact. The forget set can be an entire class, several
classes, a cohort inside one class, or a random subset spread across classes.

The unimpeachable answer is to retrain from scratch on `D_r` alone — the *retrained* (or *gold*) model.
But training a modern deep net is expensive, and deletion requests arrive repeatedly; retraining per
request is not viable in production. So the real problem is **approximate unlearning**: starting from
the already-trained weights, cheaply produce a model that *behaves like* the retrained model — it should
have lost its specific knowledge of `D_f` while keeping its accuracy on `D_r` — in a small fraction of
the cost of retraining.

Three further constraints make the problem hard rather than trivial. First, it must work on an
*already-trained* monolithic deep net, without having reorganized training in advance and without prior
information about how the model was optimized. Second, simply destroying performance on `D_f` is not
enough and can be actively harmful: a model that is *confidently wrong* on the forget data advertises
that those samples were specially treated, which a membership-inference attacker can exploit. The target
behavior on `D_f` is the *generalization-level* behavior of a model that never saw it — uncertain, not
adversarially wrong. Third, we need a way to *measure* whether forgetting actually happened that does not
itself require the expensive retrained model as a yardstick.

## Background

**Memorization and the deletion obligation.** Deep nets memorize individual training samples
(Feldman 2020; Carlini et al. 2019), so a sample's influence genuinely persists in the weights and must
be actively removed, not merely hidden by withholding the data. Machine unlearning as a named problem
was introduced by Cao & Yang (2015), who recast learning algorithms into a summation form so a deleted
point's contribution could be subtracted; this works for statistical-query-style learners but not for
the non-linear, SGD-trained deep nets we care about here.

**The retrained model as the ground truth.** The probabilistic notion of unlearning (Ginart et al. 2019),
borrowing the differential-privacy template, asks that the output distribution of the unlearned model be
close to that of the model retrained without `D_f`. Most subsequent theory (Guo et al. 2020 certified
removal; Neel et al. 2021 descent-to-delete; Sekhari et al. 2021) lives in the convex / linear regime
and gives guarantees there, but does not transfer to deep nets. Practically, the retrained model is the
thing every method is trying to imitate.

**The tradeoff and its empirical signature.** There is a clear tension: push hard on forgetting and
retain accuracy collapses; stay conservative and measurable traces of `D_f` survive. A diagnostic that
recurs in this literature: an over-aggressive method drives forget-set accuracy to *exactly zero*, which
looks like success but is a privacy tell — the retrained model, by contrast, typically retains some
nonzero, near-chance accuracy on the forget class because the class is not pathologically *avoided*,
merely not specifically learned. Confidently-wrong predictions on `D_f` are themselves information.

**Knowledge distillation.** The load-bearing technique from elsewhere in deep learning is
distillation (Hinton, Vinyals & Dean 2015): a *student* network is trained to match a *teacher's*
output distribution rather than the hard labels. With temperature-softened softmax
`p_i(z; T) = softmax(z_i / T)`, the teacher's soft probabilities carry "dark knowledge" — the relative
mass it places on the wrong classes — and the student minimizes the divergence to those soft targets,

```
L_KD = KL( p^teacher(T) || p^student(T) ) ,   p(T) = softmax(logits / T),
```

(commonly weighted by `T^2` when mixed with a hard-label term, to keep the softened-loss gradient on the
same scale). The general property that matters: distillation makes the student *copy whatever
distribution the teacher emits on a given input* — the teacher need not be a "good" model; it is just a
source of target behavior. Jensen–Shannon divergence,
`JS(p, q) = ½ KL(p‖m) + ½ KL(q‖m)` with `m = (p+q)/2`, is the symmetric, bounded cousin used to
*compare* two output distributions.

## Baselines

The prior deep-network unlearning methods a new method would be measured against, and the specific place
each one stalls.

**SISA — sharded training (Bourtoule et al. 2021).** Partition `D_c` into disjoint shards, train one
sub-model per shard, and aggregate (e.g. by voting); cache intermediate checkpoints. A deletion request
only forces retraining of the single shard (and slice) that held the point, so cost drops by the shard
count. **Where it stalls:** it is a *training-time* architecture — you must have sharded and checkpointed
from the beginning. It does nothing for an already-trained monolithic model, aggregation costs accuracy
as shards multiply, and it stores many sub-models and checkpoints.

**Fisher scrubbing and its linearized variants (Golatkar, Achille & Soatto 2020 "eternal sunshine";
Golatkar et al. 2020 NTK; Golatkar et al. 2021 mixed-linear).** Derive a closed-form weight update that
moves the parameters toward the distribution a retrained model would have, using the Fisher information
matrix (a noise injection that scrubs the readable information) or a neural-tangent-kernel linearization
of training dynamics; the mixed-linear version trains an auxiliary linearized model alongside the
original. **Where they stall:** the scrubbing derivation assumes the model was trained with SGD and uses
the optimization trajectory, so it constrains *how the model had to be trained*; the Fisher / Hessian
approximations are expensive and brittle; and the NTK / mixed-linear variants require training and
storing an extra approximating model. They carry strong assumptions and high computational cost.

**UNSIR — impair-then-repair (Tarun et al. 2021).** For class unlearning, learn an
error-maximizing *noise* tensor for the forget class, fine-tune the model on that noise to impair its
forget-class behavior, then briefly fine-tune on retain data to repair collateral damage. Fast and
effective at class removal. **Where it stalls:** it is *class-level only* — there is no natural noise
construction for a random subset of points scattered across classes — and it drives forget-class
accuracy to exactly 0, the confidently-wrong regime that diverges from how a retrained model behaves and
that a membership-inference attacker can read.

**Amnesiac unlearning (Graves et al. 2021).** Log, during training, the parameter update contributed by
each batch; to unlearn, subtract back the logged updates that touched the forget data. **Where it
stalls:** it requires storing the full per-batch update history throughout training — large storage,
tied to training bookkeeping — and the subtraction is approximate once later updates have interacted
with the removed ones.

Across these, the recurring limitations are: a dependence on *how training was done* (SGD-only,
sharding, gradient logs), reliance on expensive second-order information or auxiliary models, support for
only one forgetting mode (class-level), and — for the aggressive methods — collapsing forget accuracy to
zero, which is a privacy signal rather than faithful imitation of a never-trained model.

## Evaluation settings

The natural yardsticks already in use for deep unlearning:

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
