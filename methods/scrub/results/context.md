# Context: deep machine unlearning (circa 2020-2023)

## Research question

A neural network `f(·; w^o)` has been trained by empirical risk minimization on a dataset
`D = {(x_i, y_i)}_{i=1}^N`, producing weights `w^o`. After training, we are handed a *forget
set* `D_f ⊂ D` and asked to remove its influence from the model, leaving a *retain set*
`D_r = D \ D_f` whose information the model is still allowed to keep. The goal is to produce
new weights `w^u` such that `f(·; w^u)` has "forgotten" `D_f` — it no longer behaves as
though it was trained on those examples — while preserving *utility*: accuracy on `D_r` and
generalization to held-out data. Crucially, this must be done *efficiently*; otherwise the
trivial solution of retraining from scratch on `D_r` (which by construction never saw `D_f`)
already exists and there is nothing to improve on.

The difficulty is that deep networks memorize information about individual instances, and
their highly non-convex training makes it impossible to trace which weights encode which
example. The forget set and the retain set are entangled in the same shared representation, so
any crude attempt to delete the forget set's contribution risks tearing out retained
knowledge with it. What is needed is an update rule that removes the model's specific
competence on `D_f` without collateral damage to `D_r`, runs in a small number of passes over
the data (no Hessian, no per-example retraining), and does not depend on fragile assumptions
about how far `w^u` can be from `w^o`.

A further subtlety that frames the whole problem: what counts as "forgetting" is
application-dependent, and the right *amount* of forgetting differs across use cases. If the
goal is removing a harmful bias or correcting mislabeled data, we want the model to be
*maximally* wrong on `D_f` — it should never reproduce the unwanted behaviour. If the goal is
user privacy, we instead want the forget error to be only *as high as a model that never saw
`D_f`* — an unusually high error is itself a tell that lets a membership-inference attacker
identify the deleted examples. A solution should be able to target both regimes.

## Background

**Exact vs. approximate unlearning.** The only procedure that *provably* removes `D_f` is to
retrain from scratch on `D_r`; this is "exact unlearning" but is far too expensive to run per
deletion request on a large model. Everything else is *approximate unlearning*: modify `w^o`
in place to mimic what retraining would have produced. The dominant framing
(Ginart et al. 2019; Guo et al. 2019; Golatkar et al. 2020) borrows from differential privacy
and defines success as *indistinguishability* from the retrain-from-scratch model — the
distribution of unlearned weights (or outputs) should be close to that of a model trained
without `D_f`. This goal is mirrored in the metric these works optimize: the unlearned model's
error on `D_f` should be *just as high as* the retrained reference, no higher.

**The stability assumption these methods rest on, and where it cracks.** Indistinguishability
methods lean on the *stability of SGD*: if `D` and `D_r` differ in only a few examples, the
weights obtained by training on each are assumed close in weight space, so unlearning is a
*small* local correction. Golatkar et al. formalize this with a Forgetting Lagrangian — the
retain loss plus a KL term penalizing distance from the retrained reference — and a Local
Forgetting Bound showing the complex global KL can be controlled by a per-seed expectation.
Under a local-quadratic approximation this yields a closed-form scrubbing step, in the limit a
Newton update `w ← w − B^{-1}∇L_{D_r}(w)` (plus noise), with `B` the Hessian of the loss.
The crack: the closeness premise holds only when `D_f` is *small* relative to `D` (high
overlap between `D` and `D_r`). For a whole class, or any sizable forget set, `w^o` and the
retrain solution need not be close at all, and the linearization stops being a good proxy.

**Diagnostic facts about the design space.** Three observations shape what any update rule must
respect. (i) `D_f` and `D_r` share the same weights; a deep
net's lower layers compute features used by *every* class. So a gradient that degrades the
model on `D_f` will, through the shared representation, also move predictions on `D_r` —
pushing the model away from its behaviour on the forget set tends to drag down retain
performance as an unintended side effect. (ii) A network's softened output distribution over
classes carries far more information than its top-1 label: the relative sizes of the small
probabilities ("dark knowledge") encode a rich similarity structure over classes, so matching
a teacher's full distribution is a much stronger "behave as you did before" constraint than
matching its hard labels. (iii) Naively maximizing the loss on `D_f` (gradient *ascent*) is
unbounded — the loss can grow without limit — so on its own it drives the weights to
degenerate, utility-destroying regions.

**Membership inference as the privacy yardstick.** For privacy applications, success is
measured by whether an attacker can tell apart forgotten examples from never-seen ones, given
the unlearned model's behaviour (e.g. loss/confidence) on a target example. State-of-the-art
likelihood-ratio attacks (Carlini et al. 2022) fit per-example Gaussians over confidences from
"shadow" models trained with and without the example, and threshold the likelihood ratio. An
over-forgotten example — one the model is *abnormally* bad at — stands out under such an
attack, which is exactly why "just high enough" forget error, rather than maximal, is the
privacy target.

## Baselines

These are the prior methods a new unlearning rule would be measured against and react to.

**Retrain from scratch.** Train the same architecture on `D_r` with the original
hyperparameters. The gold standard for forget quality — it genuinely never saw `D_f` — and the
implicit reference point for indistinguishability metrics. **Gap:** the cost it is the whole
point to avoid; rerunning training per deletion request is infeasible at scale.

**Finetuning on the retain set.** Continue training `w^o` on `D_r` only (the model never sees
`D_f` again), relying on a mild form of catastrophic forgetting to erode the forget set's
influence. Cheap and utility-preserving. **Gap:** forgetting is incidental and weak — without
any explicit pressure against `D_f`, the model often retains high confidence and correct
predictions on the forget set, so it fails the forget-quality goal.

**NegGrad / NegGrad+ (gradient ascent on the forget loss).** Plain NegGrad finetunes `w^o`
by *negating* the gradient on `D_f` — gradient ascent on the forget loss — to actively raise
forget error. NegGrad+ strengthens it by mixing in descent on the retain loss:
`L = β·(CE on D_r) − (1−β)·(CE on D_f)`, with `β ∈ [0,1]` tuned to balance the two. **Gap:**
the forget-ascent term is unbounded and small `β` gives it too much weight, so it explodes
quickly; even when balanced, both terms live in raw cross-entropy-to-hard-labels space, with no
anchor to the *full behaviour* of the original model on `D_r`, so the retain term defends only
the argmax, not the function.

**Fisher / NTK forgetting (Golatkar et al. 2020a/b).** Closed-form scrubbing from the
local-quadratic analysis above: a Newton/Fisher step `w − B^{-1}∇L_{D_r}` plus
information-destroying noise (Fisher forgetting), or a linearization of the unlearning
finetuning via NTK theory (NTK forgetting). Principled and tied to an indistinguishability
guarantee. **Gap:** forming and inverting the Hessian/Fisher scales *quadratically* in the
number of training samples, so these do not scale beyond small datasets; and they inherit the
small-forget-set closeness assumption, which a class-sized `D_f` violates.

**CF-k / EU-k (Goel et al. 2022).** Freeze the first `k` layers of `w^o` and either finetune
the remaining top layers on `D_r` (Catastrophic-Forgetting-k) or reinitialize and retrain them
on `D_r` (Exact-Unlearning-k). A cheap middle ground between finetuning and retraining. **Gap:**
CF-k inherits finetuning's weak forgetting (it keeps utility but barely forgets); EU-k forgets
more reliably for whole-class removal but has a notable failure mode on *selective* (instance)
forgetting, and both presuppose that the deleted information lives only in the top layers.

**Bad-T: incompetent-teacher distillation (Chundawat et al. 2022).** A teacher-student method
that uses *two* teachers: the competent original model for the retain set and an *incompetent*
(randomly initialized) teacher for the forget set. The student is distilled (KL on softened
outputs) toward the competent teacher on `D_r` and toward the incompetent teacher on `D_f`, so
on forget examples it learns to imitate a model that knows nothing. **Gap:** moving *close to a
bad teacher* on `D_f` is a different goal than removing the original model's specific knowledge
of `D_f`; a random teacher emits near-uniform targets, a weak and noisy forget signal, and the
injected randomness tends to degrade overall model quality.

## Evaluation settings

The yardsticks already in use for unlearning experiments at the time:

- **Datasets / pretraining.** CIFAR-10 and Lacuna-10 (the latter derived from VGG-Faces) as
  the unlearning datasets, with models pretrained on CIFAR-100 / Lacuna-100 respectively, for
  consistency with prior work; data augmentation turned off (so the original model overfits
  somewhat, stressing privacy). Small-scale variants CIFAR-5 / Lacuna-5 (5 classes, 100
  train / 25 val / 100 test per class) for comparing against methods like NTK that do not
  scale.
- **Architectures.** ResNet-18 and an All-CNN variant (batch-norm before each nonlinearity);
  filters reduced ~60% for small-scale.
- **Forget-set regimes.** *Class unlearning* (the entire training set of one class, ~10%) and
  *selective unlearning* (a handful of examples, e.g. 100, from one class, ~0.25-2%); for
  resolving-confusion experiments, the forget set is exactly the mislabeled examples between
  two confused classes.
- **Metrics.** Forget-set error, retain-set error, test error (for utility); for the privacy
  scenario, a membership-inference attack — including a likelihood-ratio (LiRA-style) attack
  adapted to unlearning — where lower attacker advantage / AUC is better. Three application
  scenarios with distinct forget-quality definitions: removing biases (maximal forget error
  desired), resolving confusion (mislabel error eliminated), user privacy (forget error
  "just high enough").
- **Protocol.** Three random seeds, mean and standard deviation reported; pretrain, define the
  forget split, then run the unlearning update for a small fixed number of epochs receiving
  retain and forget minibatches, with a standard optimizer (SGD or Adam) at a small learning
  rate.

## Code framework

The update rule plugs into a fixed unlearning harness. The harness has already pretrained the
original model on the full dataset, designated a forget class, and constructed retain/forget
data loaders; it then calls the rule once per step with a retain minibatch, a forget minibatch,
a shared optimizer, and the step/epoch counters, for a fixed number of epochs. Everything about
*how* to turn those two minibatches into a weight update is the open slot — that update rule is
exactly what is to be designed.

```python
import torch
import torch.nn.functional as F


class UnlearningMethod:
    """An in-place unlearning update rule. The harness has already trained the
    original model and split the data into retain/forget; it calls unlearn_step
    once per step for a fixed number of epochs. The rule decides how to use the
    retain and forget minibatches to produce a weight update."""

    def __init__(self):
        # TODO: any state the rule needs to carry across steps
        #       (e.g. a reference to the model as it was before unlearning).
        pass

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        # retain_batch = (images, labels) from D_r, already on device
        # forget_batch = (images, labels) from D_f, already on device
        # optimizer    = shared optimizer (Adam, lr small)
        # step, epoch  = counters the rule may use to schedule its behaviour
        #
        # TODO: the unlearning update rule we will design. Using the retain and
        #       forget minibatches (and any state kept in __init__), compute a
        #       loss, backpropagate, and step the optimizer so that the model
        #       loses competence on D_f while keeping it on D_r.
        retain_x, retain_y = retain_batch
        forget_x, forget_y = forget_batch
        optimizer.zero_grad()
        loss = ...  # TODO
        loss.backward()
        optimizer.step()
        return {"loss": float(loss.item())}
```

The harness supplies the two minibatches and the optimizer; `unlearn_step` is where the
forgetting-vs-retaining update rule will live.
