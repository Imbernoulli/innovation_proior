# Context: combining a fixed-pipeline classifier with a downstream human expert (circa 2018-2020)

## Research question

A learned model is rarely the last word in a high-stakes pipeline. A radiologist still reads
the chest X-ray; a human moderator still reviews the flagged post; a clinician still signs off
on the risk score. In these deployments the model does not have to answer every instance — it
can hand some of them to a downstream **expert** who often has access to side information the
model never sees (the patient's history, the full thread, a second test). The practical
question is therefore not "what is the most accurate classifier?" but "for a given input,
*who should answer* — the model or the expert — so that the combined system is as accurate (or
as fair, or as cheap in expert time) as possible?"

Making that routing decision well is subtle for three reasons. First, the model must decide to
abstain *without* observing the expert's side information — at decision time it sees only the
covariates `x`, not the expert's context `z`. Second, the gains from routing come precisely
from the *mismatch* between where the model is weak and where the expert is strong, so the
router has to reason about both error profiles jointly. Third, the only supervision available is
samples of the expert's past decisions `m_i` — we do not get to interrogate the expert on demand
or read out their internal confidence. A solution would have to learn, from a dataset
`{(x_i, y_i, m_i)}`, a routing rule and a predictor that together minimize a combined
system loss, fit into the ordinary deep-learning stack, and come with a guarantee that the
trainable objective drives the system error toward its optimum rather than toward some unrelated
quantity.

## Background

**Abstention and the error–reject tradeoff.** The idea that a classifier should be allowed to
*not answer* goes back to Chow (1970), who analyzed the optimal tradeoff between error rate and
rejection rate when each rejection costs a fixed `c`. Chow's rule is clean: with posterior
`η(x) = P(Y=+1|x)`, predict only where you are confident enough, i.e. abstain exactly when
`max{η(x), 1-η(x)} ≤ 1 - c`. The whole machinery of abstention since then rests on this picture
of *confidence below a threshold means reject*.

**Surrogates for the reject option.** The Chow rule is defined through the unknown posterior;
to learn it, one needs a tractable loss. Bartlett & Wegkamp (2008) gave a convex *double-hinge*
surrogate for the binary reject loss `I[yh(x)≤0]·I[no reject] + c·I[reject]` and proved it
consistent, with the rejector implicitly a thresholded function of the predictor's score:
reject when `|h(x)| ≤ γ`. Ramaswamy et al. (2018) gave consistent abstain surrogates in the
multiclass setting through binary encodings. All of these keep the rejector *tied to the
predictor's own confidence*.

**Decoupling the rejector from the predictor.** Cortes, DeSalvo & Mohri (2016) observed that
forcing `r(x) = |h(x)| - γ` can be strictly too weak: there are distributions (e.g. a
one-dimensional example where the best predictor is a threshold `h(x)=x-θ` but the region that
should be rejected is `x ≤ η`, with `η ≠ θ`) where no confidence band of the *best* predictor
coincides with the *best* rejection region. They proposed learning a **pair** of functions — a
predictor `h` and a separate rejector `r`, drawn from possibly different hypothesis sets — under
the rejection loss

```
L(h, r, x, y) = I[y·h(x) ≤ 0]·I[r(x) > 0] + c·I[r(x) ≤ 0],
```

and gave a full theoretical analysis (Rademacher generalization bounds, a notion of *realizable
(H,R)-consistency*, and convex binary surrogates built from two upper-bounding convex functions
`φ, ψ` via the identity `max(a,b) ≥ (a+b)/2`, yielding a *max-hinge* and a *plus-hinge* loss).
Crucially their analysis, and their consistent surrogates, were for the **binary** case. Cortes
et al. explicitly *considered and set aside* a cost-sensitive-learning view of rejection,
remarking that the reject symbol "is not a class" and that there is no natural distribution over
the augmented set `{-1, +1, reject}`. Ni et al. (2019) then tried to push the Cortes-style
surrogates to multiclass and proved an impossibility: those particular two-function surrogates
*cannot* be made consistent for `K > 2`, which is why multiclass abstention work fell back on
confidence thresholds. They left finding a consistent multiclass reject surrogate as an open
problem.

**Selective classification.** A neighbouring line (El-Yaniv & Wiener 2010; Geifman & El-Yaniv
2017, 2019) fixes a *coverage* constraint instead of a reject cost: predict on at least a
fraction of inputs and minimize error there. SelectiveNet (Geifman & El-Yaniv 2019) even adds a
second network head to decide coverage. But selective classification assumes there is *no
downstream answerer* — a rejected point is simply not predicted — so it never models who picks
up the rejected instance or how good that answerer is on it.

**Routing to a downstream answerer.** Two recent threads put an expert at the end of the pipe.
Raghu et al. (2019) learn a classifier on the task, separately learn a model of *whether the
expert agrees with the label*, extract a confidence from each, and defer to whichever is more
confident. Madras, Pitassi & Zemel (2018) instead optimize a single differentiable objective
that softly mixes the classifier's and the expert's costs through a learned gate — a *mixture of
two answerers, one of them fixed* — borrowing the soft-gate idea from mixtures of experts
(Jordan & Jacobs). A consistent estimator should produce, in the infinite-data limit over all
measurable functions, the Bayes-optimal routing rule. For the combined system with
misclassification costs that Bayes rule is computable in closed form: with `η_y(x)=P(Y=y|x)`,
the optimal predictor is `h^B(x)=argmax_y η_y(x)` and one should hand the instance to the expert
exactly when the expert's chance of being right beats the model's best guess,
`r^B(x) = I[ max_y η_y(x) ≤ P(Y=M|X=x) ]`. This Bayes rule is the yardstick any proposed method
will be checked against.

**The diagnostic that motivates joint learning.** A documented failure mode of the
*train-the-classifier-first* recipe (the confidence approach, and any method that fits `h` to
the target while ignoring the expert) appears already under limited model capacity. Picture two
sub-populations: on group A the expert is excellent (it uses side information, or computes a
nonlinear boundary the model cannot), and on group B only the model can help. A linear `h`
trained on *all* the data tries to fit both groups at once and separates neither well. The
better system has `h` give up on group A entirely and specialize on group B, while `r` routes
group A to the expert — but a classifier trained without reference to the expert can never
discover that it *should* abandon group A. So independence of the classifier from the expert is
not a convenience; under realistic capacity limits it costs accuracy. Likewise, a soft-gate
objective that compares the classifier's *entropy* to the expert's error rate (rather than the
classifier's *confidence in its top class*) can route in the wrong places, and is observed to
collapse to "never defer" once the classifier's training loss reaches zero while the expert-cost
term stays put.

## Baselines

These are the prior methods a new routing approach would be measured against and react to.

**Confidence thresholding / Chow's rule (Chow 1970; Bartlett & Wegkamp 2008).** Threshold the
predictor's own confidence: abstain when `max_y P(y|x)` is below a cutoff set by the reject cost
(or, with a coverage target, by a quantile). Simple, consistent for the *pure-rejection*
problem. **Gap:** with a downstream expert there is no model of the expert at all, so it cannot
tell a hard-for-the-model-but-easy-for-the-expert instance from a hard-for-everyone instance; it
abstains on raw uncertainty, not on *who is better here*.

**Independent confidence comparison (Raghu et al. 2019).** Train `h` on the task; separately
train a model of `P(expert correct | x)`; defer to whichever is more confident. This *is*
consistent over all measurable functions and does model the expert. **Gap:** the classifier is
fit ignoring the expert, so with a restricted hypothesis class it cannot specialize away from
the expert's strong region (the two-subpopulation failure above); and it needs *two* trained
models, paying the statistical cost of two hypothesis classes when expert-labeled data is scarce.

**Mixtures-of-experts deferral (Madras, Pitassi & Zemel 2018).** Optimize one differentiable
objective: a soft gate `softmax(r_0, r_1)` weights the classifier's cross-entropy against the
expert's error, `E[ softmax(r0)·l(y,h(x)) + softmax(r1)·l(y,m) ]`, with the classifier's
gradient stopped from flowing through the gate. **Gap:** the objective is non-convex in the gate;
its population minimizer compares the classifier-posterior *entropy* `H(h^B(x))` against
`P(Y≠M|x)` rather than the *confidence* `max_y η_y(x)`, so it does not recover the Bayes rejector
— and in practice it tends to stop deferring as the classifier's loss vanishes, because predict
becomes the uniformly cheaper branch.

**Binary two-function reject surrogates (Cortes, DeSalvo & Mohri 2016; Bartlett & Wegkamp 2008;
extended by Ni et al. 2019).** Learn `(h, r)` jointly via convex surrogates upper-bounding the
binary reject loss. **Gap:** the consistent constructions are binary; their multiclass extension
was shown unable to be consistent (Ni et al. 2019), and the reject cost is a constant `c`, not a
per-instance expert-error that varies across the input space.

**Selective classification with a learned head (Geifman & El-Yaniv 2017, 2019).** Add a coverage
constraint and possibly a second network head for the abstain decision. **Gap:** no downstream
expert is modeled — a rejected point is simply uncovered, so the framework cannot reason about
the expert's competence on the rejected region.

## Evaluation settings

The natural yardsticks already in use at the time:

- **Image classification with synthetic experts** — CIFAR-10 and CIFAR-100 (Krizhevsky 2009),
  with a base WideResNet (Zagoruyko & Komodakis 2016) trained by SGD with momentum and cosine
  annealing. A *synthetic expert* is constructed with a controllable competence profile: e.g.
  for parameter `k`, the expert is perfect on images whose class index is `≤ k` and predicts
  uniformly at random otherwise — so the expert is strong on a known sub-region and useless
  elsewhere, exactly the regime where routing should help.
- **Human-annotation expert** — CIFAR-10H (Peterson et al. 2019), fifty crowd labels per test
  image, used to simulate a realistic non-uniform human whose accuracy varies by class, with
  only a *fraction* of points carrying expert labels (the limited-expert-data regime).
- **Text classification with a demographically biased expert** — hate-speech / offensive-language
  detection (Davidson et al. 2017), a CNN over GloVe embeddings (Kim 2014; Pennington et al.
  2014), with a synthetic expert whose error rate differs between African-American-English and
  non-AAE tweets (detected with the Blodgett et al. 2016 language model) — to probe fairness of
  the combined system.
- **Medical imaging** — CheXpert chest X-rays (Irvin et al. 2019), DenseNet-121 pretrained on
  ImageNet, five binary "competition" tasks, a synthetic expert keyed on the presence of support
  devices; temperature scaling (Guo et al. 2017) for calibration of confidence baselines.
- **Metrics & protocol** — combined-system accuracy versus expert competence; classifier accuracy
  on the non-deferred ("covered") examples versus coverage; system AU-ROC / AU-PR across the full
  coverage range (sweeping a deferral threshold so coverage runs from 0 to 100%); sample-complexity
  curves varying training-set size; and, for fairness, the gap in false-positive rates across
  demographic groups. The combined system, the classifier alone, and the expert alone are all
  reported on the same axes.

## Code framework

The ordinary supervised-classification stack already has a backbone network, an output head, a
softmax or logit-based loss, and a minibatch SGD loop. The substrate below keeps only that
generic machinery: a classifier backbone, a place where the training objective is computed from
`(model outputs, target, expert label)`, and an inference rule that turns model outputs into a
prediction or a handoff decision.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Backbone(nn.Module):
    """Generic classifier backbone. The output dimension and how the outputs encode
    prediction versus handoff are left open."""

    def __init__(self, make_features, num_outputs):
        super().__init__()
        self.features = make_features()                  # existing feature extractor
        # TODO: the output structure we'll design.
        self.head = nn.Linear(self.features.out_dim, num_outputs)

    def forward(self, x):
        return self.head(self.features(x))               # raw logits


def system_objective(outputs, target, expert_label, tuning=None):
    """Per-batch training objective for the predict-or-defer system.

    outputs:      (B, ...) raw model outputs for the batch
    target:       (B,) ground-truth labels y
    expert_label: (B,) the expert's decision m on each example
    tuning:       optional validation-selected settings

    TODO: the objective we'll design from these quantities.
    """
    raise NotImplementedError


def decide(outputs, tuning=None):
    """Inference rule: from a single example's model outputs, return either a predicted
    class label or the handoff action.

    TODO: the decision rule we'll design.
    """
    raise NotImplementedError


# existing minibatch training loop the objective plugs into
def train(model, data_loader, optimizer, expert_fn, tuning=None):
    model.train()
    for x, target in data_loader:                        # draw a minibatch
        outputs = model(x)                               # forward through the backbone
        expert_label = expert_fn(x, target)              # observed expert decisions m_i
        loss = system_objective(outputs, target, expert_label, tuning)
        optimizer.zero_grad()
        loss.backward()                                  # backprop through the backbone
        optimizer.step()
```

The training loop supplies, per example, the model's outputs, the target, and the expert's
decision; the objective and `decide` are the two empty places where the new construction has to
land.
