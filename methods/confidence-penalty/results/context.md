# Context: regularizing deep classifiers through their output distribution (circa 2016)

## Research question

Large neural networks with millions of parameters reach strong accuracy on image
classification, machine translation, language modeling, and speech recognition, and on every
one of these tasks training accuracy climbs toward perfection while test accuracy plateaus or
degrades. The standard countermeasures — early stopping, L1/L2 weight decay, dropout, batch
normalization — share a structural feature: each one acts on the *hidden activations or the
weights* of the network. The conditional distribution `p_theta(y|x)` over labels that comes
out of the softmax is the object the network ultimately exposes to the world. The question is
how to regularize a deep classifier by operating on that output distribution directly,
improving generalization across architectures and tasks while dropping into existing pipelines
without retuning the optimizer, schedule, or model.

## Background

**The functional view of a network's knowledge.** One way to think about what a trained
network knows is to identify its knowledge with the mapping from inputs to output
distributions, rather than with the learned values of its parameters. Under that view the
probabilities a network assigns to the *incorrect* classes are
themselves knowledge: shown an image of a particular car, a network that puts `1e-3` on a
visually similar make and `1e-9` on an unrelated object is, all else equal, different from one
that reverses those two, because the relative sizes of the wrong-class probabilities encode
how the network generalizes. Distillation exploits exactly this "dark knowledge" — the
example-specific ratios among incorrect classes — by training a small network to reproduce a
large network's soft outputs. One consequence for regularization: because the
output distribution has a natural scale and the weights do not, a regularizer acting on the
output is invariant to the underlying parameterization.

**The maximum-entropy principle.** Among all distributions consistent with given constraints,
the maximum-entropy one is the least committal — it assumes no structure beyond what the
constraints force (Jaynes, 1957). In supervised learning, searching for the maximum-entropy
model subject to constraints on empirical statistics is what gives rise to maximum likelihood
in log-linear models (Berger et al., 1996). For a softmax output, the entropy
`H(p_theta(y|x)) = - sum_i p_theta(y_i|x) log p_theta(y_i|x)` measures how committal a
distribution is: it is maximal for the uniform distribution and zero for a one-hot spike.

**Entropy terms in two corners.** First, in network *training as
optimization*: deterministic annealing (Rose, 1998), derivable from the maximum-entropy
principle, introduces an entropy term and slowly anneals it to avoid poor local minima; Miller
et al. (1996) applied this to train multilayer perceptrons with an annealed entropy-based
regularizer, their concern being to escape bad initialization. Second,
in *reinforcement learning*: adding the entropy of a stochastic policy to the objective,
`+ beta H(pi)`, keeps the policy from collapsing onto a deterministic choice too early and so
improves exploration. This was introduced by Williams & Peng (1991) in their REINFORCE-with-
entropy variant, found especially helpful on tasks requiring hierarchical behavior, and it is
standard in modern policy-gradient training (Mnih et al., 2016, where the policy-gradient
update carries an explicit `beta grad H(pi(s))` term with `beta` setting the strength). A
related strand penalizes low entropy when combining RL with supervised learning to teach a
speech model when to emit tokens (Luo et al., 2016), annealing the entropy term over training;
the same entropy-augmented objective linked maximum likelihood and reinforcement learning in
reward-augmented maximum likelihood (Norouzi et al., 2016).

**The over-confidence mechanism of one-hot cross-entropy.** With softmax
`p(k) = exp(z_k) / sum_i exp(z_i)` and a one-hot target `q(k) = delta_{k,y}`, the loss
`ell = - sum_k q(k) log p(k) = - log p(y)` has its minimum only in the limit `z_y >> z_k` for
every `k != y`: the maximum is unreachable at finite logits and is merely approached as the
correct logit runs away from the rest. The per-logit gradient is `partial ell / partial z_k =
p(k) - q(k)`, bounded in `[-1, 1]`. So training perpetually pushes the correct logit further
above the others, the network grows steadily more confident on its training labels, and this
runaway confidence is a documented symptom of overfitting (Szegedy et al., 2016). A diagnostic
picture of the phenomenon: histograms of softmax probabilities on a held-out set from a small
fully-connected MNIST network trained with dropout show the mass collecting almost entirely at
0 and 1 — peaked, near-deterministic outputs. The symptom is visible directly in the
probability vector, not only in the internal weights or activations. A second observed
regularity, readable straight
off the bounded gradient `p - q`: when the output is peaked on a *misclassified* example the
model receives a large gradient (the correct class has `p` near 0 against `q = 1`), so peaked
output distributions go hand in hand with large gradient norms during training.

## Baselines

These are the regularizers a new output-distribution regularizer would be measured against.

**Dropout (Srivastava et al., 2014).** Randomly zero hidden units on each forward pass, which
at test time approximates averaging an exponential ensemble of subnetworks. The dominant
generalization regularizer of the time and the standard baseline. It acts on hidden
activations; empirically it drives the softmax toward near-deterministic 0/1 outputs (the
peaked histograms above).

**L1/L2 weight decay, batch normalization.** Penalize or normalize weights and activations.
They act on the internal parameters, whose meaning is entangled across the whole network, so a
penalty's effect depends on the parameterization.

**Label smoothing / LSR (Szegedy et al., 2016).** Replace the one-hot target with a mixture of
the hard label and a fixed label distribution `u`, `q'(k) = (1 - epsilon) delta_{k,y} +
epsilon u(k)`, with `u` uniform, `u(k) = 1/K`. Reading it as a loss: the cross-entropy against
`q'` splits as `H(q', p) = (1 - epsilon) H(q, p) + epsilon H(u, p)`, ordinary hard-label
cross-entropy plus an `epsilon`-weighted term `H(u, p)` that penalizes the prediction's
deviation from the prior, and since `H(u, p) = D_KL(u || p) + H(u)` with `H(u)` constant, that
second term is a forward-KL penalty `D_KL(u || p)` between the fixed uniform `u` and the
model's output `p`. Because every entry of `q'` now has a positive floor `epsilon/K`, an
infinite logit gap incurs infinite cross-entropy, so the correct logit can no longer run away.
It reliably improves accuracy on ImageNet at `epsilon = 0.1`. It weights every class with the
constant uniform `u` and forces the incorrect classes toward a common target value `epsilon/K`;
it presupposes a prior `u` over labels, with uniform `u` natural when classes are balanced.

**Label noise / DisturbLabel (Xie et al., 2016).** With small probability, replace a training
example's label by one drawn uniformly at random. This penalizes placing very small
probability on a label that the corruption occasionally makes correct, and improves
generalization. Label smoothing is its marginalized expectation. It assumes a uniform
corruption distribution and injects stochastic noise into the targets.

**Distillation / self-distillation (distillation work; Reed et al., 2014).** Smooth the targets
with a teacher's soft outputs, or with the model's own current distribution (self-distillation,
a trust-region-like effect via self-created soft targets). These preserve incorrect-class
ratios, using a teacher model or a schedule of the model's own predictions.

**Virtual adversarial training (Miyato et al., 2015).** A smoothing regularizer that penalizes
the local distributional change under a worst-case input perturbation, with several
hyperparameters, the smoothness gradient computed from extra forward/backward passes per step.

## Evaluation settings

The natural yardsticks already in use, spanning the tasks where over-confidence shows up:

- **Image classification.** Permutation-invariant MNIST with a fully-connected ReLU network
  (two hidden layers, 1024 units), SGD, held-out validation split; and CIFAR-10 (32x32x3 RGB,
  10 classes, 50k train / 10k test) with a densely connected convolutional network (the small
  40-layer, growth-rate-12 configuration), trained with SGD at lr 0.1 dropped by 10x partway,
  batch size 50. Metric: test error.
- **Language modeling.** Word-level Penn Treebank with a 2-layer 1500-unit LSTM, dropout on
  non-recurrent connections, SGD with learning-rate decay and gradient-norm clipping; metric:
  validation and test perplexity. Words follow a steep, non-uniform frequency distribution.
- **Machine translation.** WMT'14 English-to-German with an 8-layer sequence-to-sequence model
  with attention, dropout, beam search; metric: tokenized BLEU.
- **Speech recognition.** TIMIT phoneme recognition and Wall Street Journal character-level
  recognition with attention-based sequence-to-sequence models; metrics: phoneme error rate
  and word error rate after beam search.
- Protocol: per-task grid search over each method's regularization strength while holding the
  base model's existing hyperparameters fixed; comparison read off the best setting against the
  unregularized / dropout-only baseline.

## Code framework

The regularizer plugs into a fixed supervised training pipeline that already exists. The data
loader, the model producing logits, the cross-entropy objective, the optimizer (SGD with
momentum and weight decay), and the schedule are all settled and held constant — the only
thing being designed is an *additional* differentiable penalty term computed each step from
the current batch and added to the cross-entropy loss. So the substrate is just the generic
training loop plus an empty hook for that penalty.

```python
import torch
import torch.nn.functional as F


def compute_regularization(model, inputs, outputs, targets, config):
    """Additional regularization term, added to the cross-entropy loss every step.

    Inputs available to the term:
      model   : the nn.Module (iterate named_parameters / named_modules for weight terms)
      inputs  : [B, C, H, W] input batch
      outputs : [B, num_classes] logits  (for output-distribution-based terms)
      targets : [B] integer labels
      config  : dict with num_classes, epoch, total_epochs
    Must return a differentiable scalar tensor.
    """
    # TODO: the regularization term we will design.
    return outputs.new_zeros(())


# existing fixed training loop the term plugs into
def train(model, data_loader, optimizer):
    for inputs, targets in data_loader:        # draw a minibatch
        optimizer.zero_grad()
        outputs = model(inputs)                # forward through the fixed model -> logits
        loss = F.cross_entropy(outputs, targets)            # fixed base objective
        loss = loss + compute_regularization(               # add the designed penalty
            model, inputs, outputs, targets, config)
        loss.backward()                        # backprop through base loss + penalty
        optimizer.step()                       # fixed optimizer / schedule
```

The training loop supplies the logits and labels for each batch; `compute_regularization` is
the single empty slot where the penalty will live.
