# Context

## Research question

We want fast, small networks at inference time without sacrificing accuracy. Depth helps accuracy -- deeper nets represent more complex functions with fewer parameters per layer -- but deeper nets are *harder to train* by gradient descent, because greater depth means greater non-linearity and worse-conditioned optimization. So the appealing target is a student that is **deeper and thinner** than a wide reference model: fewer parameters and multiplications per layer (cheaper), more layers (still expressive). The question is how to actually train such a thin, deep student to a good optimum, in the regime where standard supervised training and output-only distillation struggle.

## Background

Large, wide, or ensemble networks reach high accuracy but are expensive to evaluate; there is strong interest in compressing them into a single small model that runs fast. The key tools:

- **Knowledge distillation (Hinton et al., 2014).** Train a small *student* on the *softened* class probabilities of a large *teacher*. Soften with a temperature tau > 1: P^tau = softmax(a / tau), where a are pre-softmax activations (logits). The soft targets carry "dark knowledge" -- relative similarities between classes that the one-hot label hides -- so each example conveys more information than its hard label. The student is trained on a blend of the true-label loss and the teacher-matching loss.
- **Curriculum learning (Bengio et al., 2009).** Presenting training examples (or sub-objectives) from easier to harder accelerates convergence and can improve generalization. A staged training where an easier intermediate objective precedes the full objective is a form of curriculum.
- **The optimization difficulty of depth.** As the student is made deeper and thinner under a fixed compute budget, plain supervised training -- and then output-only distillation -- reach their limits. The teacher's final output supplies a useful softened target, but it still supervises the student only at the top layer, so the early and middle layers of a much deeper student remain hard to steer into a good basin.

## Baselines

- **Standard supervised training of the thin deep net.** Train the small deep student directly on labels. Gap: deep plus thin is hard to optimize; gradient descent can fail to find a good starting region for the full network.
- **Knowledge distillation, output-only (Hinton et al., 2014).** Student matches teacher softened outputs plus true labels:
  ```
  L_KD(W_S) = H(y_true, P_S) + lambda H(P_T^tau, P_S^tau),
  ```
  with H cross-entropy, lambda balancing the two terms, and tau the temperature applied to both teacher and student logits. Designed for a student of *similar depth* to the teacher. Gap: as the student is made substantially deeper than the teacher, output-only matching still suffers from the difficulty of optimizing deep nets; the supervision reaches only the top of the network.
- **Wide / ensemble teacher.** The reference high-accuracy model. Gap: expensive at inference (the thing we want to compress).

## Evaluation settings

- **Datasets.** CIFAR-10 and CIFAR-100 (32 x 32 color), SVHN (street-view digits), MNIST (28 x 28 digits), and AFLW-derived 16 x 16 face/non-face patches for a sigmoid face-recognition setting.
- **Architectures.** A wide, relatively shallow teacher, often a maxout convolutional net, versus thin, deep student candidates with many more layers but far fewer parameters or multiplications.
- **Metrics / protocol.** Classification accuracy/error of the student versus the teacher and versus output-only distillation, at matched or reduced parameter count and multiply count. Stochastic gradient training with RMSProp for the student candidates; temperature tau and loss weight lambda are the relevant distillation knobs.

## Code framework

The primitives that already exist: an autodiff framework with conv/linear layers, stochastic gradient optimizers, cross-entropy, a softmax-with-temperature, a trained teacher network exposing final logits and optional intermediate activations, and a randomly initialized deep thin student. The training procedure for the student is left open.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def softmax_with_temperature(logits, temperature):
    return F.softmax(logits / temperature, dim=1)

class Teacher(nn.Module):
    def forward(self, x, return_intermediate=False):
        # returns final logits; optionally an intermediate activation
        ...

class Student(nn.Module):          # more layers, fewer channels than the teacher
    def forward(self, x, return_intermediate=False):
        # returns final logits; optionally an intermediate activation
        ...

def train_student(student, teacher, loader):
    # TODO: train the thin deep student given the trained teacher.
    pass
```
