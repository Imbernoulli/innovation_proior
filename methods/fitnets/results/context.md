# Context

## Research question

We want fast, small networks at inference time without sacrificing accuracy. Depth helps accuracy — deeper nets represent more complex functions with fewer parameters per layer — but deeper nets are *harder to train* by gradient descent, because greater depth means greater non-linearity and worse-conditioned optimization. So the appealing target is a student that is **deeper and thinner** than a wide reference model: fewer parameters and multiplications per layer (cheaper), more layers (still expressive). The question is how to actually train such a thin, deep student to a good optimum — a regime where standard supervised training, and even output-only distillation, struggle.

## Background

The field state: large/wide/ensemble networks reach high accuracy but are expensive to evaluate; there is strong interest in compressing them into a single small model that runs fast. The key tools:

- **Knowledge distillation (Hinton et al., 2014).** Train a small *student* on the *softened* class probabilities of a large *teacher*. Soften with a temperature τ > 1: P^τ = softmax(a/τ) where a are pre-softmax activations (logits). The soft targets carry "dark knowledge" — relative similarities between classes that the one-hot label hides — so each example conveys more information than its hard label. The student is trained on a blend of the true-label loss and the teacher-matching loss.
- **Curriculum learning (Bengio et al., 2009).** Presenting training examples (or sub-objectives) from easier to harder accelerates convergence and can improve generalization. A staged training where an easier intermediate objective precedes the full objective is a form of curriculum.
- **The optimization difficulty of depth.** A diagnostic fact that motivates everything here: as the student is made deeper, plain supervised training — *and* output-only distillation — get stuck; the deep thin student fails to reach the accuracy its capacity should allow. The teacher's *final output* alone is too weak a signal to guide a much deeper student's many intermediate layers to a good basin. There must be a way to inject guidance into the student's *interior*.

## Baselines

- **Standard supervised training of the thin deep net.** Train the small deep student directly on labels. Gap: deep + thin is hard to optimize; it underfits / lands in a poor optimum.
- **Knowledge distillation, output-only (Hinton et al., 2014).** Student matches teacher softened outputs plus true labels:
  ```
  L_KD(W_S) = H(y_true, P_S) + λ H(P_T^τ, P_S^τ),
  ```
  with H cross-entropy, λ balancing the two, τ the temperature applied to both teacher and student logits. Designed for a student of *similar depth* to the teacher. Gap: as the student is made substantially deeper than the teacher, output-only matching still suffers from the difficulty of optimizing deep nets — the supervision reaches only the top of the network.
- **Wide / ensemble teacher.** The reference high-accuracy model. Gap: expensive at inference (the thing we want to compress).

## Evaluation settings

- **Datasets.** CIFAR-10 and CIFAR-100 (32×32 color), SVHN (street-view digits), MNIST (28×28 digits), and AFLW (face landmark patches, used to test beyond classification).
- **Architectures.** A wide, relatively shallow teacher (maxout convolutional nets) versus thin, deep students with many more layers but far fewer parameters/multiplications. Hint and guided layers chosen at roughly the middle of teacher and student respectively.
- **Metrics / protocol.** Classification accuracy/error of the student versus the teacher and versus output-only distillation, at matched or reduced parameter count and multiply count (the compression/speed axis). Standard SGD training; the temperature τ and the loss weight λ are the relevant knobs.

## Code framework

The primitives that already exist: an autodiff framework with conv/linear layers, an SGD optimizer, cross-entropy, a softmax-with-temperature, a trained teacher network exposing both its final logits and its intermediate feature maps, and a randomly initialized deep thin student. What does *not* yet exist is how to use the teacher's *interior* to guide the student's interior, and how to stage the training. Those are the empty slots.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def softmax_T(logits, T):
    return F.softmax(logits / T, dim=1)

class Teacher(nn.Module):
    def forward(self, x, return_hint=False):
        # returns final logits; optionally an intermediate feature map ("hint")
        ...

class Student(nn.Module):          # deeper and thinner than the teacher
    def forward(self, x, return_guided=False):
        # returns final logits; optionally an intermediate feature map ("guided")
        ...

def intermediate_supervision_loss(student, teacher, x):
    # TODO: how do we make the student's interior match the teacher's interior
    #       when the two layers have different widths/shapes?
    pass

def train_student(student, teacher, loader):
    # TODO: in what ORDER do we train? a single objective, or staged?
    pass
```
