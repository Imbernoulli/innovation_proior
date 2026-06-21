# Context

## Research question

We want fast, small networks at inference time without sacrificing accuracy. Depth helps accuracy -- deeper nets represent more complex functions with fewer parameters per layer -- but deeper nets are *harder to train* by gradient descent, because greater depth means greater non-linearity and worse-conditioned optimization. So the appealing target is a student that is **deeper and thinner** than a wide reference model: fewer parameters and multiplications per layer (cheaper), more layers (still expressive). The question is how to train such a thin, deep student to a good optimum given a trained wide teacher.

## Background

Large, wide, or ensemble networks reach high accuracy but are expensive to evaluate; there is strong interest in compressing them into a single small model that runs fast. The key tools:

- **Knowledge distillation (Hinton et al., 2014).** Train a small *student* on the *softened* class probabilities of a large *teacher*. Soften with a temperature tau > 1: P^tau = softmax(a / tau), where a are pre-softmax activations (logits). The soft targets carry "dark knowledge" -- relative similarities between classes that the one-hot label hides -- so each example conveys more information than its hard label. The student is trained on a blend of the true-label loss and the teacher-matching loss.
- **Curriculum learning (Bengio et al., 2009).** Presenting training examples (or sub-objectives) from easier to harder accelerates convergence and can improve generalization. A staged training where an easier intermediate objective precedes the full objective is a form of curriculum.
- **Depth and optimization.** Under a fixed compute budget, the student is made deeper and thinner. Knowledge distillation supplies a softened target at the student's top layer from the teacher's final output.

## Baselines

- **Standard supervised training of the thin deep net.** Train the small deep student directly on labels with stochastic gradient descent.
- **Knowledge distillation, output-only (Hinton et al., 2014).** Student matches teacher softened outputs plus true labels:
  ```
  L_KD(W_S) = H(y_true, P_S) + lambda H(P_T^tau, P_S^tau),
  ```
  with H cross-entropy, lambda balancing the two terms, and tau the temperature applied to both teacher and student logits. Designed for a student of *similar depth* to the teacher; supervision is applied at the network output.
- **Wide / ensemble teacher.** The reference high-accuracy model, expensive at inference.

## Evaluation settings

- **Datasets.** CIFAR-10 and CIFAR-100 (32 x 32 color), SVHN (street-view digits), MNIST (28 x 28 digits), and AFLW-derived 16 x 16 face/non-face patches for a sigmoid face-recognition setting.
- **Architectures.** A wide, relatively shallow teacher, often a maxout convolutional net, versus thin, deep student candidates with many more layers but far fewer parameters or multiplications.
- **Metrics / protocol.** Classification accuracy/error of the student versus the teacher and versus output-only distillation, at matched or reduced parameter count and multiply count. Stochastic gradient training with RMSProp for the student candidates; temperature tau and loss weight lambda are the relevant distillation knobs.

## Code framework

The primitives that already exist: an autodiff framework with conv/linear layers, stochastic gradient optimizers, cross-entropy, a softmax-with-temperature, a trained teacher network exposing its final logits, and a randomly initialized deep thin student. The training procedure for the student is left open.

```python
import torch, torch.nn as nn, torch.nn.functional as F

def softmax_with_temperature(logits, temperature):
    return F.softmax(logits / temperature, dim=1)

class Teacher(nn.Module):
    def forward(self, x):
        # returns final logits
        ...

class Student(nn.Module):          # more layers, fewer channels than the teacher
    def forward(self, x):
        # returns final logits
        ...

def train_student(student, teacher, loader):
    # TODO: train the thin deep student given the trained teacher.
    pass
```
