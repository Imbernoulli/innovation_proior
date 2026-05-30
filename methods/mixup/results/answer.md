# mixup

## Problem

Large neural classifiers are trained by Empirical Risk Minimization (ERM): minimize the average loss over the training set, i.e. minimize the risk under the empirical distribution P_δ(x, y) = (1/n) Σ_i δ(x = x_i, y = y_i), a sum of Dirac masses on the training points. Because this objective constrains the function only *at* those points, large nets exploit the freedom everywhere else: they memorize the data (fitting even random labels to zero training error) and behave erratically just off the data manifold (a tiny gradient-based input perturbation flips the prediction — adversarial examples). Standard data augmentation mitigates this by training on a hand-built *vicinity* of each example, but it is dataset-specific and single-class: every virtual example keeps its source's label, so the region *between* different classes is never supervised.

## Key idea

Replace the empirical distribution with a generic Vicinal Risk Minimization (VRM) vicinity that is data-agnostic and crosses class boundaries: construct virtual training examples as convex combinations of two random examples *and their labels*.

  x̃ = λ x_i + (1 − λ) x_j
  ỹ = λ y_i + (1 − λ) y_j      (y_i, y_j one-hot; ỹ a soft label)
  λ ~ Beta(α, α),  α ∈ (0, ∞)

This encodes the prior that linear interpolations of inputs should produce linear interpolations of targets — i.e. the model should behave linearly between training points. Linearity suppresses off-data oscillation and overconfidence, constrains the loss slope along chords between examples, makes random-label memorization expensive, and supports a simple complexity bound. The hyperparameter α controls where samples fall on each chord; α → 0 recovers ERM.

## Why it works

- **VRM framing.** ERM uses P_δ = (1/n) Σ δ(x_i, y_i); mixup uses the vicinity P_μ = (1/n) Σ_i μ(x̃, ỹ | x_i, y_i) with μ the law of (λ x_i + (1−λ) x_j, λ y_i + (1−λ) y_j), λ ~ Beta(α, α). One samples virtual pairs from P_μ and minimizes the empirical vicinal risk R_μ(f) = (1/m) Σ ℓ(f(x̃), ỹ).

- **Complexity bound.** Let f̃(x) = E_{x', λ}[ f̂(λ x + (1−λ) x') ] be the expected mixup model and assume zero training error, so f̂ produces the interpolated target on interpolated inputs. With Lip̂(g) = sup_{x,x'∈D} ‖g(x') − g(x)‖ / ‖x' − x‖,

    Lip̂(f̃) = sup ‖E_{x'',λ}[ f̂(λx'+(1−λ)x'') − f̂(λx+(1−λ)x'') ]‖ / ‖x'−x‖
           = sup ‖E_{x'',λ}[ λf(x') + (1−λ)f(x'') − λf(x) − (1−λ)f(x'') ]‖ / ‖x'−x‖
           = sup ‖E_λ[ λ(f(x') − f(x)) ]‖ / ‖x'−x‖
           ≤ E[λ] · Lip̂(f).

  The mixing-partner terms (1−λ)f(x'') cancel; the second equality uses the interpolated *target value*, which only mixing the labels supplies. For the symmetric Beta(α, α) family, E[λ] = 1/2 for every positive α, so this bound is a coarse half-Lipschitz statement; α still controls how close the sampled points are to endpoints or to the segment middle.

- **Loss linearity.** Cross-entropy is linear in the target, so ℓ(f(x̃), λ y_a + (1−λ) y_b) = λ ℓ(f(x̃), y_a) + (1−λ) ℓ(f(x̃), y_b). The soft label never needs to be materialized — compute hard-label CE against each endpoint and take the convex combination.

## Design choices

- **Mix raw inputs**, not hidden-layer features: the off-data constraint is needed in input space, while hidden representations can be reshaped by the network.
- **Mix across all classes**, random pairs: same-class mixing (SMOTE-style) leaves the cross-class gap unsupervised; nearest-neighbor restriction narrows the vicinity back toward local augmentation.
- **Mix the labels too**, not the dominant/closer label: required for the complexity bound and for coupling supervision to the input.
- **Two examples**, not three-plus (Dirichlet): pairwise chords express the linearity prior directly, while more partners add compute and average toward a many-point center.
- **One data loader**: pair a minibatch with a shuffled copy of itself (a random permutation) instead of running two loaders, reducing I/O without changing the sampled-pair logic.

## Code

A compact CIFAR-10 training loop keeps the network, SGD, schedule, and underlying crop/flip augmentation unchanged; only the minibatch construction and target-aware loss are filled in.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = datasets.CIFAR10(root='~/data', train=True, download=False,
                            transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=8)

net = build_network()                       # e.g. a PreAct ResNet-18
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
use_cuda = torch.cuda.is_available()
if use_cuda:
    net.cuda()
vicinity_strength = 1.0                     # alpha in the equations


def make_training_pairs(x, y, strength=vicinity_strength, use_cuda=True):
    # one Beta draw per minibatch; strength <= 0 leaves the empirical batch unchanged
    lam = np.random.beta(strength, strength) if strength > 0 else 1.0

    # pair the batch with a shuffled copy of itself -> random cross-class pairs,
    # one data loader, no extra I/O
    if use_cuda:
        index = torch.randperm(x.size(0)).cuda()
    else:
        index = torch.randperm(x.size(0))

    mixed_x = lam * x + (1 - lam) * x[index, :]                # convex combo of inputs
    target_spec = (y, y[index], lam)                           # endpoint labels + weight
    return mixed_x, target_spec


def compute_loss(criterion, outputs, target_spec):
    y_a, y_b, lam = target_spec
    # CE is linear in the target, so loss on the mixed soft label
    # lam*y_a + (1-lam)*y_b  ==  lam*CE(.,y_a) + (1-lam)*CE(.,y_b);
    # no need to build the soft-label vector
    return lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)


def train(epoch):
    net.train()
    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        inputs, target_spec = make_training_pairs(inputs, targets,
                                                  vicinity_strength, use_cuda)
        outputs = net(inputs)
        loss = compute_loss(criterion, outputs, target_spec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Typical settings (CIFAR-10): PreAct ResNet-18, batch size 128, SGD lr 0.1 with momentum 0.9 and weight decay 1e-4, learning rate divided by 10 at epochs 100 and 150, 200 epochs total, standard random-crop + horizontal-flip + normalization underneath, α = 1 (so λ ~ Uniform[0, 1]). For ImageNet, α ∈ [0.1, 0.4]; for robustness to corrupted labels, larger α (e.g. 8 or 32). A smaller weight decay (1e-4) suits mixup, reflecting that mixup itself supplies regularization.
