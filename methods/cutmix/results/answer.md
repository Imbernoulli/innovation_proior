# CutMix

## Problem

CNN classifiers tend to concentrate their class evidence on the single most discriminative part of an object, which makes them fragile to occlusion and poor at localizing the full object. Regional-dropout regularizers (Cutout, Random Erasing) counter this by deleting a contiguous region of each training image, forcing whole-object attention — but they fill the deleted region with zeros or noise, wasting a large fraction of every image's pixels in a data-hungry regime. Mixup avoids waste by blending whole images and labels, but its translucent per-pixel overlay is locally unnatural and has no coherent occluded region, so localization and detection transfer degrade. The goal: keep regional dropout's whole-object/localization benefit while making the deleted region informative and the augmented image natural.

## Key idea

Cut a rectangular region out of image A and paste in the same-shaped patch of pixels from another training image B, then set the label to the *area-proportional* mixture of the two class labels.

  x̃ = M ⊙ x_A + (1 − M) ⊙ x_B,   M ∈ {0,1}^{W×H}  (1 on the kept-A region, 0 on the pasted-B box)
  ỹ = λ y_A + (1 − λ) y_B,         λ = fraction of the image that is A
  λ ~ Beta(α, α),  α = 1 (so λ ~ Uniform(0,1))

The box of area (1−λ)·WH is a scaled copy of the image placed uniformly:

  r_x ~ Unif(0, W),  r_w = W √(1−λ)
  r_y ~ Unif(0, H),  r_h = H √(1−λ)        ⇒  r_w r_h / (WH) = 1−λ

so the surviving A-area fraction is λ, matching the label weight on y_A.

## Why it works

- **Retains regional dropout.** A contiguous rectangle of A is removed, so the network must recognize A from a partial view — the occlusion that spreads class evidence over the whole object, exactly Cutout's benefit.
- **Wastes no pixels.** The hole is filled with real pixels of a real image B, not zeros — every pixel contributes to the forward pass and gradient (Mixup's data-efficiency virtue).
- **Locally natural.** A sharp real patch on a real image has no ghosting, so the activation maps stay clean and the two objects are localizable — unlike Mixup's translucent superposition.
- **Honest soft label.** The composite is spatially part-A, part-B; the area-proportional target is the only label consistent with the pixel composition. Committing to one dominant label, or to a fixed 0.5/0.5 split, miscalibrates the supervision against the real ratio.

## Design choices

- **Fill with a real patch** (vs. zeros/noise): keeps a deleted region *and* makes it informative.
- **Contiguous rectangle** (vs. Mixup's global blend or scattered pixels): a coherent occlusion is what forces whole-object attention; it also keeps the image natural.
- **Area-proportional label λ** (vs. one-hot dominant label, vs. fixed 0.5/0.5): only the area-proportional mix matches the construction; the alternatives degrade.
- **Apply at the input level** (vs. feature maps): the occlusion/localization phenomenon lives in pixel space; mixing features lets the network route around the constraint.
- **λ ~ Beta(1,1) = Uniform**: maximally diverse patch sizes; α = 1 is the ablation sweet spot.
- **Box = √(1−λ)-scaled image, placed uniformly**: gives area exactly 1−λ from a single λ; uniform location beats center-Gaussian or fixed-size variants.
- **Re-adjust λ to the true clipped area**: border clipping shrinks the realized box, so the label weight must track the actual surviving-A fraction, not the nominal one.
- **Shuffle within the minibatch** for the partner B: random cross-image pairs with one loader, no extra I/O.

## Code

A CIFAR/ImageNet training loop with the network, SGD, schedule, and underlying crop/flip/normalize unchanged; only the patched-input construction and the area-weighted loss are added.

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
trainset = datasets.CIFAR100(root='~/data', train=True, download=False,
                             transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                           shuffle=True, num_workers=8)

net = build_network()                       # e.g. a ResNet / PyramidNet
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.25, momentum=0.9, weight_decay=1e-4,
                      nesterov=True)
net = net.cuda()
beta = 1.0                                  # alpha; 1.0 -> lambda ~ Uniform(0,1)
cutmix_prob = 0.5                           # official CIFAR example; ImageNet uses 1.0


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)             # side scale; area fraction = cut_rat^2 = 1-lam
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def train(epoch):
    net.train()
    for input, target in trainloader:
        input, target = input.cuda(), target.cuda()
        if beta > 0 and np.random.rand() < cutmix_prob:
            lam = np.random.beta(beta, beta)
            index = torch.randperm(input.size(0)).cuda()      # partner B = shuffled batch
            target_a, target_b = target, target[index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[index, :, bbx1:bbx2, bby1:bby2]
            # re-adjust lambda to the TRUE pasted-area fraction after clipping
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = net(input)
            # CE linear in target: lam*CE(.,a) + (1-lam)*CE(.,b)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            output = net(input)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Typical settings: CIFAR-100 with PyramidNet-200 or ResNet, α = 1 and cutmix probability 0.5; ImageNet with ResNet-50/101, α = 1 and cutmix probability 1.0 (applied every iteration). Both use SGD momentum 0.9 with weight decay and step-decayed learning rate, standard crop/flip/normalize underneath. Adds negligible compute and leaves the architecture, optimizer, and loss otherwise unchanged.
