## Research question

Convolutional image classifiers trained by minimizing average cross-entropy on a finite training set tend to latch onto the single most discriminative part of an object — the head of a person, the face of a dog — and ignore the rest. This is fragile: it generalizes poorly when the discriminative part is occluded or absent, and it localizes poorly, because the network's evidence for a class is concentrated on a small patch rather than the full object extent. A family of *regional dropout* regularizers addresses this by deleting a contiguous region of each training image, forcing the network to find class evidence elsewhere and thereby attend to the whole object. They work — both classification and weakly-supervised localization improve. But the deletion has a cost: the removed pixels are replaced with zeros or random noise, so they carry no signal at all. Every training image now spends a fraction of its area on uninformative content, and CNNs are data-hungry.

The question is whether the "attend to the whole object" benefit of regional dropout can be kept while making the deleted region carry useful information instead of wasting it. A good answer would: (i) retain a genuine *region* being dropped from each image, so the whole-object/localization benefit survives; (ii) ensure no training pixel is wasted; (iii) keep the augmented images locally natural (no artifacts that confuse the model about where the object is); (iv) add negligible compute and leave the network, optimizer, and loss untouched.

## Background

**The field state.** Deep CNNs (AlexNet, VGG, ResNet) are trained with SGD-with-momentum, weight decay, and a battery of regularizers and augmentations. Data augmentation — random crops, horizontal flips, color jitter — is the most reliable generalization tool, and a set of "feature removal" regularizers has grown up alongside it.

**Feature removal as regularization.** Dropout (Srivastava et al., 2014) randomly zeroes hidden activations during training, breaking co-adaptation among units so that no single feature can dominate. The same removal idea was carried to the *input*: instead of dropping hidden units, drop a region of the image. The motivating diagnostic, reported repeatedly across this line of work, is that an unregularized classifier concentrates its class evidence on a small, most-discriminative patch — visible in class activation maps as a single hot spot — and that occluding that patch causes the prediction to collapse. Deleting random regions during training measurably spreads the evidence over the whole object and improves both held-out accuracy and localization.

**Label softening.** Label smoothing (Szegedy et al., 2016) replaces the one-hot target with a softened distribution, discouraging overconfidence. It establishes that a *soft* multi-class target is a legitimate and beneficial training signal — relevant because any method that blends two images will want to blend their labels.

**Mixing whole images.** Mixup (Zhang et al., 2017) takes a different route to filling the off-data region: rather than deleting pixels, it forms a per-pixel convex blend of two images and the matching convex blend of their one-hot labels, with the mixing weight drawn from a Beta distribution. Every pixel stays informative and the label tracks the mixture. The diagnostic limitation, visible in the mixed images themselves and in their class activation maps, is that a translucent pixel-wise overlay of two photographs is *locally unnatural* — a ghosted superposition that exists in no real image — and the network's activation map on such an input is diffuse and confused about where each object is. This helps classification but degrades localization and detection transfer.

## Baselines

**Cutout (DeVries & Taylor, 2017).** Cut a single fixed-size square out of each input image at a uniformly random location and set those pixels to zero (the dataset mean after normalization). Core idea: a contiguous input-space dropout that simulates occlusion, forcing the network to use the whole object rather than one patch. Algorithm: sample a center, zero a square of fixed side length around it (clipped at borders); label unchanged. Gap: the zeroed square is informationless — a sizable fraction of every image contributes nothing to the forward pass or the gradient, which is wasteful for a data-hungry model.

**Random Erasing (Zhong et al., 2017).** Same regional deletion, but the erased rectangle is filled with random pixel values (and its size/aspect ratio are randomized). Core idea and gap as Cutout — the filled region is still noise, carrying no class signal.

**Hide-and-Seek (Singh & Lee, 2017).** Partition the image into a grid of patches and randomly hide a subset each iteration, again to spread localization evidence over the object. Gap: hidden patches are deleted content, same information-loss issue.

**DropBlock (Ghiasi et al., 2018).** Regional dropout moved into feature space — drop contiguous blocks of a feature map rather than scattered units. Core idea: contiguous removal is more effective than pointwise dropout in convolutional layers. Gap: operates on internal activations, not pixels; still a removal, not a reuse, of information.

**Mixup (Zhang et al., 2017).** x̃ = λx_A + (1−λ)x_B, ỹ = λy_A + (1−λ)y_B, λ ~ Beta(α, α). Core idea: no deletion at all — a global convex blend of two images and their labels, so every pixel is informative and the supervision is a soft mixture. Gap: the per-pixel blend is locally unnatural (ghosting), there is no contiguous region being dropped so the regional-dropout/localization benefit is absent, and the resulting activation maps are diffuse — classification improves but localization and detection-transfer suffer.

## Evaluation settings

The yardsticks are image-classification benchmarks: CIFAR-10 and CIFAR-100 (50k train / 10k test, 32×32, 10 and 100 classes) and ImageNet-2012 (1.28M train, 1000 classes, top-1/top-5 error). Standard backbones are residual networks — ResNet-50/101, PyramidNet-200, Wide-ResNet, ResNeXt — trained with SGD-with-momentum 0.9, weight decay, step-decayed learning rate, batch size 128 (CIFAR) to 256 (ImageNet), with the usual pad-crop + horizontal-flip + per-channel normalization underneath. Beyond clean accuracy, the relevant probes are: weakly-supervised object localization (CUB200-2011 and ImageNet) scored by class-activation-map overlap with ground-truth boxes; transfer to Pascal VOC detection (mAP) and MS-COCO captioning (BLEU) from a pretrained backbone; robustness to occlusion and to adversarial (FGSM) perturbations; and out-of-distribution detection. Cutout and Mixup are the natural augmentation comparison points.

## Code framework

The available ingredients are a CIFAR/ImageNet dataset with standard pad-crop/flip/normalize augmentation, a loader yielding minibatches of inputs and integer labels, a residual network, a cross-entropy criterion, and an SGD-with-momentum training loop. What is open is how a raw minibatch is transformed into the (input, target) pair the network and loss consume, and how the loss reads whatever target object that transformation returns.

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                           shuffle=True, num_workers=8)

net = build_network()                       # e.g. a ResNet / PyramidNet
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
net = net.cuda()
mix_strength = 0.0


def make_training_pairs(x, y, strength=mix_strength):
    """Turn a raw minibatch (inputs x, integer labels y) into the (input, target)
    pair fed to the network and loss."""
    # TODO: construct the virtual input and its target specification here.
    pass


def compute_loss(criterion, outputs, target_spec):
    """Evaluate the loss against the target object produced above."""
    # TODO: compute the criterion-compatible loss.
    pass


def train(epoch):
    net.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.cuda(), targets.cuda()

        inputs, target_spec = make_training_pairs(inputs, targets, mix_strength)
        outputs = net(inputs)
        loss = compute_loss(criterion, outputs, target_spec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
