Convolutional image classifiers trained with the usual cross-entropy objective have a well-known failure mode: they learn to rely on the single most discriminative patch of an object, such as a dog's face or a person's head, and largely ignore the rest. This makes them fragile when that patch is occluded and weak at weakly-supervised localization, because the class activation map collapses to one hot blob instead of covering the full object extent. Regional dropout methods like Cutout address this by deleting a contiguous square from every training image, which forces the network to gather evidence from the remaining pixels and thus spreads attention across the whole object. The trouble is that the deleted region is replaced with zeros or random noise, so a significant fraction of every image contributes nothing to the forward pass or the gradient. In a regime where CNNs are already data-hungry, deliberately throwing away pixels is an expensive way to buy regularization. Mixup, by contrast, never wastes pixels: it blends two whole images and their one-hot labels, so every location carries signal. But the resulting translucent superposition is locally unnatural, and because nothing is truly occluded, the network's activation maps become diffuse and localization suffers. The goal is to keep the whole-object benefit of regional dropout while making the replaced region informative and the composite image natural.

The method that achieves this is CutMix. Instead of filling Cutout's deleted rectangle with zeros or noise, CutMix fills it with the corresponding rectangular patch cut from another training image. More concretely, given an image A and a randomly chosen partner image B from the same minibatch, a binary mask M marks a contiguous rectangle as the region to replace. The augmented input is x̃ = M ⊙ x_A + (1 − M) ⊙ x_B. The rectangle occupies a (1 − λ) fraction of the image area, chosen by sampling λ from a Beta(α, α) distribution with α = 1, so λ is uniform on (0, 1). The rectangle's width and height each scale by √(1 − λ), which makes the area fraction exactly 1 − λ, and its center is sampled uniformly. Because the composite is literally part A and part B, the target label is the area-proportional mixture ỹ = λ y_A + (1 − λ) y_B. This is the only label that honestly matches the pixel composition; a one-hot dominant label would mislabel the pasted patch, and a fixed 0.5/0.5 split would miscalibrate cases where one class dominates. After the box is clipped to the image borders, λ is recomputed from the actual pasted area so the label stays calibrated. Cross-entropy is linear in the target, so the loss can be evaluated as λ·CE(p, y_A) + (1 − λ)·CE(p, y_B) using the ordinary integer-label criterion.

CutMix keeps every virtue it inherits and fixes the defects. A contiguous region of A is genuinely removed, so the network must recognize A from a partial view; that is the same occlusion mechanism that makes Cutout improve localization. The replacement pixels are real pixels of a real image, so no training location is wasted; every pixel contributes to the gradient. The composite is locally natural, because it is made of sharp real patches rather than a ghostly overlay, so activation maps remain clean and the two objects are localizable. The area-proportional soft label matches the construction exactly, avoiding the calibration problems of hard or fixed labels. The implementation requires only a few lines added to the data-loading side of the training loop; the network architecture, optimizer, and loss are otherwise unchanged.

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
