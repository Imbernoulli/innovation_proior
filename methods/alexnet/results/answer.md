# AlexNet, distilled

AlexNet is a large, deep convolutional neural network trained end-to-end by supervised SGD that classifies the 1000-class ImageNet (ILSVRC) data, demonstrating that a learned deep CNN beats engineered-feature pipelines at large scale. It combines a non-saturating neuron for fast training, a two-GPU split to fit a high-capacity net in limited memory, and an aggressive overfitting defense (augmentation + dropout) for its 60M parameters.

## The problem

Real-world object recognition needs a model with very large learning capacity, and fitting one needs a lot of data. ImageNet/ILSVRC now supplies ~1.2M training images over 1000 classes — but even that underdetermines a 60M-parameter model (≈10 bits/example of label constraint), so the model must (a) encode strong image priors, (b) train fast enough to run at all, and (c) not overfit.

## The key ideas

- **CNN as capacity-with-priors.** Weight sharing encodes the stationarity of image statistics; local receptive fields encode locality of pixel dependencies. This gives far fewer parameters than a dense net of similar layer sizes, with capacity tunable by depth/breadth. Depth is load-bearing: removing any single conv layer (each <1% of params) costs ~2% top-1.
- **ReLU `f(x)=max(0,x)`.** Non-saturating → gradient is 1 wherever active → no saturation slowdown. ~6× faster to a fixed training-error target than `tanh` on a 4-layer CIFAR-10 net; the gap grows with size. Makes the large-net experiment feasible.
- **Two-GPU model parallelism.** A 3GB GPU can't hold the net; put half the kernels on each GPU and let them communicate only in chosen layers (e.g. conv3 reads all maps; conv4/conv5 read only same-GPU maps), tuning communication to an acceptable fraction of compute. Beats the half-width single-GPU net by 1.7%/1.2% top-1/top-5.
- **Local Response Normalization (LRN).** Lateral-inhibition-style competition across kernel maps at each location, normalizing each activation by neighboring kernels' activity. ~1.4%/1.2% gain.
- **Overlapping max-pooling** (`z=3, s=2`): 0.4%/0.3% gain over non-overlapping, slightly harder to overfit.
- **Data augmentation:** random 224×224 crops + horizontal reflections from 256×256 images (~2048× expansion); PCA color jitter along the natural RGB principal components. 10-crop averaging at test.
- **Dropout** (p=0.5) on the first two FC layers: a cheap weight-sharing ensemble that prevents co-adaptation; test-time ×0.5 approximates the geometric mean of the ensemble.

## The architecture (original two-tower)

8 weight layers: 5 conv + 3 FC. Input 224×224×3.
- conv1: 96 kernels 11×11×3, stride 4 → ReLU → LRN → max-pool(3,2)
- conv2: 256 kernels 5×5×48 → ReLU → LRN → max-pool(3,2)
- conv3: 384 kernels 3×3×256 (reads both GPUs) → ReLU
- conv4: 384 kernels 3×3×192 → ReLU
- conv5: 256 kernels 3×3×192 → ReLU → max-pool(3,2)
- FC6: 4096 → FC7: 4096 → FC8: 1000-way softmax. ReLU after every layer; dropout on FC6, FC7.
- ≈60M parameters, 650k neurons.

**LRN:** `b^i_{x,y} = a^i_{x,y} / (k + α Σ_{j=max(0,i−n/2)}^{min(N−1,i+n/2)} (a^j_{x,y})^2)^β`, with `k=2, n=5, α=1e-4, β=0.75`, applied after ReLU on conv1, conv2.

**PCA color augmentation:** add `[p₁,p₂,p₃]·[α₁λ₁, α₂λ₂, α₃λ₃]ᵀ` to each pixel, where `(p_i, λ_i)` are eigenvectors/eigenvalues of the 3×3 RGB covariance and `α_i ~ N(0, 0.1)` drawn once per image.

## Training recipe

SGD, batch 128, momentum 0.9, weight decay 5e-4 (which also lowers *training* error here). Update: `v_{i+1}=0.9·v_i − 0.0005·ε·w_i − ε·⟨∂L/∂w⟩_{D_i}`, `w_{i+1}=w_i+v_{i+1}`. Weights ~ N(0, 0.01); biases = 1 on conv2/4/5 and all FC (feeds ReLUs positive inputs early), 0 elsewhere. LR shared, start 0.01, ÷10 on validation plateau (reduced 3× total). ~90 epochs, 5–6 days on two GTX 580 GPUs. Objective: multinomial logistic regression (softmax cross-entropy).

## Working code

The widely-used single-stream form folds the two GPU towers into one stream (channels 64/192/384/256/256); the structure is identical to the derivation above.

```python
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # overlapping pooling
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# LRN as in the original net (after ReLU on early conv layers):
lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

model = AlexNet(num_classes=1000, dropout=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

for m in model.features:
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.01)
for m in model.classifier:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.constant_(m.bias, 1.0)
```

The method reduces to one bet: a deep CNN trained purely supervised, made trainable by ReLU + GPU parallelism and kept from overfitting by augmentation + dropout, beats engineered features at ImageNet scale — and the deeper it is, the better.
