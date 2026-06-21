Real-world object recognition is hard because objects vary enormously in pose, lighting, occlusion, and background, so solving it requires a model with very large learning capacity and a dataset large enough to train it. ImageNet/ILSVRC now provides roughly 1.2 million training images across 1000 classes, which is enough data to justify a high-capacity model, but the problem is still underdetermined: a 60-million-parameter model receives only about ten bits of label constraint per example. That means the architecture must encode strong prior knowledge about images, the model must train fast enough to actually finish, and it must not simply memorize the training set. Earlier approaches fall short on at least one of these fronts. Saturating-neuron CNNs using tanh or sigmoid units train prohibitively slowly because their gradients vanish on large-magnitude inputs. Hand-engineered features combined with shallow classifiers cannot exploit the full supervised signal in 1.2 million images because the representation is fixed rather than learned. Even the promising recent GPU-trained CNNs were limited by the memory of a single GPU, which caps model size and therefore capacity. So the challenge is to build a deep CNN large enough to learn ImageNet, make it trainable on 2012-era hardware, and keep it from overfitting.

The method I propose is AlexNet. It is a large, deep convolutional neural network trained end-to-end by supervised SGD for 1000-way ImageNet classification. Its core insight is to combine three things in one system: strong image priors built into a deep CNN architecture, non-saturating ReLU units and two-GPU model parallelism to make that large model trainable, and aggressive but cheap regularization through label-preserving data augmentation and dropout. The convolutional body encodes stationarity of image statistics through weight sharing and locality of pixel dependencies through local receptive fields, giving far fewer parameters than a dense net of comparable layer size while keeping capacity tunable via depth and width. The depth is load-bearing: removing any single middle convolutional layer, each holding less than one percent of the parameters, costs about two percent top-1 accuracy. The first convolutional layer uses a large 11×11 receptive field with stride 4 to capture coarse structure cheaply, the second uses 5×5 kernels, and the next three stack smaller 3×3 kernels to add depth without blowing up computation.

To make the network fit and train, AlexNet uses ReLU activations everywhere instead of tanh or sigmoid. ReLU does not saturate on the positive side, so its gradient remains one wherever the unit is active; this produces a roughly six-fold speedup to a fixed training-error target compared with tanh on a four-layer CIFAR-10 CNN, and the gap grows with network size. Because the full model does not fit in the 3GB memory of a single GTX 580 GPU, the network is split across two GPUs. Half the kernels live on each GPU, and they communicate only in selected layers. For example, the third convolutional layer reads feature maps from both GPUs, while the fourth and fifth read only from the same GPU, keeping cross-GPU traffic small. This split not only solves the memory problem but also beats a single-GPU net with half the kernels per layer by about 1.7% top-1 and 1.2% top-5 error. Two smaller architectural choices improve accuracy slightly: local response normalization, which creates lateral-inhibition-style competition across kernel maps at each spatial location and reduces top-1 error by about 1.4%, and overlapping max-pooling with 3×3 windows stepping by 2, which reduces top-1 by another 0.4% and appears slightly harder to overfit.

Overfitting is attacked from two directions: more effective data and direct regularization. During training, each 256×256 image is randomly cropped to 224×224 and horizontally flipped, expanding the effective training set by roughly 2048×. In addition, PCA is performed on RGB pixel values across the training set, and a small perturbation along each principal component direction is added to every pixel of the image, with the same random coefficients reused across the image. This injects illumination and color invariance in the directions in which real images naturally vary. At test time, predictions are averaged over ten crops: the four corners, the center, and their horizontal reflections. For direct regularization, dropout with probability 0.5 is applied to the first two fully-connected layers, which is where most of the 60 million parameters live. Dropout trains an exponential ensemble of weight-sharing sub-networks and prevents hidden units from forming fragile co-adaptations. At test time, all units are kept active and their outputs are multiplied by 0.5 to approximate the ensemble average.

The model is trained with stochastic gradient descent using batch size 128, momentum 0.9, and weight decay 5e-4. Weight initialization is important for getting ReLUs moving: weights are drawn from a zero-mean Gaussian with standard deviation 0.01, and biases are set to 1 in the second, fourth, and fifth convolutional layers and in all fully-connected layers so that ReLUs receive positive inputs early in training. The learning rate starts at 0.01 and is divided by 10 whenever validation error plateaus, for a total of three reductions. The objective is multinomial logistic regression, equivalently softmax cross-entropy. Training runs for about ninety epochs and takes five to six days on two GTX 580 GPUs.

```python
import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
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


# Local response normalization as used after ReLU on the early conv layers.
lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0)

model = AlexNet(num_classes=1000, dropout=0.5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

for m in model.features:
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.01)
for m in model.classifier:
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.constant_(m.bias, 1.0)
```

AlexNet is therefore one coherent bet: encode image priors with a deep CNN, make large-scale training feasible with ReLU and two-GPU model parallelism, sharpen accuracy slightly with response normalization and overlapping pooling, and defend against overfitting with label-preserving augmentation and dropout, all trained end-to-end by momentum SGD to maximize the log-probability of the correct label.
