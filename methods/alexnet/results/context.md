# Context

The aim is to train an object-recognition model with enough learning capacity to handle the full variability of real-world images at the new scale of ImageNet — roughly 1.2 million labeled high-resolution images across 1000 classes — using the models, hardware, and software primitives available in 2012, and to make a high-capacity model *fast enough to train* and *resistant enough to overfitting* that it can actually be fit on this data.

## Research question

Object recognition in realistic settings is hard precisely because real objects vary enormously — in pose, illumination, occlusion, background — so a model that recognizes them needs a very large learning capacity. Until recently the labeled-image datasets were small (tens of thousands of examples: NORB, Caltech-101/256, CIFAR-10/100), and on tasks that simple — even MNIST digit recognition, where the best error is below 0.3% — relatively shallow methods plus label-preserving augmentation suffice. But those datasets are too small to learn the thousands-of-objects task at the fidelity the real world demands; their shortcomings are widely recognized. Large datasets have now arrived — LabelMe with hundreds of thousands of segmented images, and ImageNet with over 15 million labeled high-resolution images across ~22,000 categories.

The blunt question: with a dataset this large, can we train a model with capacity large enough to actually learn the task, and will it generalize? Two things make this non-trivial. First, even ImageNet does not fully specify the mapping from image to label — the object-recognition problem is so complex that the model must carry strong prior knowledge to compensate for the data it still lacks. Second, a model with capacity that large has tens of millions of free parameters, so overfitting and training time both become first-order obstacles, not afterthoughts. A solution has to (a) encode the right priors about images, (b) be trainable in a tolerable amount of wall-clock time on the hardware that exists, and (c) not overfit 60-million-parameter-scale models on ~1.2M images.

## Background

By 2012 the field's working hypothesis is that recognizing objects in realistic settings requires *both* much larger labeled datasets and much higher-capacity models, together with better techniques for preventing overfitting. Two ingredients have just become available that make the bet plausible.

**Convolutional neural networks as a capacity-with-priors model.** CNNs (LeCun et al. 1990; LeCun, Bottou et al.; Turaga et al. 2010) are a class of models whose capacity can be controlled by varying depth and breadth, and which build in strong, mostly-correct assumptions about the nature of images: **stationarity of statistics** (the same feature is useful everywhere in the image, so weights can be shared across spatial positions) and **locality of pixel dependencies** (a unit need only look at a local neighborhood). Because of weight sharing and local connectivity, a CNN has far fewer connections and parameters than a fully-connected net with similarly sized layers, so it is easier to train, while its theoretically best performance is likely only slightly worse. These priors are exactly the "prior knowledge to compensate for the data we don't have" that the problem demands.

**GPUs plus an optimized 2D-convolution implementation.** CNNs have historically been prohibitively expensive to apply at large scale to high-resolution images. Current GPUs, paired with a highly optimized implementation of 2D convolution, are now powerful enough to train interestingly large CNNs; and datasets like ImageNet are large enough to train such models without severe overfitting. A single GTX 580 GPU has only 3GB of memory, which caps the size of a net that fits on it.

The classical reasons high-capacity nets were hard to train and easy to overfit are the live pain points:
- **Slow training from saturating nonlinearities.** The standard neuron output is `f(x)=tanh(x)` or the logistic sigmoid `f(x)=(1+e^{-x})^{-1}`. Both saturate: for large |x| the derivative is ~0, so gradient descent crawls. At the scale of a large net on a large dataset, this alone can make an experiment infeasible.
- **Overfitting of large models.** A net with tens of millions of parameters can memorize even a million-image training set. The 1000 classes impose only ~10 bits of constraint per training example — far too little to pin down so many parameters without strong regularization.
- **Memory and time limits.** The net's size is bounded mainly by GPU memory and by how much training time is tolerable.

The diagnostic observation that motivates a non-saturating neuron: on a four-layer CNN trained on CIFAR-10 to a fixed 25% training-error target, a rectified `max(0,x)` unit reaches that target several times faster than a `tanh` unit — the gap widens with network size. This is a fitting-speed phenomenon (how fast the model drives down *training* error), distinct from the overfitting concern that earlier rectifier-like work focused on.

## Baselines

**Saturating-nonlinearity CNNs.** The conventional CNN uses `tanh` or sigmoid units. These train slowly because of saturation; on large nets and large data the slowdown is prohibitive. This is the direct foil for the choice of neuron.

**Jarrett et al. 2009 — `|tanh|` with local contrast normalization (ICCV).** A multi-stage architecture using the nonlinearity `f(x)=|tanh(x)|` followed by local contrast normalization and local average pooling, which works particularly well on Caltech-101. *Gap:* their primary concern is preventing overfitting on a small dataset, so the effect they observe is about regularization, not about the accelerated *fitting* of training error that matters when training a large model on a large dataset. It does, however, establish that a local normalization stage in the convolutional pipeline can help.

**Cireşan et al. 2011/2012 — high-performance / multi-column GPU CNNs.** GPU-trained CNNs, including a "columnar" architecture that runs several CNN columns and combines them. *Gap / reusable piece:* shows GPU CNNs are practical and that running multiple columns helps; the columns there are independent, whereas a single net could instead be *split across two GPUs* with the columns deliberately *not* independent, communicating in chosen layers, to fit a bigger net in limited memory.

**Hand-engineered features + shallow classifiers (the ILSVRC state of the art).** The leading large-scale recognition pipelines encode images with engineered descriptors and classify with shallow models: sparse-coding pipelines averaging predictions over features (ILSVRC-2010 winner), and Fisher-Vector encodings of densely sampled features with linear classifiers (Sánchez & Perronnin 2011). *Gap:* these features are fixed and engineered rather than learned end-to-end from the data, so they cannot exploit the full 1.2M-image signal the way a trained deep model can; they are the yardstick a learned CNN would have to beat.

**Dropout (Hinton et al. 2012).** A then-new regularization technique: during training, zero each hidden unit's output with probability 0.5; dropped units contribute to neither the forward pass nor backprop, so each presented example trains a different sub-network that shares weights with all the others. *Role:* a candidate cure for the overfitting of the large fully-connected layers, far cheaper than training and combining many separate models.

**ReLU (Nair & Hinton 2010).** The rectified linear unit `f(x)=max(0,x)`, introduced to improve restricted Boltzmann machines. *Role:* the non-saturating neuron whose fast-fitting behavior is the key to training a large net in tolerable time.

## Evaluation settings

- **ImageNet / ILSVRC (1000 classes).** ImageNet: ~15M labeled high-resolution images, ~22k categories, collected from the web and labeled via Amazon Mechanical Turk. ILSVRC uses a 1000-category subset with ~1000 images each: ~1.2M training, 50k validation, 150k test. ILSVRC-2010 is the only release with public test labels (so it allows full test-set evaluation); the 2012 release has private test labels (validation and test error track within ~0.1% of each other, so they are used interchangeably).
- **Metrics.** Top-1 and top-5 error rate, where top-5 error is the fraction of test images whose correct label is not among the model's five most probable labels. Training error itself is a first-class quantity here — fitting speed is measured as iterations to reach a fixed training-error target.
- **Input pipeline (fixed-size requirement).** The model needs constant input dimensionality, so variable-resolution images are down-sampled to 256×256: rescale so the shorter side is 256, then crop the central 256×256 patch. The only further preprocessing is subtracting the per-pixel mean over the training set; the net is trained on centered raw RGB values.
- **Testing protocol.** Multi-crop averaging at test time: extract five 224×224 patches (four corners + center) and their horizontal reflections (ten patches), and average the softmax predictions over the ten. Ensembling several independently trained nets by averaging their softmax outputs is also standard practice.
- **Comparison principle.** Compare against the contemporaneous ILSVRC leaders (engineered-feature pipelines) on the same train/val/test split and the same top-1/top-5 metrics.

## Code framework

The available code is a bare GPU CNN image-classification harness. The libraries supply 2D convolution, max-pooling, an SGD-with-momentum optimizer, a step learning-rate scheduler, a softmax cross-entropy loss, and standard image transforms (resize, crop, flip, mean-subtraction). The scaffold is the harness with one empty slot: the network body and the regularization/normalization pieces are stubs.

```python
import torch
import torch.nn as nn


class Net(nn.Module):
    # TODO: the architecture we'll design.
    # Known going in: a CNN built from conv -> nonlinearity -> (optional pool)
    # stacks that exploit weight sharing + local connectivity, ending in some
    # fully-connected layers and a 1000-way softmax. How many conv layers, their
    # kernel sizes / strides / channel counts, where pooling and any normalization
    # go, how the fully-connected head is sized and regularized, which nonlinearity
    # the units use, and how the whole thing is laid out across the limited GPU
    # memory available — all open.
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = self._make_features()   # TODO: the conv stack
        self.classifier = self._make_classifier(num_classes)  # TODO: the FC head

    def _make_features(self):
        # TODO: the convolutional body (conv / nonlinearity / pool / normalization)
        raise NotImplementedError

    def _make_classifier(self, num_classes):
        # TODO: the fully-connected head + output projection, with whatever
        # regularization the large parameter count turns out to need
        raise NotImplementedError

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# --- training harness ---
model = Net(num_classes=1000)
criterion = nn.CrossEntropyLoss()                       # multinomial logistic regression
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1)

# data pipeline: resize shorter side to 256, central 256x256 crop, subtract
# per-pixel training mean; then label-preserving augmentation TBD.
# train_transform = Compose([Resize(256), CenterCrop(256), ...,
#                            ToTensor(), Normalize(pixel_mean, ...)])

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
    return correct / total
```
