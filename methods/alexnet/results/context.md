# Context

The aim is to train an object-recognition model with enough learning capacity to handle the full variability of real-world images at the new scale of ImageNet — roughly 1.2 million labeled high-resolution images across 1000 classes — using the models, hardware, and software primitives available in 2012.

## Research question

Object recognition in realistic settings calls for a very large learning capacity, because real objects vary enormously — in pose, illumination, occlusion, background. Until recently the labeled-image datasets were small (tens of thousands of examples: NORB, Caltech-101/256, CIFAR-10/100), and on tasks that simple — even MNIST digit recognition, where the best error is below 0.3% — relatively shallow methods plus label-preserving augmentation suffice. Large datasets have now arrived — LabelMe with hundreds of thousands of segmented images, and ImageNet with over 15 million labeled high-resolution images across ~22,000 categories.

The question: with a dataset this large, can we train a model with capacity large enough to learn the thousands-of-objects task, and will it generalize? Even ImageNet does not fully specify the mapping from image to label, so the model carries prior knowledge to compensate for the data it lacks; and a model with capacity that large has tens of millions of free parameters, so overfitting and training time are both in play.

## Background

By 2012 the field's working hypothesis is that recognizing objects in realistic settings requires *both* much larger labeled datasets and much higher-capacity models, together with better techniques for preventing overfitting. Two ingredients have just become available that make the bet plausible.

**Convolutional neural networks as a capacity-with-priors model.** CNNs (LeCun et al. 1990; LeCun, Bottou et al.; Turaga et al. 2010) are a class of models whose capacity can be controlled by varying depth and breadth, and which build in strong assumptions about the nature of images: **stationarity of statistics** (the same feature is useful everywhere in the image, so weights can be shared across spatial positions) and **locality of pixel dependencies** (a unit need only look at a local neighborhood). Because of weight sharing and local connectivity, a CNN has far fewer connections and parameters than a fully-connected net with similarly sized layers, while its theoretically best performance is likely only slightly worse.

**GPUs plus an optimized 2D-convolution implementation.** CNNs have historically been expensive to apply at large scale to high-resolution images. Current GPUs, paired with a highly optimized implementation of 2D convolution, are powerful enough to train large CNNs; and datasets like ImageNet are large enough to train such models. A single GTX 580 GPU has 3GB of memory, which bounds the size of a net that fits on it.

Some standing facts about high-capacity nets:
- **Saturating nonlinearities.** The standard neuron output is `f(x)=tanh(x)` or the logistic sigmoid `f(x)=(1+e^{-x})^{-1}`. Both saturate: for large |x| the derivative is ~0.
- **Model size vs. data.** A net with tens of millions of parameters has more capacity than a million-image training set tightly constrains; the 1000 classes impose roughly 10 bits of constraint per training example.
- **Memory and time.** The net's size is bounded mainly by GPU memory and by how much training time is tolerable.

## Baselines

**Saturating-nonlinearity CNNs.** The conventional CNN uses `tanh` or sigmoid units.

**Jarrett et al. 2009 — `|tanh|` with local contrast normalization (ICCV).** A multi-stage architecture using the nonlinearity `f(x)=|tanh(x)|` followed by local contrast normalization and local average pooling, which works particularly well on Caltech-101.

**Cireşan et al. 2011/2012 — high-performance / multi-column GPU CNNs.** GPU-trained CNNs, including a "columnar" architecture that runs several CNN columns and combines them; the columns are independent models, each fitting within a single GPU's memory.

**Hand-engineered features + shallow classifiers (the ILSVRC state of the art).** The leading large-scale recognition pipelines encode images with engineered descriptors and classify with shallow models: sparse-coding pipelines averaging predictions over features (ILSVRC-2010 winner), and Fisher-Vector encodings of densely sampled features with linear classifiers (Sánchez & Perronnin 2011). These are the contemporaneous yardstick on the ILSVRC metrics.

**Dropout (Hinton et al. 2012).** A then-new regularization technique: during training, zero each hidden unit's output with probability 0.5; dropped units contribute to neither the forward pass nor backprop, so each presented example trains a different sub-network that shares weights with all the others. It is far cheaper than training and combining many separate models.

**ReLU (Nair & Hinton 2010).** The rectified linear unit `f(x)=max(0,x)`, introduced to improve restricted Boltzmann machines. Unlike `tanh`/sigmoid it does not saturate on the positive side.

## Evaluation settings

- **ImageNet / ILSVRC (1000 classes).** ImageNet: ~15M labeled high-resolution images, ~22k categories, collected from the web and labeled via Amazon Mechanical Turk. ILSVRC uses a 1000-category subset with ~1000 images each: ~1.2M training, 50k validation, 150k test. ILSVRC-2010 is the only release with public test labels (so it allows full test-set evaluation); the 2012 release has private test labels (validation and test error track within ~0.1% of each other, so they are used interchangeably).
- **Metrics.** Top-1 and top-5 error rate, where top-5 error is the fraction of test images whose correct label is not among the model's five most probable labels. Training error itself is a first-class quantity here — training speed can be measured as iterations to reach a fixed training-error target.
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
