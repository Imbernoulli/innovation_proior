## Research question

Convolutional networks have just taken over large-scale image classification. The open question is
narrow but important: **how much of recognition accuracy is governed by network depth alone, and can
depth be pushed far beyond what current architectures use?**

## Background

The field state: large labeled image repositories (ImageNet, ~1.3M training images over 1000 classes) and
GPUs have made it possible to train large ConvNets end to end. The annual ImageNet Large-Scale Visual
Recognition Challenge (ILSVRC) is the testbed; over a few years it moved from shallow high-dimensional
feature encodings (Fisher vectors, the ILSVRC-2011 winner) to deep ConvNets (the ILSVRC-2012 winner).

The load-bearing concepts:

- **A convolutional layer** slides a bank of small filters over the input; each output channel is a
  filter convolved across spatial positions, followed by a pointwise nonlinearity. Spatial **max pooling**
  downsamples and grants local translation tolerance. Stacking conv+pool builds a hierarchy of features.
- **Receptive field.** Each unit deep in the net "sees" a patch of the input image; the size of that patch
  is the receptive field. With stride-1 convolutions and no pooling between them, stacking conv layers
  grows the receptive field additively as the per-layer footprints overlap. With pooling or large strides,
  resolution is lost and the field grows faster but coarser.
- **ReLU** (rectified linear units, max(0,x)) replaced saturating nonlinearities and sped up training of
  deep nets considerably; it is the standard hidden-layer nonlinearity.
- **Regularization for big models:** dropout (randomly zeroing activations in the fully-connected layers
  during training) and L2 weight decay, both needed because the fully-connected layers hold most of the
  parameters and overfit easily.
- **The 1×1 convolution** (from the "Network in Network" line, Lin et al. 2014): a convolution is a
  generalized linear model over a local patch and can cleanly separate only linearly separable patterns.
  Replacing it with a tiny per-location multilayer perceptron — implementable as stacked 1×1 convolutions
  with a nonlinearity between them — injects extra nonlinear cross-channel mixing at a spatial location
  without enlarging the receptive field.

Motivating empirical findings already on the table:

- The ILSVRC-2013 winners (Zeiler & Fergus; OverFeat, Sermanet et al. 2014) improved on the 2012 network
  chiefly by shrinking the **first** convolutional layer — from an 11×11 filter at stride 4 down to 7×7 at
  stride 2 — and by training/testing densely over the whole image and over multiple scales. Deconvolutional
  visualization (Zeiler & Fergus) showed that finer first-layer sampling captures cleaner low-level
  structure. So smaller, finer first-layer filters were already known to help.
- Goodfellow et al. (2014) applied an 11-weight-layer ConvNet to multi-digit street-number recognition and
  reported that the added depth improved performance — a hint that depth, specifically, matters.
- Small filters in deeper-than-usual nets had been used by Ciresan et al. (2011), but only on small
  datasets (digits, traffic signs), never stress-tested at ILSVRC scale.

## Baselines

These are the prior networks a new architecture would be measured against and reacts to.

- **The ILSVRC-2012 winning ConvNet (Krizhevsky, Sutskever & Hinton, 2012).** Eight weight layers: five
  convolutional, three fully-connected. The first conv layer is 96 filters of 11×11×3 at **stride 4**; the
  second is 5×5; the rest are 3×3. Max pooling follows some conv layers; Local Response Normalization (LRN)
  follows others. ReLU throughout. Dropout 0.5 on the fully-connected layers, L2 weight decay 5×10⁻⁴.
  Trained with SGD + momentum 0.9, batch ~128, learning rate stepped down on plateau, with crop, flip and
  PCA-based color-jitter augmentation on 224×224 inputs.
- **Zeiler & Fergus (2013).** A retuned version of the above with a 7×7 stride-2 first layer; better
  accuracy, and a deconv-based diagnosis of what the layers learn.
- **OverFeat (Sermanet et al., 2014).** 7×7 stride-2 first layer; its key contribution is **dense
  evaluation** — at test time the fully-connected layers are re-expressed as convolutions (the first FC
  becomes a conv whose spatial size equals the last feature map, the rest become 1×1 convs), turning the
  net fully convolutional so it can be applied to an arbitrarily sized image in a single pass, yielding a
  spatial grid of class scores; plus multi-scale processing.
- **Network in Network (Lin et al., 2014).** Introduced the 1×1-convolution / mlpconv idea to add
  per-location nonlinearity, and global average pooling instead of fully-connected layers.
- **Ciresan et al. (2011); Goodfellow et al. (2014).** Small-filter and/or deeper nets on small
  datasets or a narrow task.

## Evaluation settings

- **Dataset:** ILSVRC-2012, 1000 classes; ~1.3M training images, 50K validation, 100K test (held-out
  labels). The validation set is commonly used as a development test set.
- **Metrics:** top-1 error (fraction of images whose top prediction is wrong) and top-5 error (fraction
  whose ground-truth class is not among the five highest-scored), the latter being the headline ILSVRC
  criterion.
- **Protocol:** isotropically rescale images to a chosen smallest-side length; subtract the training-set
  mean RGB; train on random 224×224 crops with horizontal-flip and color augmentation; evaluate with crop
  ensembling and/or whole-image dense application, optionally over several scales, optionally averaging
  several models. Transfer to other classification benchmarks (PASCAL VOC, Caltech-101/256) is done by
  using a pretrained net as a fixed feature extractor feeding a linear classifier.

## Code framework

A Caffe-style deep-learning toolbox already provides convolution, ReLU, max pooling, linear layers,
dropout, softmax cross-entropy, SGD with momentum and weight decay, and a data pipeline that rescales,
crops, flips, color-jitters, and mean-centers images. The image-classification harness can stay generic:
a symbolic layer plan builds a convolutional feature extractor, a module attaches a classifier head, and
the training step optimizes class scores against labels.

```python
from typing import Union

import torch
import torch.nn as nn

LayerSpec = Union[str, int]


layer_plans: dict[str, list[LayerSpec]] = {
    # TODO: fill with controlled convolution/pooling recipes
}


def make_layers(plan: list[LayerSpec]) -> nn.Sequential:
    # TODO: turn a symbolic plan into conv / nonlinearity / pooling layers
    pass


class ImageClassifier(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = None                              # TODO: fixed-size bridge from features to head
        self.classifier = None                           # TODO: classifier head over conv features
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        # TODO: initialize convolutional and linear layers
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: features -> fixed-size tensor -> flatten -> classifier -> class scores
        pass


def build_classifier(plan_name: str, **kwargs) -> ImageClassifier:
    # TODO: pick a plan, build its layers, and wrap them in the classifier module
    pass


def train_step(model, images, labels, optimizer, loss_fn):
    optimizer.zero_grad()
    scores = model(images)
    loss = loss_fn(scores, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def make_optimizer(model):
    # TODO: choose the optimizer and regularization recipe for this classifier
    pass


loss_fn = nn.CrossEntropyLoss()


def to_fully_convolutional(model, num_classes: int = 1000) -> nn.Sequential:
    # TODO: convert the fixed classifier head into convolutional layers for whole-image evaluation
    pass


def predict_dense(model, image, num_classes: int = 1000):
    # TODO: run the trained classifier over a whole image and aggregate class-score maps
    pass
```
