# Context

## Research question

Convolutional networks classify, localise, and segment images far better than anything that came before, yet a trained network responds differently to an object depending on where it sits in the frame, how large it is, and how it is rotated or warped. The only spatial flexibility built into the architecture is local max-pooling, a fixed, hand-wired mechanism with a tiny receptive field. The question is whether a network can be given a learnable ability to spatially normalise its input — to find the relevant region and warp it into a canonical pose before the rest of the network has to recognise it — using nothing but the ordinary task loss and backpropagation, with no extra labels describing the correct transformation, and in a form that can be inserted into any existing architecture at any depth.

## Background

The dominant model is the convolutional network (LeCun et al., 1998): stacks of learned convolutions interleaved with pointwise nonlinearities and spatial pooling, trained end-to-end by backpropagation. Convolution gives *translation equivariance* — shifting the input shifts the feature map — and local max-pooling over a small window (commonly 2×2) collapses that into a small amount of local *translation invariance*. The standard reading is that pooling lets the network "disentangle object pose from identity."

With a 2×2 support, a single pooling layer is invariant only to a one-pixel jitter; any robustness to a large shift, a change of scale, or a rotation has to be assembled across a deep hierarchy of alternating convolution and pooling. Measurements of how CNN representations actually behave under input transformations make the limit concrete: studies estimating the linear relationship between representations of an image and its transformed copy (Lenc & Vedaldi, 2015), and analyses of how learned representations transform under symmetry groups (Cohen & Welling, 2015), find that intermediate feature maps are *not* invariant to large transformations of the input.

The field's working remedy for nuisance spatial variation is data augmentation: synthesise rotated, scaled, translated, and warped copies of every training image so the network is forced to see the nuisance variation.

Two lines of work build spatial handling more directly into the architecture. One builds invariance into the *feature extractor*: tie or transform the filters across a group of transformations so the response is invariant by design — deep symmetry networks over symmetry groups (Gens & Domingos, 2014), locally scale-invariant CNNs (Kanazawa et al., 2014), filter banks of transformed filters (Sohn & Lee, 2012), and scattering networks that are provably invariant to small deformations (Bruna & Mallat, 2013). The other models objects as *transformed parts* — assigning canonical object-based frames of reference (Hinton, 1981), transforming autoencoders that predict and apply 2D affine transforms to parts (Hinton et al., 2011), and generative models that explicitly affine-transform learnt parts (Tieleman, 2014). These are trained with the transformations supplied as input or target.

A third, closely related thread is attention. Glimpse and foveation models select a sub-region to process: learning fovea trajectories for target detection (Schmidhuber & Huber, 1991), recurrent visual attention reading crops (Ba et al., 2015), attention for fine-grained categorisation (Sermanet et al., 2014). Because taking a hard crop is not differentiable, these are typically trained with reinforcement learning. The DRAW generative model (Gregor et al., 2015) instead uses a *differentiable* attention built from a grid of Gaussian read/write kernels, and image-captioning attention (Xu et al., 2015) attends over feature locations with an axis-aligned Gaussian window. Region-proposal detection (Girshick et al., 2014) and learning to regress salient boxes (Erhan et al., 2014) use attention by an external or separately-trained mechanism.

There is one more body of knowledge that predates deep learning entirely and is worth keeping in view: image resampling in computer graphics (Foley et al., 1994). Graphics has long-established, well-understood machinery for warping an image to a geometric transformation — texture mapping with source/target coordinates and interpolation between pixels — developed for rendering with fixed, externally specified transforms.

## Baselines

**CNN with local max-pooling (LeCun et al., 1998).** Convolution + pointwise nonlinearity + small max-pooling, trained by backprop. Pooling over a 2×2 window provides local translation invariance; deeper stacks accumulate a little more.

**Data augmentation.** Train on synthetically transformed copies of each image.

**Transforming-parts / capsule models (Hinton, 1981; Hinton et al., 2011; Tieleman, 2014).** Represent objects as parts with explicit 2D affine transforms predicted by the network; compose transformed parts generatively. Trained with the transformations given as supervision (input or target).

**Group / filter-bank invariance (Gens & Domingos, 2014; Kanazawa et al., 2014; Sohn & Lee, 2012; Bruna & Mallat, 2013).** Hard-wire invariance to a chosen transformation group into the filters/architecture.

**Attention / glimpse models (Schmidhuber & Huber, 1991; Ba et al., 2015; Sermanet et al., 2014; Gregor et al., 2015; Xu et al., 2015; Girshick et al., 2014).** Select and process a sub-region. Hard-crop versions are non-differentiable and trained with REINFORCE (high variance); the differentiable version (DRAW) uses an axis-aligned grid of Gaussian kernels.

## Evaluation settings

The natural testbeds are supervised image tasks where nuisance spatial variation is large and uncorrelated with the label, all trainable with only the task label:

- **Distorted MNIST.** MNIST digits subjected to rotation (R), to rotation+translation+scale (RTS), to projective distortion (P), and to elastic warping (E), placed in larger (e.g. 42×42) canvases; a translated-and-cluttered variant (60×60 with random 6×6 digit-patch distractors). Metric: classification error. Baselines for comparison are fully-connected (FCN) and convolutional (CNN, with max-pooling) networks matched for parameter count and trained with identical SGD schedules and multinomial cross-entropy.
- **Street View House Numbers (Netzer et al., 2011).** ~200k real images, sequences of 1–5 digits, large variation in scale and arrangement. Standard protocol: 64×64 crops around the sequence, and a looser 128×128 crop; a multi-softmax character-sequence model (Goodfellow et al., 2013; Jaderberg et al., 2014). Metric: sequence recognition error.
- **CUB-200-2011 birds (Wah et al., 2011).** ~6k train / ~5.8k test images, 200 species, objects at varied scale and orientation, not tightly cropped, requiring fine texture/shape discrimination. Strong baseline: an Inception architecture with batch normalisation (Ioffe & Szegedy, 2015) pre-trained on ImageNet (Russakovsky et al., 2014) and fine-tuned. Trained with image class labels only. Metric: classification accuracy.
- **Co-localisation and MNIST addition** as semi-supervised / multi-object probes: localise a common object across an image set with no class/location labels (a triplet hinge loss in an embedding space), and predict the sum of two independently-transformed digits presented in separate channels.

## Code framework

The primitives that already exist: a deep-learning framework with autograd, convolution/linear layers, pooling, ReLU, SGD, and standard image-classification data loaders. The scaffold below is a generic classifier with one empty slot for the module to be designed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialReorganiser(nn.Module):
    """A self-contained, differentiable module that takes a feature map,
    decides (conditioned on the map itself) how to spatially reorganise it,
    and returns the reorganised map of the same channel count. End-to-end
    trainable by the downstream loss alone — no supervision of the geometry.
    To be designed."""
    def __init__(self, in_channels):
        super().__init__()
        # TODO: design the module.
        pass

    def forward(self, x):
        # TODO: take the feature map x and return a spatially reorganised
        #       feature map of the same channel count, differentiably.
        raise NotImplementedError


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.reorg = SpatialReorganiser(in_channels=1)  # the slot for the new module
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.reorg(x)                                   # spatially normalise first
        x = F.relu(F.max_pool2d(self.conv1(x), 2))          # then ordinary CNN
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, loader, optimizer):
    model.train()
    for data, target in loader:
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()       # gradient must reach inside SpatialReorganiser
        optimizer.step()
```
