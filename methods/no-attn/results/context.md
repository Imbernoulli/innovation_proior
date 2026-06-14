## Research question

The task is *dense* prediction: assign an output to **every** pixel of an image, with the
output map the same spatial size as the input, and with the assignment localized precisely
to the right pixel. Image segmentation is the canonical case (a class label per pixel), but
the same shape of problem covers any image-to-image map where the network must produce a
full-resolution field aligned to the input — depth, flow, or, as the prediction target,
a per-pixel continuous quantity.

The difficulty is a structural tension between two things the network must do at once.
To decide *what* is at a pixel, the network needs a large receptive field and many layers
of abstraction — it has to integrate a wide spatial context before "membrane vs. cell"
or "cat vs. background" is even decidable. The standard way to grow the receptive field
cheaply is pooling/striding, which shrinks the spatial grid. But to say *where* — to place
the label on exactly the right pixel and trace a thin boundary — the network needs high
spatial resolution, and pooling has thrown that resolution away. A deep classification
convnet, run as-is, collapses a whole image down to a coarse grid (or a single vector):
it answers *what* well and *where* not at all. So a solution must somehow recover full
input resolution and precise localization **without** giving up the deep, pooled,
large-receptive-field path that makes the *what* decidable in the first place.

A second, sharpening constraint comes from the application domain that motivates the work:
biomedical microscopy, where annotated images are extremely scarce — tens of images, not
thousands. A solution must train end-to-end from that little data without overfitting, and
must run fast enough to segment a large image in well under a second. Whatever recovers the
resolution cannot also demand a huge labeled corpus or a slow per-pixel inference loop.

## Background

By this time deep convolutional networks have overturned the state of the art in visual
recognition. Krizhevsky, Sutskever & Hinton (2012) trained an 8-layer, multi-million-
parameter net on ImageNet's million labeled images and won by a wide margin; VGG
(Simonyan & Zisserman, 2014) pushed the same recipe deeper, stacking many 3×3 convolutions
and showing that depth made from small filters works better than fewer large ones.
Convolutional nets themselves go back to LeCun et al. (1989). But all of these are
*classifiers*: image in, a single class label out. Their architecture is built to *destroy*
spatial information on purpose — pool it away until only a global summary remains — because
that is exactly right for "what is in this image" and exactly wrong for "what is at this
pixel."

The prevailing wisdom for getting per-pixel labels out of such a net was the **sliding
window**: to label a pixel, crop a patch centered on it and run the classifier on that
patch. Ciresan et al. (2012) did this for neuronal membranes in electron microscopy and
won the ISBI 2012 segmentation challenge. It has one structural virtue worth keeping in
mind — the number of training *patches* is enormous even when the number of training
*images* is tiny, which matters in the few-image regime. But it has two well-documented
failure modes. First, it is slow and massively redundant: the network is rerun from
scratch for every pixel, and neighboring patches overlap almost completely, so nearly all
computation is repeated. Second, and more fundamentally, the patch size sets a single knob
that trades localization against context: a larger patch admits more max-pooling and thus
more context but coarser localization, while a smaller patch localizes better but sees too
little context. The two cannot be had at once.

The spatial-information-destruction problem also has a known partial remedy on the
representation side. He et al. (2015) showed that a deep stack of alternating convolution
and ReLU layers — especially one with several parallel paths through it — needs careful
initialization or parts of it saturate or never activate; drawing the initial weights from
a Gaussian of standard deviation √(2/N), where N is the fan-in of a unit (for a 3×3
convolution over 64 input channels, N = 9·64 = 576), keeps each feature map at roughly
unit variance and lets such a network train. This is the practical precondition for going
deep with many paths.

On the data side, Dosovitskiy et al. (2014) established that aggressive data augmentation
can teach a network the invariances it would otherwise need labeled examples to learn —
directly relevant when only tens of annotated images exist and the dominant real-world
variation (in tissue, deformation) can be simulated cheaply.

A normalization that matters for the modern incarnation of this kind of network is group
normalization (Wu & He, 2018): unlike batch normalization, it computes statistics over
groups of channels within a single example and so is independent of batch size, which is
essential when the effective batch is very small.

## Baselines

**Sliding-window patch classifier (Ciresan et al., 2012).** For each pixel, classify a
patch centered on it with a convnet; assemble the per-pixel predictions into a map.
**Gap:** redundant and slow (one forward pass per pixel, overlapping patches recompute
almost everything), and the patch size couples localization to context so that improving
one degrades the other — there is no patch size that both sees wide context and localizes
to the pixel.

**Fully convolutional network, FCN (Long, Shelhamer & Darrell, 2014, arXiv:1411.4038).**
The pivotal reframing. Reinterpret a classification net as *fully convolutional* by
turning its fully-connected layers into 1×1 convolutions; then a single forward pass on an
arbitrary-size image produces a coarse spatial score map directly, with no per-patch
redundancy. To get back to full resolution, FCN upsamples *in-network* with "backwards
strided convolution" (deconvolution / transposed convolution) of output stride f, a
learnable layer initialized to bilinear interpolation and trained end-to-end through the
pixelwise loss. Because a deep net's final map is at a 32-pixel stride — too coarse for
fine boundaries — FCN adds a "skip" mechanism: it puts a 1×1 convolution on an earlier,
finer-stride pooling layer (pool4 at stride 16, pool3 at stride 8) to produce extra
per-pixel class scores, 2×-upsamples the coarser prediction, and **sums** the two score
maps (FCN-16s, then FCN-8s); summing was chosen because max fusion made learning unstable
through gradient switching. Finer layers, seeing fewer pixels, supply finer detail. FCN is
trained by fine-tuning a large ImageNet-pretrained classifier (VGG-16).
**Gaps (observed limitations):** the cross-layer fusion happens at only two or three
points and combines *thin, low-dimensional class-score maps by addition*, so very little
of the rich feature content from the high-resolution early layers actually reaches the
output; the upsampling side of the network is shallow — essentially upsample-and-add, with
almost no learned processing on the way back to full resolution — so the reconstructed
detail is limited; and the whole approach leans on a big pretrained classifier and a large
labeled dataset, neither of which exists for a thirty-image biomedical problem.

**Multi-layer feature classifiers (Seyedhosseini et al., 2013, cascaded hierarchical
models; Hariharan et al., 2014, Hypercolumns, arXiv:1411.5752).** Form a per-pixel feature
by stacking activations drawn from several layers of a convnet (coarse-and-deep together
with fine-and-shallow), and run a classifier on that stacked feature, so that good
localization and the use of context become possible simultaneously.
**Gap:** the recombination is a separate per-pixel classifier sitting on top of features
hand-assembled from several layers; the layer selection and the way the layers are stacked
are fixed by hand rather than learned, and the full-resolution output is not produced in one
fast forward pass of a single end-to-end-trained network.

## Evaluation settings

The natural yardsticks already in use for dense prediction at the time:

- **EM neuronal-structure segmentation (ISBI 2012 challenge).** 30 training images
  (512×512) from serial-section transmission electron microscopy of the Drosophila VNC,
  each with a full ground-truth segmentation of cells vs. membranes; the test set's masks
  are held out and scored by the organizers. Metrics: *warping error*, *Rand error*, and
  *pixel error*, computed by thresholding the predicted membrane probability map at
  several levels.
- **ISBI cell-tracking challenge (2014/2015), 2D transmitted-light datasets.** "PhC-U373"
  (glioblastoma-astrocytoma cells, phase contrast; ~35 partially annotated training
  images) and "DIC-HeLa" (HeLa cells on glass, differential interference contrast; ~20
  partially annotated training images). Metric: average **intersection over union (IoU)**.
- **Natural-image semantic segmentation, PASCAL VOC 2011.** Per-pixel multinomial logistic
  loss; metric is mean pixel **IoU** averaged over classes including background, with
  ambiguous pixels ignored. The standard benchmark FCN reports against.
- Protocol notes: training is by stochastic
  gradient descent; because unpadded ("valid") convolutions make the output smaller than
  the input by a fixed border, one favors large input tiles over large batches to use GPU
  memory, with a correspondingly high momentum; and for arbitrarily large images one tiles
  the input and mirror-pads the borders ("overlap-tile") since GPU memory caps the tile
  size.

## Code framework

A dense-prediction training harness already exists. There is a data pipeline that yields
input images and the matching per-pixel targets; a standard SGD/Adam-style optimizer; a
per-pixel loss (e.g. softmax cross-entropy for labels, or MSE for a continuous target);
and the generic neural-network primitives — convolution, pooling/striding, in-network
upsampling, normalization, nonlinearities — out of which a network is assembled. What is
*not* settled is the network that turns an input image into a same-size output field while
resolving the *what*-vs-*where* tension; that architecture is exactly the empty slot.

```python
import torch
import torch.nn as nn


class DenseImageModel(nn.Module):
    """Maps an input image (and, where the task needs it, an auxiliary scalar
    condition such as a time/level index) to a SAME-SIZE per-pixel output field.

    Built from the primitives that already exist: nn.Conv2d, pooling / strided
    conv for downsampling, in-network upsampling, a normalization, a
    nonlinearity. The internal arrangement of those primitives -- how context is
    built up and how full input resolution and precise localization are then
    recovered -- is what we have to design and is left empty here."""

    def __init__(self, in_channels, out_channels, base_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO: the architecture we will design goes here.
        pass

    def forward(self, x, cond=None):
        # x:    [B, in_channels, H, W]
        # cond: optional [B] auxiliary scalar per example (if the task supplies one)
        # returns: [B, out_channels, H, W]   -- same spatial size as x
        # TODO: run x through the architecture we will design.
        raise NotImplementedError


# the existing dense-prediction training loop the model plugs into
def train(model, loss_fn, data_loader, optimizer, cond_fn=None):
    for images, targets in data_loader:            # images, per-pixel targets
        optimizer.zero_grad()
        cond = cond_fn(images) if cond_fn is not None else None
        pred = model(images, cond)                 # [B, out_channels, H, W]
        loss = loss_fn(pred, targets)              # per-pixel loss, averaged
        loss.backward()
        optimizer.step()
```

The harness supplies images and same-size targets; `DenseImageModel` is where the
image-to-field architecture will live, and its `forward` must return a field at the input's
spatial size.
