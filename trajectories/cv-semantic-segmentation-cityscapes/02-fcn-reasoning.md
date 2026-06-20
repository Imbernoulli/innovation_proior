69.10. The encoder-decoder with skips works — the prediction is dense and the boundaries are no longer
mush. But look at what that encoder *is*: a stack of plain two-conv blocks trained from scratch on a
few thousand Cityscapes images. It's a perfectly reasonable feature extractor, but it has never seen
ImageNet. The whole rest of computer vision has converged on one fact — that a deep residual network
pretrained on a million labeled images learns features so good that almost every downstream task does
better by starting from them than by training a bespoke encoder from scratch. My U-shaped net is
leaving that on the table. Its semantics come entirely from the Cityscapes training set, which is small
and narrow; a pretrained ResNet-50 brings a far richer notion of "what objects look like" before it
sees a single street scene.

So the question is whether I can keep the part of the last rung that worked — a fully-convolutional
net that produces a dense per-pixel prediction, no fully-connected bottleneck collapsing the spatial
grid — but mount it on a *pretrained* deep backbone instead of a from-scratch encoder. The tension is
the same one I keep running into: the pretrained classification backbone is built to *shrink*
resolution. Its deepest stage is at output stride 32. If I just read that off and upsample, I'm back
to the blurry baseline I started two rungs ago, only with better features.

There are two distinct levers here and I should separate them. One is *getting the resolution up
inside the backbone*, before the head ever runs. The other is *what the head does* with the resulting
feature map. Take them in turn.

For the first: I don't have to accept output stride 32. The downsampling in the last backbone stages
comes from stride-2 convolutions. If I replace those strides with *dilation* — spread the kernel taps
apart by the same factor instead of striding — each unit still sees the same wide receptive field, the
pretrained weights still apply (the kernel is the same size, I've only changed the spacing of where it
samples), but the feature map stops shrinking. Dilate the last two stages and the backbone now emits
at output stride 8 instead of 32. That's a 4× finer grid for free, in the sense that I keep the
pretrained weights and add no parameters — I've only paid in compute, because the convs now run on a
larger spatial map. This is the R-50-D8 backbone: ResNet-50 with `strides=(1, 2, 1, 1)` and
`dilations=(1, 1, 2, 4)`, output stride 8.

Now the head. At output stride 8 the feature map is rich (2048 channels of pretrained semantics) and
reasonably fine. I want to turn those 2048 channels into 19 class scores per location, and I want the
turning itself to be *learned* rather than a fixed 1×1 projection, because a couple of 3×3 convs over
the deep map can do local smoothing and refinement — clean up the per-pixel decision using the
neighborhood — that a pointwise 1×1 cannot. So the head is: a few 3×3 conv+norm+ReLU layers over the
backbone's deep feature, then the 1×1 classifier, then bilinear upsample by 8 to full resolution. The
whole thing is convolutional end to end — every layer is a convolution, the spatial grid is never
collapsed — so it accepts any input size and emits a correspondingly-sized dense map. The "upsampling"
back to full resolution is itself just an in-network resize the loss is computed through, so the net
learns dense prediction directly against the dense labels.

One more refinement that the last rung taught me. Reading off only the very deepest map throws away the
slightly-shallower stage, which has weaker semantics but is a touch sharper. I don't need the full
U-Net machinery here — at output stride 8 the deep map is already fairly fine — but I can cheaply fold
the head's input back in: run the 3×3 convs, then concatenate their output with the head's *input*
feature before the final classifier conv. That's a single skip, a "concat the input" option, letting
the classifier see both the refined features and the original deep features side by side. It's the same
idea as the U-Net skip — give the final decision access to a less-processed, more-faithful copy of the
features — but in miniature, because the dilated backbone has already kept the resolution I need.

Let me write the head. A couple of 3×3 conv blocks, an optional concat of the input, then the 1×1
classifier:

```python
class FCNHead(BaseDecodeHead):
    def __init__(self, num_convs=2, kernel_size=3, concat_input=True,
                 dilation=1, **kwargs):
        super().__init__(**kwargs)
        conv_padding = (kernel_size // 2) * dilation
        convs = [ConvModule(self.in_channels, self.channels, kernel_size,
                            padding=conv_padding, dilation=dilation,
                            norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)]
        for _ in range(num_convs - 1):
            convs.append(ConvModule(self.channels, self.channels, kernel_size,
                                    padding=conv_padding, dilation=dilation,
                                    norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
        self.convs = nn.Sequential(*convs)
        if concat_input:
            self.conv_cat = ConvModule(self.in_channels + self.channels,
                                       self.channels, kernel_size,
                                       padding=kernel_size // 2,
                                       norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def _forward_feature(self, inputs):
        x = self._transform_inputs(inputs)   # the backbone's deep (OS=8) feature
        feats = self.convs(x)                # learned local refinement
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))  # skip: refined + raw
        return feats

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)          # 1x1 -> 19 classes, then upsample x8
```

The prediction, against 69.10. Two things should move the number up at once: the backbone is now a
pretrained ResNet-50 rather than a from-scratch encoder, so the *what* is far stronger; and dilation
keeps it at output stride 8, so the *where* is decent without any decoder. The bet is that strong
pretrained features at a moderately-fine resolution, with a light learned head, clear the U-Net's mIoU.
The risk is the flip side of dropping the U-Net's full decoder: at output stride 8 the prediction is
still 8× coarser than the labels, so the finest boundaries will be softer than the U-Net's
full-resolution decoder produced — I'm trading some of the U-Net's localization for the pretrained
backbone's much stronger semantics, and betting the semantics win on the class-averaged metric. The
concrete change is the dilated pretrained backbone plus the fully-convolutional refine-and-classify
head; the full head is in the answer.
