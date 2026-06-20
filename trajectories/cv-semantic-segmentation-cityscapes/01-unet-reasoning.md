Let me start from where the cheap baseline stalls. I take a classification backbone, strip its global
pooling and fully-connected head, read off the deepest feature map, slap a 1×1 conv on it to turn the
2048 channels into 19 class scores, and bilinearly upsample to full resolution. It works, and the
features are semantically strong — but the prediction is *born* at a downsampled grid. By the time the
backbone has run, the spatial resolution has been pooled down by a large factor; for a 512×1024 crop
the class scores live on a grid coarser than the objects I care about. Then I blow that up by bilinear
interpolation, which is a smooth spatial average — it cannot put an edge back where pooling erased it.
So the segmentation comes out semantically right but spatially mushy: the boundaries are soft, thin
things like poles and the legs of a pedestrian wash out, and small distant objects are simply gone
because they never survived to the coarse grid in a recoverable form.

The root cause is that I asked one feature map to be two incompatible things at once. The *deep* map
is where the semantics live — "this region is a car, that is road" — but to get there the network
threw away the resolution. The *shallow* maps still have the resolution — they're near full size — but
they only see edges and textures, not classes. A single upsample of the deep map is me trying to
recover localization from a tensor that no longer contains it. The information about *where* the
boundary is hasn't gone into the deep map; it's still sitting in the shallow maps, which I'm ignoring.

So the question is how to bring the deep map's semantics and the shallow maps' resolution back
together. The obvious first idea: upsample the deep map step by step instead of in one bilinear jump,
and at each step *fuse in* the same-resolution shallow feature map from the backbone, the one that
still carries the sharp spatial detail at that scale. The backbone is already a sequence of stages at
1, 1/2, 1/4, 1/8, … resolution; that's a natural *encoder* that contracts resolution while it builds
up semantics. What I'm describing is a mirror-image *decoder* that expands resolution back up, and at
each expansion level reaches sideways to grab the encoder feature at the matching resolution.

Let me make that concrete. The encoder is the contracting path: blocks of two 3×3 convs each, with a
2× downsample between them, so after stage *i* the feature map is at 1/2ⁱ resolution and 2ⁱ·C
channels. I save the output of every encoder stage — those saved maps are the *skip* tensors. The
decoder is the expanding path: at each level I take the lower-resolution decoder feature, upsample it
2×, and concatenate it with the saved encoder map at this resolution, then run a couple of 3×3 convs
to fuse the two. The upsampled decoder feature supplies the *semantics* (it descends from the deepest,
most context-aware part of the net), and the concatenated encoder skip supplies the *high-frequency
spatial detail* at this exact resolution. Stack these levels until I'm back at (near) full resolution,
then a final 1×1 conv to class scores.

Why concatenation, and why a skip at *every* level rather than just one fusion at the end? Concatenation
rather than addition because the two tensors play different roles — coarse-semantic vs.
fine-spatial — and I want the fusing convs to *learn* how to combine them channel by channel, not be
forced to add them as if they were the same kind of signal. And a skip at every resolution because the
localization error compounds: if I only injected detail once, at full resolution, the intermediate
decoder levels would still be reconstructing structure from a blurred starting point and the final
convs couldn't fully repair it. Injecting the matching-resolution encoder map at *each* upsampling step
means every decoder level starts its fusion already aligned to the true spatial structure at that
scale, so error doesn't accumulate down the expanding path. The decoder ends up symmetric to the
encoder — same number of resolution levels going up as came down — which is what makes the skip at each
level have an exact-resolution partner to concatenate with. The shape is a "U": contract, then expand,
with horizontal wires across the U joining equal-resolution levels.

The upsampling operator inside the decoder is a small design choice. The cheapest is bilinear interp;
slightly richer is a learned upsample — a transposed convolution, or an interpolate-then-conv — that
can learn to sharpen as it expands rather than just smoothing. Either way each decoder block is:
upsample the incoming low-res feature to the skip's resolution, concatenate `[skip, upsampled]`, then
two 3×3 conv+norm+activation to fuse. The concatenation is why the conv block that follows takes
`2 × skip_channels` inputs: it sees the skip and the (channel-matched) upsampled feature side by side.

Let me write the decoder upsample-and-fuse block — the wire across the U:

```python
class UpConvBlock(nn.Module):
    def __init__(self, conv_block, in_channels, skip_channels, out_channels,
                 num_convs=2, upsample_cfg=dict(type='InterpConv')):
        super().__init__()
        # the fusing conv block consumes [skip ; upsampled] => 2*skip_channels in
        self.conv_block = conv_block(in_channels=2 * skip_channels,
                                     out_channels=out_channels,
                                     num_convs=num_convs)
        self.upsample = build_upsample_layer(cfg=upsample_cfg,
                                             in_channels=in_channels,
                                             out_channels=skip_channels)

    def forward(self, skip, x):
        x = self.upsample(x)             # bring deep feature up to skip's resolution
        out = torch.cat([skip, x], dim=1)  # fuse semantics (x) with detail (skip)
        out = self.conv_block(out)
        return out
```

and the whole U: run the encoder, stashing each stage's output, then walk the decoder back up,
consuming the stashed skips in reverse order.

```python
def forward(self, x):
    enc_outs = []
    for enc in self.encoder:          # contracting path
        x = enc(x)
        enc_outs.append(x)            # save skip at each resolution
    dec_outs = [x]
    for i in reversed(range(len(self.decoder))):   # expanding path
        x = self.decoder[i](enc_outs[i], x)        # upsample + fuse matching skip
        dec_outs.append(x)
    return dec_outs
```

What I expect: feeding the matching-resolution detail back in at every step should fix exactly the
failure I started from. The boundaries should sharpen, because the final fusion levels are working at
near-full resolution with the sharp encoder maps in hand rather than blowing up a coarse map; thin and
small structures should survive, because their spatial signature lives in the shallow skips and now
gets carried straight into the decoder instead of being discarded. The bet is that an encoder-decoder
with per-level skip connections clears the Cityscapes mIoU of the single-upsample baseline — that
giving the decoder the localization it was missing is worth more than the extra decoder compute. The
risk is the reverse of the baseline's: the skips are *shallow* features, low-level edge/texture
responses with weak semantics, so if the fusing convs lean on them too hard the decoder could sharpen
the wrong boundaries — crisp but mislabeled. The two-conv fusion at each level, with the deep semantic
feature concatenated alongside, is what's supposed to keep the labels honest while the skip supplies
the geometry. The concrete change is the symmetric encoder-decoder with a skip concatenation at every
resolution level; the full head is in the answer.
