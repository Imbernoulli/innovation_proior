The cheap baseline gets me a dense prediction but a blurry one. I take a classification backbone, strip its global pooling and fully-connected head, read off the deepest feature map, put a $1\times1$ conv on it to turn the $2048$ channels into $19$ class scores, and bilinearly upsample to full resolution. The features are semantically strong, but the prediction is *born* on a heavily downsampled grid — for a $512\times1024$ crop the class scores live on a grid coarser than the objects I care about — and bilinear interpolation, a smooth spatial average, cannot put an edge back where pooling erased it. So the segmentation is semantically right but spatially mushy: boundaries are soft, thin things like poles and a pedestrian's legs wash out, and small distant objects are gone, because they never survived to the coarse grid in a recoverable form. The root cause is that I asked one feature map to be two incompatible things. The *deep* map holds the semantics but threw away resolution; the *shallow* maps still have near-full resolution but only see edges and textures. The information about *where* a boundary sits never went into the deep map — it is still sitting in the shallow maps I am ignoring.

I propose **U-Net**: a symmetric *encoder–decoder with skip connections at every resolution level* that brings the deep map's semantics and the shallow maps' resolution back together. The encoder is a contracting path — blocks of two $3\times3$ convolutions each, with a $2\times$ downsample between them, so after stage $i$ the feature map is at $1/2^i$ resolution and $2^i\!\cdot\!C$ channels — and I save the output of every encoder stage. Those saved maps are the *skip* tensors. The decoder is a mirror-image expanding path: at each level I take the lower-resolution decoder feature, upsample it $2\times$, concatenate it with the saved encoder map at this same resolution, and fuse the two with a couple of $3\times3$ convolutions. The upsampled decoder feature supplies the *semantics* — it descends from the deepest, most context-aware part of the net — and the concatenated encoder skip supplies the *high-frequency spatial detail* at this exact resolution. Stacking these levels until I am back at (near) full resolution, a final $1\times1$ conv maps to class scores. The shape is a "U": contract, then expand, with horizontal wires across the U joining equal-resolution levels.

Two design choices carry the method, and the obvious alternatives are worse for concrete reasons. First, I fuse the skip by *concatenation* rather than addition: the two tensors play different roles — coarse-semantic versus fine-spatial — and I want the fusing convs to *learn* how to combine them channel by channel, not be forced to add them as if they were the same kind of signal. Second, I inject a skip at *every* resolution rather than once at the end. Localization error compounds down the expanding path: if I only injected detail once, at full resolution, the intermediate decoder levels would still be reconstructing structure from a blurred starting point and the final convs could not fully repair it. Reinjecting the matching-resolution encoder map at *each* upsampling step means every decoder level starts its fusion already aligned to the true spatial structure at that scale, so the error does not accumulate. This is what forces the decoder to be symmetric to the encoder — the same number of resolution levels going up as came down — so every skip has an exact-resolution partner to concatenate with. The concatenation is also why the conv block that follows takes $2\times\text{skip\_channels}$ inputs: it sees the skip and the channel-matched upsampled feature side by side. The upsampling operator itself is a small choice — the cheapest is bilinear interpolation, but a learned `InterpConv` (interpolate, then a $1\times1$ conv) can sharpen as it expands rather than merely smooth, so each decoder block is: upsample the incoming low-res feature to the skip's resolution, concatenate $[\text{skip};\text{upsampled}]$, then two $3\times3$ conv+norm+activation to fuse. The one risk I accept is that the skips are *shallow* features — low-level edge/texture responses with weak semantics — so if the fusing convs lean on them too hard the decoder could sharpen crisp but mislabeled boundaries; the two-conv fusion at each level, with the deep semantic feature concatenated alongside, is what keeps the labels honest while the skip supplies the geometry.

```python
class UpConvBlock(nn.Module):
    """Upsample the deep feature, concat the matching-resolution skip, fuse."""
    def __init__(self, conv_block, in_channels, skip_channels, out_channels,
                 num_convs=2, upsample_cfg=dict(type='InterpConv')):
        super().__init__()
        self.conv_block = conv_block(in_channels=2 * skip_channels,
                                     out_channels=out_channels,
                                     num_convs=num_convs)
        self.upsample = build_upsample_layer(cfg=upsample_cfg,
                                             in_channels=in_channels,
                                             out_channels=skip_channels)

    def forward(self, skip, x):
        x = self.upsample(x)               # deep feature -> skip resolution
        out = torch.cat([skip, x], dim=1)  # [skip ; upsampled]
        out = self.conv_block(out)
        return out

class UNet(nn.Module):
    def forward(self, x):
        enc_outs = []
        for enc in self.encoder:                       # contracting path
            x = enc(x)
            enc_outs.append(x)                         # save skip per level
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):   # expanding path
            x = self.decoder[i](enc_outs[i], x)        # upsample + fuse skip
            dec_outs.append(x)
        return dec_outs
```

Each encoder/decoder block is a `BasicConvBlock` of two $3\times3$ conv+BN+ReLU; the upsample is a learned `InterpConv` (interpolate-then-$1\times1$-conv). Output stride is set by how far the U descends and climbs back; the final decoder feature is at full resolution and goes through the $1\times1$ `conv_seg`.
