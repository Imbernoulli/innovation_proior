Let me start from the thing that actually bugs me. I train a big convolutional classifier, it has far more parameters than the data can constrain, and it overfits unless I drown it in noise and regularization. For the fully connected part I know exactly what to reach for: dropout. Zero each unit independently with probability `1 - p`, divide the survivors by `p` so the expected activation is unchanged and inference can run the plain network, and the magic is that a single forward pass is really one sample from an exponential family of thinned sub-networks — averaging that ensemble at test time is what kills the co-adaptation between feature detectors. It works beautifully on dense layers. And yet when I look at how the strong recent convolutional architectures are actually trained, dropout has quietly disappeared from the convolutional layers. People keep it on the final fully connected head if at all. That is strange — the single most successful piece of injected noise we have, and it's been abandoned exactly in the part of the network that has the most spatial parameters to regularize. So either everyone is wrong, or dropout on a convolutional feature map doesn't do what I think it does. Let me figure out which.

Why might independent unit-dropout fail on a convolutional feature map specifically? The thing that's different about a conv map versus a dense layer is correlation. A dense layer's units are, by design, fairly decorrelated arbitrary projections. But a convolutional feature map is produced by sliding one kernel over a spatially smooth input — natural images are smooth, the receptive fields of neighboring output positions overlap heavily — so two adjacent activations in the same channel are looking at almost the same patch and they carry almost the same number. Their values are strongly correlated. Now run the dropout mechanism in my head on that. I zero one activation at position `(i, j)`. Its four neighbors are still alive, and they hold essentially the same information — the same edge, the same texture, the same piece of an object part. The next layer's convolution, sliding over this region, simply reads that information off the neighbors. I removed a unit and removed nothing: the information just flowed around the hole. So the "thinned sub-network" I sampled isn't actually thinned in any meaningful way — the path through the data is still open. That's the failure. Dropout's whole power comes from the fact that on a dense layer, dropping a unit genuinely cuts off the information that unit carried; on a correlated conv map, dropping a scalar cuts off nothing because the same signal is redundantly stored in the neighborhood. And indeed this matches what's been seen directly — put standard dropout before a convolution on correlated maps and it lengthens training without preventing overfitting; the consensus phrasing is that standard dropout simply fails there.

So the diagnosis is clear, and it tells me what a fix has to do: I can't remove information by deleting *one* point from a smooth field, because the field is redundant at the scale of a point. I have to remove information at the scale on which the field is *correlated*. If a unit and its neighbors all carry the same thing, then to actually delete that thing I have to delete the unit together with the neighborhood that backs it up. The natural object to remove is not a point but a contiguous region of the feature map. Drop a whole patch, and the information that lived in that patch has nowhere to leak to: there are no surviving correlated neighbors inside the hole to carry it onward. The network is then forced to go find the evidence somewhere else in the map. The noise has to be spatially contiguous rather than independent across scalars.

Before I build that, let me check what's already on the table that's structured, because if one of them already does this I shouldn't reinvent it. The clearest precedent is SpatialDropout from Tompson and colleagues — and notably it was invented for *exactly* this reason, the observation that on correlated convolutional maps standard dropout fails. Their fix: instead of `nfeats × H × W` independent Bernoulli trials, do only `nfeats` trials and drop an entire channel at a time, so a whole feature map is either all-zero or all-alive. That genuinely removes information — if a channel encodes "this kind of texture is present," zeroing the whole channel deletes that fact, and there are no surviving spatial neighbors within the channel to leak it. So SpatialDropout already grasps the principle: drop at the scale of correlation, not the scale of a scalar. But its granularity is the whole channel, all-or-nothing, and that's a blunt instrument. On a high-resolution feature map a single channel holds an enormous amount, and throwing all of it away in one trial is harsh — it has been seen to be too aggressive on large maps, and channel-level / max-activation drop schemes have even come out worse than plain dropout when batch normalization is in the mix. There's no dial between "one scalar" (dropout, removes nothing on conv maps) and "one entire channel" (SpatialDropout, removes a lot, sometimes too much, with no spatial control). I want the dial. I want to remove a *region* of a channel — bigger than a point so information can't leak around it, smaller than the whole channel so I can tune how much I remove and where.

The other precedent worth staring at is Cutout from DeVries and Taylor. They mask out one fixed-size square at a random location of the *input image*, and because that occluded square propagates through every downstream feature map, the removed object genuinely can't be recovered from context, so the network learns to use the whole image instead of a couple of key visual features. That is the right *shape* of idea — a contiguous square that deletes semantic content — but it lives only at the input, and only as a single region. Two of their findings are useful to me, though. They found the *size* of the masked square matters more than its shape, so they just use a square and tune the side length — that says I should parameterize the region by one scalar size and not fuss over geometry. And they found accuracy versus patch size is parabolic — too small does nothing, too big destroys the image, there's a sweet spot — which I should expect for my region size too. The thing Cutout doesn't give me is generality: I want this noise injectable at *any* convolutional layer, not just bolted onto the input. The correlation problem lives at every intermediate map, so the cure should be applicable at every intermediate map.

The shape that survives these comparisons is a dropout-like layer that, during training, zeros out one or more contiguous square regions of side `block_size`, leaves the tensor shape alone, is a no-op at inference, and applies the dropout-style rescale so the activation scale stays comparable. Two parameters fall out immediately: `block_size`, the side of the square I delete, and something that controls *how many* units I end up dropping. Let me think about how to actually sample the regions and how to set that second parameter, because that's where the real work is.

How do I sample contiguous squares to drop? The clean way that reuses everything I know about dropout is: don't sample the squares directly, sample their *centers*. Draw a sparse Bernoulli mask of "seeds" — call a seed a 1 where I'm going to start a dropped block — and then expand each seed into a `block_size × block_size` zero square centered on it. That's appealing because the seed-sampling step is just ordinary independent Bernoulli sampling, the thing I already do for dropout, and the "make it structured" part is a deterministic expansion afterward. Let me say a seed is sampled with probability `gamma`, where `gamma` is the per-position probability that a position becomes the center of a dropped block.

Now I hit the first real subtlety. If I place a block centered at a position too close to the edge of the feature map, the block runs off the map — a `7 × 7` block centered one pixel from the boundary would need rows that don't exist. I don't want partial blocks hanging off the edge; I want every dropped block to be a full square fully contained in the map (that keeps the drop size honest and the bookkeeping clean). So seeds may only be sampled in the interior region where a full `block_size` square fits — for a `feat_size × feat_size` map, the valid center positions form a `(feat_size - block_size + 1) × (feat_size - block_size + 1)` sub-grid. Outside that band, no seeds.

And now the second subtlety, the one I have to get exactly right: what should `gamma` be? In ordinary dropout I'd just say "drop each unit with probability `1 - keep_prob`," one knob with a clear meaning. I'd like to keep that knob — let me think in terms of `keep_prob`, the probability that a given unit survives, so people can carry over their dropout intuition — and then *derive* the seed probability `gamma` that realizes it. The two are not equal, because one seed doesn't drop one unit, it drops `block_size²` of them, and seeds only live in the shrunken valid region. Let me just count expectations.

I want, in expectation, the fraction of dropped units to equal `1 - keep_prob`. Over a `feat_size²`-unit map that's a target of `(1 - keep_prob) · feat_size²` dropped units. Where do those come from? The number of seeds is `gamma` times the number of valid center positions, so the expected seed count is `gamma · (feat_size - block_size + 1)²`. Each seed, if its block didn't overlap any other, deletes `block_size²` units. So the expected number of dropped units is approximately

  `E[dropped] ≈ gamma · (feat_size - block_size + 1)² · block_size²`.

Set that equal to the target and solve for `gamma`:

  `gamma · (feat_size - block_size + 1)² · block_size² = (1 - keep_prob) · feat_size²`,

  `gamma = (1 - keep_prob) / block_size² · feat_size² / (feat_size - block_size + 1)²`.

Let me read back what the two factors are doing, because each one is paying for one of the subtleties I just worried about. The `1 / block_size²` factor is there because one seed deletes `block_size²` units, so to hit a target *unit* drop rate I have to sample seeds `block_size²` times more sparsely — bigger blocks mean fewer seeds. The `feat_size² / (feat_size - block_size + 1)²` factor is there because I'm only allowed to sample seeds in the shrunken valid interior; since the seeds are crammed into a smaller area than the full map but must account for dropping units across the *whole* map, I have to scale their density *up* by the ratio of full area to valid area. Sanity-check the corner cases. If `block_size = 1`, the valid region is the whole map, the ratio is 1, and `gamma = 1 - keep_prob` — which is exactly the dropout drop rate, as it must be, because a `1 × 1` block is a single unit and this whole construction collapses back to plain dropout. Good, dropout is the `block_size = 1` special case. If `block_size` covers the entire feature map, then `(feat_size - block_size + 1)² = 1`, there's a single valid center, and a seed there zeros the whole map — that's SpatialDropout, dropping a whole channel. So the two existing methods sit at the two ends of my `block_size` dial, which is a good sign that I've found the right generalization rather than a third unrelated thing — there's a continuum and the prior art is its endpoints.

I should be honest with myself about one thing in that derivation: it's an approximation, and I know precisely where. I counted `block_size²` dropped units per seed *assuming blocks don't overlap*. But seeds are sampled independently, and two nearby seeds will produce overlapping blocks, so the union of dropped units is smaller than `seeds × block_size²` — I'm double-counting the overlaps. For a fixed `gamma`, the count therefore overestimates how many units will actually be dropped, and the realized `keep_prob` won't be exactly the nominal one. I could try to correct for overlap analytically, but that gets ugly fast and depends on the spatial arrangement. For the scale bookkeeping, though, there is a cleaner fix: use the mask I actually draw.

Here's the rescaling. Like dropout, after I zero out the blocks I must compensate so the activation scale going into the next layer stays comparable and inference can use the un-noised network. Dropout divides survivors by `keep_prob`. I'll do the analogous thing, but I'll use the *realized* mask rather than the nominal rate. After I build the final block-structured keep-mask `M` (1 where kept, 0 where dropped), I count how many ones it actually has, `count_ones(M)`, out of its total `count(M)` entries, and I multiply the masked activations by `count(M) / count_ones(M)`. That ratio is one over the *actual* fraction of surviving positions in this particular mask, so overlaps and boundary effects are handled by the mask I actually drew rather than by the approximate `gamma` calculation. The nominal `gamma` aims the drop rate; the realized-survival rescale, `numel / sum`, keeps the layer's scale tied to the actual mask.

Now, do I share one block mask across all channels, or sample a fresh mask per channel? Two activations at the same `(i, j)` in different channels encode different features, so dropping the *same* spatial block in every channel removes "everything at this location" uniformly, whereas an independent mask per channel removes a different region of each channel and forces each channel separately to find alternative evidence. The more direct formulation samples a mask for each feature map. A compact PyTorch layer can also draw one spatial mask per example and broadcast it over channels; that is cheaper and still tests the central mechanism, but I should not confuse it with the exact channel-wise formulation. I can write the derivation for the exact mask and make the implementation choice explicit when I mirror the compact layer.

Let me now worry about the dynamics over training, because there's a known trap. If I switch on a strong drop rate from step zero — a small `keep_prob` right at the start — I'm tearing big holes in feature maps that haven't learned anything yet. Early in training the features are near-random, the network is just finding its footing, and removing large contiguous chunks of a not-yet-meaningful representation just starves it of signal and hurts learning. The same problem showed up for scheduled path-dropping: a fixed drop probability didn't help the searched cells, but ramping the drop probability up *linearly over the course of training* did. The principle generalizes cleanly to my setting — start gentle, get harsh. So I won't hold `keep_prob` fixed. I'll start at `keep_prob = 1` (no dropping at all, let the network learn) and linearly decrease it toward the target over training, equivalently linearly increasing the number of dropped units from zero to the target. This does two things: it protects early learning, and — the part I care about more — it makes the method robust to the choice of target `keep_prob`, because even if the final target is a bit too aggressive, the network had a long gentle runway to build features before the noise got strong. A method whose one hyperparameter you don't have to tune carefully is worth a lot. So I implement the strength schedule as a linear ramp on `keep_prob` (hence on `drop_prob = 1 - keep_prob`, hence on `gamma`), stepped once per training step.

A couple of placement questions that the residual structure forces on me. In a residual block the output is the convolution branch plus the skip connection. If I only inject the structured noise on the convolution branch, the skip connection carries the *undropped* version of the information straight around my hole — the identity path leaks exactly what I tried to delete, the same leak-around-the-hole failure as scalar dropout but at the block level. So to actually remove the information I should drop on the skip connection too, not just the conv branch. And which layers? The shallow, high-resolution maps are big and not very semantic; the deep, low-resolution maps (the last residual groups) are the small, highly-semantic ones where a fixed `block_size` covers a larger *fraction* of the map and where deleting a contiguous region deletes a genuine semantic part. So applying it in the deeper groups, with a constant `block_size` regardless of resolution (which automatically means it bites harder on the smaller deep maps), is the natural choice. For setting the actual numbers I'd estimate a `keep_prob` somewhere in the high range — units mostly survive, say between `0.75` and `0.95` — and a `block_size` on the order of the small deep-map size (around `7` for a `7×7`-ish map), and let the schedule and the per-sample rescale absorb the imprecision.

Let me also pin the inference behavior, because it has to be consistent with the ensemble story. I do not apply any of this at test time — the layer is the identity at inference. That's the same interpretation as dropout: each training pass evaluates one sub-network from an exponential ensemble, and these particular sub-networks are the special ones that have whole contiguous parts of their feature maps removed; the plain inference network approximates the average prediction over that ensemble. Because training rescales by the realized survival fraction, the activation scale is already normalized during noisy passes, so the inference network needs no change.

Now let me make this concrete, because the sampling-and-expansion has a clean implementation I should nail down rather than hand-wave. I need to (1) sample a seed mask of 1s where blocks will be dropped, with each seed probability `gamma`, restricted to the valid interior; (2) expand each seed into a `block_size × block_size` square; (3) invert to get the keep-mask; (4) multiply; (5) rescale by `numel / sum`. The elegant trick for step (2) is max-pooling. If I have a tensor that is 1 at seed positions and 0 elsewhere, then a max-pool with kernel `block_size`, stride 1, and padding `block_size // 2` produces a tensor that is 1 at every position within `block_size/2` of any seed — i.e. it dilates each seed into a `block_size × block_size` square of 1s. So the dropped region is exactly `max_pool2d(seed_mask, block_size, stride=1, padding=block_size//2)`, and the keep-mask is `1 -` that. Max-pool *is* the block expansion. (One off-by-one: for even `block_size` the symmetric padding produces a mask one larger than wanted, so I trim the last row and column.) That's the whole forward pass, and it's all standard tensor ops — no explicit loops over seeds.

Let me also be careful that I sample the seed mask the right way. The exact tensor implementation samples random noise with the same shape as the activation tensor, gates it by the valid-center region so a center near the border cannot create a clipped block, and uses the full `gamma` with the boundary correction. When `block_size` equals the map width, the block expansion degenerates into a whole-feature-map decision, so a spatial reduce-min gives the SpatialDropout endpoint. The compact PyTorch implementation I want to mirror makes two simplifications: it samples a seed field of shape `(N, H, W)` and broadcasts the resulting keep-mask over channels, and it uses `gamma ≈ drop_prob / block_size²`, dropping the boundary ratio. The max-pool expansion and realized-survival rescale are the same core operations in both versions.

So here is the layer, filling the noise-process slot in the conv harness:

```python
import torch
import torch.nn.functional as F
from torch import nn


class DropBlock2D(nn.Module):
    """Structured dropout for a 4D conv activation tensor (N, C, H, W):
    zero out contiguous block_size x block_size squares, rescale by realized
    survival fraction, identity at inference."""

    def __init__(self, drop_prob, block_size):
        super().__init__()
        self.drop_prob = drop_prob          # 1 - keep_prob (target), scheduled externally
        self.block_size = block_size        # side of the square block to drop

    def forward(self, x):
        assert x.dim() == 4, "expected (N, C, H, W)"
        if not self.training or self.drop_prob == 0.0:
            return x                        # no dropping at inference / when off

        gamma = self._compute_gamma(x)      # per-position seed (block-center) probability

        # (1) sample seed mask: 1 where a dropped block will be centered
        mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
        mask = mask.to(x.device)

        # (2) expand each seed into a block_size x block_size square -> dropped region;
        #     invert to get the keep-mask
        block_mask = self._compute_block_mask(mask)

        # (4) apply the keep-mask, broadcast over channels in this compact variant
        out = x * block_mask[:, None, :, :]

        # (5) rescale by the REALIZED survival fraction (numel / surviving),
        #     matching the actual mask rather than the nominal drop rate
        out = out * block_mask.numel() / block_mask.sum()
        return out

    def _compute_block_mask(self, mask):
        # max-pool dilates each seed into a block_size square of 1s = the dropped region
        block_mask = F.max_pool2d(mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        if self.block_size % 2 == 0:        # even block: symmetric pad overshoots by one
            block_mask = block_mask[:, :, :-1, :-1]
        return 1 - block_mask.squeeze(1)    # keep-mask: 0 inside dropped blocks, 1 elsewhere

    def _compute_gamma(self, x):
        # seeds dropped per position; one seed deletes block_size^2 units, hence /block_size^2.
        # Exact form also multiplies by feat^2 / (feat - block_size + 1)^2 for the valid
        # interior; that ratio -> 1 for large maps, so it is dropped here.
        return self.drop_prob / (self.block_size ** 2)
```

And the strength schedule that ramps `drop_prob` from 0 up to the target over training, so early learning is protected and the result is robust to the target:

```python
import numpy as np
from torch import nn


class LinearScheduler(nn.Module):
    """Linearly increase drop_prob from start_value to stop_value over nr_steps,
    advancing once per training step (start at 0 = no dropping early)."""

    def __init__(self, dropblock, start_value, stop_value, nr_steps):
        super().__init__()
        self.dropblock = dropblock
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]
        self.i += 1
```

Let me trace the whole causal chain back to make sure it holds together. I started from a concrete failure: dropout, the best noise I have, stops regularizing once it's inside the convolutional layers. Diagnosing it, the cause is that conv activations are spatially correlated, so deleting one unit deletes nothing: its neighbors carry the same information onward, and the sampled sub-network isn't really thinned. That forced the realization that I must remove information at the *scale of correlation*, which means dropping a contiguous region, not a point. SpatialDropout already does this at the coarsest scale, a whole channel, but offers no dial between a scalar and a channel and is too harsh on large maps; Cutout does it as a single square but only on the input. Combining them gives the target: drop `block_size × block_size` squares on convolutional feature maps, with `block_size` the dial whose endpoints are exactly dropout (`block_size = 1`) and SpatialDropout (`block_size =` full map). To realize a target `keep_prob`, I sample block *centers* as a Bernoulli seed field and expand each seed; max-pool does the expansion. The exact count gives `gamma = (1 - keep_prob)/block_size² · feat_size²/(feat_size - block_size + 1)²`: the `1/block_size²` appears because each seed deletes `block_size²` units, and the area ratio appears because seeds live only in the valid interior. That count is approximate because blocks overlap, so the forward pass normalizes by the *realized* survival fraction `numel/sum` and lets inference run the plain network, consistent with the ensemble-of-thinned-networks reading. Finally, a fixed strong drop rate hurts early learning, so I linearly ramp `keep_prob` from 1 down to the target, which both protects the start and makes the method robust to the target value; and because the skip connection would otherwise leak the dropped information around the block, I apply it on the skip too, in the deeper, more semantic groups, with a constant `block_size`.
