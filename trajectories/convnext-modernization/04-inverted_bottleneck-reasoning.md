Both predictions from the last rung landed. Depthwise convolution alone cost
1.23 points (79.51 to 78.28) while nearly halving FLOPs (4.42G to 2.35G) —
exactly the capacity-loss-for-compute-savings trade I expected once
cross-channel mixing was stripped out of the spatial layer. Widening to 96
channels more than recovered it, 78.28 to 80.50, confirming that the freed
compute, reinvested as width, restores and exceeds the lost capacity. But I
should look at the FLOPs line honestly before treating that as an
unqualified win: 5.27G, well above the 4.5G Swin-T reference I've been
tracking and above every prior point on this ladder. The accuracy gain is
real, but I've been spending FLOPs to get it, and the budget discipline
that's held since rung 2 has slipped. Before doing anything else structural,
I want a move that recovers some of that FLOPs overshoot without giving back
the accuracy — and I think there's a design change sitting right in the
current block that does exactly that, for reasons that have nothing to do
with wanting to save compute per se.

Look at where the block currently spends its 1x1 convolutions. The
bottleneck pattern — inherited unchanged from vanilla ResNet all the way
through this rewrite — is wide-at-the-shortcut, narrow-in-the-middle: the
block's input and output channel count is some value C, the residual branch
first *reduces* down to a smaller planes count with a 1x1 conv, does its
spatial mixing at that reduced width, then *expands* back up to C with a
second 1x1 conv before the shortcut addition. That means every downsampling
block's shortcut path — the 1x1 conv with stride 2 that reshapes the
identity when spatial resolution and channel count both change between
stages — is operating at the *full*, wide channel count C, on both ends,
since the shortcut has to match the block's wide input and wide output
directly. That's the single most expensive 1x1 conv in the network at each
stage transition, and it exists purely because of which end of the block is
wide.

There's a documented alternative pattern for exactly this arrangement, used
extensively for efficiency-focused ConvNets: invert it. Instead of
wide-narrow-wide, go narrow-wide-narrow — expand up from the block's input
width with the first 1x1 conv, do the spatial mixing (now the depthwise
conv, following rung 3) at the *expanded* width, then project back down to
the block's narrow input/output width with the second 1x1 conv. The
motivating precedent for this shape is efficiency-oriented mobile
architectures that use exactly this ordering with a roughly 4x-wide "expand"
factor and prove it can match or beat the standard bottleneck at
substantially lower cost. But there's a second, independent reason to like
this shape specifically now, not just an efficiency argument imported from
mobile networks: it's also the shape of the sublayer inside a Transformer
block. A Transformer feedforward sublayer takes the model's narrow residual-stream
width, projects up to four times that width with one linear layer,
applies a nonlinearity, and projects back down to the narrow width with a
second linear layer — narrow, expand 4x, narrow, precisely the inverted
shape. Two lines of reasoning, mobile-efficiency and Transformer-block
structure, converge on the same block shape, which is a stronger signal
than either alone; if this shape happened to only exist in mobile networks
I might read it as ConvNet-specific over-optimization for a constraint
(inference latency) I don't actually care about here, and if it only
existed in Transformers I might read it as purely a Transformer sequence-
model convention with no reason to transfer. That it independently shows up
under two different pressures — hardware efficiency and whatever makes
attention-based sequence models work well — is why I'm willing to adopt it
now rather than treat it as an arbitrary reshuffling.

Concretely: keep the depthwise conv from rung 3, but move it inside an
inverted arrangement. The block's boundary width (input, output, and what
the shortcut connects) stays at the current bottleneck's "planes" count —
narrower than the expanded middle. The first 1x1 conv expands from that
narrow width up by a factor of 4 (matching the Transformer MLP's expansion
ratio, since I have no separate reason yet to pick a different multiplier).
The depthwise conv operates at this wide, expanded width. The second 1x1
conv projects back down to the narrow boundary width. Crucially, this
changes what the *downsampling shortcut* has to do: previously it connected
wide-to-wide (full channel count on both sides, since wide was the block
boundary); now the block boundary is narrow, so the shortcut's 1x1 conv
connects narrow-to-narrow instead, at the fraction of the cost.

What do I expect this to do to FLOPs and accuracy separately, since they
should move somewhat independently here? On FLOPs: two forces pull opposite
directions and I don't have a confident prior on which wins. Working
against savings: the depthwise conv itself, which now operates at 4x the
channel count it used to (the expanded middle width, not the narrow
boundary width), and depthwise conv FLOPs scale linearly with channel
count, so that layer alone should get meaningfully more expensive. Working
toward savings: every downsampling block's shortcut 1x1 conv, which used to
connect at the wide boundary and now connects at the narrow one, and there
are several such shortcuts (one per stage transition) each processing every
spatial position in that stage. My directional guess is that the shortcut
savings should dominate the depthwise cost increase, since the shortcut
convs process full spatial resolution while the depthwise conv's added cost
is confined to a single already-cheap operation, and the FLOPs total should
come down from the 5.27G I just measured, hopefully back toward the ~4.5G
band, but I'm not confident of the exact landing point without running it.
On accuracy: this is a pure rearrangement of *where* the same operations
sit relative to the shortcut, not a change to what operations exist or how
much total capacity the block has (still one depthwise conv, still two 1x1
convs, still a roughly 4x expansion somewhere in the middle) — so I don't
expect a large swing either direction. If anything I'd lean toward a small
positive: the shortcut path, now narrower, is a slightly "cleaner" identity
connection with less to project, which is generally considered favorable
for residual learning, but I hold this loosely since it's a secondary
effect at best.

So the test: apply the inverted-bottleneck rearrangement on top of the
rung-3 configuration (depthwise conv, width 96, stage ratio (3,3,9,3),
patchify stem), same frozen recipe, and read off top-1 accuracy and GFLOPs.
The number that matters most for judging whether this rung was worth doing
is FLOPs relative to 5.27G — I'm looking for a real reduction toward the
Swin-T-matched budget without giving back the 80.50% accuracy point, which
would mean the rearrangement bought back budget discipline for free.
