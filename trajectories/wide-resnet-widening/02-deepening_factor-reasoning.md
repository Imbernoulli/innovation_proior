The block-type sweep landed almost exactly where I predicted: close together, `B(3,3)` on top by a
small margin (5.73%), with `B(3,1)` and `B(3,1,3)` right behind it (5.78% each) at meaningfully fewer
parameters and, in `B(3,1,3)`'s case, less wall-clock per epoch (59.9s versus 67.5s for `B(3,3)`). The
worse performers — `B(1,3,1)` at 6.06%, and especially `B(1,3)` at 6.42%, the single clearly-separated
result in the table — share a structural feature: both put a `1x1` convolution first, immediately
after the shortcut split, so the very first operation on the branch cannot mix spatial information at
all. That's a plausible enough story for why they trail, but it's a small, close table, and my
decision rule from before this experiment was explicit: when results cluster tightly, don't chase a
difference that might not be real, prefer the cheapest of the near-ties. `B(3,1)` and `B(3,1,3)` are
both legitimate candidates on that rule. But I'm going to fix the block at `B(3,3)` anyway, for a
reason external to this table: I already know I'm about to spend the next several rungs almost
entirely on the width axis, running dozens of configurations at various `k` and depth. Every one of
those configurations needs a block, and if I pick `B(3,1)` now to save a small, possibly-noise-level
margin of parameters, I lose the ability to compare cleanly against every published thin-residual
number I have (all built on the two-`3x3` convention) for the rest of this process. `B(3,3)` costing
7-8% more parameters than its closest rivals, when I'm about to scale parameters by factors of 10-60x
on the width axis, is not the trade I want to be careful about. So: block fixed at `B(3,3)`, and I
stop spending experiments on the internal kernel pattern.

That closes half of the original question — the *pattern* of convolutions inside a block doesn't move
the needle much. The other half is still open: does the *count* of convolutions per block matter, at
fixed total budget? This is a genuinely different question from the one I just answered. Block-type
compared kernel-size lists at roughly matched depth-and-parameter pairs chosen ad hoc per variant; it
never isolated "more layers per residual unit, fewer residual units" as a single controlled axis. I
want to do that now, cleanly: fix the block to two-`3x3`-equivalent convolutions per unit as the
baseline (`l=2`), and ask what happens if I deepen each individual block — pack more sequential
convolutions inside one residual unit — while shrinking the number of units so total convolution count
and total parameter count both stay fixed. Call the number of convolutions per block `l`. The current
default has `l=2`.

There's a real tension here, not a foregone conclusion. The case for larger `l`: if a block is the
unit that's supposed to learn something, a deeper block should be able to learn a strictly more
expressive per-unit transformation than a shallow one — two convolutions can express less than three
or four stacked convolutions at the same total parameter count spent within one unit, in the same way
a deeper plain network can express more than a shallower one at matched width. Pushed to its logical
end, this argument says: since I'm holding total depth (total conv count) fixed either way, I should
just always prefer fewer, deeper blocks, because each unit gets to do more work before handing off.

The case against is exactly the machinery this whole design process starts from: the identity
shortcut. Every residual unit is also a shortcut — an unimpeded path for both the forward activation
and the backward gradient. Stochastic depth's result (competitive accuracy even when whole blocks are
randomly dropped during training) is direct evidence that these shortcuts are doing real optimization
work, independent of whatever the residual branch itself computes; the more of them a network has, the
more places gradient has an easy path and the more places training has a chance to route around a
still-poorly-conditioned branch. Fixed total convolution count and fixed total parameter count is
exactly the setting where the two effects trade off cleanly against each other: raising `l` from 2 to
3 or 4 does not add capacity for free — every convolution moved inside a block is a convolution *not*
spent instantiating another block, so `d`, the number of blocks (equivalently the number of shortcut
connections), must fall as `l` rises to keep the total fixed. So the real question is not "do deeper
blocks help" in isolation — it's whether the per-unit expressiveness gained by increasing `l` is worth
more than the shortcut density lost by decreasing `d` correspondingly. I genuinely don't know which
side wins without measuring it; the stochastic-depth evidence says shortcuts matter a lot, but it
doesn't say by how much relative to a modest capacity increase inside each remaining block, and I
don't want to assume the answer just because the shortcut argument sounds appealing on paper.

I also want to check `l=1` — a block with a single convolution, at the same fixed total budget, which
means the most blocks (most shortcuts) of any configuration in the sweep, and the least
per-unit expressiveness. If the shortcut-density story is right, `l=1` should sit at one extreme of
whatever pattern emerges; if it's simply too weak per-unit to represent anything useful regardless of
how many units there are, that would be informative too, and distinguishing those two failure modes is
part of why I'm running the full range rather than just comparing `l=2` against one alternative.

To isolate `l` from every other confound, I'll fix the network to `k=2`, `3x3` convolutions
throughout, and pick one total parameter budget (`~2.2M`, `WRN-40-2`'s own budget under `l=2`) that
every `l` value has to hit by adjusting block count `d` accordingly — so `l=1`, `l=2` (the current
default, included as its own point in the sweep rather than assumed), `l=3`, and `l=4` all land at the
same total convolutions and the same rough parameter count, with only the internal grouping into units
differing. Same CIFAR-10 protocol, median test error over 5 runs, as before. If capacity-per-unit
dominates, error should fall monotonically as `l` rises from 1 to 4. If shortcut density dominates,
error should rise past `l=2`, or even monotonically rise across the whole range from `l=1`. Either
clean pattern tells me something the block-type sweep couldn't: not just which internal layout is
marginally better, but which of the two structural stories — more expressive units, or more numerous
shortcuts — is actually driving what a residual network can learn at fixed budget. That's the
result I need before I can reason honestly about spending the *next* budget increase on width instead
of depth, which is the whole point of this design process.
