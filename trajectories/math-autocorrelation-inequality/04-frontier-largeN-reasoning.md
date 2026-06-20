The hierarchical lifts kept paying — `0.8848` at `20` pieces, into the high `0.89`s at `500` — and the
gradient was still moving when I stopped at `500`. The feedback said the resolution was not yet exhausted,
so the natural next move is to lift once more, into the thousands of pieces, and spend a much longer, more
carefully annealed refinement there. The published step-function frontier — Boyer–Li's `575`-piece
`0.901564`, Jaech–Joseph's `539`-piece `~0.9016` — lives exactly in the range a few thousand well-optimized
pieces can reach, so that is the target for this endpoint rung: push the construction up to and, if I can,
through that band, knowing the absolute record (AlphaEvolve-V2's irregular `~50000`-piece `0.96102`) sits
far above and was bought with vastly more compute than I have here.

I take the optimized `500`-piece profile and upscale `×4` to `2000`. As before the upscale is free — same
function, same `R` — and as before the upscaled point is a degenerate plateau of flat blocks, so I kick it
with a small multiplicative perturbation to break the block symmetry and give the gradient traction. Then I
refine. The mechanics are the same `β`-annealed Adam gradient ascent as the previous rung, but the *budget
and the schedule* are where the endpoint differs, and getting them right is the whole game at this
resolution.

Three things change at `2000` pieces. First, the sheer length of the run. At `500` pieces a few thousand
Adam steps per pass was enough to settle; at `2000` the optimum is a finer, more irregular shape with many
more coordinates to coordinate, and the gradient keeps finding small improvements for tens of thousands of
steps. So I budget a long final pass — tens of thousands of iterations — and let it grind. The FFT
evaluator makes this affordable: each step is `O(N log N)`, so even `40000` steps at `2000` pieces is a
couple of minutes, not hours.

Second, the `β` schedule has to be pushed *much* sharper at the end. The softmax stand-in for `||f*f||_inf`
is only a faithful proxy for the true `max` when `β` is large relative to the spread of the node values; at
`2000` pieces with a tall spike the node values span a wide range, and a `β` that was sharp enough at `500`
is too soft here — it lets the surrogate's peak sit below the true peak, so the optimizer chases a slightly
wrong objective and the true `R` lags. So I anneal `β` up to several hundred times `N`, far sharper than at
the coarse levels, in the final passes, so the surrogate genuinely tracks the hard `max` and the optimum it
finds is the optimum of the real ratio.

Third, periodic kicks during the long run, not just at the start. A single kick at the upscale unsticks the
initial plateau, but over tens of thousands of steps the optimizer can settle into a shallow local basin
and stop improving. A small multiplicative kick every few thousand steps — gentle enough not to wreck the
shape, strong enough to jostle it out of a shallow trap — acts like a mild restart that keeps the long run
exploring. I shrink the kick as the run sharpens, so the late phase is pure refinement.

I run this as a short ladder of passes at `2000`: a first pass with a moderate `β` ceiling and a kick to
let the lifted shape reorganize, then a long pass with a high `β` ceiling and periodic kicks to grind out
the fine structure, then a final low-learning-rate pass with the sharpest `β` to polish. I keep the best
true `R` ever seen across all of it, because the surrogate and the true ratio diverge slightly and I want
the genuinely best vector, not the surrogate-best one.

What do I expect? The first reorganizing pass should clear the `500`-piece value comfortably — more
resolution always helped before — landing somewhere around `0.899`. The long sharp-`β` grind is where the
real endpoint number comes from: this is the phase that carves the irregular fine structure the published
constructions rely on, and I expect it to push into the low `0.90`s, into and ideally through the
`0.9016` band of the Boyer–Li / Jaech–Joseph step functions. I am honest that I will *not* reach
`0.96102`: that record is a `~50000`-piece deliberately irregular function found by an evolutionary search
spending orders of magnitude more compute, and the gap from the low `0.90`s to `0.96` is the part of the
problem that a single bounded gradient run on a few thousand pieces does not close. So the endpoint of this
ladder is the step-function frontier that careful gradient refinement on thousands of pieces can actually
reach — matching the best published step-function results — with the AlphaEvolve-V2 record standing above as
the still-open distance, exactly the way the maximal-determinant record stood above the entry-flip frontier
in the analogous combinatorial ladder. The honest measured number I report is the one the evaluator returns
on the returned `2000`-piece vector, and the remaining gap to `0.96102` is the measure of how much further
this open problem still has to go.
